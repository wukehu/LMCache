"""Server-side sequence/ACK binding for owned SHM stores."""
from dataclasses import dataclass,replace
from threading import Lock
import hashlib
import msgspec
from lmcache.v1.multiprocess.shm_sequence_window import ShmSequenceWindow,SequenceRejected,TransferTicket
from lmcache.v1.multiprocess.shm_wire import ShmRegistrationIdentity
from lmcache.v1.multiprocess.modules.server_transfer import ShmStoreAdmissionRejected

@dataclass
class _Operation:
    ticket: TransferTicket
    key: object
    instance_id: int
    state: str = 'preparing'
    response: object = None
    kind: str = 'store'
    complete: object = None

class SequencedShmStore:
    def __init__(self,strategy,capacity):
        self.strategy=strategy
        self.window=ShmSequenceWindow(min(capacity,65536))
        self._ops={}
        self._read_claims={}
        self._lock=Lock()

    def handshake(self):
        if self.strategy._retired:
            raise SequenceRejected('registration retired')
        return ShmRegistrationIdentity(self.window.epoch,self.window.capacity,operations=('store','retrieve'))

    def prepare(self,key,instance_id,ref,context,resolve):
        if ref.nonce:
            raise SequenceRejected('PREPARE cannot supply a completion nonce')
        fingerprint=hashlib.sha256(msgspec.msgpack.encode(['store',key])).digest()
        with self._lock:
            if self.strategy._retired:
                raise SequenceRejected('registration retired')
            ticket,new=self.window.reserve(ref.epoch,ref.sequence,fingerprint)
            if not new:
                op=self._ops[ref.sequence]
                if op.state=='prepared':return op.response
                if op.state!='admission_rejected':
                    raise SequenceRejected('store prepare is in flight, completed or quarantined')
                if op.instance_id!=instance_id:
                    raise SequenceRejected('store instance changed')
                op.state='preparing'
            # A private request identity separates different sequences without
            # changing the reusable ObjectKey resolved from the original key.
            private=replace(key,request_id='shm-v2:'+ref.epoch+':'+str(ref.sequence),lookup_id='')
            if new:
                op=_Operation(ticket,private,instance_id)
                self._ops[ref.sequence]=op
        try:
            response=self.strategy.prepare_store(private,instance_id,context,lambda _:resolve(key))
            response.context['shm_ticket']=dict(epoch=ticket.epoch,sequence=ticket.sequence,nonce=ticket.nonce)
            with self._lock:
                if self.strategy._retired:raise SequenceRejected('registration retired during prepare')
                op.response=response;op.state='prepared'
            return response
        except ShmStoreAdmissionRejected:
            # Exact admission boundary guarantees no receipt or SHM descriptor
            # escaped. Keep fingerprint/ticket and retry the same sequence.
            with self._lock:
                op.state='admission_rejected'
            raise
        except BaseException:
            with self._lock:
                op.state='quarantined';self.window.quarantine(ticket)
            raise

    def prepare_read(self,key,instance_id,ref,prepare,complete):
        if ref.nonce:
            raise SequenceRejected('PREPARE cannot supply a completion nonce')
        fingerprint=hashlib.sha256(msgspec.msgpack.encode(['retrieve',key])).digest()
        claim=(key.request_id,key.lookup_id,key)
        with self._lock:
            if self.strategy._retired:
                raise SequenceRejected('registration retired')
            claimed=self._read_claims.get(claim)
            if claimed is not None and claimed!=ref.sequence:
                raise SequenceRejected('read already bound to another sequence')
            ticket,new=self.window.reserve(ref.epoch,ref.sequence,fingerprint)
            if not new:
                op=self._ops[ref.sequence]
                if op.instance_id!=instance_id:
                    raise SequenceRejected('read instance changed')
                if op.state=='prepared':
                    if op.response.success:
                        checked=prepare()
                        if checked is not op.response or not checked.success:
                            raise SequenceRejected('read ownership no longer exportable')
                    return op.response
                raise SequenceRejected('read prepare is in flight, completed or quarantined')
            op=_Operation(ticket,key,instance_id,kind='retrieve',complete=complete)
            self._ops[ref.sequence]=op
            self._read_claims[claim]=ref.sequence
        try:
            response=prepare()
            response.context['shm_ticket']=dict(epoch=ticket.epoch,sequence=ticket.sequence,nonce=ticket.nonce)
            with self._lock:
                if self.strategy._retired:raise SequenceRejected('registration retired during prepare')
                op.response=response;op.state='prepared'
                if not response.success:op.complete=lambda:True
            return response
        except BaseException:
            with self._lock:
                op.state='quarantined';self.window.quarantine(ticket)
            raise

    def commit(self,instance_id,ref,kind='store'):

        ref.require_completion_ticket()
        with self._lock:
            if self.strategy._retired or ref.epoch!=self.window.epoch:
                raise SequenceRejected('registration retired or epoch mismatch')
            op=self._ops.get(ref.sequence)
            if op is None or op.ticket!=TransferTicket(ref.epoch,ref.sequence,ref.nonce) or op.instance_id!=instance_id:
                raise SequenceRejected('stale or unknown completion ticket')
            if op.kind!=kind:raise SequenceRejected('completion direction mismatch')
            if op.state=='committed':return True
            if op.state!='prepared':raise SequenceRejected('store completion not ready or unknown')
            op.state='committing'
        try:
            result=(self.strategy.commit_store(op.key,instance_id,b'',None,None)
                    if op.kind=='store' else op.complete())
            if not result:raise SequenceRejected('owned SHM store did not complete')
            with self._lock:
                self.window.commit(op.ticket)
                op.state='committed';op.response=None
            return True
        except BaseException:
            with self._lock:
                op.state='quarantined'
            raise

    def abort_store(self,instance_id,ref):
        # This ticketed request attests the client has stopped accessing slots.
        # It is never a timeout-based server reaper action.
        ref.require_completion_ticket()
        with self._lock:
            if self.strategy._retired or ref.epoch!=self.window.epoch:
                raise SequenceRejected('registration retired or epoch mismatch')
            op=self._ops.get(ref.sequence)
            if op is None or op.ticket!=TransferTicket(ref.epoch,ref.sequence,ref.nonce) or op.instance_id!=instance_id or op.kind!='store':
                raise SequenceRejected('invalid store abort ticket')
            if op.state=='aborted':return True
            if op.state!='prepared':raise SequenceRejected('store is not abortable')
            op.state='aborting'
        try:
            result=self.strategy.abort_store(op.key,instance_id)
            if not result:raise SequenceRejected('store abort was not confirmed')
            with self._lock:
                self.window.commit(op.ticket)
                op.state='aborted';op.response=None
            return True
        except BaseException:
            with self._lock:op.state='quarantined'
            raise

    def acknowledge(self,instance_id,ref):
        ref.require_completion_ticket()
        with self._lock:
            if self.strategy._retired:raise SequenceRejected('registration retired')
            op=self._ops.get(ref.sequence)
            if op is not None:
                if op.ticket!=TransferTicket(ref.epoch,ref.sequence,ref.nonce) or op.instance_id!=instance_id:
                    raise SequenceRejected('ACK ticket mismatch')
                if op.state not in ('committed','aborted'):raise SequenceRejected('ACK before confirmed completion')
            frontier=self.window.acknowledge(ref.epoch,ref.sequence)
            # If deletion fails, the watermark already forbids recreation.
            # Keep remaining bindings so an ACK retry can finish cleanup.
            with self.strategy._pending_lock:
                for sequence in tuple(self._ops):
                    if sequence>frontier:continue
                    finished=self._ops[sequence]
                    if finished.kind=='store':
                        key=self.strategy._transfer_key_factory(finished.key,finished.instance_id)
                        receipt=self.strategy._pending_writes.get(key)
                        if receipt is not None:
                            if receipt.owner is not self.strategy._registration_token or receipt.state!=finished.state:
                                raise SequenceRejected('cannot retire unconfirmed SHM ownership')
                            del self.strategy._pending_writes[key]
                    else:
                        if finished.key.lookup_id:
                            key=self.strategy._owned_transfer_key(finished.key,finished.instance_id)
                        else:
                            key=self.strategy._transfer_key_factory(finished.key,finished.instance_id)
                        receipt=self.strategy._pending_reads.get(key)
                        if receipt is not None:
                            if finished.key.lookup_id:
                                if not receipt.completion.done():
                                    raise SequenceRejected('cannot retire pending owned read')
                            elif receipt.owner is not self.strategy._registration_token or receipt.state!='committed':
                                raise SequenceRejected('cannot retire unconfirmed legacy read')
                            del self.strategy._pending_reads[key]
                        self._read_claims.pop((finished.key.request_id,finished.key.lookup_id,finished.key),None)
                    del self._ops[sequence]
            return frontier
