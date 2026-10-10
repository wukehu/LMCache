"""Client-side retry state for negotiated sequenced SHM stores."""
from dataclasses import dataclass,field
from threading import Lock,RLock
import time
import hashlib,msgspec
from lmcache.v1.multiprocess.shm_wire import ShmTransferRef,CAPABILITY

STORE_FEATURE='shm_sequence_ack_store_v1'
ABORT_FEATURE='shm_sequence_abort_store_v1'

@dataclass
class _Pending:
    ref: ShmTransferRef
    fingerprint: bytes
    ticket: ShmTransferRef | None = None
    response: object = None
    committed: bool = False
    commit_attempted: bool = False
    aborting: bool = False
    aborted: bool = False
    recovery_attempts: int = 0
    recovery_after: float = 0
    recovery_error: str = ""
    lock: object = field(default_factory=RLock)

class ShmSequenceClient:
    def __init__(self,client,instance_id,timeout,identity,*,supports_abort=False):
        if identity.capability!=CAPABILITY or 'store' not in identity.operations:
            raise ValueError('server did not negotiate sequenced SHM stores')
        self.supports_abort=supports_abort
        self.client=client;self.instance_id=instance_id;self.timeout=timeout
        self.identity=identity;self._next_sequence=1;self._pending={};self._lock=Lock();self._recovery_cursor=0

    @staticmethod
    def _key(key):return key.request_id,key.lookup_id,key

    @staticmethod
    def _fingerprint(key):return hashlib.sha256(msgspec.msgpack.encode(key)).digest()

    def prepare(self,key,kind="store"):
        identity=(kind,self._key(key))
        if kind not in self.identity.operations:raise ValueError("SHM operation not negotiated")
        fingerprint=self._fingerprint(key)
        with self._lock:
            op=self._pending.get(identity)
            if op is None:
                if len(self._pending)>=self.identity.window_capacity:
                    raise RuntimeError('client SHM sequence window full')
                op=_Pending(ShmTransferRef(self.identity.epoch,self._next_sequence),fingerprint)
                self._pending[identity]=op;self._next_sequence+=1
            elif op.fingerprint!=fingerprint:
                raise ValueError('SHM retry changed request metadata')
        with op.lock:
            if op.aborting:raise RuntimeError('SHM cancellation started')
            if op.committed:
                if kind=='retrieve' and op.response is not None and not op.response.success:return op.response
                raise RuntimeError('cannot access after confirmed SHM commit')
            if op.commit_attempted:
                raise RuntimeError('cannot access after attempted SHM commit')
            if op.response is None:
                response=getattr(self.client,"shm_v2_prepare_"+kind)(key,self.instance_id,op.ref).result(timeout=self.timeout)
                ticket=ShmTransferRef(**response.context['shm_ticket'])
                ticket.require_completion_ticket()
                if (ticket.epoch,ticket.sequence)!=(op.ref.epoch,op.ref.sequence):
                    raise RuntimeError('server returned a different SHM operation')
                op.ticket=ticket;op.response=response
            return op.response

    def commit(self,key,kind="store",_timeout=None):
        timeout=self.timeout if _timeout is None else _timeout
        identity=(kind,self._key(key))
        with self._lock:op=self._pending.get(identity)
        if op is None:return False
        if op.fingerprint!=self._fingerprint(key):raise ValueError('SHM commit changed request metadata')
        with op.lock:
            if op.aborting:raise RuntimeError('SHM cancellation started')
            if op.ticket is None:return False
            if not op.committed:
                op.commit_attempted=True
                result=getattr(self.client,"shm_v2_commit_"+kind)(self.instance_id,op.ticket).result(timeout=timeout)
                if not result:return False
                op.committed=True
                if kind!='retrieve' or op.response.success:op.response=None
            # Preserve committed=True on ACK timeout: retry only ACK, never
            # re-issue COMMIT after the server may have retired its record.
            self.client.shm_v2_ack(self.instance_id,op.ticket).result(timeout=timeout)
            with self._lock:
                if self._pending.get(identity) is op:del self._pending[identity]
            return True

    def abort_store(self,key,*,access_complete,_timeout=None):
        if access_complete is not True:
            raise ValueError('abort requires confirmed stopped SHM access')
        if not self.supports_abort:
            raise RuntimeError('server did not negotiate STORE abort')
        timeout=self.timeout if _timeout is None else _timeout
        identity=('store',self._key(key))
        with self._lock:op=self._pending.get(identity)
        if op is None:return True
        if op.fingerprint!=self._fingerprint(key):raise ValueError('SHM abort changed request metadata')
        with op.lock:
            if op.commit_attempted or op.committed:
                raise RuntimeError('cannot abort an attempted COMMIT')
            op.aborting=True
            if op.ticket is None:
                # Recover the original receipt after PREPARE response loss.
                # The recovered descriptors are never exposed to the caller.
                response=self.client.shm_v2_prepare_store(key,self.instance_id,op.ref).result(timeout=timeout)
                ticket=ShmTransferRef(**response.context['shm_ticket'])
                ticket.require_completion_ticket()
                if (ticket.epoch,ticket.sequence)!=(op.ref.epoch,op.ref.sequence):
                    raise RuntimeError('server returned a different SHM operation')
                op.ticket=ticket
            if not op.aborted:
                result=self.client.shm_v2_abort_store(self.instance_id,op.ticket).result(timeout=timeout)
                if not result:return False
                op.aborted=True;op.response=None
            self.client.shm_v2_ack(self.instance_id,op.ticket).result(timeout=timeout)
            with self._lock:
                if self._pending.get(identity) is op:del self._pending[identity]
            return True

    def abandon_retrieve(self,key,*,access_complete,_timeout=None):
        if access_complete is not True:
            raise ValueError('read abandon requires confirmed stopped SHM access')
        if 'retrieve' not in self.identity.operations:
            raise RuntimeError('server did not negotiate sequenced reads')
        timeout=self.timeout if _timeout is None else _timeout
        identity=('retrieve',self._key(key))
        with self._lock:op=self._pending.get(identity)
        if op is None:return True
        if op.fingerprint!=self._fingerprint(key):raise ValueError('SHM read abandon changed metadata')
        with op.lock:
            op.aborting=True
            if op.ticket is None:
                response=self.client.shm_v2_prepare_retrieve(key,self.instance_id,op.ref).result(timeout=timeout)
                ticket=ShmTransferRef(**response.context['shm_ticket'])
                ticket.require_completion_ticket()
                if (ticket.epoch,ticket.sequence)!=(op.ref.epoch,op.ref.sequence):
                    raise RuntimeError('server returned a different SHM operation')
                op.ticket=ticket
            if not op.committed:
                op.commit_attempted=True
                if not self.client.shm_v2_commit_retrieve(self.instance_id,op.ticket).result(timeout=timeout):return False
                op.committed=True;op.response=None
            self.client.shm_v2_ack(self.instance_id,op.ticket).result(timeout=timeout)
            with self._lock:
                if self._pending.get(identity) is op:del self._pending[identity]
            return True

    def recover_pending(self,limit=4,timeout=.25):
        """Bounded scan; never acquire a foreground operation's busy lock."""
        if type(limit) is not int or not 1<=limit<=16 or not 0<timeout<=1:
            raise ValueError('invalid recovery budget')
        with self._lock:items=tuple(self._pending.items())
        if not items:return 0
        attempted=0
        start=self._recovery_cursor%len(items)
        for offset in range(min(16,len(items))):
            identity,op=items[(start+offset)%len(items)]
            self._recovery_cursor=start+offset+1
            if attempted>=limit:break
            if not op.lock.acquire(blocking=False):continue
            try:
                with self._lock:
                    if self._pending.get(identity) is not op:continue
                if not (op.aborting or op.commit_attempted) or op.recovery_attempts>=8 or time.monotonic()<op.recovery_after:continue
                attempted+=1;op.recovery_attempts+=1
                kind,key_identity=identity;key=key_identity[-1]
                try:
                    if op.aborting:
                        method=self.abort_store if kind=='store' else self.abandon_retrieve
                        done=method(key,access_complete=True,_timeout=timeout)
                    else:done=self.commit(key,kind=kind,_timeout=timeout)
                    op.recovery_error='' if done else 'completion not confirmed'
                except Exception as exc:
                    op.recovery_error=type(exc).__name__+': '+str(exc)[:192]
                op.recovery_after=time.monotonic()+min(5.,.25*2**(op.recovery_attempts-1))
            finally:op.lock.release()
        return attempted

    def recovery_snapshot(self):
        with self._lock:ops=tuple(self._pending.values())
        return dict(pending=len(ops),exhausted=sum(op.recovery_attempts>=8 for op in ops),attempts=sum(op.recovery_attempts for op in ops))

    def pending_count(self):
        with self._lock:return len(self._pending)
