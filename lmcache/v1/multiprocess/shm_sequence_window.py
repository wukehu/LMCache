"""Bounded acknowledged sequence window for a future SHM protocol revision.

Not wired into RPC. Completion alone cannot retire a record: the client must
acknowledge receiving the completion, preserving COMMIT retry after ACK loss.
"""
from dataclasses import dataclass
from threading import Lock
import uuid

class SequenceRejected(RuntimeError):
    pass

@dataclass(frozen=True)
class TransferTicket:
    epoch: str
    sequence: int
    nonce: str

@dataclass
class _Record:
    ticket: TransferTicket
    fingerprint: bytes
    state: str = 'active'

class ShmSequenceWindow:
    def __init__(self, capacity):
        if type(capacity) is not int or not 1 <= capacity <= 65536:
            raise ValueError('sequence window capacity must be in [1,65536]')
        self.epoch=uuid.uuid4().hex
        self.capacity=capacity
        self._retired_through=0
        self._records={}
        self._closed=False
        self._lock=Lock()

    def _validate(self, epoch, sequence):
        if epoch != self.epoch or self._closed:
            raise SequenceRejected('registration epoch is retired or different')
        if type(sequence) is not int or not 1 <= sequence <= (1 << 64)-1:
            raise SequenceRejected('invalid transfer sequence')

    def reserve(self, epoch, sequence, fingerprint):
        if type(fingerprint) is not bytes or len(fingerprint) != 32:
            raise ValueError('fingerprint must be an exact 32-byte digest')
        with self._lock:
            self._validate(epoch,sequence)
            if sequence <= self._retired_through:
                raise SequenceRejected('transfer sequence already retired')
            if sequence > self._retired_through+self.capacity:
                raise SequenceRejected('transfer sequence window exhausted')
            existing=self._records.get(sequence)
            if existing is not None:
                if existing.fingerprint != fingerprint:
                    raise SequenceRejected('transfer sequence changed content')
                return existing.ticket,False
            ticket=TransferTicket(self.epoch,sequence,uuid.uuid4().hex)
            self._records[sequence]=_Record(ticket,fingerprint)
            return ticket,True

    def commit(self, ticket):
        """Called only after the exact operation's ownership release succeeds."""
        with self._lock:
            self._validate(ticket.epoch,ticket.sequence)
            record=self._records.get(ticket.sequence)
            if record is None or record.ticket != ticket:
                raise SequenceRejected('stale or unknown completion ticket')
            if record.state == 'quarantined':
                raise SequenceRejected('completion outcome unknown')
            if record.state in ('committed','acknowledged'):
                return False
            record.state='committed'
            return True

    def quarantine(self,ticket):
        with self._lock:
            self._validate(ticket.epoch,ticket.sequence)
            record=self._records.get(ticket.sequence)
            if record is None or record.ticket != ticket:
                raise SequenceRejected('stale or unknown quarantine ticket')
            if record.state != 'active':
                raise SequenceRejected('cannot quarantine a confirmed completion')
            record.state='quarantined'

    def acknowledge(self,epoch,sequence):
        """Retire a contiguous prefix only after explicit client completion ACK.

        ACK replay below the watermark is harmless and needs no retained record.
        Unknown/pending/quarantined operations cannot be acknowledged or skipped.
        """
        with self._lock:
            self._validate(epoch,sequence)
            if sequence <= self._retired_through:
                return self._retired_through
            record=self._records.get(sequence)
            if record is None or record.state not in ('committed','acknowledged'):
                raise SequenceRejected('completion is not confirmed')
            record.state='acknowledged'
            while (next_record := self._records.get(self._retired_through+1)) is not None and next_record.state == 'acknowledged':
                del self._records[self._retired_through+1]
                self._retired_through+=1
            return self._retired_through

    def close(self):
        # Closing admission does not prove consumers stopped using SHM.
        with self._lock:
            self._closed=True

    def snapshot(self):
        with self._lock:
            return dict(epoch=self.epoch,capacity=self.capacity,retired_through=self._retired_through,
                        retained=len(self._records),closed=self._closed,
                        states={k:v.state for k,v in self._records.items()})
