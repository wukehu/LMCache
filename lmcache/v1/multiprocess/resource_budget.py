"""Atomic per-request and server reservations; completion policy belongs to callers."""
from dataclasses import dataclass
from threading import Lock

class ResourceCapacityError(RuntimeError):
    pass

@dataclass(frozen=True)
class RequestLimits:
    """Explicit decoded-request caps; transport decoding remains separate."""
    max_tokens: int
    max_workers: int
    max_objects: int
    max_identifier_bytes: int

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (
            self.max_tokens, self.max_workers, self.max_objects,
            self.max_identifier_bytes,
        )):
            raise ValueError('request limits must be positive integers')

    def validate_key(self, key):
        if len(key.token_ids) > self.max_tokens:
            raise ResourceCapacityError('request token limit exceeded')
        workers = key.lookup_world_size
        if workers < 1 or workers > self.max_workers:
            raise ResourceCapacityError('request worker fan-out limit exceeded')
        # Check the explicit list even if worker_id reduces effective fan-out.
        if key.lookup_worker_ids is not None and len(key.lookup_worker_ids) > self.max_workers:
            raise ResourceCapacityError('request worker-list limit exceeded')
        for value in (key.model_name, key.request_id, key.lookup_id, key.cache_salt):
            if not isinstance(value, str):
                raise ValueError('request identifiers must be strings')
            # The character check bounds the temporary UTF-8 encoding too.
            if len(value) > self.max_identifier_bytes or len(value.encode('utf-8')) > self.max_identifier_bytes:
                raise ResourceCapacityError('request identifier limit exceeded')

    def validate_expansion(self, key, chunk_size, num_groups):
        if type(chunk_size) is not int or chunk_size <= 0 or type(num_groups) is not int or num_groups <= 0:
            raise ValueError('invalid lookup expansion dimensions')
        objects = (len(key.token_ids) // chunk_size) * key.lookup_world_size * num_groups
        if objects > self.max_objects:
            raise ResourceCapacityError('request object expansion limit exceeded')
        return objects

    def validate_transfer(self, key, block_ids, event_handle, chunk_size, num_groups):
        self.validate_key(key)
        self.validate_expansion(key, chunk_size, num_groups)
        # At most one block per token, per selected object; this conservative
        # cap works across heterogeneous groups without constructing tensors.
        if len(block_ids) > self.max_objects:
            raise ResourceCapacityError('transfer block-group limit exceeded')
        count = 0
        for group in block_ids:
            count += len(group)
            if count > self.max_objects * chunk_size:
                raise ResourceCapacityError('transfer block-index limit exceeded')
        if len(event_handle) > self.max_identifier_bytes:
            raise ResourceCapacityError('transfer event handle limit exceeded')

@dataclass(frozen=True)
class Usage:
    metadata_bytes: int = 0
    payload_bytes: int = 0
    operations: int = 0

    def validate(self):
        if any(type(v) is not int or v < 0 for v in self.values()):
            raise ValueError('resource dimensions must be nonnegative integers')

    def values(self):
        return self.metadata_bytes, self.payload_bytes, self.operations

    def plus(self, other):
        return Usage(*(a+b for a,b in zip(self.values(),other.values())))

    def minus(self, other):
        return Usage(*(a-b for a,b in zip(self.values(),other.values())))

    def fits(self, limit):
        return all(a<=b for a,b in zip(self.values(),limit.values()))

class Reservation:
    def __init__(self, account, usage):
        self._account = account
        self._usage = usage

    def resize(self, usage):
        self._account._budget._resize(self, usage)

    def release(self):
        return self._account._budget._release(self)

    def release_payload(self):
        """Keep replay metadata/operation charges after payload ownership ends."""
        return self._account._budget._release_payload(self)

class Account:
    def __init__(self, budget):
        self._budget = budget
        self._used = Usage()
        self._reservations = {}

    def reserve(self, usage):
        return self._budget._reserve(self, usage)

    def snapshot(self):
        with self._budget._lock:
            return self._used

class ResourceBudget:
    def __init__(self, per_request, total):
        per_request.validate();total.validate()
        if not per_request.fits(total):
            raise ValueError('per-request budget exceeds total budget')
        self._per_request = per_request
        self._total = total
        self._used = Usage()
        self._lock = Lock()

    def account(self):
        return Account(self)

    def snapshot(self):
        with self._lock:
            return self._used

    def _checked_totals(self, account, old, new):
        new.validate()
        # Every live reservation consumes a slot, including zero-byte work.
        if new.operations < 1:
            raise ValueError('a reservation needs at least one operation slot')
        request_used = account._used.minus(old).plus(new)
        total_used = self._used.minus(old).plus(new)
        if not request_used.fits(self._per_request) or not total_used.fits(self._total):
            raise ResourceCapacityError('request or server resource budget exhausted')
        return request_used, total_used

    def _reserve(self, account, usage):
        if account._budget is not self:
            raise ValueError('foreign budget account')
        with self._lock:
            request_used,total_used = self._checked_totals(account,Usage(),usage)
            reservation = Reservation(account,usage)
            # Allocate and publish the ledger entry before changing counters.
            account._reservations[id(reservation)] = reservation
            account._used,self._used = request_used,total_used
            return reservation

    def _resize(self, reservation, usage):
        account = reservation._account
        with self._lock:
            if account._reservations.get(id(reservation)) is not reservation:
                raise ValueError('reservation has already been released')
            request_used,total_used = self._checked_totals(account,reservation._usage,usage)
            reservation._usage = usage
            account._used,self._used = request_used,total_used

    def _release_payload(self, reservation):
        account = reservation._account
        with self._lock:
            if account._reservations.get(id(reservation)) is not reservation:
                return False
            old = reservation._usage
            if old.payload_bytes == 0:
                return False
            usage = Usage(old.metadata_bytes, 0, old.operations)
            request_used,total_used = self._checked_totals(account,old,usage)
            reservation._usage = usage
            account._used,self._used = request_used,total_used
            return True

    def _release(self, reservation):
        account = reservation._account
        with self._lock:
            if account._reservations.get(id(reservation)) is not reservation:
                return False
            request_used = account._used.minus(reservation._usage)
            total_used = self._used.minus(reservation._usage)
            del account._reservations[id(reservation)]
            account._used,self._used = request_used,total_used
            return True
