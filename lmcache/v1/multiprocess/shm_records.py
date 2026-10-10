"""Shared record-count admission for both SHM directions.

Count bounds are not byte/RSS bounds. No record expires here: only explicit
deletion reclaims a slot, so unknown ownership cannot be released by a timer.
"""
from collections.abc import MutableMapping
from threading import Lock

class ShmRecordCapacityError(RuntimeError):
    pass

class ShmRecords:
    def __init__(self, limit):
        if type(limit) is not int or limit <= 0:
            raise ValueError('max_shm_transfer_records must be a positive integer')
        self.limit=limit
        self._lock=Lock()
        self._data={'writes':{},'reads':{}}
        self.writes=_Direction(self,'writes')
        self.reads=_Direction(self,'reads')

    def snapshot(self):
        with self._lock:
            counts={name:len(values) for name,values in self._data.items()}
            return dict(limit=self.limit,used=sum(counts.values()),**counts)

class _Direction(MutableMapping):
    def __init__(self, owner, direction):
        self._owner=owner;self._direction=direction

    def __getitem__(self,key):
        with self._owner._lock:
            return self._owner._data[self._direction][key]

    def __setitem__(self,key,value):
        with self._owner._lock:
            data=self._owner._data[self._direction]
            if key not in data and sum(len(d) for d in self._owner._data.values()) >= self._owner.limit:
                raise ShmRecordCapacityError('SHM transfer record capacity exhausted')
            data[key]=value

    def __delitem__(self,key):
        with self._owner._lock:
            del self._owner._data[self._direction][key]

    def __iter__(self):
        with self._owner._lock:
            return iter(tuple(self._owner._data[self._direction]))

    def __len__(self):
        with self._owner._lock:
            return len(self._owner._data[self._direction])
