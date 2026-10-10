"""Wire identities for the negotiated SHM sequence/ACK protocol.

Separate from reusable KV cache identity. Schemas do not advertise support;
handlers and client lifecycle integration must precede capability activation.
"""
from dataclasses import dataclass

CAPABILITY='shm_sequence_ack_v1'

def _hex_id(value):
    return type(value) is str and len(value)==32 and all(c in '0123456789abcdef' for c in value)

@dataclass(frozen=True)
class ShmTransferRef:
    epoch: str
    sequence: int
    nonce: str = ''

    def __post_init__(self):
        if not _hex_id(self.epoch):
            raise ValueError('SHM epoch must be a 32-character lowercase hex ID')
        if type(self.sequence) is not int or not 1 <= self.sequence <= (1<<64)-1:
            raise ValueError('SHM sequence must be a positive uint64')
        if self.nonce != '' and not _hex_id(self.nonce):
            raise ValueError('SHM nonce must be empty or a 32-character lowercase hex ID')

    def require_completion_ticket(self):
        if not self.nonce:
            raise ValueError('SHM completion requires the server-issued nonce')

@dataclass(frozen=True)
class ShmRegistrationIdentity:
    epoch: str
    window_capacity: int
    capability: str = CAPABILITY
    operations: tuple[str, ...] = ('store',)

    def __post_init__(self):
        if not _hex_id(self.epoch):
            raise ValueError('invalid SHM registration epoch')
        if type(self.window_capacity) is not int or not 1 <= self.window_capacity <= 65536:
            raise ValueError('invalid SHM sequence window capacity')
        if self.capability != CAPABILITY:
            raise ValueError('unsupported SHM capability')
        if self.operations not in (('store',),('store','retrieve')):
            raise ValueError('unsupported SHM operation set')
