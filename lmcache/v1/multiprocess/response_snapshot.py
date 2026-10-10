"""Bounded wire snapshot for audited non-custom MessagePack responses."""
from dataclasses import fields,is_dataclass
import enum
class ResponsePreflightError(ValueError):pass

def snapshot_response(value,limit,*,max_depth=64):
    if type(limit) is not int or limit<0:raise ValueError('invalid limit')
    used=0;ancestors=set()
    def charge(n):
        nonlocal used
        if n>limit-used:raise ResponsePreflightError('response capacity exceeded')
        used+=n
    def freeze(obj,depth):
        if depth>max_depth:raise ResponsePreflightError('response nesting too deep')
        t=type(obj)
        if obj is None or t is bool:charge(1);return obj
        if t is int:
            if not -(1<<63)<=obj<(1<<64):raise ResponsePreflightError('integer range')
            charge(9);return obj
        if t is float:charge(9);return obj
        if t is bytes:charge(5+len(obj));return obj
        if t is str:
            if len(obj)>limit-used:raise ResponsePreflightError('string capacity')
            charge(5+sum(1 if ord(c)<128 else 2 if ord(c)<2048 else 3 if ord(c)<65536 else 4 for c in obj));return obj
        if isinstance(obj,enum.Enum):return freeze(obj.value,depth+1)
        oid=id(obj)
        if oid in ancestors:raise ResponsePreflightError('cyclic response')
        ancestors.add(oid)
        try:
            charge(5)
            if t in (tuple,list):return tuple(freeze(v,depth+1) for v in obj)
            if t is dict:
                result={}
                for k,v in obj.items():result[freeze(k,depth+1)]=freeze(v,depth+1)
                return result
            if is_dataclass(obj) and not isinstance(obj,type):
                result={}
                for f in fields(obj):result[freeze(f.name,depth+1)]=freeze(getattr(obj,f.name),depth+1)
                return result
            raise ResponsePreflightError('unsupported response type: '+t.__name__)
        finally:ancestors.remove(oid)
    snapshot=freeze(value,0)
    return snapshot,used
