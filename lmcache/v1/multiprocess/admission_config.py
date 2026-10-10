"""Explicit experimental MP admission configuration; no inferred defaults."""
import json
from dataclasses import dataclass
from lmcache.v1.multiprocess.resource_budget import ResourceBudget,Usage,RequestLimits
from lmcache.v1.multiprocess.mq import RequestFrameLimits,ResponseLimits

@dataclass(frozen=True)
class AdmissionConfig:
    session_budget: ResourceBudget
    rpc_budget: ResourceBudget
    request_limits: RequestLimits
    frame_limits: RequestFrameLimits
    response_limits: ResponseLimits | None = None

def load_admission_config(raw):
    if raw is None:
        return None
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('duplicate admission config key: '+key)
            result[key]=value
        return result
    data=json.loads(raw,object_pairs_hook=unique)
    required={'session_per_request','session_total','rpc_per_request','rpc_total','request','frames'}
    if type(data) is not dict or set(data) not in (required,required|{'responses'}):
        raise ValueError('admission config requires '+','.join(sorted(required))+'; optional responses')
    def construct(cls,name,fields):
        value=data[name]
        if type(value) is not dict or set(value)!=set(fields):
            raise ValueError('invalid admission config fields: '+name)
        return cls(**value)
    usage_fields=('metadata_bytes','payload_bytes','operations')
    session=ResourceBudget(construct(Usage,'session_per_request',usage_fields),construct(Usage,'session_total',usage_fields))
    rpc_request=construct(Usage,'rpc_per_request',usage_fields)
    rpc_total=construct(Usage,'rpc_total',usage_fields)
    rpc=ResourceBudget(rpc_request,rpc_total)
    if rpc_request.operations!=1 or rpc_request.payload_bytes or rpc_total.payload_bytes:
        raise ValueError('RPC budget requires one operation per request and zero payload_bytes')
    limits=construct(RequestLimits,'request',('max_tokens','max_workers','max_objects','max_identifier_bytes'))
    frames=construct(RequestFrameLimits,'frames',('max_frame_bytes','max_request_bytes','max_frames','max_header_bytes'))
    if not frames.max_header_bytes<=frames.max_frame_bytes<=frames.max_request_bytes<=rpc_request.metadata_bytes or frames.max_frames<3:
        raise ValueError('inconsistent frame and RPC byte limits')
    responses = construct(ResponseLimits,'responses',('max_response_bytes','total_response_bytes','max_responses')) if 'responses' in data else None
    return AdmissionConfig(session,rpc,limits,frames,responses)
