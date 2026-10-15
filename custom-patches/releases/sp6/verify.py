# SPDX-License-Identifier: Apache-2.0
import json,hashlib,ast
from pathlib import Path
m=json.loads(Path('/opt/intellif/releases/sp5-async-c3/manifest.json').read_text())
for f in m['files']:
 p=Path('/usr/local/lib/python3.12/dist-packages')/f['path']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'],str(p)
for rel in m['changed_files']:ast.parse((Path('/usr/local/lib/python3.12/dist-packages')/rel).read_text())
import torch,lmcache.cuda_ops,vllm._C_stable_libtorch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
assert hasattr(KVConnectorBase_V1,'on_model_step_completed_with_metadata')
print('Pinned sources, frame hook and native imports passed')
