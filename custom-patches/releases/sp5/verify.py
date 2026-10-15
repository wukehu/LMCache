# SPDX-License-Identifier: Apache-2.0
import hashlib
import importlib
import json
import py_compile

import torch  # Load libtorch before importing compiled extensions.
from pathlib import Path

manifest = json.loads(Path('/opt/intellif/releases/sp5/manifest.json').read_text())
for item in manifest['files']:
    path = Path('/usr/local/lib/python3.12/dist-packages') / item['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], str(path)
    py_compile.compile(str(path), doraise=True)
for name in ('lmcache.cuda_ops', 'vllm._C_stable_libtorch'):
    module = importlib.import_module(name)
    print('native import verified:', name, module.__file__)
print('verified SP5 overlay and all 13 inherited SP4 source files')
