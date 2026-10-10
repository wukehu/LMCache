# SPDX-License-Identifier: Apache-2.0
"""Verify installed Git source, native bytes, versions, and optionally native imports."""
import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path


def verify(native_imports: bool) -> dict:
    result = {}
    for name, version in [('vllm', '0.26.0'), ('lmcache', '0.5.5')]:
        dist = metadata.distribution(name)
        if dist.version != version:
            raise RuntimeError(f'{name} version mismatch: {dist.version}')
        root = Path(dist.locate_file(''))
        manifest = json.loads((root / name / '_custom_source_manifest.json').read_text())
        for category in ('sources', 'native'):
            for relative, expected in manifest[category].items():
                actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
                if actual != expected:
                    raise RuntimeError(f'{category} hash mismatch: {relative}')
        result[name] = dict(version=version, source_files=len(manifest['sources']),
                            native_files=len(manifest['native']))
    if native_imports:
        import torch
        if torch.__version__ != '2.11.0+cu130':
            raise RuntimeError(f'Torch ABI changed: {torch.__version__}')
        modules = ['lmcache.lmcache_native', 'lmcache.cuda_ops',
                   'vllm._C_stable_libtorch', 'vllm._moe_C_stable_libtorch', 'vllm._qutlass_C',
                   'lmcache.v1.multiprocess.server', 'lmcache.integration.vllm.lmcache_mp_connector']
        result['imports'] = {name: importlib.import_module(name).__file__ for name in modules}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-imports', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.native_imports)), flush=True)
