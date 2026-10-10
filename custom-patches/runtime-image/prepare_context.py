# SPDX-License-Identifier: Apache-2.0
"""Export clean Git source and recipe into a separate immutable build context."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(['git', '-C', str(repo), *args])


def prepare(vllm_repo: Path, lmcache_repo: Path, output: Path, base_image: str) -> None:
    if '@sha256:' not in base_image or len(base_image.rsplit('@sha256:', 1)[1]) != 64:
        raise ValueError('A digest-pinned base image is required')
    if output.exists():
        raise FileExistsError(output)
    revisions = {}
    for name, repo in [('vllm', vllm_repo), ('lmcache', lmcache_repo)]:
        if git(repo, 'status', '--porcelain').strip():
            raise RuntimeError(f'{name} worktree must be clean')
        revisions[name] = git(repo, 'rev-parse', 'HEAD').decode().strip()
    output.mkdir(parents=True)
    for name, repo in [('vllm', vllm_repo), ('lmcache', lmcache_repo)]:
        destination = output / 'sources' / name
        destination.mkdir(parents=True)
        data = git(repo, 'archive', '--format=tar', revisions[name], name)
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archive.extractall(destination, filter='data')
    recipe = 'custom-patches/runtime-image/'
    for filename in ['Dockerfile', 'repack.py', 'verify.py']:
        (output / filename).write_bytes(git(lmcache_repo, 'show', revisions['lmcache'] + ':' + recipe + filename))
    tests = output / 'tests'
    tests.mkdir()
    for filename in ['test_pp_namespace.py', 'test_pp_node_endpoints.py',
                     'test_pp_parallel_strategy.py', 'test_pp_topology.py']:
        (tests / filename).write_bytes(git(lmcache_repo, 'show', revisions['lmcache'] + ':tests/v1/' + filename))
    lock = dict(base_image=base_image, revisions=revisions,
                versions=dict(vllm='0.26.0', lmcache='0.5.5', torch='2.11.0+cu130'),
                native_policy='Reuse verified binaries from pinned base; no native source changes')
    (output / 'source-lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    print(json.dumps(lock), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vllm-repo', type=Path, required=True)
    parser.add_argument('--lmcache-repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-image', required=True)
    args = parser.parse_args()
    prepare(args.vllm_repo, args.lmcache_repo, args.output, args.base_image)
