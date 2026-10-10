# SPDX-License-Identifier: Apache-2.0
"""Build source-pinned wheels while retaining the pinned runtime's native ABI."""
import base64
import csv
import hashlib
import importlib.metadata as metadata
import io
import json
from pathlib import Path
import shutil
import zipfile


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(name: str, version: str) -> dict:
    dist = metadata.distribution(name)
    if dist.version != version:
        raise RuntimeError(f"Unexpected base {name}: {dist.version}")
    installed = Path(dist.locate_file(name))
    source = Path('/build/sources') / name / name
    stage = Path('/build/stage') / name
    shutil.copytree(installed, stage / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    info_name = f'{name}-{version}.dist-info'
    info = next(Path(dist.locate_file(f)).parent for f in dist.files if str(f).endswith('.dist-info/WHEEL'))
    shutil.copytree(info, stage / info_name)
    hashes = {}
    for path in sorted(source.rglob('*.py')):
        relative = Path(name) / path.relative_to(source)
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[relative.as_posix()] = digest(path.read_bytes())
    native = {}
    for path in sorted((stage / name).rglob('*.so')):
        native[path.relative_to(stage).as_posix()] = digest(path.read_bytes())
    if not native:
        raise RuntimeError(f'{name}: native extensions missing from wheel')
    tag = next(line[5:] for line in (stage / info_name / 'WHEEL').read_text().splitlines() if line.startswith('Tag: '))
    for filename in ('RECORD', 'direct_url.json', 'INSTALLER', 'REQUESTED'):
        (stage / info_name / filename).unlink(missing_ok=True)
    provenance = dict(name=name, version=version, sources=hashes, native=native,
                      native_policy='Reused unchanged from the digest-pinned base; not recompiled')
    (stage / name / '_custom_source_manifest.json').write_text(json.dumps(provenance, indent=2))
    wheel = Path('/wheels') / f'{name}-{version}-{tag}.whl'
    wheel.parent.mkdir(exist_ok=True)
    records = []
    with zipfile.ZipFile(wheel, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(stage.rglob('*')):
            if not path.is_file():
                continue
            relative = path.relative_to(stage).as_posix()
            data = path.read_bytes()
            archive.writestr(relative, data)
            encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
            records.append([relative, f'sha256={encoded}', str(len(data))])
        record_name = info_name + '/RECORD'
        records.append([record_name, '', ''])
        buffer = io.StringIO()
        csv.writer(buffer).writerows(records)
        archive.writestr(record_name, buffer.getvalue())
    with zipfile.ZipFile(wheel) as archive:
        for filename in native:
            if digest(archive.read(filename)) != native[filename]:
                raise RuntimeError(f'Native artifact changed: {filename}')
    return dict(wheel=wheel.name, sha256=digest(wheel.read_bytes()),
                source_count=len(hashes), native=native)


if __name__ == '__main__':
    result = {name: build(name, version) for name, version in [('vllm', '0.26.0'), ('lmcache', '0.5.5')]}
    Path('/build/wheel-manifest.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
