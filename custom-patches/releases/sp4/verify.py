import hashlib
import json
import py_compile
from pathlib import Path

manifest = json.loads(Path('/opt/source-token-candidate/manifest.json').read_text())
for item in manifest['files']:
    path = Path('/usr/local/lib/python3.12/dist-packages') / item['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], str(path)
    py_compile.compile(str(path), doraise=True)
print(f"verified {len(manifest['files'])} candidate Python sources")
