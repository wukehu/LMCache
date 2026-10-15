FROM intellif.io/dynamo/vllm-runtime@sha256:dc305f33a17b6702dbbc924f07ecffdbb8e0919832ad056fc7d64c1f8285da5f
RUN python -c "import hashlib; from pathlib import Path; p=Path('/usr/local/lib/python3.12/dist-packages/lmcache/v1/platform/cuda/ipc_wrapper.py'); assert hashlib.sha256(p.read_bytes()).hexdigest()=='3a69f8d399a9c57f29614c8921c45549124fb0ae7edcf1cb6f10e33e01a97bd2'"
COPY lmcache/v1/platform/cuda/ipc_wrapper.py /usr/local/lib/python3.12/dist-packages/lmcache/v1/platform/cuda/ipc_wrapper.py
COPY manifest.json verify.py /opt/intellif/releases/sp5/
RUN LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH} python /opt/intellif/releases/sp5/verify.py
LABEL io.intellif.release="SP5" io.intellif.release-base="SP4" io.intellif.lmcache-source="51c9ebd15c7cea2a48aebb79188ad4d2dc7c6ea0" io.intellif.lmcache-upstream-pr="5169" io.intellif.pp-gt1="excluded-from-acceptance"
