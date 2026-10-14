FROM intellif.io/dynamo/vllm-runtime@sha256:d8d83d33c73f7f9e4f2cff478e5bd9167e124f66ee26208f81910b49aca6064c
COPY lmcache/ /usr/local/lib/python3.12/dist-packages/lmcache/
COPY vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY manifest.json verify.py /opt/source-token-candidate/
RUN /opt/uv/bin/uv venv --system-site-packages /tmp/source-verify && /opt/uv/bin/uv run --no-project --python /tmp/source-verify/bin/python python /opt/source-token-candidate/verify.py
LABEL io.intellif.candidate="source-token-PP1-sync-v1" io.intellif.release-base="SP3" io.intellif.pp-gt1="excluded"
