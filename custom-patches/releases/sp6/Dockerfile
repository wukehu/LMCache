FROM intellif.io/dynamo/vllm-runtime@sha256:620a26047f46fa6d3e323ac66f13f74ab6c2716f8fdf7ef92b4bd2ba17c39557
COPY lmcache/v1/mp_observability/source_attribution.py /usr/local/lib/python3.12/dist-packages/lmcache/v1/mp_observability/source_attribution.py
COPY lmcache/integration/vllm/lmcache_mp_metadata.py /usr/local/lib/python3.12/dist-packages/lmcache/integration/vllm/lmcache_mp_metadata.py
COPY lmcache/integration/vllm/lmcache_mp_connector.py /usr/local/lib/python3.12/dist-packages/lmcache/integration/vllm/lmcache_mp_connector.py
COPY vllm/v1/core/sched/scheduler.py /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py
COPY vllm/distributed/kv_transfer/kv_connector/v1/base.py /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/base.py
COPY vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py
COPY manifest.json verify.py /opt/intellif/releases/sp5-async-c3/
RUN LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH} python /opt/intellif/releases/sp5-async-c3/verify.py
LABEL io.intellif.release="SP5-async-c3"
