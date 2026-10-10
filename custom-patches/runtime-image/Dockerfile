# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS wheels
USER root
COPY sources/ /build/sources/
COPY source-lock.json repack.py verify.py /build/
RUN uv venv --system-site-packages /build/.venv && \
    /build/.venv/bin/python /build/repack.py

ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
ARG VLLM_SHA
ARG LMCACHE_SHA
LABEL io.runtime.vllm.revision=${VLLM_SHA} \
      io.runtime.lmcache.revision=${LMCACHE_SHA} \
      io.runtime.native-policy="reuse pinned ABI-compatible native binaries; Python source from Git"
COPY --from=wheels /build/source-lock.json /opt/custom-runtime/source-lock.json
COPY --from=wheels /build/wheel-manifest.json /opt/custom-runtime/wheel-manifest.json
COPY verify.py /opt/custom-runtime/verify.py
COPY tests/ /opt/custom-runtime/tests/
RUN --mount=from=wheels,source=/wheels,target=/wheels \
    uv pip install --system --no-deps --reinstall /wheels/*.whl && \
    uv venv --system-site-packages /opt/custom-runtime/.venv && \
    /opt/custom-runtime/.venv/bin/python /opt/custom-runtime/verify.py
