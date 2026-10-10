# Unified runtime candidate: vLLM 0.26.0 + LMCache 0.5.5

Both forks use `feature-runtime-v026-lmc055`. The default branches remain upstream-only.

The candidate combines the source-verified exp30 MP / PP > 1 implementation (44 LMCache files and six vLLM NIXL/scheduler files), the LMCache partial-prefix Tensor/pointer fallback fix already present in that implementation, and the DeepSeek R1 MLA W8A8 activation-dtype fix. Never replay the older fallback patch over the integrated implementation.

## Historical branch decisions

- `feature-runtime-pp-exp30-preserved-20261010`: included, retaining the PP namespaces, epoch recovery, request ownership, END_SESSION rank barrier, and admission/retired-resource lifecycle.
- `feature-partial-prefix-fallback-v055`: included by equivalent code in the runtime branch.
- `feature-mla-w8a8-v026`: included in the vLLM runtime branch.
- `feature-v016-negative-counter-preserved-20261010`: the PrefillStats mechanism is already present in upstream v0.26.0; do not replace its scheduler with v0.16 files.
- `feature-metrics-preserved-20261010`: archival only; its incomplete `local_cache_` member access must not enter the runtime.
- LMCache development snapshots based on `1a4b40b1`: retained separately. Their newer role/lazy-offload APIs and completion rules cannot overwrite the 0.5.5 exp30 END_SESSION/ownership protocol. Independent P2P and memcpy experiments in those snapshots are not asserted to be part of this candidate.

## Image contract

The recipe is `custom-patches/runtime-image/` in the LMCache fork. Build context is generated from clean, exact Git commits of both repositories. It builds two wheels with source from those commits and unmodified native binaries/generated dependencies from a digest-pinned runtime. It does not claim a fresh CUDA or Rust compilation. Torch must remain 2.11.0+cu130.

Each wheel carries a source/native SHA256 manifest. The build checks wheel native contents; final image verification checks every installed source/native file and, with GPU driver access, imports native extensions and the MP connector/server. Runtime regression and full DeepSeek multi-node inference are distinct validation layers.
