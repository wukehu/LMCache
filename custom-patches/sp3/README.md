# SP3 lmcache source archive

Branch: `release-sp3`; base: `b2dfb89963791d9b57dde2c720592bcb7bf49a0e`.
Image: `intellif.io/dynamo/vllm-runtime:1.4.0-mooncake-0.3.12-lmcache-0.5.5-cuda13-sp3`.
Digest: `sha256:d8d83d33c73f7f9e4f2cff478e5bd9167e124f66ee26208f81910b49aca6064c`.

This worktree archives existing SP3 changes. It introduces no new runtime behavior.
The component manifest records source hashes; other working trees are preserved.
Experimental PP > 1 is excluded. Existing PP compatibility code is preserved.
Cold/cache output numerical equivalence remains deferred.
This branch archives the SP3 source. No release tag or clean-image rebuild is claimed.

Production files: 18. See `manifest.json` for the image binding.

Git sources use LF. Image byte hashes and per-file newline policy are recorded separately; the overlay exporter restores original line endings for exact byte verification.

CPU regression: 230 tests and 4 subtests passed; 2 test files could not collect because Mooncake imports require libcuda.so.1 in the GPU-disabled container. This is not a full-suite pass.
