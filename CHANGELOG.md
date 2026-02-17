# Changelog

## [0.2.1](https://github.com/KalleBylin/town-elder/compare/v0.2.0...v0.2.1) (2026-02-17)


### Bug Fixes

* **packaging:** add README metadata and require pillow&gt;=12.1.1 ([fc358d1](https://github.com/KalleBylin/town-elder/commit/fc358d12ef68c0b59e92710dbb77fa43db37aeb2))
* **packaging:** add README metadata and require pillow&gt;=12.1.1 ([966b5cd](https://github.com/KalleBylin/town-elder/commit/966b5cdaffca4b4cf65fd3f6dc087e85b83a0eca))

## 0.2.0 (2026-02-17)


### ⚠ BREAKING CHANGES

* **cli:** replace legacy aliases with canonical index subcommands

### Features

* Add --version flag to CLI ([b9daf2b](https://github.com/KalleBylin/town-elder/commit/b9daf2bdb1155528771f2811ca97b868304c51f8))
* Batch embedding generation in commit-index ([872d187](https://github.com/KalleBylin/town-elder/commit/872d187ccc8b67f5cf4d594622dea75dde306b2d))
* **cli:** add canonical initial-indexing entrypoint (te index --all) ([9e160b6](https://github.com/KalleBylin/town-elder/commit/9e160b68860508d0a01cd23b82d4f080359baf22))
* **cli:** add stage telemetry for file indexing ([f28f442](https://github.com/KalleBylin/town-elder/commit/f28f4420401d113f279a2dc5ec7c9247df16ef10))
* **cli:** replace legacy aliases with canonical index subcommands ([a9fc056](https://github.com/KalleBylin/town-elder/commit/a9fc0567b3e66f0e8ed449a5b02c0d5f73e14785))
* Expand _is_te_hook detection to support python3/python3.x interpreter names ([aca9f12](https://github.com/KalleBylin/town-elder/commit/aca9f127e3714fc3d8f53f12546e905f79618046))
* improve --data-dir discoverability and placement UX ([f5fc5ff](https://github.com/KalleBylin/town-elder/commit/f5fc5ff5c2b3c7e30f16c3cde7029e7af7bee7f5))
* improve indexing, fix backlog loss, add init hook, align CLI ([71690ee](https://github.com/KalleBylin/town-elder/commit/71690ee47544875b963748e1b4ed02d955a25013))
* improve te add input ergonomics for shell workflows ([fb618ed](https://github.com/KalleBylin/town-elder/commit/fb618ed4e0d581a1ba5663d8ea800b15c128f738))
* **indexing:** add git blob-hash change detection for incremental file indexing ([686f583](https://github.com/KalleBylin/town-elder/commit/686f5837ea60ea9270daf4b5c7636ed51299eead))
* **indexing:** add producer-worker file parsing pipeline ([dd3d6fc](https://github.com/KalleBylin/town-elder/commit/dd3d6fc78b78d60e99787355cb951c6c8d4a1bc0))
* **indexing:** batch file embeddings and bulk vector upserts ([571e951](https://github.com/KalleBylin/town-elder/commit/571e951fca06181763ace1fb5ca5eae408bd4b11))
* **indexing:** refactor file scanner with .rst support and _build exclusion ([2900943](https://github.com/KalleBylin/town-elder/commit/2900943d43732e06324ec579a493868695c76710))
* **parsers:** add RST parser with semantic chunking and directive extraction ([01856e5](https://github.com/KalleBylin/town-elder/commit/01856e5ebd4bb25157ce61c4f33a6ff88d1c51de))


### Bug Fixes

* Add missing embed() method to _FakeEmbedder for commit-index tests ([1b93ff0](https://github.com/KalleBylin/town-elder/commit/1b93ff0a9f6a3b1a62d664dc17e1543cd025615c))
* add thread lock to ZvecStore collection access ([ca840a6](https://github.com/KalleBylin/town-elder/commit/ca840a65235e402d48d8a335f5285f094a7e75e7))
* add thread lock to ZvecStore.upsert to prevent race condition ([8a7ef38](https://github.com/KalleBylin/town-elder/commit/8a7ef383e2a7c5953e429327784683bb128ecfa5))
* always include --data-dir in hook to support env var config ([007b488](https://github.com/KalleBylin/town-elder/commit/007b4888e731583f4d0a6a3a14a264540a4b6288))
* check command exit code in hook fallback chain ([e24fe9a](https://github.com/KalleBylin/town-elder/commit/e24fe9ad38f91235963d95a7b623a60599941baf))
* check TOWN_ELDER_DATA_DIR env var in get_config ([77f666a](https://github.com/KalleBylin/town-elder/commit/77f666af18be081f01d3058cb6ae0137e2fd3684))
* **cli:** add executable entrypoints and honor --data-dir across services; add regression tests and beads follow-up tickets ([7b7686f](https://github.com/KalleBylin/town-elder/commit/7b7686f25b9c4d6e8d00e532697bf4a89a8bd097))
* distinguish empty repo errors from fatal git errors ([92e3ed7](https://github.com/KalleBylin/town-elder/commit/92e3ed76e3e12e04cb745ee23c3922d72c46f02c))
* embed absolute data_dir path in hook scripts ([ba8d5fa](https://github.com/KalleBylin/town-elder/commit/ba8d5faa5927db55f925b05e4ca1e41a5d5f68f2))
* get_diffs_batch returns exactly requested hashes ([bfdec44](https://github.com/KalleBylin/town-elder/commit/bfdec443ab29757799618cb2ee5329486243aa51))
* handle corrupted JSON metadata gracefully in ZvecStore ([f0dd0a8](https://github.com/KalleBylin/town-elder/commit/f0dd0a80caf389bd4e4827e12fb4bf2d51ef692d))
* handle corrupted JSON metadata gracefully in ZvecStore ([b09fec5](https://github.com/KalleBylin/town-elder/commit/b09fec5edd2ef5b5efe5d2affaa368aadc59dd47))
* handle invalid --repo paths gracefully in hook commands ([03375c4](https://github.com/KalleBylin/town-elder/commit/03375c46230e884cde1fb0c4c43501ab7f222834))
* handle invalid date parsing in GitRunner gracefully ([8f8192f](https://github.com/KalleBylin/town-elder/commit/8f8192f6e17ff405bce388bafaf03a802cf68c11))
* handle non-dict repos in index_state.json ([bb52f8a](https://github.com/KalleBylin/town-elder/commit/bb52f8a26ea588b2f975580f99f274271001a51c))
* handle ServiceInitError in commit-index and make embedder optional ([653c2bb](https://github.com/KalleBylin/town-elder/commit/653c2bb56763d8fb26356a4437ef49c718b49171))
* handle valid JSON that is not an object in index_state ([dd6e61a](https://github.com/KalleBylin/town-elder/commit/dd6e61a8ccd74425293928c685bce336e8d4c905))
* harden GitRunner commit parsing against delimiter collisions ([44118f7](https://github.com/KalleBylin/town-elder/commit/44118f753abc5a4db4b1e1bdb1404b1968c5110d))
* hide redundant alias commands from te --help ([f2b62e5](https://github.com/KalleBylin/town-elder/commit/f2b62e5cb42ee48c390927c0d3249519b5e3101c))
* Honor configured embed_model in IndexService and QueryService ([6321859](https://github.com/KalleBylin/town-elder/commit/632185979380571d56df7e949689aef0adbdae99))
* **indexing:** correct incremental file hash state handling ([c89cc56](https://github.com/KalleBylin/town-elder/commit/c89cc56349bab940a14a9d8c63d37f515fcd781f))
* **indexing:** prevent index --all double dispatch and correct RST chunk boundaries ([8b9d099](https://github.com/KalleBylin/town-elder/commit/8b9d099848c307e00178d9a246a2783c2320d7df))
* install hooks in common gitdir for worktree support ([b879638](https://github.com/KalleBylin/town-elder/commit/b879638ddbba56d88bab40bac2715df37e691962))
* optimize te index file discovery to avoid full-list materialization ([9ae83b4](https://github.com/KalleBylin/town-elder/commit/9ae83b499b4446a0103631a17e48b76eae3f6355))
* pass embed_dimension to ZvecStore and Embedder in services ([cb4106b](https://github.com/KalleBylin/town-elder/commit/cb4106b4ec1745269fc93672d4532da7e3b86f3b))
* propagate --data-dir into hook and fail fast on zvec open errors ([64261da](https://github.com/KalleBylin/town-elder/commit/64261daace38cac85cc873aec2280b8fedd2982d))
* refuse to install hook over symlinks ([d48fbc1](https://github.com/KalleBylin/town-elder/commit/d48fbc14ba7e9182cbfaad5fcbc759cd5e29ea5e))
* **release:** add release-please pipeline and trusted PyPI publish ([632f897](https://github.com/KalleBylin/town-elder/commit/632f897820002dd714c2915a6ee1dd4cf1cc93d8))
* resolve relative .git gitdir paths in hook commands ([598d754](https://github.com/KalleBylin/town-elder/commit/598d75464e24ac36e36d1cb9dfdfcf9899e5cd1e))
* resolve repo path to root when --repo points to subdirectory ([9705233](https://github.com/KalleBylin/town-elder/commit/9705233b59973b5c6ea72fd01f55cdbd469d6312))
* Ruff lint errors (S607, PLR2004, SIM110, S603) ([7437927](https://github.com/KalleBylin/town-elder/commit/74379274878e961c2b529a683e0068b4ed5ab00a))
* shell-escape --data-dir paths in hook scripts ([aa9cee0](https://github.com/KalleBylin/town-elder/commit/aa9cee0cec3060efbe30e5ece2cee81d5e98e93d))
* show friendly message for empty repos in commit-index ([05b6e6b](https://github.com/KalleBylin/town-elder/commit/05b6e6b9d3f279067c67e5bf99a274c68ff4102c))
* Tighten init --force safe-path checks to prevent deleting arbitrary hidden directories ([ed01e28](https://github.com/KalleBylin/town-elder/commit/ed01e28b8f70fd4c3091c635e60ded247347319f))
* treat top_k=0 explicitly instead of as unset in QueryService ([5734b14](https://github.com/KalleBylin/town-elder/commit/5734b144aafc6217c1515afebff50799b423f3de))
* use atomic writes for index_state to prevent data loss ([d804854](https://github.com/KalleBylin/town-elder/commit/d804854e9ecafdfc77d06109c53b12e3ef4b03f8))
* Use hash for doc_id in IndexService.index_file to satisfy zvec ID regex ([07f4f76](https://github.com/KalleBylin/town-elder/commit/07f4f76efbb0b8dfaeff16283f11998c5bfe34d2))
* use shutil.which to find git instead of hardcoded path ([aee3d3e](https://github.com/KalleBylin/town-elder/commit/aee3d3e23726fcec83ffe000b27eff28927e5044))
* validate --top-k and reject zero/negative values in search/query ([3028939](https://github.com/KalleBylin/town-elder/commit/3028939fab54b8f1e47e3a6b578cd87d0cb88d72))
* validate data_dir exists even when explicitly provided ([051b516](https://github.com/KalleBylin/town-elder/commit/051b5165e1b6f217c4ada611dc47651c9e51cc50))
* validate data-dir path at CLI entry ([f4386bf](https://github.com/KalleBylin/town-elder/commit/f4386bff6a6fb94cae26142d5f26d2dc464eda0a))
* warn on diff header parse failures instead of silently losing content ([e3c73cb](https://github.com/KalleBylin/town-elder/commit/e3c73cbb1710093651956a6b476347c6cf80c374))


### Documentation

* add explicit agent workflow and dogfood usage ([4244aa9](https://github.com/KalleBylin/town-elder/commit/4244aa95a26c9222c1baac9f2ffc07510e630dff))
* Add onboarding troubleshooting for model download and hooks ([5be0cbc](https://github.com/KalleBylin/town-elder/commit/5be0cbcddaa7b8da3d1b25284a2cf6911f87f508))
* Fix README clone URL and update pip to uv for source install ([b36a8e6](https://github.com/KalleBylin/town-elder/commit/b36a8e6df0b19d6a88cfeb55a00cf49d56f4d4cf))
* update CLI documentation for usability improvements ([3411ddb](https://github.com/KalleBylin/town-elder/commit/3411ddb53d9637904f21e4c36afe83468bfe930c))
* Update README to match CLI reality ([1663721](https://github.com/KalleBylin/town-elder/commit/1663721e9cb5712580101d85c9cbf94817c74967))
* Update stale context docs to use te instead of replay ([f5fffca](https://github.com/KalleBylin/town-elder/commit/f5fffca0c96caa971c8e91b719e8aa507025dc88))


### Miscellaneous Chores

* correct initial version ([fbac858](https://github.com/KalleBylin/town-elder/commit/fbac858ef1e872e11db71e17d0b9bcdc1c9ead49))
* correct initial version ([2ecac87](https://github.com/KalleBylin/town-elder/commit/2ecac87e50696f0040153b96e73cef5210e0bed7))
