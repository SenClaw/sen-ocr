# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

`sen-ocr` is the SenClaw OCR runtime — PaddleOCR (PP-OCRv4/v5) on the MNN
inference backend, ported unchanged from the daemon's `src/local_model/ocr`
+ `src/gateway/ui_server/ocr.rs`. It is a standalone binary the SenClaw
daemon installs, launches as a child process, and talks to over loopback
HTTP: see `../senclaw/docs/runtime-protocol.md` §4.4 for the exact contract
(routes, bodies, status codes) this repo must keep serving **verbatim** —
desktop and web call these paths through the daemon's proxy, and the
daemon's own OCR-fallback (text-only chat models "reading" an attached
image) and the `senclaw-ocr` MCP server call `POST /api/ocr/recognize`
directly.

## Rules for Claude

- **The bundled latin model is honestly partial Vietnamese, not full
  Tiếng Việt.** `PP-OCRv5_mobile_latin`'s charset
  (`ppocr_keys_latin.txt`) has no precomposed stacked-tone vowels (ế ộ ệ ạ ố
  …) — verified at ~58% word recall via `examples/ocr_roundtrip.rs`. Never
  relabel its `default_language` as `"vi"` (a test in `src/ocr/catalog.rs`
  guards this); the honest fix is a full-Vietnamese MNN model, which does not
  exist upstream yet.
- **Every catalog URL must be the GitHub-raw mirror**, never the gated
  HuggingFace one (`zibo-chen/rust-paddle-ocr-models` 401s for anonymous
  downloads). Tests in `src/ocr/catalog.rs` enforce this and the `_infer`
  filename suffix on the latin recognition model.
- **The engine is lazy-loaded and explicitly unloaded after every
  `recognize` call** (`OcrEngine::unload`, called from both the HTTP handler
  and the round-trip harness) to keep idle RAM low — mirrors the pattern the
  old Whisper sidecar used. Do not cache a loaded session across requests.
- **Models are keyed by a "safe dirname"** (`id.replace('/', "__")`) under
  `SENCLAW_OCR_MODELS_DIR` (default `<SENCLAW_HOME>/ocr-models/`) — an
  engine-private root, read directly from the environment
  (`http::models_root`) rather than through a [`sen_runtime_sdk::env::LaunchEnv`]
  field, because it is not the shared `local-models` root the daemon passes
  through unconditionally.
- **Settings live in `<SENCLAW_RUNTIME_DATA_DIR>/settings.json`**, seeded
  once from the daemon's old `config.json["ocrConfig"]` via
  `sen_runtime_sdk::legacy::load_or_import` (`src/settings_store.rs`). Read
  per request so a save applies to the next call without a restart, same as
  the old daemon behaviour.
- **`ocr-paddle` is a default-on feature, not an always-on dependency.** A
  build without it still serves the whole API — `recognize` answers `501`
  instead of failing to compile — because MNN (via `ocr-rs`) is a heavy C++
  native dependency (cmake) some CI/dev builds want to skip.
- **Loopback bind + bearer auth is the SDK server scaffold's job**
  (`sen_runtime_sdk::server::serve`), not this repo's — never hand-roll
  another listener or auth check here.
- No plan ids, phase numbers, or finding codes in code, comments, or test
  names — explain the invariant or behaviour directly.
