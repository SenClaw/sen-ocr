# sen-ocr

SenClaw's OCR runtime: PaddleOCR (PP-OCRv4/v5) on the MNN inference backend,
via the pure-Rust [`ocr-rs`](https://github.com/zibo-chen/rust-paddle-ocr)
crate. Launched by the SenClaw daemon as a child process and driven over
loopback HTTP — see
[`senclaw/docs/runtime-protocol.md`](../senclaw/docs/runtime-protocol.md)
§4.4 for the wire contract this repo implements.

Ported from the SenClaw daemon's `src/local_model/ocr/**` and
`src/gateway/ui_server/ocr.rs`. `POST /api/ocr/recognize` is also the route
the daemon's internal OCR-fallback client (text-only chat models "reading"
an attached image) and the `senclaw-ocr` MCP server call — unchanged.

## Build & run

```bash
cargo build --release                 # or: make build (adds --features ocr-paddle-metal on macOS)
cargo test                            # or: make test
make package                          # dist/sen-ocr-<version>-<platform>.tar.gz + .sha256
make install-local                    # senclaw runtime install-local, or extract into ~/.senclaw/runtimes
make run-dev                          # standalone serve on :4962, no token, no watchdog
```

Standalone (no daemon) for development:

```bash
cargo run --features ocr-paddle-metal -- serve --host 127.0.0.1 --port 4962
```

With no `SENCLAW_RUNTIME_TOKEN` set there is no auth; with no
`SENCLAW_PARENT_PID` there is no parent watchdog — see
[`sen_runtime_sdk::env`](../senclaw/crates/sen-runtime-sdk/src/env.rs).

## Round-trip harness

`examples/ocr_roundtrip.rs` renders known phrases to PNG, recognizes them
with the production engine, and reports word recall — the way to check
whether a downloaded model actually reads Vietnamese/English text, and how
fast:

```bash
cargo run --release --features ocr-paddle-metal --example ocr_roundtrip -- \
  --model ~/.senclaw/ocr-models/PP-OCRv5_mobile_latin --lang vi --iters 3
```

## Routes

Every route the daemon proxied at `/api/ocr/*` before the split, served
**verbatim** (same paths, bodies, status codes), plus the common
`/health` · `/runtime/info` · `/runtime/shutdown` from the SDK server
scaffold:

```
GET    /api/ocr/models
POST   /api/ocr/models/custom
POST   /api/ocr/models/:id/download
GET    /api/ocr/models/:id/status
POST   /api/ocr/models/:id/cancel
DELETE /api/ocr/models/:id
GET    /api/ocr/settings
PUT    /api/ocr/settings
POST   /api/ocr/recognize
```

## Models on disk

`SENCLAW_OCR_MODELS_DIR`, default `<SENCLAW_HOME>/ocr-models/<safe-id>/`
(`det.mnn` + `rec.mnn` + `keys.txt`) — unchanged from the daemon. Settings
persist at `<SENCLAW_RUNTIME_DATA_DIR>/settings.json`, seeded once from the
daemon's old `config.json["ocrConfig"]`.

## Feature flags

- `ocr-paddle` (default on) — the MNN-backed engine (`ocr-rs` + `image`).
  Off, the binary still serves the full API; `recognize` answers 501.
- `ocr-paddle-metal` (additive, macOS) — Metal + CoreML acceleration.
