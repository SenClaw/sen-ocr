//! `sen-ocr` — the SenClaw OCR runtime: PaddleOCR (PP-OCRv4/v5) on the MNN
//! inference backend. Launched by the SenClaw daemon as a child process and
//! driven over loopback HTTP (see `senclaw/docs/runtime-protocol.md`); also
//! runnable standalone for development.

use std::sync::Arc;

use sen_runtime_sdk::env::LaunchEnv;
use sen_runtime_sdk::manifest::{Capability, RunMode};
use sen_runtime_sdk::server::{serve, Readiness, ServeArgs, ServeOptions};

fn usage() -> ! {
    eprintln!(
        "usage: sen-ocr serve [--host HOST] [--port PORT]\n\n\
         Serves the SenClaw OCR API on loopback HTTP. Standalone: with no\n\
         SENCLAW_RUNTIME_TOKEN set there is no auth, and with no\n\
         SENCLAW_PARENT_PID there is no parent watchdog."
    );
    std::process::exit(1);
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    sen_runtime_sdk::server::init_tracing();
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    let cmd = args.remove(0);
    match cmd.as_str() {
        "serve" => run_serve(args).await,
        "-h" | "--help" => usage(),
        other => {
            eprintln!("unknown subcommand `{other}`");
            usage();
        }
    }
}

async fn run_serve(args: Vec<String>) -> anyhow::Result<()> {
    let parsed = ServeArgs::parse(args).map_err(|e| anyhow::anyhow!(e))?;
    let env = LaunchEnv::from_env(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
    let state = Arc::new(sen_ocr::http::AppState { env: env.clone() });
    let routes = sen_ocr::http::router(state);
    let opts = ServeOptions {
        env,
        mode: RunMode::Service,
        // A service answers /health before any weights load — the OCR engine
        // lazy-loads on the first recognize call, never at boot.
        capabilities: vec![Capability::Ocr],
        readiness: Readiness::ready(),
        info_detail: Some(Arc::new(|| {
            serde_json::json!({ "compiledOcrPaddle": cfg!(feature = "ocr-paddle") })
        })),
        args: parsed,
    };
    serve(routes, opts).await
}
