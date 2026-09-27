//! On-device OCR powered by PaddleOCR (PP-OCRv4/v5) on the MNN inference
//! backend, via the pure-Rust crate [`ocr-rs`](https://github.com/zibo-chen/rust-paddle-ocr).
//!
//! ## Features
//!
//! - Cross-platform CPU inference (Linux/Windows/macOS).
//! - macOS Metal/CoreML acceleration when built with the additive
//!   `ocr-paddle-metal` feature.
//! - ~10 supported languages via per-language recognition models. The default
//!   `latin_PP-OCRv5_mobile_rec` model covers English + 40 Latin scripts and
//!   recognises Vietnamese only **partially** — its charset omits precomposed
//!   stacked-tone vowels (ế ộ ệ ạ ố …), so full Tiếng Việt is not supported by
//!   any MNN model currently published upstream. See [`catalog`].
//! - Lazy-loaded engine with explicit [`OcrEngine::unload`] to release RAM
//!   between requests.
//!
//! ## Layout
//!
//! ```text
//!     {ocr_models_dir}/{safe-id}/
//!         ├── det.mnn    PP-OCRv5 detection model
//!         ├── rec.mnn    Recognition model (per language / latin)
//!         └── keys.txt   Charset (ppocr_keys_v5.txt or latin variant)
//! ```
//!
//! Catalog entries (default model URLs) live in [`catalog`].
//!
//! Ported unchanged from the SenClaw daemon's `src/local_model/ocr`, where it
//! ran behind the `ocr-paddle` build feature; it moved into this standalone
//! `sen-ocr` runtime so the daemon links no inference code at all.

// `catalog` has no dependency on `ocr-rs` and stays available in every build
// (the model list must render even where MNN isn't compiled in). `engine` is
// the actual MNN-backed session and needs the `ocr-paddle` feature: it holds
// an `ocr_rs::OcrEngine` unconditionally, so it cannot compile without the
// crate `ocr-paddle` pulls in.
pub mod catalog;
#[cfg(feature = "ocr-paddle")]
pub mod engine;

pub use catalog::{
    default_entry, installed_model_files, CatalogEntry, CATALOG, DEFAULT_MODEL_ID, DET_FILE,
    KEYS_FILE, REC_FILE,
};
#[cfg(feature = "ocr-paddle")]
pub use engine::{OcrBlock, OcrEngine, OcrResult};
