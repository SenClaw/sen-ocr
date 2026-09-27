//! Library surface for `sen-ocr` — split out of the binary so the OCR
//! round-trip example (`examples/ocr_roundtrip.rs`) can drive [`ocr::OcrEngine`]
//! directly, the same way it did as `senclaw::local_model::OcrEngine` before
//! this runtime moved out of the daemon.

pub mod http;
pub mod ocr;
pub mod settings_store;
