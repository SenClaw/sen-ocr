//! High-level OCR engine wrapper around `ocr_rs::OcrEngine`.
//!
//! - Lazy load on first `recognize_*` call.
//! - Explicit [`OcrEngine::unload`] frees the underlying MNN session (called
//!   from the HTTP layer after each request to keep idle RAM low).
//! - macOS Metal backend when built with `ocr-paddle-metal`; CPU otherwise.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, Context, Result};
use serde::Serialize;

use super::catalog::installed_model_files;

/// One detected text region.
#[derive(Debug, Clone, Serialize)]
pub struct OcrBlock {
    pub text: String,
    pub confidence: f32,
    /// Axis-aligned bounding box: `(x, y, width, height)` in image pixels.
    pub bbox: (i32, i32, u32, u32),
}

/// Result of running OCR on a single image.
#[derive(Debug, Clone, Serialize)]
pub struct OcrResult {
    /// All recognized text joined with newlines (in reading order as returned
    /// by the detector).
    pub text: String,
    pub blocks: Vec<OcrBlock>,
}

struct Loaded {
    inner: ocr_rs::OcrEngine,
}

pub struct OcrEngine {
    model_dir: PathBuf,
    /// Optional language hint, currently informational only — the underlying
    /// PaddleOCR recognition model is chosen at model-download time.
    #[allow(dead_code)]
    lang: String,
    loaded: Mutex<Option<Loaded>>,
}

impl OcrEngine {
    pub fn new(model_dir: impl Into<PathBuf>, lang: impl Into<String>) -> Self {
        Self {
            model_dir: model_dir.into(),
            lang: lang.into(),
            loaded: Mutex::new(None),
        }
    }

    /// Drop the loaded MNN session. Lazily re-instantiated on the next call.
    pub fn unload(&self) {
        *self.loaded.lock().unwrap() = None;
    }

    /// Recognize text from an in-memory image (PNG/JPEG/WebP/BMP/GIF bytes).
    pub fn recognize_bytes(&self, bytes: &[u8]) -> Result<OcrResult> {
        let image = image::load_from_memory(bytes).context("decoding image")?;
        self.recognize_image(&image)
    }

    /// Recognize text from an image file on disk.
    pub fn recognize_path(&self, path: &Path) -> Result<OcrResult> {
        let image =
            image::open(path).with_context(|| format!("opening image {}", path.display()))?;
        self.recognize_image(&image)
    }

    fn recognize_image(&self, image: &image::DynamicImage) -> Result<OcrResult> {
        self.ensure_loaded()?;
        let guard = self.loaded.lock().unwrap();
        let loaded = guard.as_ref().expect("ensure_loaded populates");
        let raw = loaded
            .inner
            .recognize(image)
            .map_err(|e| anyhow!("OCR recognize failed: {e}"))?;
        let blocks: Vec<OcrBlock> = raw
            .into_iter()
            .map(|r| OcrBlock {
                text: r.text,
                confidence: r.confidence,
                bbox: (
                    r.bbox.rect.left(),
                    r.bbox.rect.top(),
                    r.bbox.rect.width(),
                    r.bbox.rect.height(),
                ),
            })
            .collect();
        let text = blocks
            .iter()
            .map(|b| b.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        Ok(OcrResult { text, blocks })
    }

    fn ensure_loaded(&self) -> Result<()> {
        let mut guard = self.loaded.lock().unwrap();
        if guard.is_some() {
            return Ok(());
        }
        let (det, rec, keys) = installed_model_files(&self.model_dir);
        for (label, p) in [("det", &det), ("rec", &rec), ("keys", &keys)] {
            if !p.exists() {
                return Err(anyhow!(
                    "OCR model file missing ({}): {}",
                    label,
                    p.display()
                ));
            }
        }
        let cfg = build_config();
        let inner = ocr_rs::OcrEngine::new(&det, &rec, &keys, cfg)
            .map_err(|e| anyhow!("OcrEngine::new failed: {e}"))?;
        *guard = Some(Loaded { inner });
        Ok(())
    }
}

#[cfg(feature = "ocr-paddle-metal")]
fn build_config() -> Option<ocr_rs::OcrEngineConfig> {
    Some(ocr_rs::OcrEngineConfig::new().with_backend(ocr_rs::Backend::Metal))
}

#[cfg(not(feature = "ocr-paddle-metal"))]
fn build_config() -> Option<ocr_rs::OcrEngineConfig> {
    // Default CPU backend on non-Metal builds; the crate uses sensible defaults.
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unload_resets_state() {
        let e = OcrEngine::new("/nonexistent", "vi");
        e.unload(); // should not panic on an empty engine
        assert!(e.loaded.lock().unwrap().is_none());
    }

    #[test]
    fn ensure_loaded_fails_when_files_missing() {
        let e = OcrEngine::new("/definitely/does/not/exist", "vi");
        let err = e.ensure_loaded().unwrap_err();
        assert!(err.to_string().contains("missing"));
    }
}
