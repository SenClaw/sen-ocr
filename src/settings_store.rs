//! OCR UI settings: selected PaddleOCR model id + default language.
//!
//! Persisted as `<data_dir>/settings.json`, seeded once from the daemon's old
//! `ocrConfig` key ([`sen_runtime_sdk::legacy::load_or_import`]) and read per
//! request so a save applies to the next call without a restart. Unlike the
//! old daemon file this one is private to `sen-ocr`, so there are no other
//! keys to preserve on a save.

use std::path::Path;

use sen_runtime_sdk::env::LaunchEnv;
use serde::{Deserialize, Serialize};

const LEGACY_KEY: &str = "ocrConfig";

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct OcrSettings {
    #[serde(rename = "modelId", default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

pub fn load(env: &LaunchEnv) -> OcrSettings {
    match sen_runtime_sdk::legacy::load_or_import(&env.data_dir, &env.config_path, LEGACY_KEY) {
        Some(v) => serde_json::from_value(v).unwrap_or_default(),
        None => OcrSettings::default(),
    }
}

pub fn save(env: &LaunchEnv, settings: &OcrSettings) -> std::io::Result<()> {
    write_atomic(&env.data_dir, settings)
}

fn write_atomic(data_dir: &Path, settings: &OcrSettings) -> std::io::Result<()> {
    std::fs::create_dir_all(data_dir)?;
    let body = serde_json::to_string_pretty(settings).map_err(std::io::Error::other)?;
    let tmp = data_dir.join("settings.json.tmp");
    std::fs::write(&tmp, body)?;
    std::fs::rename(tmp, data_dir.join("settings.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_at(dir: &Path) -> LaunchEnv {
        LaunchEnv::from_lookup("sen-ocr", "0.0.0-test", |k| match k {
            "SENCLAW_RUNTIME_DATA_DIR" => Some(dir.join("data").to_string_lossy().into_owned()),
            "SENCLAW_CONFIG_PATH" => Some(dir.join("config.json").to_string_lossy().into_owned()),
            "SENCLAW_HOME" => Some(dir.to_string_lossy().into_owned()),
            _ => None,
        })
    }

    #[test]
    fn imports_the_legacy_key_once_then_reads_its_own_file() {
        let tmp = tempfile::tempdir().unwrap();
        let env = env_at(tmp.path());
        std::fs::write(&env.config_path, r#"{"ocrConfig": {"modelId": "PP-OCRv5_mobile_latin", "language": "vi"}}"#).unwrap();

        let first = load(&env);
        assert_eq!(first.model_id.as_deref(), Some("PP-OCRv5_mobile_latin"));

        let mut updated = first;
        updated.language = Some("en".into());
        save(&env, &updated).unwrap();
        std::fs::write(&env.config_path, r#"{"ocrConfig": {"modelId": "other"}}"#).unwrap();
        let second = load(&env);
        assert_eq!(second.model_id.as_deref(), Some("PP-OCRv5_mobile_latin"));
        assert_eq!(second.language.as_deref(), Some("en"));
    }

    #[test]
    fn no_legacy_file_means_defaults() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(load(&env_at(tmp.path())), OcrSettings::default());
    }
}
