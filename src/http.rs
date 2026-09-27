//! `sen-ocr`'s own HTTP surface: the old daemon namespace `/api/ocr/*`, served
//! **verbatim** — same paths, request and response bodies, status codes —
//! because the daemon's proxy, its internal OCR-fallback client and the
//! `senclaw-ocr` MCP server keep calling `/api/ocr/recognize` through it.
//! Ported from the daemon's `src/gateway/ui_server/ocr.rs`; only the plumbing
//! that read the daemon's `Config`/`UiState` changed, none of the behaviour.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::{
    extract::{Path as AxumPath, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::get,
    Router,
};
use axum_extra::extract::Multipart;
use futures::StreamExt;
use once_cell::sync::Lazy;
use sen_runtime_sdk::api::ErrorBody;
use sen_runtime_sdk::env::LaunchEnv;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::io::AsyncWriteExt;
use tokio_util::sync::CancellationToken;

use crate::ocr::{installed_model_files, CatalogEntry, CATALOG, DEFAULT_MODEL_ID, DET_FILE, KEYS_FILE, REC_FILE};
use crate::settings_store::{self, OcrSettings};

/// Everything a handler needs besides the request itself.
pub struct AppState {
    pub env: LaunchEnv,
}

/// `SENCLAW_OCR_MODELS_DIR`, else `<SENCLAW_HOME>/ocr-models` — engine-private,
/// so (unlike the shared `local-models` root) it is read directly rather than
/// through a field on [`LaunchEnv`].
fn models_root(env: &LaunchEnv) -> PathBuf {
    std::env::var("SENCLAW_OCR_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| env.home.join("ocr-models"))
}

pub struct AppError(pub StatusCode, pub String);

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        (self.0, Json(ErrorBody::new(self.1))).into_response()
    }
}

fn catalog_get(id: &str) -> Option<&'static CatalogEntry> {
    CATALOG.iter().find(|e| e.id == id)
}

fn safe_dirname(id: &str) -> String {
    id.replace('/', "__")
}

fn model_dir(state: &AppState, id: &str) -> PathBuf {
    models_root(&state.env).join(safe_dirname(id))
}

fn is_installed(dir: &PathBuf) -> bool {
    let (det, rec, keys) = installed_model_files(dir);
    [det, rec, keys]
        .iter()
        .all(|p| std::fs::metadata(p).map(|m| m.len() > 0).unwrap_or(false))
}

// ── Download progress (process-global) ───────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DownloadStatus {
    Queued,
    Downloading,
    Done,
    Error,
    Cancelled,
}

#[derive(Debug, Clone, Serialize)]
struct DownloadState {
    model_id: String,
    status: DownloadStatus,
    total_bytes: u64,
    downloaded_bytes: u64,
    current_file: Option<String>,
    files_total: u32,
    files_done: u32,
    error: Option<String>,
}

#[derive(Clone)]
struct DownloadHandle {
    state: Arc<Mutex<DownloadState>>,
    cancel: CancellationToken,
}

static DOWNLOADS: Lazy<Mutex<HashMap<String, DownloadHandle>>> = Lazy::new(|| Mutex::new(HashMap::new()));

// ── Routes: model listing ────────────────────────────────────────────────────

async fn ocr_models_list(State(state): State<Arc<AppState>>) -> Result<impl IntoResponse, AppError> {
    let downloads = DOWNLOADS.lock().unwrap();
    let mut models = Vec::new();
    for e in CATALOG {
        let dir = model_dir(&state, e.id);
        let download = downloads.get(e.id).map(|h| h.state.lock().unwrap().clone());
        models.push(json!({
            "id": e.id,
            "label": e.label,
            "description": e.description,
            "approx_size_mb": e.approx_size_mb,
            "default_language": e.default_language,
            "version": e.version,
            "is_default": e.is_default,
            "installed": is_installed(&dir),
            "on_disk_path": dir.to_string_lossy(),
            "download": download,
        }));
    }
    // Also surface any download entries the user kicked off for ids not in the
    // bundled catalog (custom URLs etc.).
    for (id, handle) in downloads.iter() {
        if catalog_get(id).is_some() || models.iter().any(|m| m["id"] == *id) {
            continue;
        }
        let dir = model_dir(&state, id);
        models.push(json!({
            "id": id,
            "label": format!("OCR custom ({id})"),
            "description": "User-supplied model URLs",
            "approx_size_mb": 0.0,
            "default_language": "vi",
            "version": 0,
            "is_default": false,
            "installed": is_installed(&dir),
            "on_disk_path": dir.to_string_lossy(),
            "download": handle.state.lock().unwrap().clone(),
        }));
    }
    // Also scan on-disk dir for any custom-installed models the user dropped
    // in manually (not in catalog, no download record).
    if let Ok(entries) = std::fs::read_dir(models_root(&state.env)) {
        for entry in entries.flatten() {
            let Ok(ft) = entry.file_type() else { continue };
            if !ft.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            // dir-name → id (reverse of safe_dirname; OCR ids never contain '/')
            let id = name.clone();
            if catalog_get(&id).is_some() || models.iter().any(|m| m["id"] == id) {
                continue;
            }
            let dir = entry.path();
            if !is_installed(&dir) {
                continue;
            }
            models.push(json!({
                "id": id,
                "label": format!("OCR custom ({id})"),
                "description": "Manually-installed model directory",
                "approx_size_mb": 0.0,
                "default_language": "vi",
                "version": 0,
                "is_default": false,
                "installed": true,
                "on_disk_path": dir.to_string_lossy(),
                "download": serde_json::Value::Null,
            }));
        }
    }
    Ok(Json(json!({
        "models": models,
        "default_model_id": DEFAULT_MODEL_ID,
    })))
}

// ── Routes: download ─────────────────────────────────────────────────────────

async fn ocr_download(
    State(state): State<Arc<AppState>>,
    AxumPath(id): AxumPath<String>,
) -> Result<impl IntoResponse, AppError> {
    let entry = catalog_get(&id).ok_or_else(|| AppError(StatusCode::BAD_REQUEST, format!("unknown OCR model id `{id}`")))?;

    {
        let downloads = DOWNLOADS.lock().unwrap();
        if let Some(h) = downloads.get(&id) {
            let s = h.state.lock().unwrap();
            if matches!(s.status, DownloadStatus::Queued | DownloadStatus::Downloading) {
                return Err(AppError(StatusCode::CONFLICT, format!("download for {id} already in progress")));
            }
        }
    }

    let dir = model_dir(&state, &id);
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let progress = Arc::new(Mutex::new(DownloadState {
        model_id: id.clone(),
        status: DownloadStatus::Queued,
        total_bytes: 0,
        downloaded_bytes: 0,
        current_file: None,
        files_total: 3,
        files_done: 0,
        error: None,
    }));
    let cancel = CancellationToken::new();
    DOWNLOADS.lock().unwrap().insert(id.clone(), DownloadHandle { state: progress.clone(), cancel: cancel.clone() });

    let files = vec![
        (entry.det_url.to_string(), DET_FILE.to_string()),
        (entry.rec_url.to_string(), REC_FILE.to_string()),
        (entry.keys_url.to_string(), KEYS_FILE.to_string()),
    ];

    tokio::spawn(async move {
        let result = run_ocr_download(files, &dir, progress.clone(), cancel).await;
        let mut s = progress.lock().unwrap();
        match result {
            Ok(()) if s.status != DownloadStatus::Cancelled => s.status = DownloadStatus::Done,
            Ok(()) => {}
            Err(e) => {
                s.status = DownloadStatus::Error;
                s.error = Some(e.to_string());
            }
        }
    });

    Ok(Json(json!({ "ok": true, "id": id })))
}

// ── Route: custom URL download ───────────────────────────────────────────────

#[derive(Deserialize)]
struct OcrCustomDownloadBody {
    /// Free-form id chosen by the user (e.g. `my-vietnamese-v5`). Used as the
    /// directory name; must not contain `/`.
    id: String,
    det_url: String,
    rec_url: String,
    keys_url: String,
}

async fn ocr_custom_download(
    State(state): State<Arc<AppState>>,
    Json(body): Json<OcrCustomDownloadBody>,
) -> Result<impl IntoResponse, AppError> {
    let id = body.id.trim().to_string();
    if id.is_empty() || id.contains('/') || id.contains('\\') || id.contains("..") {
        return Err(AppError(StatusCode::BAD_REQUEST, "invalid model id (no '/', '\\\\', '..')".into()));
    }
    for (label, url) in [("det_url", &body.det_url), ("rec_url", &body.rec_url), ("keys_url", &body.keys_url)] {
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err(AppError(StatusCode::BAD_REQUEST, format!("{label} must start with http(s)://")));
        }
    }

    {
        let downloads = DOWNLOADS.lock().unwrap();
        if let Some(h) = downloads.get(&id) {
            let s = h.state.lock().unwrap();
            if matches!(s.status, DownloadStatus::Queued | DownloadStatus::Downloading) {
                return Err(AppError(StatusCode::CONFLICT, format!("download for {id} already in progress")));
            }
        }
    }

    let dir = model_dir(&state, &id);
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let progress = Arc::new(Mutex::new(DownloadState {
        model_id: id.clone(),
        status: DownloadStatus::Queued,
        total_bytes: 0,
        downloaded_bytes: 0,
        current_file: None,
        files_total: 3,
        files_done: 0,
        error: None,
    }));
    let cancel = CancellationToken::new();
    DOWNLOADS.lock().unwrap().insert(id.clone(), DownloadHandle { state: progress.clone(), cancel: cancel.clone() });

    let files = vec![
        (body.det_url, DET_FILE.to_string()),
        (body.rec_url, REC_FILE.to_string()),
        (body.keys_url, KEYS_FILE.to_string()),
    ];

    tokio::spawn(async move {
        let result = run_ocr_download(files, &dir, progress.clone(), cancel).await;
        let mut s = progress.lock().unwrap();
        match result {
            Ok(()) if s.status != DownloadStatus::Cancelled => s.status = DownloadStatus::Done,
            Ok(()) => {}
            Err(e) => {
                s.status = DownloadStatus::Error;
                s.error = Some(e.to_string());
            }
        }
    });

    Ok(Json(json!({ "ok": true, "id": id })))
}

async fn ocr_status(AxumPath(id): AxumPath<String>) -> Result<impl IntoResponse, AppError> {
    let downloads = DOWNLOADS.lock().unwrap();
    let progress = downloads.get(&id).map(|h| h.state.lock().unwrap().clone());
    Ok(Json(json!({ "id": id, "download": progress })))
}

async fn ocr_cancel(AxumPath(id): AxumPath<String>) -> Result<impl IntoResponse, AppError> {
    let downloads = DOWNLOADS.lock().unwrap();
    if let Some(h) = downloads.get(&id) {
        h.cancel.cancel();
        h.state.lock().unwrap().status = DownloadStatus::Cancelled;
    }
    Ok(Json(json!({ "ok": true })))
}

async fn ocr_delete(State(state): State<Arc<AppState>>, AxumPath(id): AxumPath<String>) -> Result<impl IntoResponse, AppError> {
    let dir = model_dir(&state, &id);
    if dir.exists() {
        tokio::fs::remove_dir_all(&dir)
            .await
            .map_err(|e| AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }
    DOWNLOADS.lock().unwrap().remove(&id);
    drop_engine(&dir);
    Ok(Json(json!({ "ok": true })))
}

// ── Routes: settings ─────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct OcrSettingsBody {
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    language: Option<String>,
}

/// Pick the first installed model — preferring the catalog default, then the
/// rest of the catalog in order, then any custom-installed dirs.
fn auto_select_model_id(state: &AppState) -> Option<String> {
    if !DEFAULT_MODEL_ID.is_empty() && is_installed(&model_dir(state, DEFAULT_MODEL_ID)) {
        return Some(DEFAULT_MODEL_ID.to_string());
    }
    for e in CATALOG {
        if is_installed(&model_dir(state, e.id)) {
            return Some(e.id.to_string());
        }
    }
    if let Ok(entries) = std::fs::read_dir(models_root(&state.env)) {
        for entry in entries.flatten() {
            let Ok(ft) = entry.file_type() else { continue };
            if !ft.is_dir() {
                continue;
            }
            if is_installed(&entry.path()) {
                return Some(entry.file_name().to_string_lossy().to_string());
            }
        }
    }
    None
}

async fn ocr_settings_get(State(state): State<Arc<AppState>>) -> Result<impl IntoResponse, AppError> {
    let mut s = settings_store::load(&state.env);
    // Auto-promote the first installed model so the user doesn't have to
    // manually pick after their first download.
    if s.model_id.is_none() {
        if let Some(id) = auto_select_model_id(&state) {
            s.model_id = Some(id.clone());
            let _ = settings_store::save(&state.env, &s);
        }
    }
    Ok(Json(json!({
        "model_id": s.model_id,
        "language": s.language.unwrap_or_else(|| "vi".to_string()),
        "default_model_id": DEFAULT_MODEL_ID,
    })))
}

async fn ocr_settings_put(
    State(state): State<Arc<AppState>>,
    Json(body): Json<OcrSettingsBody>,
) -> Result<impl IntoResponse, AppError> {
    let settings = OcrSettings { model_id: body.model_id, language: body.language };
    settings_store::save(&state.env, &settings).map_err(|e| AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(json!({ "ok": true })))
}

// ── Route: recognize (multipart image) ───────────────────────────────────────

async fn ocr_recognize(State(state): State<Arc<AppState>>, mut multipart: Multipart) -> Result<impl IntoResponse, AppError> {
    let mut image: Option<Vec<u8>> = None;
    let mut language: Option<String> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError(StatusCode::BAD_REQUEST, format!("read multipart: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        if name == "language" {
            language = field.text().await.ok().filter(|s| !s.is_empty());
        } else {
            let bytes = field.bytes().await.map_err(|e| AppError(StatusCode::BAD_REQUEST, format!("read image: {e}")))?;
            image = Some(bytes.to_vec());
        }
    }

    let bytes = image.ok_or_else(|| AppError(StatusCode::BAD_REQUEST, "no image field".into()))?;

    let settings = settings_store::load(&state.env);
    let model_id = settings
        .model_id
        .clone()
        .or_else(|| CATALOG.iter().map(|e| e.id.to_string()).find(|id| is_installed(&model_dir(&state, id))))
        .ok_or_else(|| AppError(StatusCode::BAD_REQUEST, "no OCR model selected or installed".into()))?;
    let dir = model_dir(&state, &model_id);
    if !is_installed(&dir) {
        return Err(AppError(StatusCode::BAD_REQUEST, format!("model `{model_id}` is not installed")));
    }
    let lang = language.or(settings.language).unwrap_or_else(|| "vi".into());

    recognize_impl(dir, bytes, lang).await
}

// ── Download worker ──────────────────────────────────────────────────────────

async fn run_ocr_download(
    files: Vec<(String, String)>,
    dir: &PathBuf,
    progress: Arc<Mutex<DownloadState>>,
    cancel: CancellationToken,
) -> anyhow::Result<()> {
    let client = reqwest::Client::builder().connect_timeout(std::time::Duration::from_secs(30)).build()?;

    progress.lock().unwrap().status = DownloadStatus::Downloading;

    for (url, filename) in files {
        if cancel.is_cancelled() {
            progress.lock().unwrap().status = DownloadStatus::Cancelled;
            return Ok(());
        }
        progress.lock().unwrap().current_file = Some(filename.clone());

        let dst = dir.join(&filename);
        if let Some(parent) = dst.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // HEAD probe for size (best-effort; non-fatal if it fails).
        let expected_size = client.head(&url).send().await.ok().and_then(|r| r.content_length());
        if let Some(size) = expected_size {
            if let Ok(meta) = tokio::fs::metadata(&dst).await {
                if meta.len() == size {
                    let mut s = progress.lock().unwrap();
                    s.files_done += 1;
                    s.downloaded_bytes += size;
                    continue;
                }
            }
            let mut s = progress.lock().unwrap();
            s.total_bytes += size;
        }

        let resp = client.get(&url).send().await?.error_for_status()?;
        let mut stream = resp.bytes_stream();
        let mut file = tokio::fs::File::create(&dst).await?;
        while let Some(chunk) = stream.next().await {
            if cancel.is_cancelled() {
                drop(file);
                let _ = tokio::fs::remove_file(&dst).await;
                progress.lock().unwrap().status = DownloadStatus::Cancelled;
                return Ok(());
            }
            let bytes = chunk?;
            file.write_all(&bytes).await?;
            progress.lock().unwrap().downloaded_bytes += bytes.len() as u64;
        }
        file.flush().await?;
        progress.lock().unwrap().files_done += 1;
    }

    Ok(())
}

// ── Engine cache + recognize bridge ──────────────────────────────────────────

#[cfg(feature = "ocr-paddle")]
static ENGINES: Lazy<Mutex<HashMap<String, Arc<crate::ocr::OcrEngine>>>> = Lazy::new(|| Mutex::new(HashMap::new()));

#[cfg(feature = "ocr-paddle")]
fn get_or_create_engine(dir: &PathBuf, lang: &str) -> Arc<crate::ocr::OcrEngine> {
    let key = dir.to_string_lossy().to_string();
    let mut map = ENGINES.lock().unwrap();
    map.entry(key).or_insert_with(|| Arc::new(crate::ocr::OcrEngine::new(dir.clone(), lang))).clone()
}

#[cfg(feature = "ocr-paddle")]
fn drop_engine(dir: &PathBuf) {
    ENGINES.lock().unwrap().remove(&dir.to_string_lossy().to_string());
}

#[cfg(not(feature = "ocr-paddle"))]
fn drop_engine(_dir: &PathBuf) {}

#[cfg(feature = "ocr-paddle")]
async fn recognize_impl(dir: PathBuf, bytes: Vec<u8>, language: String) -> Result<axum::response::Json<serde_json::Value>, AppError> {
    let engine = get_or_create_engine(&dir, &language);
    let result = tokio::task::spawn_blocking(move || {
        let res = engine.recognize_bytes(&bytes);
        engine.unload();
        res
    })
    .await
    .map_err(|e| AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(json!({ "ok": true, "text": result.text, "blocks": result.blocks })))
}

#[cfg(not(feature = "ocr-paddle"))]
async fn recognize_impl(_dir: PathBuf, _bytes: Vec<u8>, _language: String) -> Result<axum::response::Json<serde_json::Value>, AppError> {
    Err(AppError(
        StatusCode::NOT_IMPLEMENTED,
        "OCR requires building with `--features ocr-paddle` (or `ocr-paddle-metal` on macOS)".into(),
    ))
}

/// The full router this runtime serves.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/ocr/models", get(ocr_models_list))
        .route("/api/ocr/models/custom", axum::routing::post(ocr_custom_download))
        .route("/api/ocr/models/:id/download", axum::routing::post(ocr_download))
        .route("/api/ocr/models/:id/status", get(ocr_status))
        .route("/api/ocr/models/:id/cancel", axum::routing::post(ocr_cancel))
        .route("/api/ocr/models/:id", axum::routing::delete(ocr_delete))
        .route("/api/ocr/settings", get(ocr_settings_get).put(ocr_settings_put))
        .route("/api/ocr/recognize", axum::routing::post(ocr_recognize))
        // OCR images can exceed axum's 2 MB default body limit.
        .layer(axum::extract::DefaultBodyLimit::max(25 * 1024 * 1024))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_at(dir: &std::path::Path) -> Arc<AppState> {
        let env = LaunchEnv::from_lookup("sen-ocr", "0.0.0-test", |k| match k {
            "SENCLAW_RUNTIME_DATA_DIR" => Some(dir.join("data").to_string_lossy().into_owned()),
            "SENCLAW_OCR_MODELS_DIR" => Some(dir.join("ocr-models").to_string_lossy().into_owned()),
            "SENCLAW_CONFIG_PATH" => Some(dir.join("config.json").to_string_lossy().into_owned()),
            "SENCLAW_HOME" => Some(dir.to_string_lossy().into_owned()),
            _ => None,
        });
        Arc::new(AppState { env })
    }

    #[tokio::test]
    async fn models_and_settings_answer_with_the_old_shapes() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let tmp = tempfile::tempdir().unwrap();
        let app = router(state_at(tmp.path()));

        let resp = app
            .clone()
            .oneshot(Request::get("/api/ocr/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["default_model_id"], DEFAULT_MODEL_ID);
        assert!(v["models"].as_array().unwrap().iter().any(|m| m["id"] == DEFAULT_MODEL_ID));

        let resp = app
            .oneshot(Request::get("/api/ocr/settings").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["language"], "vi");
    }

    #[tokio::test]
    async fn recognize_without_an_image_field_is_a_bad_request() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let tmp = tempfile::tempdir().unwrap();
        let app = router(state_at(tmp.path()));
        let resp = app
            .oneshot(
                Request::post("/api/ocr/recognize")
                    .header("content-type", "multipart/form-data; boundary=X")
                    .body(Body::from("--X--\r\n"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
