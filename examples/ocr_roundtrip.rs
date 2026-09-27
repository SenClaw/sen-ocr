//! OCR round-trip test + benchmark — render text, then read it back with OCR.
//!
//! Rasterizes each known phrase onto a clean white image, runs the production
//! on-device PaddleOCR engine (`sen_ocr::ocr::OcrEngine`, the same type the
//! `/api/ocr/recognize` route holds in its cache), and reports word-recall
//! vs. the original phrase. Use it to confirm whether a downloaded OCR model
//! actually recognizes text — and how fast.
//!
//! Self-contained: no external image fixtures needed (it draws its own), unlike
//! `whisper_bench` which consumes audio clips. This mirrors `tts_roundtrip`,
//! which generates audio then verifies it with Whisper.
//!
//! ```bash
//! # Default model dir is ~/.senclaw/ocr-models/PP-OCRv5_mobile_latin :
//! cargo run --release --features ocr-paddle-metal --example ocr_roundtrip -- \
//!   --model ~/.senclaw/ocr-models/PP-OCRv5_mobile_latin --lang vi --iters 3
//!
//! # English, custom phrases, keep the rendered PNGs for inspection:
//! cargo run --release --features ocr-paddle --example ocr_roundtrip -- \
//!   --model ~/.senclaw/ocr-models/PP-OCRv5_mobile_latin --lang en \
//!   --phrase "Hello world" --phrase "Invoice #12345" --save-png /tmp/ocr-out
//! ```
//!
//! Flags: `--model <dir>` (det.mnn + rec.mnn + keys.txt), `--lang vi|en|...`,
//! `--iters N` (recognize passes per phrase, for timing + determinism),
//! `--font <path.ttf>` (override font autodetection — needs glyph coverage for
//! the language), `--phrase "..."` (repeatable; overrides the built-in set),
//! `--font-size <px>`, `--save-png <dir>`, `--threshold <0..1>` word recall to
//! pass. Exit code 0 = all passed, 3 = some failed, 2 = nothing ran.

#[cfg(not(feature = "ocr-paddle"))]
fn main() {
    eprintln!("build with --features ocr-paddle (or ocr-paddle-metal on macOS)");
}

#[cfg(feature = "ocr-paddle")]
fn main() {
    if let Err(e) = run() {
        eprintln!("ocr_roundtrip failed: {e:#}");
        std::process::exit(1);
    }
}

#[cfg(feature = "ocr-paddle")]
fn run() -> anyhow::Result<()> {
    use ab_glyph::{FontVec, PxScale};
    use anyhow::{anyhow, Context};
    use image::{Rgb, RgbImage};
    use imageproc::drawing::{draw_text_mut, text_size};
    use sen_ocr::ocr::OcrEngine;
    use std::path::PathBuf;
    use std::time::Instant;

    // ── Args (tiny ad-hoc parser; keep clap out of examples) ─────────────────
    let argv: Vec<String> = std::env::args().collect();
    let mut model_dir: Option<PathBuf> = None;
    let mut lang = "vi".to_string();
    let mut iters: usize = 1;
    let mut font_path: Option<PathBuf> = None;
    let mut font_size: f32 = 44.0;
    let mut phrases: Vec<String> = Vec::new();
    let mut save_dir: Option<PathBuf> = None;
    let mut threshold: f64 = 0.6;

    let mut i = 1;
    while i < argv.len() {
        let a = argv[i].as_str();
        let next = || -> anyhow::Result<String> {
            argv.get(i + 1)
                .cloned()
                .ok_or_else(|| anyhow!("missing value for {a}"))
        };
        match a {
            "--model" => {
                model_dir = Some(PathBuf::from(next()?));
                i += 2;
            }
            "--lang" => {
                lang = next()?;
                i += 2;
            }
            "--iters" => {
                iters = next()?.parse().context("--iters")?;
                i += 2;
            }
            "--font" => {
                font_path = Some(PathBuf::from(next()?));
                i += 2;
            }
            "--font-size" => {
                font_size = next()?.parse().context("--font-size")?;
                i += 2;
            }
            "--phrase" => {
                phrases.push(next()?);
                i += 2;
            }
            "--save-png" => {
                save_dir = Some(PathBuf::from(next()?));
                i += 2;
            }
            "--threshold" => {
                threshold = next()?.parse().context("--threshold")?;
                i += 2;
            }
            "-h" | "--help" => {
                usage();
                return Ok(());
            }
            other => return Err(anyhow!("unknown arg: {other}")),
        }
    }

    let model_dir = match model_dir {
        Some(d) => d,
        None => PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join(".senclaw/ocr-models/PP-OCRv5_mobile_latin"),
    };
    if phrases.is_empty() {
        phrases = default_phrases(&lang);
    }
    if let Some(dir) = &save_dir {
        std::fs::create_dir_all(dir).ok();
    }

    // ── Load font (autodetect, with --font override) ─────────────────────────
    let font_file = match font_path.clone() {
        Some(p) => p,
        None => find_font().ok_or_else(|| {
            anyhow!(
                "no TTF font found automatically — pass --font <path.ttf> \
                 (needs glyph coverage for `{lang}`)"
            )
        })?,
    };
    let font_bytes = std::fs::read(&font_file)
        .with_context(|| format!("reading font {}", font_file.display()))?;
    let font = FontVec::try_from_vec(font_bytes)
        .map_err(|_| anyhow!("invalid font file: {}", font_file.display()))?;
    let scale = PxScale::from(font_size);

    println!("════════════════════════════════════════════════════════════════");
    println!(" OCR round-trip (render text → OCR → word recall)");
    println!("════════════════════════════════════════════════════════════════");
    println!("  model     : {}", model_dir.display());
    println!("  language  : {lang}");
    println!("  font      : {}", font_file.display());
    println!("  font size : {font_size} px");
    println!("  iters     : {iters} (recognize passes per phrase)");
    println!(
        "  threshold : {:.0}% word recall to pass",
        threshold * 100.0
    );
    println!("  phrases   : {}", phrases.len());
    println!("  backend   : {}", ocr_backend());
    println!("════════════════════════════════════════════════════════════════");
    println!();

    let engine = OcrEngine::new(model_dir.clone(), lang.clone());

    struct PhraseResult {
        idx: usize,
        recall: f64,
        recognized: String,
        avg_conf: f32,
        blocks: usize,
        ocr_ms_med: f64,
        deterministic: bool,
    }
    let mut results: Vec<PhraseResult> = Vec::new();
    let mut any_failed = false;

    for (idx, phrase) in phrases.iter().enumerate() {
        println!(
            "┌─ phrase {} ────────────────────────────────────────────────",
            idx + 1
        );
        println!("│  expected: {phrase}");

        // ── 1. Render the phrase to a white PNG with black text ──────────────
        let pad = (font_size * 0.6) as i32;
        let (tw, th) = text_size(scale, &font, phrase);
        let w = (tw as i32 + pad * 2).max(64) as u32;
        let h = (th as i32 + pad * 2).max(48) as u32;
        let mut img = RgbImage::from_pixel(w, h, Rgb([255, 255, 255]));
        draw_text_mut(&mut img, Rgb([0, 0, 0]), pad, pad, scale, &font, phrase);

        let mut png: Vec<u8> = Vec::new();
        image::DynamicImage::ImageRgb8(img.clone())
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .context("encoding PNG")?;

        if let Some(dir) = &save_dir {
            let path = dir.join(format!("phrase_{:02}.png", idx + 1));
            img.save(&path).ok();
            println!("│  saved   : {}", path.display());
        }

        // ── 2. Recognize (iters passes for timing + determinism) ─────────────
        let mut ocr_times = Vec::new();
        let mut recognized_runs: Vec<String> = Vec::new();
        let mut last_avg_conf = 0.0_f32;
        let mut last_blocks = 0usize;
        let mut rec_err: Option<String> = None;
        for _ in 0..iters.max(1) {
            let t0 = Instant::now();
            let r = engine.recognize_bytes(&png);
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            engine.unload(); // mirror the HTTP handler: free MNN session per call
            match r {
                Ok(res) => {
                    ocr_times.push(elapsed);
                    last_blocks = res.blocks.len();
                    last_avg_conf = if res.blocks.is_empty() {
                        0.0
                    } else {
                        res.blocks.iter().map(|b| b.confidence).sum::<f32>()
                            / res.blocks.len() as f32
                    };
                    recognized_runs.push(res.text);
                }
                Err(e) => {
                    rec_err = Some(e.to_string());
                    break;
                }
            }
        }

        if let Some(err) = rec_err {
            println!("│  ocr     : ✗ FAILED — {err}");
            println!("└──────────────────────────────────────────────────────────");
            println!();
            any_failed = true;
            continue;
        }

        let deterministic = recognized_runs.windows(2).all(|w| w[0] == w[1]);
        let recognized = recognized_runs[0].replace('\n', " ");
        let recall = word_accuracy(phrase, &recognized);
        let ocr_med = median(&mut ocr_times);

        println!("│  got     : {recognized}");
        println!(
            "│  recall  : {:.1}%   blocks={last_blocks}   avg conf={:.2}",
            recall * 100.0,
            last_avg_conf
        );
        println!("│  timing  : ocr median {ocr_med:.0} ms");
        println!(
            "│  verdict : {} {}",
            if recall >= threshold {
                "✓ PASS"
            } else {
                "✗ FAIL"
            },
            if deterministic {
                "(deterministic)"
            } else {
                "(VARIED — non-deterministic)"
            }
        );
        println!("└──────────────────────────────────────────────────────────");
        println!();

        if recall < threshold {
            any_failed = true;
        }
        results.push(PhraseResult {
            idx,
            recall,
            recognized,
            avg_conf: last_avg_conf,
            blocks: last_blocks,
            ocr_ms_med: ocr_med,
            deterministic,
        });
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    println!("════════════════════════════════════════════════════════════════");
    println!(" Summary");
    println!("════════════════════════════════════════════════════════════════");
    if results.is_empty() {
        println!("  ✗ No phrases recognized — OCR pipeline non-functional.");
        std::process::exit(2);
    }
    let n = results.len();
    let passed = results.iter().filter(|r| r.recall >= threshold).count();
    let mean_recall: f64 = results.iter().map(|r| r.recall).sum::<f64>() / n as f64;
    let mean_ocr: f64 = results.iter().map(|r| r.ocr_ms_med).sum::<f64>() / n as f64;

    println!(
        "  {passed}/{n} phrases passed (≥ {:.0}% word recall)",
        threshold * 100.0
    );
    println!("  mean recall : {:.1}%", mean_recall * 100.0);
    println!("  mean ocr    : {mean_ocr:.0} ms");
    println!();
    println!(
        "  {:<3} {:<8} {:>5} {:>6} {:>9} {:>4}  {}",
        "#", "recall", "blk", "conf", "ocr(ms)", "det", "recognized (first chars)"
    );
    for r in &results {
        let head: String = r.recognized.chars().take(46).collect();
        println!(
            "  {:<3} {:<7.1}% {:>5} {:>6.2} {:>9.0} {:>4}  {}",
            r.idx + 1,
            r.recall * 100.0,
            r.blocks,
            r.avg_conf,
            r.ocr_ms_med,
            if r.deterministic { "✓" } else { "✗" },
            head,
        );
    }
    println!();

    if passed < n || any_failed {
        std::process::exit(3);
    }
    Ok(())
}

/// Multiset word-recall of `got` against `expected` in [0,1] (1 = every
/// reference word present). Tolerant of word order and extra words. Mirrors
/// `whisper_transcribe::word_accuracy`, inlined here to avoid coupling the OCR
/// harness to the `local-mlx-whisper` feature.
#[cfg(feature = "ocr-paddle")]
fn word_accuracy(expected: &str, got: &str) -> f64 {
    fn norm(s: &str) -> Vec<String> {
        s.to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { ' ' })
            .collect::<String>()
            .split_whitespace()
            .map(|w| w.to_string())
            .collect()
    }
    let exp = norm(expected);
    if exp.is_empty() {
        return 1.0;
    }
    let mut got_counts: std::collections::HashMap<String, i32> = std::collections::HashMap::new();
    for w in norm(got) {
        *got_counts.entry(w).or_default() += 1;
    }
    let mut hits = 0usize;
    for w in &exp {
        if let Some(c) = got_counts.get_mut(w) {
            if *c > 0 {
                *c -= 1;
                hits += 1;
            }
        }
    }
    hits as f64 / exp.len() as f64
}

#[cfg(feature = "ocr-paddle")]
fn ocr_backend() -> &'static str {
    if cfg!(feature = "ocr-paddle-metal") {
        "Metal/CoreML (ocr-paddle-metal)"
    } else {
        "CPU (ocr-paddle)"
    }
}

#[cfg(feature = "ocr-paddle")]
fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Search a few well-known font locations with broad Unicode (incl. Vietnamese)
/// coverage. Returns the first that exists.
#[cfg(feature = "ocr-paddle")]
fn find_font() -> Option<std::path::PathBuf> {
    const CANDIDATES: &[&str] = &[
        // macOS — Arial Unicode has full Vietnamese coverage.
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        // Linux — DejaVu / Noto are the usual suspects.
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        // Windows.
        "C:\\Windows\\Fonts\\arial.ttf",
    ];
    CANDIDATES
        .iter()
        .map(std::path::PathBuf::from)
        .find(|p| p.exists())
}

#[cfg(feature = "ocr-paddle")]
fn usage() {
    eprintln!(
        "usage: ocr_roundtrip --model <dir> [--lang vi] [--iters N] [--font <path.ttf>]\n\
         \t[--font-size 44] [--phrase \"...\"]* [--save-png <dir>] [--threshold 0.6]\n\n\
         <dir> holds det.mnn + rec.mnn + keys.txt (download via the Settings UI or\n\
         drop them into ~/.senclaw/ocr-models/<id>/)."
    );
}

#[cfg(feature = "ocr-paddle")]
fn default_phrases(lang: &str) -> Vec<String> {
    match lang {
        "vi" => vec![
            "Xin chào thế giới".to_string(),
            "Hà Nội là thủ đô Việt Nam".to_string(),
            "Trí tuệ nhân tạo".to_string(),
            "Hóa đơn số 12345".to_string(),
        ],
        _ => vec![
            "Hello world".to_string(),
            "The quick brown fox".to_string(),
            "Invoice number 12345".to_string(),
            "Artificial intelligence".to_string(),
        ],
    }
}
