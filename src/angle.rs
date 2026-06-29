//! Per-crop rotation/angle detection.
//!
//! Primary pass: run the PP-OCRv6 text detector (a DBNet) via `ort`, then
//! follow PaddleOCR's `DBPostProcess` to turn the probability map into oriented
//! text boxes, and derive a skew angle from those boxes' long axes.
//!
//! Fallback pass: when too little text is found, estimate orientation from the
//! component mask itself via the minimum-area rectangle of its contour.

use std::f64::consts::{FRAC_PI_2, PI};

use anyhow::{Context, Result, anyhow};
use image::{GrayImage, Luma, Rgb, RgbImage, RgbaImage};
use imageproc::contours::find_contours;
use imageproc::geometric_transformations::{
    Border, Interpolation, Projection, rotate_about_center_no_crop, warp_into,
};
use ort::execution_providers::{CUDAExecutionProvider, DirectMLExecutionProvider};
use ort::session::Session;
use ort::value::Tensor;

/// Direct download of the PP-OCRv6 mobile detector ONNX. `commit_from_url`
/// caches it (keyed by URL hash) in the OS cache dir, so it downloads once.
const MODEL_URL: &str =
    "https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx/resolve/main/inference.onnx";

/// PP-LCNet_x1_0 textline orientation classifier (0Â° vs 180Â°). It runs on
/// individual deskewed line crops, so it is independent of the detector;
/// `commit_from_url` caches it alongside the detector.
///
/// Pinned to an immutable commit (not `main`): a moved branch ref could serve a
/// different ONNX, and a future bug in the inference engine could turn a swapped
/// model into an RCE vector. Bump this hash deliberately to update the weights.
/// Commit `06c3603` of marsena/paddleocr-onnx-models (last modified 2025-09-01).
const ORIENT_MODEL_URL: &str = "https://huggingface.co/marsena/paddleocr-onnx-models/resolve/06c3603ca8002e22e1f41d47c1aae0a251b4d940/PP-LCNet_x1_0_textline_ori_infer.onnx";

// Preprocessing, from the model's inference.yml (note: BGR channel order).
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const LIMIT_SIDE_LEN: f32 = 960.0;

// Classifier input, from PP-LCNet_x1_0_textline_ori's inference.yml: RGB, resized
// to 160x80 (WxH), scaled to 0..1 then normalized with the shared ImageNet
// `MEAN`/`STD`. At most `ORIENT_MAX_SAMPLES` line crops are voted per object.
const ORIENT_W: u32 = 160;
const ORIENT_H: u32 = 80;
const ORIENT_MAX_SAMPLES: usize = 5;

/// PP-LCNet_x1_0 document orientation classifier (4-class: 0/90/180/270). Unlike
/// the textline model it runs on the whole object image, so it is used for
/// rectangular crops where there is no reliable text line to vote on. Pinned to
/// the same immutable commit as the textline model (see [`ORIENT_MODEL_URL`]).
const DOC_ORI_MODEL_URL: &str = "https://huggingface.co/marsena/paddleocr-onnx-models/resolve/06c3603ca8002e22e1f41d47c1aae0a251b4d940/PP-LCNet_x1_0_doc_ori_infer.onnx";

// Doc-orientation input, from PP-LCNet_x1_0_doc_ori's inference.yml: resize the
// short side to 256, center-crop 224, RGB, /255 then shared ImageNet MEAN/STD.
// The object is downscaled to this long side before deskewing (orientation is
// coarse, so full resolution is wasted work).
const DOC_ORI_RESIZE: u32 = 256;
const DOC_ORI_CROP: u32 = 224;
const DOC_ORI_PRESCALE: u32 = 768;
// The classifier only reads horizontal lines (0Â° vs 180Â°). A box whose long axis
// lands more than this far from horizontal after deskew is a 90/270 line it
// can't judge, so it is excluded from the vote. 45Â° is the landscape boundary:
// at or below it, the deskewed box's longer side is horizontal.
const ORIENT_MAX_SKEW_DEG: f64 = 45.0;

// DBPostProcess parameters, from the model's inference.yml.
const DB_THRESH: f32 = 0.2;
const BOX_THRESH: f32 = 0.45;
const MIN_SIZE: f64 = 3.0;
const UNCLIP_RATIO: f64 = 1.4;

/// The result of consensus over the orientation samples: the dominant
/// orientation relative to horizontal, in degrees within `(-90, 90]` (deskew by
/// rotating `-degrees`), plus how strong the agreement was.
/// A detected text box in crop coordinates: its four corners, the long-axis
/// orientation it votes for (degrees in `(-90, 90]`), the long-side length used
/// as its weight, and the DB confidence score.
#[derive(Debug, Clone, Copy)]
pub struct TextBox {
    pub corners: [(f64, f64); 4],
    pub angle: f64,
    pub length: f64,
    pub score: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct AngleEstimate {
    pub degrees: f32,
    /// Angle of the anchor (longest) sample, before refinement.
    pub anchor: f32,
    /// Fraction of total sample weight that agrees with the anchor.
    pub agreement: f32,
    /// Number of samples agreeing with the anchor, and in total.
    pub inliers: usize,
    pub total: usize,
}

/// The PP-OCRv6 text detector session.
pub struct Detector {
    session: Session,
}

impl Detector {
    /// Build the detector, downloading the ONNX weights on first use. Execution
    /// providers are tried in order CUDA -> DirectML -> CPU. DirectML must be
    /// registered without other GPU EPs, so each provider gets its own session
    /// attempt.
    pub fn load() -> Result<Self> {
        let mut errors = Vec::new();

        let cuda = Session::builder()
            .and_then(|s| s.with_memory_pattern(false))
            .and_then(|s| s.with_execution_providers([CUDAExecutionProvider::default().build()]))
            .and_then(|s| s.commit_from_url(MODEL_URL));
        match cuda {
            Ok(session) => return Ok(Self { session }),
            Err(e) => errors.push(format!("CUDA text detector: {e:#}")),
        }

        let directml = Session::builder()
            .and_then(|s| s.with_memory_pattern(false))
            .and_then(|s| {
                s.with_execution_providers([DirectMLExecutionProvider::default().build()])
            })
            .and_then(|s| s.commit_from_url(MODEL_URL));
        match directml {
            Ok(session) => return Ok(Self { session }),
            Err(e) => errors.push(format!("DirectML text detector: {e:#}")),
        }

        let cpu = Session::builder()
            .and_then(|s| s.with_memory_pattern(false))
            .and_then(|s| s.commit_from_url(MODEL_URL));
        match cpu {
            Ok(session) => Ok(Self { session }),
            Err(e) => {
                errors.push(format!("CPU text detector: {e:#}"));
                Err(anyhow!("{}", errors.join("; ")))
                    .context("failed to download/load the PP-OCRv6 detector")
            }
        }
    }

    /// Run text detection on a crop and return the detected text boxes (corners,
    /// long-axis angle, length, score) in crop coordinates. An empty result
    /// means no confident text was found.
    pub fn text_boxes(&mut self, crop: &RgbaImage, debug: bool) -> Result<Vec<TextBox>> {
        let (input, in_w, in_h) = preprocess(crop);
        let tensor = Tensor::from_array(([1usize, 3, in_h, in_w], input))?;
        let outputs = self.session.run(ort::inputs![tensor])?;
        let (shape, prob) = outputs[0].try_extract_tensor::<f32>()?;

        // Probability map is (N, 1, H, W); take the trailing two dims.
        let pw = shape[shape.len() - 1] as usize;
        let ph = shape[shape.len() - 2] as usize;

        // Map the probability-map coordinates back to the original crop. The
        // two scales differ (each side is rounded to a multiple of 32
        // independently), so measuring the angle in map space distorts it â€” we
        // rescale every box to crop space before computing its orientation.
        let (crop_w, crop_h) = crop.dimensions();
        let sx = crop_w as f64 / pw as f64;
        let sy = crop_h as f64 / ph as f64;

        // Binarize, then trace contours of the text regions.
        let mut bitmap = GrayImage::new(pw as u32, ph as u32);
        for (i, px) in bitmap.pixels_mut().enumerate() {
            px[0] = if prob[i] > DB_THRESH { 255 } else { 0 };
        }

        let (mut n_contours, mut n_small, mut n_low_score) = (0usize, 0usize, 0usize);
        let mut boxes = Vec::new();
        for contour in find_contours::<i32>(&bitmap) {
            if contour.points.len() < 3 {
                continue;
            }
            n_contours += 1;
            let pts: Vec<(f64, f64)> = contour
                .points
                .iter()
                .map(|p| (p.x as f64, p.y as f64))
                .collect();

            // get_mini_boxes + score + box_score_fast run in map coordinates,
            // matching PaddleOCR's DBPostProcess.
            let Some(rect) = min_area_rect(&pts) else {
                continue;
            };
            if rect.w.min(rect.h) < MIN_SIZE {
                n_small += 1;
                continue;
            }
            let score = box_score(prob, pw, ph, &rect.corners);
            if score < BOX_THRESH {
                n_low_score += 1;
                continue;
            }

            // unclip: expand the shrunken DB core back to the text-line
            // envelope, then re-fit the mini box (still in map coordinates).
            let expanded = unclip(&rect.corners, UNCLIP_RATIO);
            let Some(rect) = min_area_rect(&expanded) else {
                continue;
            };

            // Rescale the box corners to crop coordinates, then re-fit so the
            // angle and length are measured in true crop pixels.
            let scaled: Vec<(f64, f64)> = rect
                .corners
                .iter()
                .map(|&(x, y)| (x * sx, y * sy))
                .collect();
            let Some(rect) = min_area_rect(&scaled) else {
                continue;
            };

            let angle = rect.long_axis_deg();
            let length = rect.w.max(rect.h);
            if debug {
                elog!("      box  angle {angle:+7.2}Â°  len {length:7.1}  score {score:.3}");
            }
            boxes.push(TextBox {
                corners: rect.corners,
                angle,
                length,
                score,
            });
        }
        if debug {
            elog!(
                "    detector: input {in_w}x{in_h}, prob map {pw}x{ph}, \
                 contours {n_contours} -> kept {} (dropped {n_small} small, {n_low_score} low-score)",
                boxes.len()
            );
        }
        Ok(boxes)
    }
}

/// Composite the crop over white, resize per PaddleOCR's `DetResizeForTest`
/// (longest side capped at `LIMIT_SIDE_LEN`, each side rounded to a multiple of
/// 32), and produce a normalized BGR `CHW` tensor.
fn preprocess(crop: &RgbaImage) -> (Vec<f32>, usize, usize) {
    let (w0, h0) = crop.dimensions();

    // Flatten transparency onto a neutral white background.
    let mut rgb = RgbImage::new(w0, h0);
    for (x, y, p) in crop.enumerate_pixels() {
        let a = p[3] as f32 / 255.0;
        let blend = |c: u8| (c as f32 * a + 255.0 * (1.0 - a)).round() as u8;
        rgb.put_pixel(x, y, image::Rgb([blend(p[0]), blend(p[1]), blend(p[2])]));
    }

    let max_side = w0.max(h0) as f32;
    let ratio = if max_side > LIMIT_SIDE_LEN {
        LIMIT_SIDE_LEN / max_side
    } else {
        1.0
    };
    let round32 = |v: f32| (((v / 32.0).round() as i32) * 32).max(32) as u32;
    let rw = round32(w0 as f32 * ratio);
    let rh = round32(h0 as f32 * ratio);
    let resized = image::imageops::resize(&rgb, rw, rh, image::imageops::FilterType::Triangle);

    let (pw, ph) = (rw as usize, rh as usize);
    let plane = pw * ph;
    let mut data = vec![0f32; 3 * plane];
    for y in 0..ph {
        for x in 0..pw {
            let px = resized.get_pixel(x as u32, y as u32);
            let (r, g, b) = (
                px[0] as f32 / 255.0,
                px[1] as f32 / 255.0,
                px[2] as f32 / 255.0,
            );
            let idx = y * pw + x;
            // Channel order is BGR, with mean/std indexed in that same order.
            data[idx] = (b - MEAN[0]) / STD[0];
            data[plane + idx] = (g - MEAN[1]) / STD[1];
            data[2 * plane + idx] = (r - MEAN[2]) / STD[2];
        }
    }
    (data, pw, ph)
}

/// The PP-LCNet textline orientation classifier session. Resolves the 180Â°
/// ambiguity the detector leaves: a line's long-axis angle is only known modulo
/// 180Â°, so deskewing alone can leave text upside-down.
pub struct OrientClassifier {
    session: Session,
}

impl OrientClassifier {
    /// Build the classifier, downloading the ONNX weights on first use.
    /// Execution providers are tried CUDA -> DirectML -> CPU, like [`Detector`].
    pub fn load() -> Result<Self> {
        let mut errors = Vec::new();

        let cuda = Session::builder()
            .and_then(|s| s.with_memory_pattern(false))
            .and_then(|s| s.with_execution_providers([CUDAExecutionProvider::default().build()]))
            .and_then(|s| s.commit_from_url(ORIENT_MODEL_URL));
        match cuda {
            Ok(session) => return Ok(Self { session }),
            Err(e) => errors.push(format!("CUDA orientation classifier: {e:#}")),
        }

        let directml = Session::builder()
            .and_then(|s| s.with_memory_pattern(false))
            .and_then(|s| {
                s.with_execution_providers([DirectMLExecutionProvider::default().build()])
            })
            .and_then(|s| s.commit_from_url(ORIENT_MODEL_URL));
        match directml {
            Ok(session) => return Ok(Self { session }),
            Err(e) => errors.push(format!("DirectML orientation classifier: {e:#}")),
        }

        let cpu = Session::builder()
            .and_then(|s| s.with_memory_pattern(false))
            .and_then(|s| s.commit_from_url(ORIENT_MODEL_URL));
        match cpu {
            Ok(session) => Ok(Self { session }),
            Err(e) => {
                errors.push(format!("CPU orientation classifier: {e:#}"));
                Err(anyhow!("{}", errors.join("; ")))
                    .context("failed to download/load the textline orientation classifier")
            }
        }
    }

    /// Decide whether `cutout`, once deskewed by `theta`, is upside-down.
    ///
    /// Samples up to [`ORIENT_MAX_SAMPLES`] representative boxes (the two
    /// longest, the median, and the two shortest by length), warps each into an
    /// upright crop with the *same* rotation the output receives, classifies it
    /// as 0Â° or 180Â°, and majority-votes. `Ok(Some(true))` means the object
    /// should be rotated an extra 180Â°; `Ok(None)` means there was nothing to
    /// decide from (no boxes, or every sampled crop was degenerate).
    pub fn vote_flip(
        &mut self,
        cutout: &RgbaImage,
        boxes: &[TextBox],
        theta: f32,
        debug: bool,
    ) -> Result<Option<bool>> {
        if boxes.is_empty() {
            return Ok(None);
        }

        let white = composite_over_white(cutout);
        let theta_deg = theta.to_degrees() as f64;
        let (mut upright, mut flipped) = (0usize, 0usize);
        for &i in &sample_box_indices(boxes) {
            // The rotation maps a long-axis orientation `a` to `a + theta`; only
            // lines that come out near-horizontal are something a 0/180 model can
            // judge. Skip the rest (90/270 lines) so they don't pollute the vote.
            let skew = orientation_dist(boxes[i].angle + theta_deg, 0.0);
            if skew > ORIENT_MAX_SKEW_DEG {
                if debug {
                    elog!(
                        "      line len {:7.1} -> skipped ({skew:.0}Â° from horizontal)",
                        boxes[i].length,
                    );
                }
                continue;
            }
            let Some(line) = deskew_line_crop(&white, &boxes[i].corners, theta) else {
                continue;
            };
            let (is_flipped, conf) = self.classify(&line)?;
            if is_flipped {
                flipped += 1;
            } else {
                upright += 1;
            }
            if debug {
                elog!(
                    "      line len {:7.1} -> {} ({:.2})",
                    boxes[i].length,
                    if is_flipped { "180Â°" } else { "0Â°" },
                    conf,
                );
            }
        }

        if upright == 0 && flipped == 0 {
            return Ok(None);
        }
        // Tie favours leaving the deskew result as-is.
        let flip = flipped > upright;
        if debug {
            elog!("    orientation vote: {upright} upright / {flipped} flipped -> flip={flip}");
        }
        Ok(Some(flip))
    }

    /// Classify one upright line crop. Returns `(is_180_degrees, confidence)`.
    fn classify(&mut self, line: &RgbImage) -> Result<(bool, f32)> {
        let resized = image::imageops::resize(
            line,
            ORIENT_W,
            ORIENT_H,
            image::imageops::FilterType::Triangle,
        );
        let plane = (ORIENT_W * ORIENT_H) as usize;
        let mut data = vec![0f32; 3 * plane];
        for (i, px) in resized.pixels().enumerate() {
            // RGB channel order (unlike the detector's BGR), shared ImageNet stats.
            data[i] = (px[0] as f32 / 255.0 - MEAN[0]) / STD[0];
            data[plane + i] = (px[1] as f32 / 255.0 - MEAN[1]) / STD[1];
            data[2 * plane + i] = (px[2] as f32 / 255.0 - MEAN[2]) / STD[2];
        }

        let tensor = Tensor::from_array(([1usize, 3, ORIENT_H as usize, ORIENT_W as usize], data))?;
        let outputs = self.session.run(ort::inputs![tensor])?;
        let (_, scores) = outputs[0].try_extract_tensor::<f32>()?;
        // Two scores: index 0 = 0Â°, index 1 = 180Â°.
        let (p0, p1) = softmax2(scores[0], scores[1]);
        Ok((p1 > p0, p1.max(p0)))
    }
}

/// Flatten a transparent cutout onto a white background (matching the detector's
/// preprocessing), so the classifier sees the same kind of input it was trained on.
fn composite_over_white(cutout: &RgbaImage) -> RgbImage {
    let (w, h) = cutout.dimensions();
    let mut rgb = RgbImage::new(w, h);
    for (x, y, p) in cutout.enumerate_pixels() {
        let a = p[3] as f32 / 255.0;
        let blend = |c: u8| (c as f32 * a + 255.0 * (1.0 - a)).round() as u8;
        rgb.put_pixel(x, y, Rgb([blend(p[0]), blend(p[1]), blend(p[2])]));
    }
    rgb
}

/// Pick up to [`ORIENT_MAX_SAMPLES`] box indices spanning the length range: the
/// two longest, the median, and the two shortest. Deduplicated; returns every
/// index (longest-first) when there are five or fewer boxes.
fn sample_box_indices(boxes: &[TextBox]) -> Vec<usize> {
    let n = boxes.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| boxes[b].length.total_cmp(&boxes[a].length));
    if n <= ORIENT_MAX_SAMPLES {
        return order;
    }
    let mut chosen = Vec::with_capacity(ORIENT_MAX_SAMPLES);
    for p in [0, 1, n / 2, n - 2, n - 1] {
        if !chosen.contains(&order[p]) {
            chosen.push(order[p]);
        }
    }
    chosen
}

/// Warp one detected box into an upright crop using the same forward rotation
/// (`Projection::rotate(theta)` about the origin) the deskewed output uses, so
/// the classifier sees the line exactly as it will appear in the result.
/// Returns `None` for a degenerate (zero-area) box.
fn deskew_line_crop(white: &RgbImage, corners: &[(f64, f64); 4], theta: f32) -> Option<RgbImage> {
    let (sin, cos) = (theta as f64).sin_cos();
    // Rotate the corners about the origin, matching `rotated_component_bounds`
    // and `deskew_from_src`'s `Projection::rotate(theta)`.
    let rotated: [(f64, f64); 4] = std::array::from_fn(|i| {
        let (x, y) = corners[i];
        (x * cos - y * sin, x * sin + y * cos)
    });
    let min_x = rotated.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let max_x = rotated
        .iter()
        .map(|p| p.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = rotated.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let max_y = rotated
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, f64::max);

    let new_w = (max_x - min_x).round().max(1.0) as u32;
    let new_h = (max_y - min_y).round().max(1.0) as u32;
    if new_w < 2 || new_h < 2 {
        return None;
    }

    let proj = Projection::translate(-min_x as f32, -min_y as f32) * Projection::rotate(theta);
    let mut out = RgbImage::from_pixel(new_w, new_h, Rgb([255, 255, 255]));
    warp_into(
        white,
        proj,
        Interpolation::Bilinear,
        Border::Constant(Rgb([255, 255, 255])),
        &mut out,
    );
    Some(out)
}

/// Numerically stable two-class softmax.
fn softmax2(a: f32, b: f32) -> (f32, f32) {
    let m = a.max(b);
    let (ea, eb) = ((a - m).exp(), (b - m).exp());
    let s = ea + eb;
    (ea / s, eb / s)
}

/// The PP-LCNet document orientation classifier (0/90/180/270). Operates on the
/// whole object image, so it resolves orientation for rectangular crops that have
/// no text line to vote on — and unlike the textline model it can detect when the
/// deskew aligned the object onto the wrong axis (a 90/270 result).
pub struct DocOrientClassifier {
    session: Session,
}

impl DocOrientClassifier {
    /// Build the classifier. With `model_path` it loads that ONNX from disk (a
    /// custom/finetuned model); otherwise it downloads the default weights on
    /// first use. Execution providers are tried CUDA -> DirectML -> CPU, like
    /// [`Detector`]. A custom model is assumed to share the default's
    /// preprocessing and 0/90/180/270 output.
    pub fn load(model_path: Option<&std::path::Path>) -> Result<Self> {
        let mut errors = Vec::new();

        let cuda = Session::builder()
            .and_then(|s| s.with_memory_pattern(false))
            .and_then(|s| s.with_execution_providers([CUDAExecutionProvider::default().build()]))
            .and_then(|s| match model_path {
                Some(p) => s.commit_from_file(p),
                None => s.commit_from_url(DOC_ORI_MODEL_URL),
            });
        match cuda {
            Ok(session) => return Ok(Self { session }),
            Err(e) => errors.push(format!("CUDA doc orientation classifier: {e:#}")),
        }

        let directml = Session::builder()
            .and_then(|s| s.with_memory_pattern(false))
            .and_then(|s| {
                s.with_execution_providers([DirectMLExecutionProvider::default().build()])
            })
            .and_then(|s| match model_path {
                Some(p) => s.commit_from_file(p),
                None => s.commit_from_url(DOC_ORI_MODEL_URL),
            });
        match directml {
            Ok(session) => return Ok(Self { session }),
            Err(e) => errors.push(format!("DirectML doc orientation classifier: {e:#}")),
        }

        let cpu = Session::builder()
            .and_then(|s| s.with_memory_pattern(false))
            .and_then(|s| match model_path {
                Some(p) => s.commit_from_file(p),
                None => s.commit_from_url(DOC_ORI_MODEL_URL),
            });
        match cpu {
            Ok(session) => Ok(Self { session }),
            Err(e) => {
                errors.push(format!("CPU doc orientation classifier: {e:#}"));
                Err(anyhow!("{}", errors.join("; ")))
                    .context("failed to load the doc orientation classifier")
            }
        }
    }

    /// Classify the document orientation of `cutout` after deskewing by `theta`,
    /// returning the number of 90-degree clockwise turns to apply to the deskewed
    /// output to make it upright (0..=3) and the softmax confidence. Running on
    /// the deskewed image means the model only has to resolve the residual
    /// quarter turn (usually upright vs upside-down).
    pub fn orient(&mut self, cutout: &RgbaImage, theta: f32) -> Result<(u8, f32)> {
        let white = composite_over_white(cutout);
        // Downscale before rotating; orientation is a coarse, whole-image call.
        let (w, h) = white.dimensions();
        let long = w.max(h);
        let small = if long > DOC_ORI_PRESCALE {
            let s = DOC_ORI_PRESCALE as f32 / long as f32;
            image::imageops::resize(
                &white,
                ((w as f32 * s).round() as u32).max(1),
                ((h as f32 * s).round() as u32).max(1),
                image::imageops::FilterType::Triangle,
            )
        } else {
            white
        };
        let desk = if theta.abs() < 1e-6 {
            small
        } else {
            rotate_about_center_no_crop(
                &small,
                theta,
                Interpolation::Bilinear,
                Border::Constant(Rgb([255, 255, 255])),
            )
        };

        let input = resize_short_center_crop(&desk, DOC_ORI_RESIZE, DOC_ORI_CROP);
        let plane = (DOC_ORI_CROP * DOC_ORI_CROP) as usize;
        let mut data = vec![0f32; 3 * plane];
        for (i, px) in input.pixels().enumerate() {
            data[i] = (px[0] as f32 / 255.0 - MEAN[0]) / STD[0];
            data[plane + i] = (px[1] as f32 / 255.0 - MEAN[1]) / STD[1];
            data[2 * plane + i] = (px[2] as f32 / 255.0 - MEAN[2]) / STD[2];
        }

        let tensor = Tensor::from_array((
            [1usize, 3, DOC_ORI_CROP as usize, DOC_ORI_CROP as usize],
            data,
        ))?;
        let outputs = self.session.run(ort::inputs![tensor])?;
        let (_, scores) = outputs[0].try_extract_tensor::<f32>()?;
        // Classes: 0 = 0deg, 1 = 90, 2 = 180, 3 = 270 (clockwise rotation of the
        // input). Correcting a cls*90 CW rotation needs (4 - cls) % 4 CW turns.
        let probs = softmax4([scores[0], scores[1], scores[2], scores[3]]);
        let (cls, conf) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, &p)| (i, p))
            .unwrap_or((0, 0.0));
        Ok((((4 - cls) % 4) as u8, conf))
    }
}

/// Resize so the shorter side is `short` (preserving aspect), then center-crop a
/// `crop` x `crop` square — PaddleCls's `ResizeImage(resize_short)` + `CropImage`.
fn resize_short_center_crop(img: &RgbImage, short: u32, crop: u32) -> RgbImage {
    let (w, h) = img.dimensions();
    let scale = short as f32 / w.min(h).max(1) as f32;
    let rw = ((w as f32 * scale).round() as u32).max(crop);
    let rh = ((h as f32 * scale).round() as u32).max(crop);
    let resized = image::imageops::resize(img, rw, rh, image::imageops::FilterType::Triangle);
    let x0 = (rw - crop) / 2;
    let y0 = (rh - crop) / 2;
    image::imageops::crop_imm(&resized, x0, y0, crop, crop).to_image()
}

/// Numerically stable four-class softmax.
fn softmax4(s: [f32; 4]) -> [f32; 4] {
    let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut e = [0f32; 4];
    let mut sum = 0.0;
    for i in 0..4 {
        e[i] = (s[i] - m).exp();
        sum += e[i];
    }
    for v in &mut e {
        *v /= sum;
    }
    e
}

/// Reduce the text boxes to a single angle: reject outliers, then length-weight.
///
/// The longest line anchors the estimate (its angular error scales like
/// 1/length, so it is the most reliable single measurement). Lines within
/// `tol_deg` of the anchor are kept and averaged with a lengthÂ²-weighted mean
/// (inverse-variance optimal); everything else â€” a stray logo word, vertical
/// stylized text â€” is discarded *first*, so it never pulls the mean. The mask
/// is not involved here; it is handled separately as a geometric candidate.
///
/// Each sample is `(angle_degrees, length)`.
pub fn robust_text_angle(text: &[(f64, f64)], tol_deg: f64) -> Option<AngleEstimate> {
    let anchor = text
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|s| s.0)?;
    let total = text.len();
    let total_w: f64 = text.iter().map(|s| s.1).sum();
    if total_w <= 0.0 {
        return None;
    }

    let (mut sw, mut acc, mut inlier_w, mut inliers) = (0.0, 0.0, 0.0, 0usize);
    for &(deg, weight) in text {
        if orientation_dist(deg, anchor) <= tol_deg {
            let w2 = weight * weight;
            acc += w2 * wrap_orientation(deg - anchor);
            sw += w2;
            inlier_w += weight;
            inliers += 1;
        }
    }
    let center = if sw > 0.0 {
        wrap_orientation(anchor + acc / sw)
    } else {
        anchor
    };

    Some(AngleEstimate {
        degrees: center as f32,
        anchor: anchor as f32,
        agreement: (inlier_w / total_w) as f32,
        inliers,
        total,
    })
}

/// The mask's geometric orientation: the long-axis angle of its minimum-area
/// rectangle (degrees in `(-90, 90]`).
/// How the mask's rectangle is fitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskFit {
    /// Maximal enclosing rectangle (the "safezone"): sides touch the extreme
    /// points, fully enclosing any extrusion.
    Max,
    /// Tight body rectangle: the smallest inner rectangle supported across the
    /// mask's rows and columns. Only safe on rectangular masks, so callers gate
    /// it on rectangularity.
    Min,
}

fn fit_mask_rect(points: &[(f64, f64)], fit: MaskFit) -> Option<MinRect> {
    match fit {
        MaskFit::Max => min_area_rect(points),
        MaskFit::Min => min_area_rect_body(points),
    }
}

pub fn mask_angle(mask: &GrayImage, fit: MaskFit) -> Option<f64> {
    fit_mask_rect(&mask_contour_points(mask), fit).map(|r| r.long_axis_deg())
}

/// How strongly the mask looks like the minimum-area rectangle that would drive
/// mask-based rotation, in `[0, 1]`. Boundary support dominates the score so a
/// rectangular outline with imperfect fill still counts as rectangular, while
/// circles, triangles, and irregular cutouts score lower.
pub fn mask_rectangularity(mask: &GrayImage, fit: MaskFit) -> Option<f64> {
    let area = mask.pixels().filter(|p| p[0] > 0).count();
    if area == 0 {
        return None;
    }

    let contour = mask_contour_points(mask);
    let rect = fit_mask_rect(&contour, fit)?;
    let rect_area = (rect.w + 1.0).max(1.0) * (rect.h + 1.0).max(1.0);
    if rect_area <= 0.0 {
        return None;
    }

    let fill_ratio = (area as f64 / rect_area).clamp(0.0, 1.0);
    let side_support = rectangle_side_support(&contour, &rect);
    Some((0.25 * fill_ratio + 0.75 * side_support).clamp(0.0, 1.0))
}

/// The four corners of the mask's fitted rectangle (crop coordinates), for debug
/// visualization.
pub fn mask_rect_corners(mask: &GrayImage, fit: MaskFit) -> Option<[(f64, f64); 4]> {
    fit_mask_rect(&mask_contour_points(mask), fit).map(|r| r.corners)
}

fn mask_contour_points(mask: &GrayImage) -> Vec<(f64, f64)> {
    let (w, h) = mask.dimensions();
    let mut points = Vec::new();

    for y in 0..h {
        for x in 0..w {
            if mask.get_pixel(x, y)[0] == 0 {
                continue;
            }

            let at_edge = x == 0 || y == 0 || x + 1 == w || y + 1 == h;
            let touches_background = !at_edge && {
                let mut touches = false;
                for ny in y - 1..=y + 1 {
                    for nx in x - 1..=x + 1 {
                        if nx == x && ny == y {
                            continue;
                        }
                        if mask.get_pixel(nx, ny)[0] == 0 {
                            touches = true;
                        }
                    }
                }
                touches
            };

            if at_edge || touches_background {
                points.push((x as f64, y as f64));
            }
        }
    }

    points
}

/// Count enclosed background regions (holes) of at least `min_area` pixels in the
/// mask. A hole is background not 4-connected to the mask border — e.g. the two
/// reel windows of a cassette. Lets a holed-but-rectangular object get the same
/// padding as a non-rectangular one.
pub fn mask_hole_count(mask: &GrayImage, min_area: usize) -> usize {
    let (wd, ht) = mask.dimensions();
    let (w, h) = (wd as usize, ht as usize);
    if w == 0 || h == 0 {
        return 0;
    }
    let bg: Vec<bool> = mask.pixels().map(|p| p[0] == 0).collect();
    let mut visited = vec![false; w * h];
    let mut stack: Vec<usize> = Vec::new();

    // Seed the exterior flood from every border background pixel.
    let seed = |i: usize, visited: &mut [bool], stack: &mut Vec<usize>| {
        if bg[i] && !visited[i] {
            visited[i] = true;
            stack.push(i);
        }
    };
    for x in 0..w {
        seed(x, &mut visited, &mut stack);
        seed((h - 1) * w + x, &mut visited, &mut stack);
    }
    for y in 0..h {
        seed(y * w, &mut visited, &mut stack);
        seed(y * w + (w - 1), &mut visited, &mut stack);
    }
    flood_bg_4(&bg, w, h, &mut visited, &mut stack);

    // Any background pixel the exterior flood didn't reach is enclosed; count its
    // component if it is large enough to be a real hole rather than mask noise.
    let mut holes = 0;
    for start in 0..w * h {
        if visited[start] || !bg[start] {
            continue;
        }
        visited[start] = true;
        stack.push(start);
        if flood_bg_4(&bg, w, h, &mut visited, &mut stack) >= min_area {
            holes += 1;
        }
    }
    holes
}

/// 4-connected flood fill over background pixels from a seeded `stack`, marking
/// `visited` and returning the number of pixels filled.
fn flood_bg_4(bg: &[bool], w: usize, h: usize, visited: &mut [bool], stack: &mut Vec<usize>) -> usize {
    let mut area = 0;
    while let Some(i) = stack.pop() {
        area += 1;
        let (x, y) = (i % w, i / w);
        if x > 0 && bg[i - 1] && !visited[i - 1] {
            visited[i - 1] = true;
            stack.push(i - 1);
        }
        if x + 1 < w && bg[i + 1] && !visited[i + 1] {
            visited[i + 1] = true;
            stack.push(i + 1);
        }
        if y > 0 && bg[i - w] && !visited[i - w] {
            visited[i - w] = true;
            stack.push(i - w);
        }
        if y + 1 < h && bg[i + w] && !visited[i + w] {
            visited[i + w] = true;
            stack.push(i + w);
        }
    }
    area
}

// Fraction of the fitted rectangle perimeter covered by nearby mask boundary.
fn rectangle_side_support(points: &[(f64, f64)], rect: &MinRect) -> f64 {
    if rect.w < 1.0 || rect.h < 1.0 {
        return 0.0;
    }

    let c0 = rect.corners[0];
    let c1 = rect.corners[1];
    let c3 = rect.corners[3];
    let ux = ((c1.0 - c0.0) / rect.w, (c1.1 - c0.1) / rect.w);
    let vy = ((c3.0 - c0.0) / rect.h, (c3.1 - c0.1) / rect.h);
    let eps = (rect.w.min(rect.h) * 0.03).clamp(1.0, 4.0);

    let w_bins = side_bins(rect.w);
    let h_bins = side_bins(rect.h);
    let mut bottom = vec![false; w_bins];
    let mut top = vec![false; w_bins];
    let mut left = vec![false; h_bins];
    let mut right = vec![false; h_bins];

    for &(x, y) in points {
        let dx = x - c0.0;
        let dy = y - c0.1;
        let u = dx * ux.0 + dy * ux.1;
        let v = dx * vy.0 + dy * vy.1;

        if (-eps..=rect.w + eps).contains(&u) {
            if v.abs() <= eps {
                mark_side_bin(&mut bottom, u, rect.w);
            }
            if (rect.h - v).abs() <= eps {
                mark_side_bin(&mut top, u, rect.w);
            }
        }
        if (-eps..=rect.h + eps).contains(&v) {
            if u.abs() <= eps {
                mark_side_bin(&mut left, v, rect.h);
            }
            if (rect.w - u).abs() <= eps {
                mark_side_bin(&mut right, v, rect.h);
            }
        }
    }

    let side_fraction =
        |bins: &[bool]| bins.iter().filter(|&&covered| covered).count() as f64 / bins.len() as f64;

    (side_fraction(&bottom) + side_fraction(&top) + side_fraction(&left) + side_fraction(&right))
        / 4.0
}

fn side_bins(length: f64) -> usize {
    ((length / 8.0).ceil() as usize).clamp(4, 64)
}

fn mark_side_bin(bins: &mut [bool], coord: f64, length: f64) {
    if bins.is_empty() || length <= 0.0 {
        return;
    }
    let t = (coord / length).clamp(0.0, 1.0);
    let idx = (t * bins.len() as f64).floor() as usize;
    bins[idx.min(bins.len() - 1)] = true;
}

/// Distance between two orientations on the 180Â°-periodic circle, in `[0, 90]`.
pub fn orientation_dist(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(180.0);
    d.min(180.0 - d)
}

/// Wrap an orientation difference to `(-90, 90]`.
fn wrap_orientation(x: f64) -> f64 {
    let a = x.rem_euclid(180.0);
    if a > 90.0 { a - 180.0 } else { a }
}

/// A minimum-area rectangle: side lengths, the angle of the `w` side, and its
/// four corners (consistent winding).
struct MinRect {
    w: f64,
    h: f64,
    angle: f64,
    corners: [(f64, f64); 4],
}

impl MinRect {
    /// Long-axis orientation in `(-90, 90]` degrees.
    fn long_axis_deg(&self) -> f64 {
        let a = if self.w >= self.h {
            self.angle
        } else {
            self.angle + FRAC_PI_2
        };
        normalize_orientation(a).to_degrees()
    }
}

/// Wrap an orientation angle to `(-pi/2, pi/2]` (line orientations, period pi).
fn normalize_orientation(mut a: f64) -> f64 {
    while a > FRAC_PI_2 {
        a -= PI;
    }
    while a <= -FRAC_PI_2 {
        a += PI;
    }
    a
}

/// Expand a convex polygon outward â€” PaddleOCR's `unclip`. The DB probability
/// map is a shrunken text core, so the detected box is smaller than the real
/// text line; this offsets it back out by `area * ratio / perimeter` (the same
/// distance pyclipper uses), recovering the text-line envelope.
///
/// pyclipper rounds the corners (`JT_ROUND`); we use a miter join instead,
/// which is exact for the rectangular mini boxes this is called on (their
/// minimum-area rectangle is identical either way) and far simpler.
fn unclip(poly: &[(f64, f64)], ratio: f64) -> Vec<(f64, f64)> {
    let n = poly.len();
    if n < 3 {
        return poly.to_vec();
    }
    let area = polygon_area(poly);
    let perimeter = polygon_perimeter(poly);
    if perimeter < 1e-9 {
        return poly.to_vec();
    }
    let distance = area * ratio / perimeter;

    // Work on a counter-clockwise copy so the outward normal is well-defined.
    let mut p = poly.to_vec();
    if signed_area(&p) < 0.0 {
        p.reverse();
    }

    // Offset each edge outward by `distance` along its outward normal.
    let mut offset_edges: Vec<((f64, f64), (f64, f64))> = Vec::with_capacity(n);
    for i in 0..n {
        let a = p[i];
        let b = p[(i + 1) % n];
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let len = dx.hypot(dy);
        if len < 1e-9 {
            continue;
        }
        // For a CCW polygon the interior is left of each edge, so the outward
        // normal is to the right: rotate the direction by -90Â°.
        let (nx, ny) = (dy / len * distance, -dx / len * distance);
        offset_edges.push(((a.0 + nx, a.1 + ny), (b.0 + nx, b.1 + ny)));
    }

    // Each expanded vertex is where consecutive offset edges meet (miter join).
    let m = offset_edges.len();
    let mut out = Vec::with_capacity(m);
    for i in 0..m {
        let (pa, pb) = offset_edges[(i + m - 1) % m];
        let (qa, qb) = offset_edges[i];
        out.push(line_intersection(pa, pb, qa, qb).unwrap_or(qa));
    }
    out
}

/// Signed polygon area (positive when counter-clockwise) via the shoelace sum.
fn signed_area(poly: &[(f64, f64)]) -> f64 {
    let n = poly.len();
    let mut s = 0.0;
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        s += a.0 * b.1 - b.0 * a.1;
    }
    s / 2.0
}

fn polygon_area(poly: &[(f64, f64)]) -> f64 {
    signed_area(poly).abs()
}

fn polygon_perimeter(poly: &[(f64, f64)]) -> f64 {
    let n = poly.len();
    (0..n)
        .map(|i| {
            let a = poly[i];
            let b = poly[(i + 1) % n];
            (b.0 - a.0).hypot(b.1 - a.1)
        })
        .sum()
}

/// Intersection of the infinite lines through `p1p2` and `p3p4`, or `None` if
/// they are parallel.
fn line_intersection(
    p1: (f64, f64),
    p2: (f64, f64),
    p3: (f64, f64),
    p4: (f64, f64),
) -> Option<(f64, f64)> {
    let d = (p1.0 - p2.0) * (p3.1 - p4.1) - (p1.1 - p2.1) * (p3.0 - p4.0);
    if d.abs() < 1e-9 {
        return None;
    }
    let a = p1.0 * p2.1 - p1.1 * p2.0;
    let b = p3.0 * p4.1 - p3.1 * p4.0;
    let x = (a * (p3.0 - p4.0) - (p1.0 - p2.0) * b) / d;
    let y = (a * (p3.1 - p4.1) - (p1.1 - p2.1) * b) / d;
    Some((x, y))
}

/// Minimum-area enclosing rectangle via rotating calipers over the convex hull
/// (mirrors OpenCV's `minAreaRect`, which PaddleOCR uses). The sides touch the
/// extreme points, so a thin extrusion inflates the box with blank space.
fn min_area_rect(points: &[(f64, f64)]) -> Option<MinRect> {
    let hull = convex_hull(points);
    if hull.len() < 3 {
        return None;
    }
    let n = hull.len();
    let mut best: Option<MinRect> = None;
    for i in 0..n {
        let p0 = hull[i];
        let p1 = hull[(i + 1) % n];
        let edge = (p1.0 - p0.0, p1.1 - p0.1);
        let len = edge.0.hypot(edge.1);
        if len < 1e-9 {
            continue;
        }
        // Orthonormal basis: u along the edge, v perpendicular.
        let (ux, uy) = (edge.0 / len, edge.1 / len);
        let (vx, vy) = (-uy, ux);

        // The extreme in any direction is always a hull vertex, so the hull
        // alone gives the maximal bounds.
        let (mut min_u, mut max_u, mut min_v, mut max_v) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for &q in &hull {
            let pu = q.0 * ux + q.1 * uy;
            let pv = q.0 * vx + q.1 * vy;
            min_u = min_u.min(pu);
            max_u = max_u.max(pu);
            min_v = min_v.min(pv);
            max_v = max_v.max(pv);
        }

        let (w, h) = (max_u - min_u, max_v - min_v);
        if best.as_ref().is_none_or(|b| w * h < b.w * b.h) {
            let to_xy = |pu: f64, pv: f64| (pu * ux + pv * vx, pu * uy + pv * vy);
            best = Some(MinRect {
                w,
                h,
                angle: uy.atan2(ux),
                corners: [
                    to_xy(min_u, min_v),
                    to_xy(max_u, min_v),
                    to_xy(max_u, max_v),
                    to_xy(min_u, max_v),
                ],
            });
        }
    }
    best
}

/// Tight "body" rectangle for a mask the caller has confirmed is rectangular.
///
/// The orientation comes from the maximal rectangle. In that frame each point
/// is binned into a column (by `u`) and a row (by `v`); every slice contributes
/// its edge position, weighted by its span. Each side lands on the edge position
/// with the most cross-section support, which makes a thin extrusion drop out
/// without a percentile hyperparameter. This is only safe after the caller's
/// rectangularity gate.
fn min_area_rect_body(points: &[(f64, f64)]) -> Option<MinRect> {
    let base = min_area_rect(points)?;
    let (s, c) = base.angle.sin_cos();
    let (ux, uy) = (c, s);
    let (vx, vy) = (-uy, ux);

    // Per-slice extremes: columns keyed by rounded u carry the v-range; rows
    // keyed by rounded v carry the u-range.
    use std::collections::HashMap;
    let mut col_v: HashMap<i64, (f64, f64)> = HashMap::new();
    let mut row_u: HashMap<i64, (f64, f64)> = HashMap::new();
    let (mut max_u_lo, mut max_u_hi, mut max_v_lo, mut max_v_hi) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for &(x, y) in points {
        let u = x * ux + y * uy;
        let v = x * vx + y * vy;
        max_u_lo = max_u_lo.min(u);
        max_u_hi = max_u_hi.max(u);
        max_v_lo = max_v_lo.min(v);
        max_v_hi = max_v_hi.max(v);
        let cv = col_v.entry(u.round() as i64).or_insert((v, v));
        cv.0 = cv.0.min(v);
        cv.1 = cv.1.max(v);
        let ru = row_u.entry(v.round() as i64).or_insert((u, u));
        ru.0 = ru.0.min(u);
        ru.1 = ru.1.max(u);
    }

    let pick_edge = |scores: &HashMap<i64, f64>, pick_outer_low: bool| -> f64 {
        let mut best_key = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (&key, &score) in scores {
            let better_score = score > best_score + f64::EPSILON;
            let wider_tie = (score - best_score).abs() <= f64::EPSILON
                && ((pick_outer_low && key < best_key) || (!pick_outer_low && key > best_key));
            if better_score || wider_tie {
                best_key = key;
                best_score = score;
            }
        }
        best_key as f64
    };

    let dominant_bounds = |slices: &HashMap<i64, (f64, f64)>| -> (f64, f64) {
        let mut lo_scores: HashMap<i64, f64> = HashMap::new();
        let mut hi_scores: HashMap<i64, f64> = HashMap::new();
        for &(lo, hi) in slices.values() {
            let weight = (hi - lo).abs().max(1.0);
            *lo_scores.entry(lo.round() as i64).or_insert(0.0) += weight;
            *hi_scores.entry(hi.round() as i64).or_insert(0.0) += weight;
        }
        (pick_edge(&lo_scores, true), pick_edge(&hi_scores, false))
    };

    let (v_lo, v_hi) = dominant_bounds(&col_v);
    let (u_lo, u_hi) = dominant_bounds(&row_u);
    let (u_lo, u_hi) = if u_hi > u_lo {
        (u_lo, u_hi)
    } else {
        (max_u_lo, max_u_hi)
    };
    let (v_lo, v_hi) = if v_hi > v_lo {
        (v_lo, v_hi)
    } else {
        (max_v_lo, max_v_hi)
    };
    if u_hi <= u_lo || v_hi <= v_lo {
        return Some(base);
    }

    let to_xy = |pu: f64, pv: f64| (pu * ux + pv * vx, pu * uy + pv * vy);
    Some(MinRect {
        w: u_hi - u_lo,
        h: v_hi - v_lo,
        angle: base.angle,
        corners: [
            to_xy(u_lo, v_lo),
            to_xy(u_hi, v_lo),
            to_xy(u_hi, v_hi),
            to_xy(u_lo, v_hi),
        ],
    })
}

/// Convex hull (Andrew's monotone chain), counter-clockwise, collinear points
/// removed.
fn convex_hull(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap()
            .then(a.1.partial_cmp(&b.1).unwrap())
    });
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }

    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };

    let mut hull: Vec<(f64, f64)> = Vec::with_capacity(pts.len() + 1);
    for &p in &pts {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    let lower_len = hull.len() + 1;
    for &p in pts.iter().rev().skip(1) {
        while hull.len() >= lower_len && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
        {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    hull
}

/// Mean probability inside the (convex) box â€” PaddleOCR's `box_score_fast`.
fn box_score(prob: &[f32], pw: usize, ph: usize, corners: &[(f64, f64); 4]) -> f32 {
    let xmin = corners.iter().map(|c| c.0).fold(f64::MAX, f64::min).floor();
    let xmax = corners.iter().map(|c| c.0).fold(f64::MIN, f64::max).ceil();
    let ymin = corners.iter().map(|c| c.1).fold(f64::MAX, f64::min).floor();
    let ymax = corners.iter().map(|c| c.1).fold(f64::MIN, f64::max).ceil();

    let x0 = (xmin.max(0.0) as usize).min(pw.saturating_sub(1));
    let x1 = (xmax.max(0.0) as usize).min(pw.saturating_sub(1));
    let y0 = (ymin.max(0.0) as usize).min(ph.saturating_sub(1));
    let y1 = (ymax.max(0.0) as usize).min(ph.saturating_sub(1));

    let (mut sum, mut count) = (0.0f64, 0usize);
    for y in y0..=y1 {
        for x in x0..=x1 {
            if point_in_quad((x as f64, y as f64), corners) {
                sum += prob[y * pw + x] as f64;
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        (sum / count as f64) as f32
    }
}

/// Whether `p` lies inside the convex quadrilateral `c` (consistent winding).
fn point_in_quad(p: (f64, f64), c: &[(f64, f64); 4]) -> bool {
    let mut sign = 0i32;
    for i in 0..4 {
        let a = c[i];
        let b = c[(i + 1) % 4];
        let cr = (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0);
        let s = if cr > 0.0 {
            1
        } else if cr < 0.0 {
            -1
        } else {
            0
        };
        if s != 0 {
            if sign == 0 {
                sign = s;
            } else if s != sign {
                return false;
            }
        }
    }
    true
}

/// Build a binary mask (255 = object) for `label` within `[min..=max]` bounds.
pub fn object_mask(
    labels: &[usize],
    w: usize,
    label: usize,
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
) -> GrayImage {
    let ow = (max_x - min_x + 1) as u32;
    let oh = (max_y - min_y + 1) as u32;
    let mut mask = GrayImage::new(ow, oh);
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            if labels[y * w + x] == label {
                mask.put_pixel((x - min_x) as u32, (y - min_y) as u32, Luma([255]));
            }
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rotate(p: (f64, f64), deg: f64) -> (f64, f64) {
        let (s, c) = deg.to_radians().sin_cos();
        (p.0 * c - p.1 * s, p.0 * s + p.1 * c)
    }

    #[test]
    fn rectangularity_is_high_for_filled_rectangle() {
        let mut mask = GrayImage::new(20, 10);
        for y in 0..10 {
            for x in 0..20 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }

        let r = mask_rectangularity(&mask, MaskFit::Max).unwrap();
        assert!(r > 0.98, "rectangularity was {r}");
    }

    #[test]
    fn rectangularity_is_lower_for_triangle() {
        let mut mask = GrayImage::new(20, 20);
        for y in 0..20 {
            for x in 0..=y {
                mask.put_pixel(x, y, Luma([255]));
            }
        }

        let r = mask_rectangularity(&mask, MaskFit::Max).unwrap();
        assert!(r < 0.7, "rectangularity was {r}");
    }

    #[test]
    fn min_body_ignores_thin_extrusion_when_fitting_rectangle() {
        // A wide 100x6 body with a thin spike poking 30px below its
        // bottom edge. The spike is ~5% of the points, so the maximal fit must
        // enclose it (tall, blank-filled box) while the min/body fit drops it
        // and hugs the body.
        let mut points = Vec::new();
        for y in 0..6 {
            for x in 0..100 {
                points.push((x as f64, y as f64));
            }
        }
        // Spike: a 3px-wide column dropping 30px below the body.
        for y in 6..36 {
            for x in 49..52 {
                points.push((x as f64, y as f64));
            }
        }

        let maximal = min_area_rect(&points).unwrap();
        let body = min_area_rect_body(&points).unwrap();

        // The spike makes the maximal box at least as tall as the spike reach.
        assert!(
            maximal.w.min(maximal.h) > 25.0,
            "maximal short side was {}",
            maximal.w.min(maximal.h)
        );
        // Min/body drops the sparse spike, so the short side returns to ~body.
        assert!(
            body.w.min(body.h) < 10.0,
            "body short side was {}",
            body.w.min(body.h)
        );
        assert!(
            body.w.max(body.h) > 90.0,
            "body long side was {}",
            body.w.max(body.h)
        );
    }

    #[test]
    fn body_fit_drops_spike_on_both_axes_without_collapsing() {
        // A solid 60x40 body with a 1px-wide spike rising 40px above it. The
        // maximal box must enclose the spike (tall), while the body fit drops it
        // on the v-axis and must NOT collapse the u-axis (the cross axis).
        let mut points = Vec::new();
        for y in 0..40 {
            for x in 0..60 {
                points.push((x as f64, y as f64));
            }
        }
        for y in 40..80 {
            points.push((30.0, y as f64));
        }

        let maximal = min_area_rect(&points).unwrap();
        let body = min_area_rect_body(&points).unwrap();

        // Maximal encloses the spike: long side spans body + spike (~80).
        assert!(
            maximal.w.max(maximal.h) > 75.0,
            "maximal long side too small"
        );
        // Body drops the spike: both sides hug the 60x40 body.
        assert!(
            body.w.max(body.h) < 66.0,
            "body long side was {}",
            body.w.max(body.h)
        );
        // And it does not collapse the cross axis.
        assert!(
            body.w.min(body.h) > 30.0,
            "body short side was {}",
            body.w.min(body.h)
        );
    }

    #[test]
    fn hole_count_finds_enclosed_regions_above_min_area() {
        // A filled 20x20 square with two 3x3 enclosed holes (cassette-like).
        let mut mask = GrayImage::new(20, 20);
        for y in 0..20 {
            for x in 0..20 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        for (cx, cy) in [(5u32, 10u32), (14, 10)] {
            for y in cy..cy + 3 {
                for x in cx..cx + 3 {
                    mask.put_pixel(x, y, Luma([0]));
                }
            }
        }

        // Both 9px holes counted when the threshold allows them.
        assert_eq!(mask_hole_count(&mask, 4), 2);
        // Raising the minimum area past the hole size drops them (noise filter).
        assert_eq!(mask_hole_count(&mask, 16), 0);
        // A solid square (border-touching background only) has no holes.
        let solid = GrayImage::from_pixel(20, 20, Luma([255]));
        assert_eq!(mask_hole_count(&solid, 1), 0);
    }

    #[test]
    fn unclip_expands_rect_by_offset_distance() {
        // A 40x10 axis-aligned rectangle. PaddleOCR's offset distance is
        // area * ratio / perimeter = (400 * 1.4) / 100 = 5.6.
        let rect = [(0.0, 0.0), (40.0, 0.0), (40.0, 10.0), (0.0, 10.0)];
        let expanded = unclip(&rect, 1.4);
        let r = min_area_rect(&expanded).unwrap();
        let long = r.w.max(r.h);
        let short = r.w.min(r.h);
        // Each side grows by 2 * 5.6 = 11.2.
        assert!((long - 51.2).abs() < 1e-6, "long was {long}");
        assert!((short - 21.2).abs() < 1e-6, "short was {short}");
    }

    #[test]
    fn unclip_preserves_angle() {
        // unclip must not rotate the box â€” only enlarge it.
        for deg in [0.0, 12.5, 33.0, -20.0, 80.0] {
            let base = [(0.0, 0.0), (40.0, 0.0), (40.0, 10.0), (0.0, 10.0)];
            let rotated: Vec<(f64, f64)> = base.iter().map(|&p| rotate(p, deg)).collect();
            let before = min_area_rect(&rotated).unwrap().long_axis_deg();
            let after = min_area_rect(&unclip(&rotated, 1.4))
                .unwrap()
                .long_axis_deg();
            assert!(
                orientation_dist(before, after) < 1e-6,
                "angle moved from {before} to {after} at {deg}Â°"
            );
        }
    }

    #[test]
    fn deskew_formula_returns_bar_to_horizontal() {
        use image::{Rgb, RgbImage};
        use imageproc::geometric_transformations::{
            Border, Interpolation, rotate_about_center_no_crop,
        };

        let measure = |im: &RgbImage| {
            let pts: Vec<(f64, f64)> = im
                .enumerate_pixels()
                .filter(|(_, _, p)| p[0] < 128)
                .map(|(x, y, _)| (x as f64, y as f64))
                .collect();
            min_area_rect(&pts).unwrap().long_axis_deg()
        };

        // A horizontal black bar, tilted to +35Â° to act as skewed input.
        let mut img = RgbImage::from_pixel(180, 110, Rgb([255, 255, 255]));
        for y in 51..59 {
            for x in 20..160 {
                img.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }
        let white = Border::Constant(Rgb([255, 255, 255]));
        let skewed =
            rotate_about_center_no_crop(&img, 35f32.to_radians(), Interpolation::Bilinear, white);

        // Measure independently, then deskew with main's exact formula (-est).
        let est = measure(&skewed);
        let theta = -(est as f32).to_radians();
        let deskewed = rotate_about_center_no_crop(&skewed, theta, Interpolation::Bilinear, white);
        let after = measure(&deskewed);
        assert!(
            orientation_dist(after, 0.0) < 2.0,
            "measured est {est:+.2}, after deskew {after:+.2} (should be ~0)"
        );
    }

    #[test]
    fn anisotropic_rescale_rotates_long_axis() {
        // The bug the rescale fixes: measuring a 26.57Â° line in a space where x
        // and y were scaled differently reads a different angle. Here the line
        // has slope 1/2 (26.57Â°); halving x makes it slope 1 (45Â°).
        let line = [(0.0, 0.0), (40.0, 20.0), (40.0, 21.0), (0.0, 1.0)];
        let map_angle = min_area_rect(&line).unwrap().long_axis_deg();
        let scaled: Vec<(f64, f64)> = line.iter().map(|&(x, y)| (x * 0.5, y)).collect();
        let crop_angle = min_area_rect(&scaled).unwrap().long_axis_deg();
        assert!((map_angle - 26.565).abs() < 0.1, "map_angle {map_angle}");
        assert!((crop_angle - 45.0).abs() < 0.5, "crop_angle {crop_angle}");
    }
}
