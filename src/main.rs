#[macro_use]
mod logging;
mod angle;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use image::{DynamicImage, GenericImageView, ImageBuffer, ImageFormat, Pixel, Rgb, RgbImage, Rgba};
use imageproc::definitions::Clamp;
use imageproc::drawing::draw_line_segment_mut;
use imageproc::geometric_transformations::{
    Border, Interpolation, Projection, rotate_about_center_no_crop, warp_into,
};
use tiff::encoder::{
    Compression as TiffCompression, DeflateLevel, TiffEncoder as TiffImageEncoder,
    colortype::{RGB8 as TiffRgb8, RGB16 as TiffRgb16, RGBA8 as TiffRgba8, RGBA16 as TiffRgba16},
};
use usls::{Config, Device, Image, models::RMBG};

use crate::angle::{Detector, DocOrientClassifier, OrientClassifier};

type Rgba16Image = ImageBuffer<Rgba<u16>, Vec<u16>>;

/// Remove the background from an image using the RMBG model.
#[derive(Parser, Debug)]
#[command(name = "kropp", about = "RMBG background remover")]
struct Args {
    /// Path to the input image or directory of images.
    #[arg(short, long)]
    input: String,

    /// Output path for a single input file; a `_<index>` suffix is added per
    /// object. Defaults to the input path's stem and format, e.g.
    /// `photo.jpg` -> `photo_0.jpg`.
    #[arg(short, long)]
    output: Option<String>,

    /// Output directory when the input is a directory. Each input file keeps
    /// its own name and format under this directory.
    #[arg(long)]
    output_dir: Option<String>,

    /// Alpha threshold as a percentage: pixels below this become fully
    /// transparent, at or above become fully opaque.
    #[arg(short, long, default_value_t = 95.0)]
    threshold: f32,

    /// Minimum object area in pixels; connected components smaller than this
    /// are discarded as noise.
    #[arg(short, long, default_value_t = 0)]
    min_area: usize,

    /// Minimum component size as a percentage of the smaller image dimension:
    /// a component is dropped as noise unless its longer bounding-box side
    /// reaches at least this fraction of `min(width, height)`. Set to 0 to
    /// disable.
    #[arg(long, default_value_t = 10.0)]
    min_side_percent: f64,

    /// Agreement tolerance in degrees: text boxes within this of the longest
    /// line are inliers; the rest are discarded as outliers.
    #[arg(short = 'a', long, default_value_t = 10.0)]
    angle_tol: f64,

    /// Force text-based rotation for every object: run the text detector and
    /// prefer its angle when text is found. Without this flag, text rotation is
    /// only auto-enabled for sufficiently non-rectangular masks.
    #[arg(long, default_value_t = false)]
    text: bool,

    /// In default mode, run text rotation for masks with rectangularity below
    /// this value. Rectangularity scores how strongly the mask boundary and fill
    /// support its fitted rectangle. Set to 0 to disable auto text. --text
    /// ignores this threshold and always tries text rotation.
    #[arg(long, default_value_t = 0.30)]
    auto_text_rectangularity_threshold: f64,

    /// Padding percentage to add around non-rectangular objects after the final
    /// tight crop. Applied per side; 2 means 2% of the crop width on left/right
    /// and 5% of the crop height on top/bottom.
    #[arg(long, default_value_t = 2.0)]
    non_rectangular_padding: f64,

    /// Cut the object out using the mask as an alpha channel (transparent
    /// background, RGBA output). By default crops keep their original
    /// background as a plain rectangular RGB image.
    #[arg(long, default_value_t = false)]
    alpha: bool,

    /// Report the detected angle but skip rotating crops to upright.
    #[arg(long, default_value_t = false)]
    no_deskew: bool,

    /// Skip the document-orientation model for rectangular crops, leaving their
    /// 0/90/180/270 orientation uncorrected. Text-driven crops still use the
    /// textline 0/180 vote.
    #[arg(long, default_value_t = false)]
    no_doc_orient: bool,

    /// Path to a custom (e.g. finetuned) document-orientation ONNX, loaded from
    /// disk instead of downloading the default. Assumed to share the default's
    /// preprocessing and 0/90/180/270 output.
    #[arg(long)]
    doc_orient_model: Option<String>,

    /// Reprocess directory inputs even when matching outputs already exist.
    /// Without this, an input whose `<stem>_` outputs are already present is
    /// skipped.
    #[arg(long, default_value_t = false)]
    overwrite: bool,

    /// Fit every mask rectangle to the maximal extent (the "safezone"),
    /// enclosing object extrusions fully. By default only non-rectangular
    /// objects use the safezone; rectangular objects use the tight body fit.
    #[arg(long, default_value_t = false)]
    safezone: bool,

    /// Print detailed per-object angle diagnostics (text boxes, mask vote,
    /// consensus inliers/outliers) to stderr.
    #[arg(short = 'v', long, default_value_t = false)]
    debug: bool,

    /// Mirror diagnostic output (everything normally printed to stderr) to this
    /// file as well. Defaults to `kropp-debug.log` when --debug is set and this
    /// is not given.
    #[arg(long)]
    log_file: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let input_path = PathBuf::from(&args.input);

    // Install the log-file sink before any diagnostics are emitted. An explicit
    // --log-file wins; otherwise --debug writes to a default file.
    let log_path = args
        .log_file
        .clone()
        .map(PathBuf::from)
        .or_else(|| args.debug.then(|| PathBuf::from("kropp-debug.log")));
    if let Some(path) = &log_path {
        logging::init_log_file(path)?;
    }

    if !(0.0..=100.0).contains(&args.threshold) {
        bail!(
            "threshold must be between 0 and 100, got {}",
            args.threshold
        );
    }
    if !(0.0..=1.0).contains(&args.auto_text_rectangularity_threshold) {
        bail!(
            "auto text rectangularity threshold must be between 0 and 1, got {}",
            args.auto_text_rectangularity_threshold
        );
    }
    if !(0.0..=100.0).contains(&args.non_rectangular_padding) {
        bail!(
            "non-rectangular padding must be between 0 and 100, got {}",
            args.non_rectangular_padding
        );
    }
    if !(0.0..=100.0).contains(&args.min_side_percent) {
        bail!(
            "min side percent must be between 0 and 100, got {}",
            args.min_side_percent
        );
    }
    // Convert the percentage cutoff to the 0..=255 alpha scale.
    let cutoff = (args.threshold / 100.0 * 255.0).round() as u8;

    if input_path.is_dir() {
        if args.output.is_some() {
            bail!("--output cannot be used when the input is a directory; use --output-dir");
        }
        let output_dir = args
            .output_dir
            .as_deref()
            .context("input is a directory; --output-dir is required")?;
        std::fs::create_dir_all(output_dir)
            .with_context(|| format!("failed to create output dir: {output_dir}"))?;

        let inputs = collect_input_paths(&input_path)?;
        let mut model = build_rmbg()?;
        let mut detector = if args.text {
            match Detector::load() {
                Ok(d) => Some(d),
                Err(e) => {
                    elog!(
                        "warning: text detector unavailable ({e:#}); using mask orientation only"
                    );
                    None
                }
            }
        } else {
            None
        };
        let mut detector_unavailable = args.text && detector.is_none();
        let mut classifier: Option<OrientClassifier> = None;
        let mut classifier_unavailable = false;
        let mut doc_classifier: Option<DocOrientClassifier> = None;
        let mut doc_unavailable = false;

        for input_file in inputs {
            if !args.overwrite && output_dir_has_prefix(Path::new(output_dir), &input_file)? {
                if args.debug {
                    elog!("skipping {} (already processed)", input_file.display());
                }
                continue;
            }
            process_input_file(
                &input_file,
                &args,
                cutoff,
                &mut model,
                &mut detector,
                &mut detector_unavailable,
                &mut classifier,
                &mut classifier_unavailable,
                &mut doc_classifier,
                &mut doc_unavailable,
                Path::new(output_dir),
            )?;
        }
        return Ok(());
    }

    if args.output_dir.is_some() {
        bail!("--output-dir can only be used when the input is a directory");
    }

    let (src_original, input_format) = read_original_image(&input_path)?;
    let output_plan = OutputPlan::new(
        &input_path,
        args.output.as_deref(),
        None,
        args.alpha,
        input_format,
        false,
    )?;

    // Build the RMBG model (downloads the ONNX weights on first run), trying
    // GPU execution providers in order and falling back to CPU.
    let mut model = build_rmbg()?;

    // Load the input image and run inference.
    let model_image = Image::try_read(&args.input)
        .with_context(|| format!("failed to read image: {}", args.input))?;
    let ys = model.forward(std::slice::from_ref(&model_image))?;

    // The model returns a grayscale alpha mask at the source resolution.
    let mask = ys
        .first()
        .and_then(|y| y.masks.first())
        .context("model returned no mask")?;
    let alpha = mask.to_vec();

    let src = model_image.to_rgba8();
    let (w, h) = src.dimensions();
    if src_original.dimensions() != (w, h) {
        bail!(
            "original image dimensions ({:?}) do not match model image dimensions ({w}x{h})",
            src_original.dimensions()
        );
    }
    if (w * h) as usize != alpha.len() {
        bail!(
            "mask size ({}) does not match image size ({}x{})",
            alpha.len(),
            w,
            h
        );
    }

    // Binary foreground mask from the thresholded alpha.
    let fg: Vec<bool> = alpha.iter().map(|&a| a >= cutoff).collect();

    // Label disjoint objects (8-connectivity) and split each into its own file.
    let (labels, mut objects) = connected_components(&fg, w as usize, h as usize, args.min_area);
    // Drop noise: components no side of which is large relative to the image.
    objects.retain(|obj| meets_min_side(obj, w as usize, h as usize, args.min_side_percent));
    if objects.is_empty() {
        bail!("no objects found above the threshold");
    }

    // Load the text detector once when text rotation is enabled; if unavailable
    // (e.g. offline), fall back to mask-based orientation for every object.
    let mut detector = if args.text {
        match Detector::load() {
            Ok(d) => Some(d),
            Err(e) => {
                elog!("warning: text detector unavailable ({e:#}); using mask orientation only");
                None
            }
        }
    } else {
        None
    };
    let mut detector_unavailable = args.text && detector.is_none();
    // Orientation classifiers, loaded lazily per object: the textline 0/180 model
    // for text-driven crops, the doc-orientation model for rectangular ones.
    let mut classifier: Option<OrientClassifier> = None;
    let mut classifier_unavailable = false;
    let mut doc_classifier: Option<DocOrientClassifier> = None;
    let mut doc_unavailable = false;

    for (idx, obj) in objects.iter().enumerate() {
        // The masked cutout isolates the object — used for text detection (and
        // for the `--alpha` output), so background text/objects don't pollute
        // the angle signal.
        let cutout = crop_object(&src, w as usize, &labels, obj);

        if args.debug {
            elog!(
                "[obj {idx}] bbox {}x{} area {} px",
                obj.max_x - obj.min_x + 1,
                obj.max_y - obj.min_y + 1,
                obj.area
            );
        }

        let mask = angle::object_mask(
            &labels, w as usize, obj.label, obj.min_x, obj.min_y, obj.max_x, obj.max_y,
        );
        // Rectangularity is measured against the safezone rectangle. A
        // rectangular mask gets the tight Min/body fit; an irregular mask keeps
        // the Max/safezone fit so no Min/body logic is applied to it.
        let rectangularity = angle::mask_rectangularity(&mask, angle::MaskFit::Max);
        let is_non_rectangular =
            rectangularity.is_some_and(|r| r < args.auto_text_rectangularity_threshold);
        let angle_fit = angle::MaskFit::Max;
        let crop_fit = if args.safezone || is_non_rectangular {
            angle::MaskFit::Max
        } else {
            angle::MaskFit::Min
        };
        // Rotation always comes from the Max/safezone rectangle. The fitted
        // rectangle drawn blue is used only for RGB crop bounds after deskew.
        let fit_corners = angle::mask_rect_corners(&mask, crop_fit);
        let auto_text = !args.text && is_non_rectangular;
        let use_text = args.text || auto_text;
        // Rectangular objects with enclosed holes (e.g. a cassette's reel
        // windows) are complex like non-rectangular ones, so give them the same
        // breathing-room padding. Skip the (cheap-ish) hole scan when the object
        // is already non-rectangular and padded anyway.
        let hole_count = if is_non_rectangular {
            0
        } else {
            angle::mask_hole_count(&mask, (obj.area / 1000).max(64))
        };
        let crop_padding = if is_non_rectangular || hole_count > 0 {
            args.non_rectangular_padding
        } else {
            0.0
        };

        // Run the detector only when text drives the deskew (--text or a
        // non-rectangular mask); rectangular crops get their orientation from the
        // doc-orientation model instead, so they need no text boxes.
        let text_boxes = if use_text && !detector_unavailable {
            if detector.is_none() {
                match Detector::load() {
                    Ok(d) => detector = Some(d),
                    Err(e) => {
                        elog!(
                            "warning: text detector unavailable ({e:#}); using mask orientation only"
                        );
                        detector_unavailable = true;
                    }
                }
            }

            match detector.as_mut() {
                Some(det) => det.text_boxes(&cutout, args.debug)?,
                None => Vec::new(),
            }
        } else {
            Vec::new()
        };
        // (angle, length) samples; the boxes keep their corners for the overlay.
        let text_samples: Vec<(f64, f64)> =
            text_boxes.iter().map(|b| (b.angle, b.length)).collect();

        // Angle source: the outlier-rejected, length-weighted text angle, but
        // only when text rotation is in effect (--text, or a sufficiently
        // non-rectangular mask). Otherwise the mask's Max/safezone geometry
        // wins. Min/body is crop-only and must not affect rotation.
        let mask_cand = angle::mask_angle(&mask, angle_fit);
        let text_est = if use_text {
            angle::robust_text_angle(&text_samples, args.angle_tol)
        } else {
            None
        };

        let (final_angle, source) = match (text_est, mask_cand) {
            (Some(te), _) => (Some(te.degrees as f64), AngleSource::Text),
            (None, Some(ma)) => (Some(ma), AngleSource::Mask),
            (None, None) => (None, AngleSource::None),
        };

        // Deskew rotation: -angle brings the chosen direction to horizontal.
        let theta: f32 = match (final_angle, args.no_deskew) {
            (Some(a), false) => -(a as f32).to_radians(),
            _ => 0.0,
        };

        // The fitted rectangle is oriented along the mask, so it stays
        // axis-aligned after deskew only when the mask drove the angle. With a
        // text-driven deskew it is tilted in the deskewed frame and its bounding
        // box balloons, so crop to the component bounds (tight at the reading
        // orientation) instead.
        let crop_corners = if matches!(source, AngleSource::Text) {
            None
        } else {
            fit_corners
        };

        // Orientation correction: textline 0/180 vote for text-driven crops, the
        // doc-orientation model (0/90/180/270) for rectangular ones.
        let quarter_turns = decide_quarter_turns(
            use_text,
            args.no_deskew,
            args.no_doc_orient,
            args.doc_orient_model.as_deref().map(Path::new),
            &cutout,
            &text_boxes,
            theta,
            &mut classifier,
            &mut classifier_unavailable,
            &mut doc_classifier,
            &mut doc_unavailable,
            args.debug,
        )?;

        if args.debug {
            match rectangularity {
                Some(r) => {
                    let mode = if args.text {
                        "forced by --text"
                    } else if auto_text {
                        "auto text enabled"
                    } else {
                        "mask only"
                    };
                    elog!(
                        "    rectangularity: {r:.3} (threshold {:.3}; {mode}; padding {crop_padding:.1}%)",
                        args.auto_text_rectangularity_threshold,
                    );
                }
                None => elog!("    rectangularity: n/a (mask only)"),
            }
            if hole_count > 0 {
                elog!(
                    "    holes: {hole_count} enclosed -> padded {:.1}% like non-rectangular",
                    args.non_rectangular_padding
                );
            }
            log_angle_debug(&text_samples, text_est, mask_cand, source, args.angle_tol);
            log_crop_geometry(
                crop_fit,
                fit_corners,
                crop_corners,
                &labels,
                w as usize,
                obj,
                theta,
                crop_padding,
            );
            // Visual overlay: draw the detector's boxes and the final angle so we
            // can see whether the geometry, not the aggregation, is at fault.
            let overlay = draw_debug_overlay(
                &cutout,
                &mask,
                &text_boxes,
                fit_corners,
                (crop_fit != angle::MaskFit::Max)
                    .then(|| angle::mask_rect_corners(&mask, angle::MaskFit::Max))
                    .flatten(),
                final_angle.map(|a| a as f32),
            );
            let object_path = output_plan.object_path(idx);
            let dpath = debug_image_path(&object_path);
            overlay
                .save(&dpath)
                .with_context(|| format!("failed to write debug image: {}", dpath.display()))?;
            elog!("    debug image: {}", dpath.display());

            // Deskewed overlay: the same scene rotated by the chosen theta, with
            // a gray horizontal reference. If the estimate is right, the magenta
            // line should land flat on the reference — this proves the rotation
            // follows the estimate (or shows it doesn't).
            let mut desk = rotate_about_center_no_crop(
                &overlay,
                theta,
                Interpolation::Bilinear,
                Border::Constant(Rgb([255, 255, 255])),
            );
            let (dw, dh) = desk.dimensions();
            let cy = dh as f32 / 2.0;
            draw_line_segment_mut(&mut desk, (0.0, cy), (dw as f32, cy), Rgb([150, 150, 150]));
            let ddpath = suffixed_path(&object_path, ".deskewed");
            desk.save(&ddpath)
                .with_context(|| format!("failed to write debug image: {}", ddpath.display()))?;
            elog!(
                "    deskewed overlay: {} (magenta should lie on the gray line)",
                ddpath.display()
            );
        }

        // Write the cutout (RGBA, transparent background) or, by default, a
        // plain rectangular RGB crop that keeps the original background.
        let path = output_plan.object_path(idx);
        let (out_w, out_h) = if args.alpha {
            write_alpha_output(
                &src_original,
                AlphaOutputRequest {
                    labels: &labels,
                    w: w as usize,
                    obj,
                    theta,
                    quarter_turns,
                    crop_padding,
                    path: &path,
                    format: output_plan.format,
                },
            )?
        } else {
            // Crop to the fitted rectangle when the mask drove the angle; for a
            // text-driven deskew `crop_corners` is None and we use component bounds.
            let mut bounds = rotated_crop_bounds(crop_corners, &labels, w as usize, obj, theta)
                .context("object has no foreground pixels while cropping")?;
            bounds.expand_percent(crop_padding);
            // Deskew by sampling the full source, so rotation reveals the real
            // pixels around the object; padding is used only out of bounds.
            write_rgb_output(
                &src_original,
                bounds,
                theta,
                quarter_turns,
                &path,
                output_plan.format,
            )?
        };

        let angle_note = angle_note(final_angle, source, text_est, quarter_turns);
        println!(
            "wrote {} ({out_w}x{out_h}, {} px) angle {angle_note}",
            path.display(),
            obj.area
        );
    }
    println!("found {} object(s)", objects.len());

    Ok(())
}

/// Format the per-object angle summary, noting any orientation correction.
fn angle_note(
    final_angle: Option<f64>,
    source: AngleSource,
    text_est: Option<angle::AngleEstimate>,
    quarter_turns: u8,
) -> String {
    let base = match (final_angle, source) {
        (Some(a), AngleSource::Text) => {
            let e = text_est.expect("text source implies an estimate");
            format!(
                "{a:+.2}° (text, {}/{} boxes, agree {:.0}%)",
                e.inliers,
                e.total,
                e.agreement * 100.0
            )
        }
        (Some(a), AngleSource::Mask) => format!("{a:+.2}° (mask geometry)"),
        _ => "n/a".to_string(),
    };
    match quarter_turns {
        0 => base,
        n => format!("{base} + {}° turn (orientation cls)", n as u32 * 90),
    }
}

/// Decide how many 90-degree clockwise turns to apply to a deskewed object so it
/// ends up upright. Text-driven crops (`--text` or a non-rectangular mask) use
/// the textline 0/180 vote over the detected boxes; rectangular crops use the
/// whole-image doc-orientation model (0/90/180/270). Both classifiers load lazily
/// and fall back to no rotation when unavailable (e.g. offline).
#[allow(clippy::too_many_arguments)]
fn decide_quarter_turns(
    use_text: bool,
    no_deskew: bool,
    no_doc_orient: bool,
    doc_model_path: Option<&Path>,
    cutout: &image::RgbaImage,
    text_boxes: &[angle::TextBox],
    theta: f32,
    textline: &mut Option<OrientClassifier>,
    textline_unavailable: &mut bool,
    doc: &mut Option<DocOrientClassifier>,
    doc_unavailable: &mut bool,
    debug: bool,
) -> Result<u8> {
    if no_deskew {
        return Ok(0);
    }

    if use_text {
        // Non-rectangular / forced: vote 0 vs 180 over the text lines.
        if text_boxes.is_empty() || *textline_unavailable {
            return Ok(0);
        }
        if textline.is_none() {
            match OrientClassifier::load() {
                Ok(c) => *textline = Some(c),
                Err(e) => {
                    elog!(
                        "warning: textline orientation classifier unavailable ({e:#}); skipping 180° correction"
                    );
                    *textline_unavailable = true;
                }
            }
        }
        return Ok(match textline.as_mut() {
            Some(c) => {
                let flip = c.vote_flip(cutout, text_boxes, theta, debug)?.unwrap_or(false);
                if flip { 2 } else { 0 }
            }
            None => 0,
        });
    }

    // Rectangular: classify the whole deskewed object (0/90/180/270).
    if no_doc_orient || *doc_unavailable {
        return Ok(0);
    }
    if doc.is_none() {
        match DocOrientClassifier::load(doc_model_path) {
            Ok(c) => *doc = Some(c),
            Err(e) => {
                elog!(
                    "warning: doc orientation classifier unavailable ({e:#}); skipping orientation correction"
                );
                *doc_unavailable = true;
            }
        }
    }
    Ok(match doc.as_mut() {
        Some(c) => {
            let (turns, conf) = c.orient(cutout, theta)?;
            if debug {
                elog!("    doc orientation: {turns} quarter-turn(s) CW (conf {conf:.2})");
            }
            turns
        }
        None => 0,
    })
}

fn collect_input_paths(input: &Path) -> Result<Vec<PathBuf>> {
    let meta = std::fs::metadata(input)
        .with_context(|| format!("failed to read input path metadata: {}", input.display()))?;
    if meta.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    if !meta.is_dir() {
        bail!(
            "input path is neither a file nor a directory: {}",
            input.display()
        );
    }

    let mut paths = Vec::new();
    for entry in std::fs::read_dir(input)
        .with_context(|| format!("failed to read input directory: {}", input.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_file() && ImageFormat::from_path(&path).is_ok() {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn output_dir_has_prefix(output_dir: &Path, input_file: &Path) -> Result<bool> {
    let stem = input_file
        .file_stem()
        .and_then(|s| s.to_str())
        .context("input file name is not valid UTF-8")?;
    let prefix = format!("{stem}_");

    for entry in std::fs::read_dir(output_dir)
        .with_context(|| format!("failed to read output directory: {}", output_dir.display()))?
    {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        if entry
            .file_name()
            .to_str()
            .is_some_and(|name| name.starts_with(&prefix))
        {
            return Ok(true);
        }
    }

    Ok(false)
}

#[allow(clippy::too_many_arguments)]
fn process_input_file(
    input_path: &Path,
    args: &Args,
    cutoff: u8,
    model: &mut RMBG,
    detector: &mut Option<Detector>,
    detector_unavailable: &mut bool,
    classifier: &mut Option<OrientClassifier>,
    classifier_unavailable: &mut bool,
    doc_classifier: &mut Option<DocOrientClassifier>,
    doc_unavailable: &mut bool,
    output_dir: &Path,
) -> Result<()> {
    let (src_original, input_format) = read_original_image(input_path)?;
    let output_dir = output_dir
        .to_str()
        .context("output directory path is not valid UTF-8")?;
    let output_plan = OutputPlan::new(
        input_path,
        None,
        Some(output_dir),
        args.alpha,
        input_format,
        true,
    )?;

    let input_str = input_path.to_string_lossy().to_string();
    let model_image = Image::try_read(&input_str)
        .with_context(|| format!("failed to read image: {}", input_path.display()))?;
    let ys = model.forward(std::slice::from_ref(&model_image))?;

    let mask = ys
        .first()
        .and_then(|y| y.masks.first())
        .context("model returned no mask")?;
    let alpha = mask.to_vec();

    let src = model_image.to_rgba8();
    let (w, h) = src.dimensions();
    if src_original.dimensions() != (w, h) {
        bail!(
            "original image dimensions ({:?}) do not match model image dimensions ({w}x{h})",
            src_original.dimensions()
        );
    }
    if (w * h) as usize != alpha.len() {
        bail!(
            "mask size ({}) does not match image size ({}x{})",
            alpha.len(),
            w,
            h
        );
    }

    let fg: Vec<bool> = alpha.iter().map(|&a| a >= cutoff).collect();
    let (labels, mut objects) = connected_components(&fg, w as usize, h as usize, args.min_area);
    // Drop noise: components no side of which is large relative to the image.
    objects.retain(|obj| meets_min_side(obj, w as usize, h as usize, args.min_side_percent));
    if objects.is_empty() {
        bail!("no objects found above the threshold");
    }

    for (idx, obj) in objects.iter().enumerate() {
        let cutout = crop_object(&src, w as usize, &labels, obj);

        if args.debug {
            elog!(
                "[obj {idx}] bbox {}x{} area {} px",
                obj.max_x - obj.min_x + 1,
                obj.max_y - obj.min_y + 1,
                obj.area
            );
        }

        let mask = angle::object_mask(
            &labels, w as usize, obj.label, obj.min_x, obj.min_y, obj.max_x, obj.max_y,
        );
        // Rectangularity is measured against the safezone rectangle. A
        // rectangular mask gets the tight Min/body fit; an irregular mask keeps
        // the Max/safezone fit so no Min/body logic is applied to it.
        let rectangularity = angle::mask_rectangularity(&mask, angle::MaskFit::Max);
        let is_non_rectangular =
            rectangularity.is_some_and(|r| r < args.auto_text_rectangularity_threshold);
        let angle_fit = angle::MaskFit::Max;
        let crop_fit = if args.safezone || is_non_rectangular {
            angle::MaskFit::Max
        } else {
            angle::MaskFit::Min
        };
        // Rotation always comes from the Max/safezone rectangle. The fitted
        // rectangle drawn blue is used only for RGB crop bounds after deskew.
        let fit_corners = angle::mask_rect_corners(&mask, crop_fit);
        let auto_text = !args.text && is_non_rectangular;
        let use_text = args.text || auto_text;
        // Rectangular objects with enclosed holes (e.g. a cassette's reel
        // windows) are complex like non-rectangular ones, so give them the same
        // breathing-room padding. Skip the (cheap-ish) hole scan when the object
        // is already non-rectangular and padded anyway.
        let hole_count = if is_non_rectangular {
            0
        } else {
            angle::mask_hole_count(&mask, (obj.area / 1000).max(64))
        };
        let crop_padding = if is_non_rectangular || hole_count > 0 {
            args.non_rectangular_padding
        } else {
            0.0
        };

        // Run the detector only when text drives the deskew (--text or a
        // non-rectangular mask); rectangular crops use the doc-orientation model.
        let text_boxes = if use_text && !*detector_unavailable {
            if detector.is_none() {
                match Detector::load() {
                    Ok(d) => *detector = Some(d),
                    Err(e) => {
                        elog!(
                            "warning: text detector unavailable ({e:#}); using mask orientation only"
                        );
                        *detector_unavailable = true;
                    }
                }
            }

            match detector.as_mut() {
                Some(det) => det.text_boxes(&cutout, args.debug)?,
                None => Vec::new(),
            }
        } else {
            Vec::new()
        };
        let text_samples: Vec<(f64, f64)> =
            text_boxes.iter().map(|b| (b.angle, b.length)).collect();

        // Text angle only drives the deskew when text rotation is in effect;
        // otherwise mask Max/safezone geometry wins. Min/body is crop-only and
        // must not affect rotation.
        let mask_cand = angle::mask_angle(&mask, angle_fit);
        let text_est = if use_text {
            angle::robust_text_angle(&text_samples, args.angle_tol)
        } else {
            None
        };

        let (final_angle, source) = match (text_est, mask_cand) {
            (Some(te), _) => (Some(te.degrees as f64), AngleSource::Text),
            (None, Some(ma)) => (Some(ma), AngleSource::Mask),
            (None, None) => (None, AngleSource::None),
        };

        let theta: f32 = match (final_angle, args.no_deskew) {
            (Some(a), false) => -(a as f32).to_radians(),
            _ => 0.0,
        };

        // The fitted rectangle is oriented along the mask, so it stays
        // axis-aligned after deskew only when the mask drove the angle. With a
        // text-driven deskew it is tilted in the deskewed frame and its bounding
        // box balloons, so crop to the component bounds (tight at the reading
        // orientation) instead.
        let crop_corners = if matches!(source, AngleSource::Text) {
            None
        } else {
            fit_corners
        };

        // Orientation correction: textline 0/180 vote for text-driven crops, the
        // doc-orientation model (0/90/180/270) for rectangular ones.
        let quarter_turns = decide_quarter_turns(
            use_text,
            args.no_deskew,
            args.no_doc_orient,
            args.doc_orient_model.as_deref().map(Path::new),
            &cutout,
            &text_boxes,
            theta,
            classifier,
            classifier_unavailable,
            doc_classifier,
            doc_unavailable,
            args.debug,
        )?;

        if args.debug {
            match rectangularity {
                Some(r) => {
                    let mode = if args.text {
                        "forced by --text"
                    } else if auto_text {
                        "auto text enabled"
                    } else {
                        "mask only"
                    };
                    elog!(
                        "    rectangularity: {r:.3} (threshold {:.3}; {mode}; padding {crop_padding:.1}%)",
                        args.auto_text_rectangularity_threshold,
                    );
                }
                None => elog!("    rectangularity: n/a (mask only)"),
            }
            if hole_count > 0 {
                elog!(
                    "    holes: {hole_count} enclosed -> padded {:.1}% like non-rectangular",
                    args.non_rectangular_padding
                );
            }
            log_angle_debug(&text_samples, text_est, mask_cand, source, args.angle_tol);
            log_crop_geometry(
                crop_fit,
                fit_corners,
                crop_corners,
                &labels,
                w as usize,
                obj,
                theta,
                crop_padding,
            );
            let overlay = draw_debug_overlay(
                &cutout,
                &mask,
                &text_boxes,
                fit_corners,
                (crop_fit != angle::MaskFit::Max)
                    .then(|| angle::mask_rect_corners(&mask, angle::MaskFit::Max))
                    .flatten(),
                final_angle.map(|a| a as f32),
            );
            let object_path = output_plan.object_path(idx);
            let dpath = debug_image_path(&object_path);
            overlay
                .save(&dpath)
                .with_context(|| format!("failed to write debug image: {}", dpath.display()))?;
            elog!("    debug image: {}", dpath.display());

            let mut desk = rotate_about_center_no_crop(
                &overlay,
                theta,
                Interpolation::Bilinear,
                Border::Constant(Rgb([255, 255, 255])),
            );
            let (dw, dh) = desk.dimensions();
            let cy = dh as f32 / 2.0;
            draw_line_segment_mut(&mut desk, (0.0, cy), (dw as f32, cy), Rgb([150, 150, 150]));
            let ddpath = suffixed_path(&object_path, ".deskewed");
            desk.save(&ddpath)
                .with_context(|| format!("failed to write debug image: {}", ddpath.display()))?;
            elog!(
                "    deskewed overlay: {} (magenta should lie on the gray line)",
                ddpath.display()
            );
        }

        let path = output_plan.object_path(idx);
        let (out_w, out_h) = if args.alpha {
            write_alpha_output(
                &src_original,
                AlphaOutputRequest {
                    labels: &labels,
                    w: w as usize,
                    obj,
                    theta,
                    quarter_turns,
                    crop_padding,
                    path: &path,
                    format: output_plan.format,
                },
            )?
        } else {
            // Crop to the fitted rectangle when the mask drove the angle; for a
            // text-driven deskew `crop_corners` is None and we use component bounds.
            let mut bounds = rotated_crop_bounds(crop_corners, &labels, w as usize, obj, theta)
                .context("object has no foreground pixels while cropping")?;
            bounds.expand_percent(crop_padding);
            write_rgb_output(
                &src_original,
                bounds,
                theta,
                quarter_turns,
                &path,
                output_plan.format,
            )?
        };

        let angle_note = angle_note(final_angle, source, text_est, quarter_turns);
        println!(
            "wrote {} ({out_w}x{out_h}, {} px) angle {angle_note}",
            path.display(),
            obj.area
        );
    }
    println!(
        "found {} object(s) in {}",
        objects.len(),
        input_path.display()
    );

    Ok(())
}

/// Build the RMBG model, trying CUDA -> DirectML -> CPU and using the first
/// device that initializes. usls registers a single execution provider and
/// errors when it is unavailable, so we drive the fallback ourselves.
fn build_rmbg() -> Result<RMBG> {
    let mut last_err = None;
    for (device, name) in [
        (Device::Cuda(0), "CUDA"),
        (Device::DirectMl(0), "DirectML"),
        (Device::Cpu(0), "CPU"),
    ] {
        let config = match Config::rmbg2_0().with_device_all(device).commit() {
            Ok(c) => c,
            Err(e) => {
                last_err = Some(e);
                continue;
            }
        };
        match RMBG::new(config) {
            Ok(model) => {
                elog!("RMBG running on {name}");
                return Ok(model);
            }
            Err(e) => {
                elog!("RMBG unavailable on {name}: {e:#}");
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no execution provider available")))
        .context("could not initialize RMBG on any device")
}

fn read_original_image(path: &Path) -> Result<(DynamicImage, ImageFormat)> {
    let reader = image::ImageReader::open(path)
        .with_context(|| format!("failed to open image: {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("failed to guess image format: {}", path.display()))?;
    let format = reader
        .format()
        .or_else(|| ImageFormat::from_path(path).ok())
        .context("could not determine input image format")?;
    let image = reader
        .decode()
        .with_context(|| format!("failed to decode image: {}", path.display()))?;
    Ok((image, format))
}

#[derive(Debug, Clone)]
struct OutputPlan {
    base: PathBuf,
    format: ImageFormat,
}

impl OutputPlan {
    fn new(
        input: &Path,
        output: Option<&str>,
        output_dir: Option<&str>,
        alpha: bool,
        input_format: ImageFormat,
        input_is_dir: bool,
    ) -> Result<Self> {
        if input_is_dir {
            let output_dir =
                output_dir.context("output-dir is required when input is a directory")?;
            if output.is_some() {
                bail!("--output cannot be used when the input is a directory; use --output-dir");
            }
            let mut base = PathBuf::from(output_dir);
            base.push(
                input
                    .file_name()
                    .context("directory inputs must have a file name")?,
            );
            let mut format = input_format;

            if alpha && !format_supports_alpha(format) {
                format = ImageFormat::Png;
                base.set_extension("png");
            } else if base.extension().is_none() {
                base.set_extension(preferred_extension(format));
            }

            if !format.writing_enabled() {
                bail!("output format {format:?} is not enabled for writing");
            }

            return Ok(Self { base, format });
        }

        if output_dir.is_some() {
            bail!("--output-dir can only be used when the input is a directory");
        }

        let mut base = output
            .map(PathBuf::from)
            .unwrap_or_else(|| input.to_path_buf());
        let mut format = output
            .and_then(|p| ImageFormat::from_path(p).ok())
            .unwrap_or(input_format);

        if alpha && !format_supports_alpha(format) {
            format = ImageFormat::Png;
            base.set_extension("png");
        } else if base.extension().is_none() {
            base.set_extension(preferred_extension(format));
        }

        if !format.writing_enabled() {
            bail!("output format {format:?} is not enabled for writing");
        }

        Ok(Self { base, format })
    }

    fn object_path(&self, idx: usize) -> PathBuf {
        suffixed_path(&self.base, &format!("_{idx}"))
    }
}

fn preferred_extension(format: ImageFormat) -> &'static str {
    format.extensions_str().first().copied().unwrap_or("png")
}

fn format_supports_alpha(format: ImageFormat) -> bool {
    matches!(
        format,
        ImageFormat::Png
            | ImageFormat::Ico
            | ImageFormat::Tiff
            | ImageFormat::Tga
            | ImageFormat::Pnm
            | ImageFormat::Farbfeld
            | ImageFormat::Avif
            | ImageFormat::WebP
            | ImageFormat::OpenExr
            | ImageFormat::Qoi
    )
}

/// Which candidate angle was chosen for an object.
#[derive(Debug, Clone, Copy, PartialEq)]
enum AngleSource {
    Text,
    Mask,
    None,
}

/// Emit a detailed breakdown of how an object's angle was decided: the mask
/// geometry candidate, the outlier-rejected text candidate, the winner, and
/// which text boxes were kept (inliers) vs. discarded (outliers).
fn log_angle_debug(
    text_boxes: &[(f64, f64)],
    text_est: Option<angle::AngleEstimate>,
    mask_cand: Option<f64>,
    source: AngleSource,
    tol: f64,
) {
    match mask_cand {
        Some(a) => elog!("    mask : {a:+7.2}°"),
        None => elog!("    mask : (no rectangle — degenerate component)"),
    }
    match text_est {
        Some(e) => elog!(
            "    text : {:+7.2}°  (anchor {:+.2}°, {}/{} inliers, agree {:.0}%)",
            e.degrees,
            e.anchor,
            e.inliers,
            e.total,
            e.agreement * 100.0,
        ),
        None => elog!("    text : (none)"),
    }
    elog!("    chosen: {source:?}");

    // Per-box inlier/outlier breakdown relative to the text anchor.
    if let Some(e) = text_est {
        let anchor = e.anchor as f64;
        for (i, &(deg, wt)) in text_boxes.iter().enumerate() {
            let d = angle::orientation_dist(deg, anchor);
            let tag = if d <= tol { "IN " } else { "OUT" };
            elog!("      text #{i:<3} angle {deg:+7.2}°  len {wt:7.1}  [{tag} d={d:4.1}]");
        }
    }
}

/// Render an annotated overlay for one object so the detector geometry can be
/// inspected directly: the cutout (the exact pixels the detector saw, on white)
/// with the fine mask tinted green, each text box (faded orange) and its
/// long-axis (red), the fitted mask rectangle (blue) and maximal safezone
/// rectangle (cyan), and the final estimated angle through the center (magenta).
fn draw_debug_overlay(
    cutout: &image::RgbaImage,
    mask: &image::GrayImage,
    text_boxes: &[angle::TextBox],
    mask_corners: Option<[(f64, f64); 4]>,
    safezone_corners: Option<[(f64, f64); 4]>,
    estimate_deg: Option<f32>,
) -> RgbImage {
    let (w, h) = cutout.dimensions();
    let mut img = RgbImage::from_pixel(w, h, Rgb([255, 255, 255]));
    for (x, y, p) in cutout.enumerate_pixels() {
        let a = p[3] as f32 / 255.0;
        let blend = |c: u8| (c as f32 * a + 255.0 * (1.0 - a)).round() as u8;
        img.put_pixel(x, y, Rgb([blend(p[0]), blend(p[1]), blend(p[2])]));
    }

    // Tint the fine mask (the exact per-pixel component) green so the rectangle
    // fits can be judged against the real shape, spikes and all. The mask shares
    // the cutout's coordinate frame.
    let (mw, mh) = mask.dimensions();
    for y in 0..h.min(mh) {
        for x in 0..w.min(mw) {
            if mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            let p = img.get_pixel(x, y);
            let tint = |c: u8, t: u8| ((c as f32 * 0.6) + (t as f32 * 0.4)).round() as u8;
            img.put_pixel(x, y, Rgb([tint(p[0], 0), tint(p[1], 200), tint(p[2], 0)]));
        }
    }

    let red = Rgb([220, 0, 0]);
    let blue = Rgb([0, 90, 255]);
    let cyan = Rgb([0, 200, 200]);
    let magenta = Rgb([230, 0, 230]);

    // Maximal "safezone" rectangle (cyan), drawn first so the active fit sits
    // on top. Only present when it differs from the active rectangle.
    if let Some(c) = safezone_corners {
        draw_quad(&mut img, &c, cyan);
    }
    // Mask rectangle actually used for the angle (blue).
    if let Some(c) = mask_corners {
        draw_quad(&mut img, &c, blue);
    }

    // Each text box and the long-axis line it votes with (red). Boxes are
    // colored by DB score: green = confident, shading to orange as it drops.
    for b in text_boxes {
        let fade = (1.0 - b.score).clamp(0.0, 1.0);
        let box_color = Rgb([(fade * 220.0) as u8, 170, 0]);
        draw_quad(&mut img, &b.corners, box_color);
        let cx = b.corners.iter().map(|p| p.0).sum::<f64>() / 4.0;
        let cy = b.corners.iter().map(|p| p.1).sum::<f64>() / 4.0;
        let (s, c) = b.angle.to_radians().sin_cos();
        let half = b.length / 2.0;
        draw_line_segment_mut(
            &mut img,
            ((cx - c * half) as f32, (cy - s * half) as f32),
            ((cx + c * half) as f32, (cy + s * half) as f32),
            red,
        );
    }

    // Final estimate through the image center (magenta), spanning the crop.
    if let Some(deg) = estimate_deg {
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        let (s, c) = deg.to_radians().sin_cos();
        let half = w.max(h) as f32;
        draw_line_segment_mut(
            &mut img,
            (cx - c * half, cy - s * half),
            (cx + c * half, cy + s * half),
            magenta,
        );
    }

    img
}

/// Draw the four edges of a quadrilateral.
fn draw_quad(img: &mut RgbImage, corners: &[(f64, f64); 4], color: Rgb<u8>) {
    for i in 0..4 {
        let a = corners[i];
        let b = corners[(i + 1) % 4];
        draw_line_segment_mut(
            img,
            (a.0 as f32, a.1 as f32),
            (b.0 as f32, b.1 as f32),
            color,
        );
    }
}

/// Path for an object's debug overlay, e.g. `out_0.png` -> `out_0.debug.png`.
fn debug_image_path(path: &Path) -> PathBuf {
    suffixed_path(path, ".debug")
}

/// Insert `suffix` before the file extension of `path`.
fn suffixed_path(path: &Path, suffix: &str) -> PathBuf {
    let mut out = path.to_path_buf();
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let file_name = match path.extension().and_then(|s| s.to_str()) {
        Some(ext) => format!("{stem}{suffix}.{ext}"),
        None => format!("{stem}{suffix}"),
    };
    out.set_file_name(file_name);
    out
}

/// A labelled connected component: its bounding box and pixel count.
struct Object {
    label: usize,
    area: usize,
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
}

/// Whether a component is large enough to keep, given a noise threshold of
/// `min_side_percent` of the smaller image dimension. A component survives when
/// its longer bounding-box side reaches that fraction of `min(w, h)`; this drops
/// specks that no single side makes large relative to the image. `0` keeps all.
fn meets_min_side(obj: &Object, w: usize, h: usize, min_side_percent: f64) -> bool {
    if min_side_percent <= 0.0 {
        return true;
    }
    let longer_side = (obj.max_x - obj.min_x + 1).max(obj.max_y - obj.min_y + 1);
    longer_side as f64 >= min_side_percent / 100.0 * w.min(h) as f64
}

/// Label 8-connected foreground regions via iterative flood fill, keeping only
/// those with at least `min_area` pixels. Returns the per-pixel label map (0 =
/// background) and the surviving objects sorted largest-first. Each object's
/// `label` indexes into the label map.
fn connected_components(
    fg: &[bool],
    w: usize,
    h: usize,
    min_area: usize,
) -> (Vec<usize>, Vec<Object>) {
    let mut labels = vec![0usize; fg.len()]; // 0 = unvisited/background
    let mut objects: Vec<Object> = Vec::new();
    let mut stack: Vec<(usize, usize)> = Vec::new();

    for sy in 0..h {
        for sx in 0..w {
            let start = sy * w + sx;
            if !fg[start] || labels[start] != 0 {
                continue;
            }

            let label = objects.len() + 1;
            let mut obj = Object {
                label,
                area: 0,
                min_x: sx,
                min_y: sy,
                max_x: sx,
                max_y: sy,
            };

            labels[start] = label;
            stack.push((sx, sy));
            while let Some((x, y)) = stack.pop() {
                obj.area += 1;
                obj.min_x = obj.min_x.min(x);
                obj.min_y = obj.min_y.min(y);
                obj.max_x = obj.max_x.max(x);
                obj.max_y = obj.max_y.max(y);

                let x0 = x.saturating_sub(1);
                let y0 = y.saturating_sub(1);
                let x1 = (x + 1).min(w - 1);
                let y1 = (y + 1).min(h - 1);
                for ny in y0..=y1 {
                    for nx in x0..=x1 {
                        let n = ny * w + nx;
                        if fg[n] && labels[n] == 0 {
                            labels[n] = label;
                            stack.push((nx, ny));
                        }
                    }
                }
            }

            if obj.area >= min_area.max(1) {
                objects.push(obj);
            }
        }
    }

    // Sorting only reorders the metadata; each object keeps its `label`, so the
    // label map stays valid for membership tests.
    objects.sort_by_key(|o| std::cmp::Reverse(o.area));
    (labels, objects)
}

/// Build an RGBA image cropped to `obj`'s bounding box, keeping opaque only the
/// pixels whose label matches this object (so overlapping bounding boxes from
/// other objects don't leak in); everything else is transparent.
fn crop_object(
    rgb: &image::RgbaImage,
    w: usize,
    labels: &[usize],
    obj: &Object,
) -> image::RgbaImage {
    let ow = (obj.max_x - obj.min_x + 1) as u32;
    let oh = (obj.max_y - obj.min_y + 1) as u32;
    let mut out = image::RgbaImage::new(ow, oh);

    for y in obj.min_y..=obj.max_y {
        for x in obj.min_x..=obj.max_x {
            if labels[y * w + x] == obj.label {
                let mut px = *rgb.get_pixel(x as u32, y as u32);
                px[3] = 255;
                out.put_pixel((x - obj.min_x) as u32, (y - obj.min_y) as u32, px);
            }
        }
    }
    out
}

fn crop_object_rgba16(rgb: &Rgba16Image, w: usize, labels: &[usize], obj: &Object) -> Rgba16Image {
    let ow = (obj.max_x - obj.min_x + 1) as u32;
    let oh = (obj.max_y - obj.min_y + 1) as u32;
    let mut out = Rgba16Image::new(ow, oh);

    for y in obj.min_y..=obj.max_y {
        for x in obj.min_x..=obj.max_x {
            if labels[y * w + x] == obj.label {
                let mut px = *rgb.get_pixel(x as u32, y as u32);
                px[3] = u16::MAX;
                out.put_pixel((x - obj.min_x) as u32, (y - obj.min_y) as u32, px);
            }
        }
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChannelDepth {
    U8,
    U16,
}

fn channel_depth(img: &DynamicImage) -> Result<ChannelDepth> {
    match img.color() {
        image::ColorType::L8
        | image::ColorType::La8
        | image::ColorType::Rgb8
        | image::ColorType::Rgba8 => Ok(ChannelDepth::U8),
        image::ColorType::L16
        | image::ColorType::La16
        | image::ColorType::Rgb16
        | image::ColorType::Rgba16 => Ok(ChannelDepth::U16),
        other => bail!("unsupported source color type {other:?}; refusing to lose precision"),
    }
}

/// Rotate an image by `turns` quarter-turns clockwise (0..=3), the orientation
/// correction the classifier asked for. `turns % 4 == 0` is a no-op.
fn apply_quarter_turns<P>(
    img: &ImageBuffer<P, Vec<P::Subpixel>>,
    turns: u8,
) -> ImageBuffer<P, Vec<P::Subpixel>>
where
    P: Pixel + 'static,
{
    match turns % 4 {
        1 => image::imageops::rotate90(img),
        2 => image::imageops::rotate180(img),
        3 => image::imageops::rotate270(img),
        _ => img.clone(),
    }
}

fn write_rgb_output(
    src: &DynamicImage,
    bounds: RotatedBounds,
    theta: f32,
    quarter_turns: u8,
    path: &Path,
    format: ImageFormat,
) -> Result<(u32, u32)> {
    match channel_depth(src)? {
        ChannelDepth::U8 => {
            let src_rgb = src.to_rgb8();
            let mut img = deskew_from_src(&src_rgb, bounds, theta, Rgb([255, 255, 255]));
            if !quarter_turns.is_multiple_of(4) {
                img = apply_quarter_turns(&img, quarter_turns);
            }
            let dims = img.dimensions();
            match format {
                ImageFormat::Tiff => save_tiff_rgb8(path, &img)?,
                _ => img
                    .save_with_format(path, format)
                    .with_context(|| format!("failed to write image: {}", path.display()))?,
            }
            Ok(dims)
        }
        ChannelDepth::U16 => {
            let src_rgb = src.to_rgb16();
            let mut img =
                deskew_from_src(&src_rgb, bounds, theta, Rgb([u16::MAX, u16::MAX, u16::MAX]));
            if !quarter_turns.is_multiple_of(4) {
                img = apply_quarter_turns(&img, quarter_turns);
            }
            let dims = img.dimensions();
            match format {
                ImageFormat::Tiff => save_tiff_rgb16(path, &img)?,
                _ => img
                    .save_with_format(path, format)
                    .with_context(|| format!("failed to write image: {}", path.display()))?,
            }
            Ok(dims)
        }
    }
}

struct AlphaOutputRequest<'a> {
    labels: &'a [usize],
    w: usize,
    obj: &'a Object,
    theta: f32,
    quarter_turns: u8,
    crop_padding: f64,
    path: &'a Path,
    format: ImageFormat,
}

fn write_alpha_output(src: &DynamicImage, req: AlphaOutputRequest<'_>) -> Result<(u32, u32)> {
    match channel_depth(src)? {
        ChannelDepth::U8 => {
            let src_rgba = src.to_rgba8();
            let mut img = crop_object(&src_rgba, req.w, req.labels, req.obj);
            if req.theta != 0.0 {
                img = rotate_about_center_no_crop(
                    &img,
                    req.theta,
                    Interpolation::Bilinear,
                    Border::Constant(Rgba([0, 0, 0, 0])),
                );
            }
            img = trim_transparent(&img).unwrap_or(img);
            if req.crop_padding > 0.0 {
                img = pad_rgba(&img, req.crop_padding);
            }
            if !req.quarter_turns.is_multiple_of(4) {
                img = apply_quarter_turns(&img, req.quarter_turns);
            }
            let dims = img.dimensions();
            match req.format {
                ImageFormat::Tiff => save_tiff_rgba8(req.path, &img)?,
                _ => img
                    .save_with_format(req.path, req.format)
                    .with_context(|| format!("failed to write image: {}", req.path.display()))?,
            }
            Ok(dims)
        }
        ChannelDepth::U16 => {
            let src_rgba = src.to_rgba16();
            let mut img = crop_object_rgba16(&src_rgba, req.w, req.labels, req.obj);
            if req.theta != 0.0 {
                img = rotate_about_center_no_crop(
                    &img,
                    req.theta,
                    Interpolation::Bilinear,
                    Border::Constant(Rgba([0, 0, 0, 0])),
                );
            }
            img = trim_transparent16(&img).unwrap_or(img);
            if req.crop_padding > 0.0 {
                img = pad_rgba16(&img, req.crop_padding);
            }
            if !req.quarter_turns.is_multiple_of(4) {
                img = apply_quarter_turns(&img, req.quarter_turns);
            }
            let dims = img.dimensions();
            match req.format {
                ImageFormat::Tiff => save_tiff_rgba16(req.path, &img)?,
                _ => img
                    .save_with_format(req.path, req.format)
                    .with_context(|| format!("failed to write image: {}", req.path.display()))?,
            }
            Ok(dims)
        }
    }
}

fn save_tiff_rgb8(path: &Path, img: &image::RgbImage) -> Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("failed to create image: {}", path.display()))?;
    let mut encoder = TiffImageEncoder::new(file)?
        .with_compression(TiffCompression::Deflate(DeflateLevel::default()));
    encoder
        .write_image::<TiffRgb8>(img.width(), img.height(), img.as_raw())
        .with_context(|| format!("failed to write image: {}", path.display()))?;
    Ok(())
}

fn save_tiff_rgb16(path: &Path, img: &ImageBuffer<image::Rgb<u16>, Vec<u16>>) -> Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("failed to create image: {}", path.display()))?;
    let mut encoder = TiffImageEncoder::new(file)?
        .with_compression(TiffCompression::Deflate(DeflateLevel::default()));
    encoder
        .write_image::<TiffRgb16>(img.width(), img.height(), img.as_raw())
        .with_context(|| format!("failed to write image: {}", path.display()))?;
    Ok(())
}

fn save_tiff_rgba8(path: &Path, img: &image::RgbaImage) -> Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("failed to create image: {}", path.display()))?;
    let mut encoder = TiffImageEncoder::new(file)?
        .with_compression(TiffCompression::Deflate(DeflateLevel::default()));
    encoder
        .write_image::<TiffRgba8>(img.width(), img.height(), img.as_raw())
        .with_context(|| format!("failed to write image: {}", path.display()))?;
    Ok(())
}

fn save_tiff_rgba16(path: &Path, img: &Rgba16Image) -> Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("failed to create image: {}", path.display()))?;
    let mut encoder = TiffImageEncoder::new(file)?
        .with_compression(TiffCompression::Deflate(DeflateLevel::default()));
    encoder
        .write_image::<TiffRgba16>(img.width(), img.height(), img.as_raw())
        .with_context(|| format!("failed to write image: {}", path.display()))?;
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct RotatedBounds {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

impl RotatedBounds {
    fn include(&mut self, x: f64, y: f64) {
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
    }

    fn expand_percent(&mut self, percent: f64) {
        if percent <= 0.0 {
            return;
        }
        let pad_x = (self.max_x - self.min_x).max(1.0) * percent / 100.0;
        let pad_y = (self.max_y - self.min_y).max(1.0) * percent / 100.0;
        self.min_x -= pad_x;
        self.min_y -= pad_y;
        self.max_x += pad_x;
        self.max_y += pad_y;
    }
}

/// Bounds of the actual component pixels after applying the deskew transform.
fn rotated_component_bounds(
    labels: &[usize],
    w: usize,
    obj: &Object,
    theta: f32,
) -> Option<RotatedBounds> {
    let (sin, cos) = (theta as f64).sin_cos();
    let mut bounds: Option<RotatedBounds> = None;

    for y in obj.min_y..=obj.max_y {
        for x in obj.min_x..=obj.max_x {
            if labels[y * w + x] != obj.label {
                continue;
            }

            let corners = [
                (x as f64, y as f64),
                ((x + 1) as f64, y as f64),
                ((x + 1) as f64, (y + 1) as f64),
                (x as f64, (y + 1) as f64),
            ];
            for (px, py) in corners {
                let rx = px * cos - py * sin;
                let ry = px * sin + py * cos;
                match bounds.as_mut() {
                    Some(b) => b.include(rx, ry),
                    None => {
                        bounds = Some(RotatedBounds {
                            min_x: rx,
                            min_y: ry,
                            max_x: rx,
                            max_y: ry,
                        });
                    }
                }
            }
        }
    }

    bounds
}

/// Bounding box, in the theta-rotated frame, of a mask-local rectangle's four
/// corners (offset into full-image coordinates by the object's top-left). This
/// lets the crop follow the fitted mask rectangle instead of every component
/// pixel, so body-fit exclusions are cropped away too.
fn rotated_corner_bounds(
    corners: &[(f64, f64); 4],
    off_x: usize,
    off_y: usize,
    theta: f32,
) -> RotatedBounds {
    let (sin, cos) = (theta as f64).sin_cos();
    let rotate = |&(lx, ly): &(f64, f64)| {
        let (x, y) = (lx + off_x as f64, ly + off_y as f64);
        (x * cos - y * sin, x * sin + y * cos)
    };
    let (fx, fy) = rotate(&corners[0]);
    let mut bounds = RotatedBounds {
        min_x: fx,
        min_y: fy,
        max_x: fx,
        max_y: fy,
    };
    for c in &corners[1..] {
        let (rx, ry) = rotate(c);
        bounds.include(rx, ry);
    }
    bounds
}

fn rotated_crop_bounds(
    fit_corners: Option<[(f64, f64); 4]>,
    labels: &[usize],
    w: usize,
    obj: &Object,
    theta: f32,
) -> Option<RotatedBounds> {
    match fit_corners {
        Some(corners) => Some(rotated_corner_bounds(&corners, obj.min_x, obj.min_y, theta)),
        None => rotated_component_bounds(labels, w, obj, theta),
    }
}

/// Log the fitted rectangle (the blue overlay box, in cutout-local coordinates)
/// and the actual RGB crop window in the deskewed frame.
#[allow(clippy::too_many_arguments)]
fn log_crop_geometry(
    fit: angle::MaskFit,
    fit_corners: Option<[(f64, f64); 4]>,
    crop_corners: Option<[(f64, f64); 4]>,
    labels: &[usize],
    w: usize,
    obj: &Object,
    theta: f32,
    crop_padding: f64,
) {
    if let Some(c) = fit_corners {
        elog!(
            "    blue rect ({fit:?}, cutout-local): ({:.0},{:.0}) ({:.0},{:.0}) ({:.0},{:.0}) ({:.0},{:.0})",
            c[0].0,
            c[0].1,
            c[1].0,
            c[1].1,
            c[2].0,
            c[2].1,
            c[3].0,
            c[3].1,
        );
    } else {
        elog!("    blue rect ({fit:?}): none (degenerate mask)");
    }

    let Some(mut b) = rotated_crop_bounds(crop_corners, labels, w, obj, theta) else {
        elog!("    crop window: none (component has no foreground pixels)");
        return;
    };
    b.expand_percent(crop_padding);
    let crop_basis = if crop_corners.is_some() {
        "blue rect"
    } else {
        "component bounds (text-driven)"
    };
    elog!(
        "    crop window ({crop_basis}, deskewed frame): x[{:.0}..{:.0}] y[{:.0}..{:.0}] -> {}x{} px (+{crop_padding:.1}% pad)",
        b.min_x,
        b.max_x,
        b.min_y,
        b.max_y,
        (b.max_x - b.min_x).ceil().max(1.0) as i64,
        (b.max_y - b.min_y).ceil().max(1.0) as i64,
    );
}

fn trim_transparent(img: &image::RgbaImage) -> Option<image::RgbaImage> {
    let (w, h) = img.dimensions();
    let mut min_x = w;
    let mut min_y = h;
    let mut max_x = 0;
    let mut max_y = 0;
    let mut found = false;

    for (x, y, px) in img.enumerate_pixels() {
        if px[3] == 0 {
            continue;
        }
        found = true;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    found.then(|| {
        image::imageops::crop_imm(img, min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
            .to_image()
    })
}

fn trim_transparent16(img: &Rgba16Image) -> Option<Rgba16Image> {
    let (w, h) = img.dimensions();
    let mut min_x = w;
    let mut min_y = h;
    let mut max_x = 0;
    let mut max_y = 0;
    let mut found = false;

    for (x, y, px) in img.enumerate_pixels() {
        if px[3] == 0 {
            continue;
        }
        found = true;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    found.then(|| {
        image::imageops::crop_imm(img, min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
            .to_image()
    })
}

fn pad_rgba(img: &image::RgbaImage, percent: f64) -> image::RgbaImage {
    if percent <= 0.0 {
        return img.clone();
    }

    let (w, h) = img.dimensions();
    let pad_x = (w as f64 * percent / 100.0).ceil() as u32;
    let pad_y = (h as f64 * percent / 100.0).ceil() as u32;
    if pad_x == 0 && pad_y == 0 {
        return img.clone();
    }

    let mut out = image::RgbaImage::from_pixel(
        w + pad_x.saturating_mul(2),
        h + pad_y.saturating_mul(2),
        Rgba([0, 0, 0, 0]),
    );
    image::imageops::overlay(&mut out, img, i64::from(pad_x), i64::from(pad_y));
    out
}

fn pad_rgba16(img: &Rgba16Image, percent: f64) -> Rgba16Image {
    if percent <= 0.0 {
        return img.clone();
    }

    let (w, h) = img.dimensions();
    let pad_x = (w as f64 * percent / 100.0).ceil() as u32;
    let pad_y = (h as f64 * percent / 100.0).ceil() as u32;
    if pad_x == 0 && pad_y == 0 {
        return img.clone();
    }

    let mut out = Rgba16Image::from_pixel(
        w + pad_x.saturating_mul(2),
        h + pad_y.saturating_mul(2),
        Rgba([0, 0, 0, 0]),
    );
    image::imageops::overlay(&mut out, img, i64::from(pad_x), i64::from(pad_y));
    out
}

/// Deskew an object by warping straight out of the full RGB source. The output
/// is cropped tight to the transformed component pixels. Area inside the crop
/// but outside the object is filled from the real surrounding pixels; the white
/// border is used only where the sample falls outside the source image.
fn deskew_from_src<P>(
    src_rgb: &ImageBuffer<P, Vec<P::Subpixel>>,
    bounds: RotatedBounds,
    theta: f32,
    border: P,
) -> ImageBuffer<P, Vec<P::Subpixel>>
where
    P: Pixel + Send + Sync + 'static,
    P::Subpixel: Send + Sync + Into<f32> + Clamp<f32> + 'static,
{
    let min_x = bounds.min_x;
    let min_y = bounds.min_y;
    let max_x = bounds.max_x;
    let max_y = bounds.max_y;

    let new_w = (max_x - min_x).ceil().max(1.0) as u32;
    let new_h = (max_y - min_y).ceil().max(1.0) as u32;

    // Forward map (source -> output): rotate, then shift the crop to the origin.
    let proj = Projection::translate(-min_x as f32, -min_y as f32) * Projection::rotate(theta);

    let mut out = ImageBuffer::from_pixel(new_w, new_h, border);
    warp_into(
        src_rgb,
        proj,
        Interpolation::Bilinear,
        Border::Constant(border),
        &mut out,
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tiff::{decoder::Decoder as TiffDecoder, tags::Tag};

    fn temp_image_path(ext: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("kropp-test-{}-{nanos}.{ext}", std::process::id()))
    }

    #[test]
    fn default_output_uses_input_stem_and_extension() {
        let plan = OutputPlan::new(
            Path::new("fixtures/photo.jpeg"),
            None,
            None,
            false,
            ImageFormat::Jpeg,
            false,
        )
        .unwrap();

        assert_eq!(plan.object_path(3), PathBuf::from("fixtures/photo_3.jpeg"));
        assert_eq!(plan.format, ImageFormat::Jpeg);
    }

    #[test]
    fn alpha_output_falls_back_to_png_when_format_lacks_alpha() {
        let plan = OutputPlan::new(
            Path::new("photo.jpg"),
            None,
            None,
            true,
            ImageFormat::Jpeg,
            false,
        )
        .unwrap();

        assert_eq!(plan.object_path(0), PathBuf::from("photo_0.png"));
        assert_eq!(plan.format, ImageFormat::Png);
    }

    #[test]
    fn alpha_output_keeps_alpha_capable_format() {
        let plan = OutputPlan::new(
            Path::new("photo.tiff"),
            None,
            None,
            true,
            ImageFormat::Tiff,
            false,
        )
        .unwrap();

        assert_eq!(plan.object_path(0), PathBuf::from("photo_0.tiff"));
        assert_eq!(plan.format, ImageFormat::Tiff);
    }

    #[test]
    fn dir_input_uses_output_dir_for_each_object_path() {
        let plan = OutputPlan::new(
            Path::new("inputs/photo.jpg"),
            None,
            Some("out"),
            false,
            ImageFormat::Jpeg,
            true,
        )
        .unwrap();

        assert_eq!(plan.object_path(2), PathBuf::from("out/photo_2.jpg"));
    }

    #[test]
    fn output_dir_prefix_check_matches_processed_files() {
        let dir = temp_image_path("dir");
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("photo_0.jpg");
        std::fs::write(&out, b"done").unwrap();

        let skipped = output_dir_has_prefix(&dir, Path::new("input/photo.jpg")).unwrap();
        let other = output_dir_has_prefix(&dir, Path::new("input/other.jpg")).unwrap();

        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_dir(&dir);

        assert!(skipped);
        assert!(!other);
    }

    #[test]
    fn rotated_bounds_follow_component_pixels_not_bbox() {
        let mut labels = vec![0usize; 100];
        labels[5 * 10 + 5] = 1;
        let obj = Object {
            label: 1,
            area: 1,
            min_x: 0,
            min_y: 0,
            max_x: 9,
            max_y: 9,
        };

        let bounds = rotated_component_bounds(&labels, 10, &obj, 45f32.to_radians()).unwrap();
        let width = bounds.max_x - bounds.min_x;
        let height = bounds.max_y - bounds.min_y;

        assert!(width < 1.5, "width was {width}");
        assert!(height < 1.5, "height was {height}");
    }

    #[test]
    fn rotated_crop_bounds_prefers_fitted_rectangle_over_component() {
        let mut labels = vec![0usize; 400];
        for y in 3..=12 {
            for x in 2..=11 {
                labels[y * 20 + x] = 1;
            }
        }
        let obj = Object {
            label: 1,
            area: 100,
            min_x: 2,
            min_y: 3,
            max_x: 11,
            max_y: 12,
        };
        let fit_corners = Some([(1.0, 1.0), (4.0, 1.0), (4.0, 5.0), (1.0, 5.0)]);

        let bounds = rotated_crop_bounds(fit_corners, &labels, 20, &obj, 0.0).unwrap();

        assert!((bounds.min_x - 3.0).abs() < 1e-6);
        assert!((bounds.max_x - 6.0).abs() < 1e-6);
        assert!((bounds.min_y - 4.0).abs() < 1e-6);
        assert!((bounds.max_y - 8.0).abs() < 1e-6);
    }

    #[test]
    fn rotated_crop_bounds_falls_back_to_component_without_fit_rectangle() {
        let mut labels = vec![0usize; 400];
        for y in 3..=12 {
            for x in 2..=11 {
                labels[y * 20 + x] = 1;
            }
        }
        let obj = Object {
            label: 1,
            area: 100,
            min_x: 2,
            min_y: 3,
            max_x: 11,
            max_y: 12,
        };

        let bounds = rotated_crop_bounds(None, &labels, 20, &obj, 0.0).unwrap();

        assert!((bounds.min_x - 2.0).abs() < 1e-6);
        assert!((bounds.max_x - 12.0).abs() < 1e-6);
        assert!((bounds.min_y - 3.0).abs() < 1e-6);
        assert!((bounds.max_y - 13.0).abs() < 1e-6);
    }

    #[test]
    fn min_side_filter_drops_specks_relative_to_smaller_image_dim() {
        let obj = |min_x, min_y, max_x, max_y| Object {
            label: 1,
            area: 1,
            min_x,
            min_y,
            max_x,
            max_y,
        };
        // Image 200x100 -> threshold is 10% of min(200,100) = 10 px.
        let (w, h) = (200, 100);

        // 9x9 box: longer side 9 < 10 -> dropped.
        assert!(!meets_min_side(&obj(0, 0, 8, 8), w, h, 10.0));
        // 3x40 sliver: longer side 40 >= 10 -> kept (one side suffices).
        assert!(meets_min_side(&obj(0, 0, 2, 39), w, h, 10.0));
        // Exactly at the threshold (10 px) is kept.
        assert!(meets_min_side(&obj(0, 0, 9, 0), w, h, 10.0));
        // Disabled: the tiny box survives.
        assert!(meets_min_side(&obj(0, 0, 1, 1), w, h, 0.0));
    }

    #[test]
    fn rotated_bounds_padding_expands_each_side_by_percent() {
        let mut bounds = RotatedBounds {
            min_x: 10.0,
            min_y: 20.0,
            max_x: 110.0,
            max_y: 60.0,
        };

        bounds.expand_percent(5.0);

        assert!((bounds.min_x - 5.0).abs() < 1e-6);
        assert!((bounds.max_x - 115.0).abs() < 1e-6);
        assert!((bounds.min_y - 18.0).abs() < 1e-6);
        assert!((bounds.max_y - 62.0).abs() < 1e-6);
    }

    #[test]
    fn trim_transparent_crops_to_visible_alpha() {
        let mut img = image::RgbaImage::new(6, 5);
        img.put_pixel(2, 1, Rgba([10, 20, 30, 255]));
        img.put_pixel(4, 3, Rgba([40, 50, 60, 128]));

        let trimmed = trim_transparent(&img).unwrap();
        assert_eq!(trimmed.dimensions(), (3, 3));
        assert_eq!(trimmed.get_pixel(0, 0), &Rgba([10, 20, 30, 255]));
        assert_eq!(trimmed.get_pixel(2, 2), &Rgba([40, 50, 60, 128]));
    }

    #[test]
    fn pad_rgba_adds_transparent_border() {
        let mut img = image::RgbaImage::new(20, 10);
        img.put_pixel(0, 0, Rgba([10, 20, 30, 255]));

        let padded = pad_rgba(&img, 5.0);

        assert_eq!(padded.dimensions(), (22, 12));
        assert_eq!(padded.get_pixel(0, 0), &Rgba([0, 0, 0, 0]));
        assert_eq!(padded.get_pixel(1, 1), &Rgba([10, 20, 30, 255]));
    }

    #[test]
    fn rgb_output_preserves_16_bit_channels() {
        let path = temp_image_path("png");
        let src =
            DynamicImage::ImageRgb16(ImageBuffer::from_pixel(1, 1, Rgb([0x1234, 0x8000, 0xffff])));
        let bounds = RotatedBounds {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 1.0,
            max_y: 1.0,
        };

        write_rgb_output(&src, bounds, 0.0, 0, &path, ImageFormat::Png).unwrap();
        let decoded = image::open(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(decoded.color(), image::ColorType::Rgb16);
        assert_eq!(
            decoded.to_rgb16().get_pixel(0, 0),
            &Rgb([0x1234, 0x8000, 0xffff])
        );
    }

    #[test]
    fn alpha_output_preserves_16_bit_channels_and_writes_16_bit_alpha() {
        let path = temp_image_path("png");
        let src =
            DynamicImage::ImageRgb16(ImageBuffer::from_pixel(1, 1, Rgb([0x1111, 0x8888, 0xeeee])));
        let labels = vec![1usize];
        let obj = Object {
            label: 1,
            area: 1,
            min_x: 0,
            min_y: 0,
            max_x: 0,
            max_y: 0,
        };

        write_alpha_output(
            &src,
            AlphaOutputRequest {
                labels: &labels,
                w: 1,
                obj: &obj,
                theta: 0.0,
                quarter_turns: 0,
                crop_padding: 0.0,
                path: &path,
                format: ImageFormat::Png,
            },
        )
        .unwrap();
        let decoded = image::open(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(decoded.color(), image::ColorType::Rgba16);
        assert_eq!(
            decoded.to_rgba16().get_pixel(0, 0),
            &Rgba([0x1111, 0x8888, 0xeeee, u16::MAX])
        );
    }

    #[test]
    fn tiff_output_uses_deflate_compression_by_default() {
        let path = temp_image_path("tiff");
        let src =
            DynamicImage::ImageRgb16(ImageBuffer::from_pixel(1, 1, Rgb([0x2222, 0x4444, 0x6666])));
        let labels = vec![1usize];
        let obj = Object {
            label: 1,
            area: 1,
            min_x: 0,
            min_y: 0,
            max_x: 0,
            max_y: 0,
        };

        write_alpha_output(
            &src,
            AlphaOutputRequest {
                labels: &labels,
                w: 1,
                obj: &obj,
                theta: 0.0,
                quarter_turns: 0,
                crop_padding: 0.0,
                path: &path,
                format: ImageFormat::Tiff,
            },
        )
        .unwrap();

        let file = File::open(&path).unwrap();
        let mut decoder = TiffDecoder::new(file).unwrap();
        assert_eq!(decoder.get_tag_u32(Tag::Compression).unwrap(), 8);
        assert_eq!(decoder.dimensions().unwrap(), (1, 1));
        let decoded = decoder.read_image().unwrap();
        let _ = std::fs::remove_file(&path);

        assert!(
            matches!(decoded, tiff::decoder::DecodingResult::U16(v) if v == vec![0x2222, 0x4444, 0x6666, u16::MAX])
        );
    }
}
