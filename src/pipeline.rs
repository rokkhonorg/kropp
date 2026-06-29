//! The processing pipeline: model + lazily-loaded orientation classifiers, and
//! the shared per-object loop that turns one source image into cropped,
//! deskewed, orientation-corrected outputs.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use image::{DynamicImage, GenericImageView, ImageFormat, Rgb};
use imageproc::drawing::draw_line_segment_mut;
use imageproc::geometric_transformations::{
    Border, Interpolation, rotate_about_center_no_crop,
};
use usls::{Image, models::RMBG};

use crate::angle::{self, Detector, DocOrientClassifier, OrientClassifier};
use crate::cli::Args;
use crate::components::{Object, connected_components, crop_object, meets_min_side};
use crate::debug_overlay::{
    AngleSource, debug_image_path, draw_debug_overlay, log_angle_debug, log_crop_geometry,
};
use crate::model::build_rmbg;
use crate::output::{
    AlphaOutputRequest, OutputPlan, rotated_crop_bounds, suffixed_path, write_alpha_output,
    write_rgb_output,
};

/// Holds the RMBG model and the orientation classifiers. The text detector and
/// classifiers load lazily and remember when they are unavailable (e.g. offline)
/// so a missing model is reported once and then skipped. A single `Pipeline` is
/// reused across every input file in directory mode.
pub struct Pipeline {
    model: RMBG,
    detector: Option<Detector>,
    detector_unavailable: bool,
    textline: Option<OrientClassifier>,
    textline_unavailable: bool,
    doc: Option<DocOrientClassifier>,
    doc_unavailable: bool,
}

impl Pipeline {
    /// Build the RMBG model and, when `--text` forces text rotation, eagerly load
    /// the text detector so its absence is reported up front.
    pub fn new(args: &Args) -> Result<Self> {
        let model = build_rmbg()?;
        let detector = if args.text {
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
        let detector_unavailable = args.text && detector.is_none();
        Ok(Self {
            model,
            detector,
            detector_unavailable,
            textline: None,
            textline_unavailable: false,
            doc: None,
            doc_unavailable: false,
        })
    }

    /// Run inference on one source image, split it into objects, and write a
    /// crop per object. `model_input` is the path passed to the model reader;
    /// `summary_path`, when set, names the input in the final "found N" line
    /// (directory mode); otherwise the single-file form is printed.
    pub fn process_image(
        &mut self,
        src_original: &DynamicImage,
        model_input: &str,
        args: &Args,
        cutoff: u8,
        output_plan: &OutputPlan,
        summary_path: Option<&Path>,
    ) -> Result<()> {
        let model_image = Image::try_read(model_input)
            .with_context(|| format!("failed to read image: {model_input}"))?;
        let ys = self.model.forward(std::slice::from_ref(&model_image))?;

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
        let (labels, mut objects) =
            connected_components(&fg, w as usize, h as usize, args.min_area);
        // Drop noise: components no side of which is large relative to the image.
        objects.retain(|obj| meets_min_side(obj, w as usize, h as usize, args.min_side_percent));
        if objects.is_empty() {
            bail!("no objects found above the threshold");
        }

        for (idx, obj) in objects.iter().enumerate() {
            self.process_object(idx, obj, &src, &labels, w, src_original, args, output_plan)?;
        }

        match summary_path {
            Some(path) => println!("found {} object(s) in {}", objects.len(), path.display()),
            None => println!("found {} object(s)", objects.len()),
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn process_object(
        &mut self,
        idx: usize,
        obj: &Object,
        src: &image::RgbaImage,
        labels: &[usize],
        w: u32,
        src_original: &DynamicImage,
        args: &Args,
        output_plan: &OutputPlan,
    ) -> Result<()> {
        // The masked cutout isolates the object — used for text detection (and
        // for the `--alpha` output), so background text/objects don't pollute
        // the angle signal.
        let cutout = crop_object(src, w as usize, labels, obj);

        if args.debug {
            elog!(
                "[obj {idx}] bbox {}x{} area {} px",
                obj.max_x - obj.min_x + 1,
                obj.max_y - obj.min_y + 1,
                obj.area
            );
        }

        let mask = angle::object_mask(
            labels, w as usize, obj.label, obj.min_x, obj.min_y, obj.max_x, obj.max_y,
        );
        // Rectangularity is measured against the safezone rectangle. A
        // rectangular mask gets the tight Min/body fit; an irregular mask keeps
        // the Max/safezone fit so no Min/body logic is applied to it.
        let rectangularity = angle::mask_rectangularity(&mask, angle::MaskFit::Max);
        let is_non_rectangular =
            rectangularity.is_some_and(|r| r < args.rectangularity_threshold());
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
        let text_boxes = if use_text && !self.detector_unavailable {
            if self.detector.is_none() {
                match Detector::load() {
                    Ok(d) => self.detector = Some(d),
                    Err(e) => {
                        elog!(
                            "warning: text detector unavailable ({e:#}); using mask orientation only"
                        );
                        self.detector_unavailable = true;
                    }
                }
            }

            match self.detector.as_mut() {
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
        let doc_model = args.doc_orient_model_path();
        let quarter_turns = self.decide_quarter_turns(
            use_text,
            args.no_deskew,
            args.no_doc_orient,
            doc_model.as_deref(),
            &cutout,
            &text_boxes,
            theta,
            args.debug,
        )?;

        if args.debug {
            self.log_object_debug(
                obj,
                idx,
                labels,
                w,
                &cutout,
                &mask,
                &text_boxes,
                &text_samples,
                rectangularity,
                auto_text,
                hole_count,
                crop_padding,
                crop_fit,
                fit_corners,
                crop_corners,
                mask_cand,
                text_est,
                final_angle,
                source,
                theta,
                args,
                output_plan,
            )?;
        }

        // Write the cutout (RGBA, transparent background) or, by default, a
        // plain rectangular RGB crop that keeps the original background.
        let path = output_plan.object_path(idx);
        let (out_w, out_h) = if args.alpha {
            write_alpha_output(
                src_original,
                AlphaOutputRequest {
                    labels,
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
            let mut bounds = rotated_crop_bounds(crop_corners, labels, w as usize, obj, theta)
                .context("object has no foreground pixels while cropping")?;
            bounds.expand_percent(crop_padding);
            // Deskew by sampling the full source, so rotation reveals the real
            // pixels around the object; padding is used only out of bounds.
            write_rgb_output(
                src_original,
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
        Ok(())
    }

    /// Decide how many 90-degree clockwise turns to apply to a deskewed object so
    /// it ends up upright. Text-driven crops (`--text` or a non-rectangular mask)
    /// use the textline 0/180 vote over the detected boxes; rectangular crops use
    /// the whole-image doc-orientation model (0/90/180/270). Both classifiers load
    /// lazily and fall back to no rotation when unavailable (e.g. offline).
    #[allow(clippy::too_many_arguments)]
    fn decide_quarter_turns(
        &mut self,
        use_text: bool,
        no_deskew: bool,
        no_doc_orient: bool,
        doc_model_path: Option<&Path>,
        cutout: &image::RgbaImage,
        text_boxes: &[angle::TextBox],
        theta: f32,
        debug: bool,
    ) -> Result<u8> {
        if no_deskew {
            return Ok(0);
        }

        if use_text {
            // Non-rectangular / forced: vote 0 vs 180 over the text lines.
            if text_boxes.is_empty() || self.textline_unavailable {
                return Ok(0);
            }
            if self.textline.is_none() {
                match OrientClassifier::load() {
                    Ok(c) => self.textline = Some(c),
                    Err(e) => {
                        elog!(
                            "warning: textline orientation classifier unavailable ({e:#}); skipping 180° correction"
                        );
                        self.textline_unavailable = true;
                    }
                }
            }
            return Ok(match self.textline.as_mut() {
                Some(c) => {
                    let flip = c.vote_flip(cutout, text_boxes, theta, debug)?.unwrap_or(false);
                    if flip { 2 } else { 0 }
                }
                None => 0,
            });
        }

        // Rectangular: classify the whole deskewed object (0/90/180/270).
        if no_doc_orient || self.doc_unavailable {
            return Ok(0);
        }
        if self.doc.is_none() {
            match DocOrientClassifier::load(doc_model_path) {
                Ok(c) => self.doc = Some(c),
                Err(e) => {
                    elog!(
                        "warning: doc orientation classifier unavailable ({e:#}); skipping orientation correction"
                    );
                    self.doc_unavailable = true;
                }
            }
        }
        Ok(match self.doc.as_mut() {
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

    /// Emit the full `--debug` breakdown for one object and write its overlay and
    /// deskewed-overlay images next to the object's output path.
    #[allow(clippy::too_many_arguments)]
    fn log_object_debug(
        &self,
        obj: &Object,
        idx: usize,
        labels: &[usize],
        w: u32,
        cutout: &image::RgbaImage,
        mask: &image::GrayImage,
        text_boxes: &[angle::TextBox],
        text_samples: &[(f64, f64)],
        rectangularity: Option<f64>,
        auto_text: bool,
        hole_count: usize,
        crop_padding: f64,
        crop_fit: angle::MaskFit,
        fit_corners: Option<[(f64, f64); 4]>,
        crop_corners: Option<[(f64, f64); 4]>,
        mask_cand: Option<f64>,
        text_est: Option<angle::AngleEstimate>,
        final_angle: Option<f64>,
        source: AngleSource,
        theta: f32,
        args: &Args,
        output_plan: &OutputPlan,
    ) -> Result<()> {
        match rectangularity {
            Some(r) => {
                let mode = if args.text {
                    "forced by --text"
                } else if args.force_rectangular {
                    "forced rectangular"
                } else if auto_text {
                    "auto text enabled"
                } else {
                    "mask only"
                };
                elog!(
                    "    rectangularity: {r:.3} (threshold {:.3}; {mode}; padding {crop_padding:.1}%)",
                    args.rectangularity_threshold(),
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
        log_angle_debug(text_samples, text_est, mask_cand, source, args.angle_tol);
        log_crop_geometry(
            crop_fit,
            fit_corners,
            crop_corners,
            labels,
            w as usize,
            obj,
            theta,
            crop_padding,
        );
        // Visual overlay: draw the detector's boxes and the final angle so we
        // can see whether the geometry, not the aggregation, is at fault.
        let overlay = draw_debug_overlay(
            cutout,
            mask,
            text_boxes,
            fit_corners,
            (crop_fit != angle::MaskFit::Max)
                .then(|| angle::mask_rect_corners(mask, angle::MaskFit::Max))
                .flatten(),
            final_angle.map(|a| a as f32),
        );
        let object_path = output_plan.object_path(idx);
        let dpath = debug_image_path(&object_path);
        overlay
            .save(&dpath)
            .with_context(|| format!("failed to write debug image: {}", dpath.display()))?;
        elog!("    debug image: {}", dpath.display());

        // Deskewed overlay: the same scene rotated by the chosen theta, with a
        // gray horizontal reference. If the estimate is right, the magenta line
        // should land flat on the reference — this proves the rotation follows
        // the estimate (or shows it doesn't).
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
        Ok(())
    }
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

/// Resolve an input path to the list of image files to process. A file yields
/// itself; a directory yields its image-format entries, sorted.
pub fn collect_input_paths(input: &Path) -> Result<Vec<PathBuf>> {
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

/// Whether `output_dir` already holds an output for `input_file` (a file whose
/// name starts with `<stem>_`), used to skip already-processed inputs.
pub fn output_dir_has_prefix(output_dir: &Path, input_file: &Path) -> Result<bool> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir_path() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("kropp-test-{}-{nanos}.dir", std::process::id()))
    }

    #[test]
    fn output_dir_prefix_check_matches_processed_files() {
        let dir = temp_dir_path();
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
}
