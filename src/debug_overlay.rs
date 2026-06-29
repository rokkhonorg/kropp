//! Diagnostic rendering and logging for `--debug`: angle breakdowns, crop
//! geometry, and the annotated per-object overlay image.

use std::path::{Path, PathBuf};

use image::{Rgb, RgbImage};
use imageproc::drawing::draw_line_segment_mut;

use crate::angle;
use crate::components::Object;
use crate::output::{rotated_crop_bounds, suffixed_path};

/// Which candidate angle was chosen for an object.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AngleSource {
    Text,
    Mask,
    None,
}

/// Emit a detailed breakdown of how an object's angle was decided: the mask
/// geometry candidate, the outlier-rejected text candidate, the winner, and
/// which text boxes were kept (inliers) vs. discarded (outliers).
pub fn log_angle_debug(
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
pub fn draw_debug_overlay(
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
pub fn debug_image_path(path: &Path) -> PathBuf {
    suffixed_path(path, ".debug")
}

/// Log the fitted rectangle (the blue overlay box, in cutout-local coordinates)
/// and the actual RGB crop window in the deskewed frame.
#[allow(clippy::too_many_arguments)]
pub fn log_crop_geometry(
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
