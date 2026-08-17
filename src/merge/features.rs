//! The SIFT adapter: the only place that talks to `sift-features`.
//!
//! Everything downstream works on `Point` and `[u8; 128]`, so the ndarray types
//! `sift-features` returns never leave this file. That is also why kropp has no
//! direct ndarray dependency — the descriptor array is read through inherent
//! methods and converted immediately.

use image::imageops::FilterType;
use image::{DynamicImage, GrayImage};

use super::geometry::Point;
use super::matching::Descriptor;

/// Descriptors are 128-dimensional, per Lowe and OpenCV alike.
const DESCRIPTOR_LEN: usize = 128;

/// Features per megapixel.
///
/// The previous implementation ran `nfeatures=500`, but only after OpenCV had
/// downscaled each pass to 0.6 megapixels — so 500 features covered the whole
/// image densely. Applying that same *count* at full scan resolution is around
/// sixty times sparser, and sparse features clump onto whatever has the
/// strongest local contrast, leaving the fit pinned by a few tight clusters and
/// drifting away from them. What has to be carried over is the density, not the
/// number.
const FEATURES_PER_MEGAPIXEL: usize = 833;

/// Floor and ceiling on the budget. The ceiling exists because matching is
/// brute force and quadratic: doubling the features quadruples the work.
const MIN_FEATURE_LIMIT: usize = 1_500;
const MAX_FEATURE_LIMIT: usize = 4_000;

/// Kept for tests and callers that want the old fixed budget.
pub const DEFAULT_FEATURE_LIMIT: usize = 500;

/// Feature budget for an image of this size, at the density the previous
/// implementation effectively used.
pub fn feature_limit(width: u32, height: u32) -> usize {
    let megapixels = (width as f64 * height as f64) / 1_000_000.0;
    let scaled = (megapixels * FEATURES_PER_MEGAPIXEL as f64) as usize;
    scaled.clamp(MIN_FEATURE_LIMIT, MAX_FEATURE_LIMIT)
}

/// Keypoint positions and their descriptors, in matching order.
#[derive(Debug, Clone, Default)]
pub struct Features {
    pub points: Vec<Point>,
    pub descriptors: Vec<Descriptor>,
}

impl Features {
    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// Flatten a pass to the 8-bit grayscale SIFT wants.
///
/// This 8-bit reduction is confined to *feature detection*. The compositor
/// warps and blends from the original samples, so a 16-bit scan stays 16-bit
/// through the merge — unlike the OpenCV path, which loaded 8-bit and
/// composited through signed 16-bit.
pub fn detection_luma(image: &DynamicImage) -> GrayImage {
    image.to_luma8()
}

/// A pass reduced for detection, and what it takes to read its coordinates as
/// full-resolution ones again.
pub struct Detection {
    pub luma: GrayImage,
    /// Multiply a detection-space coordinate by these to get a full-resolution
    /// one. Held per axis, and derived from the rounded output dimensions, so
    /// the mapping back is exact rather than off by the rounding.
    pub scale: (f64, f64),
}

impl Detection {
    /// Whether detection ran on a reduced copy at all.
    pub fn reduced(&self) -> bool {
        self.scale.0 != 1.0 || self.scale.1 != 1.0
    }

    /// The larger of the two axis factors — how much a keypoint's localization
    /// error is magnified when read at full resolution.
    pub fn error_factor(&self) -> f64 {
        self.scale.0.max(self.scale.1)
    }
}

/// Flatten a pass and shrink it to about `target_megapixels` for detection.
///
/// Detection dominates a merge, and its cost is proportional to area, so this
/// is the one lever that really moves it. What makes it affordable is that the
/// alignment is four numbers fitted to hundreds of correspondences: individual
/// keypoints get coarser, but averaging over that many recovers a transform far
/// more accurate than any one of them. The previous OpenCV implementation
/// registered at 0.6 MP and its transforms matched full-resolution ones to
/// sub-pixel translations.
///
/// A target of zero (or one no smaller than the pass) detects at full size.
pub fn detection_target(image: &DynamicImage, target_megapixels: f64) -> Detection {
    let luma = image.to_luma8();
    let (width, height) = (luma.width(), luma.height());
    let pixels = width as f64 * height as f64;
    let target = target_megapixels * 1_000_000.0;

    if target <= 0.0 || pixels <= target || width == 0 || height == 0 {
        return Detection {
            luma,
            scale: (1.0, 1.0),
        };
    }

    let factor = (pixels / target).sqrt();
    let small_width = ((width as f64 / factor).round() as u32).max(1);
    let small_height = ((height as f64 / factor).round() as u32).max(1);
    // Triangle, not Nearest: dropping samples outright would alias scan grain
    // into exactly the corner-like structure SIFT keys on, and those keypoints
    // would be artifacts of the resampling rather than of the document.
    let small = image::imageops::resize(&luma, small_width, small_height, FilterType::Triangle);

    Detection {
        scale: (
            width as f64 / small_width as f64,
            height as f64 / small_height as f64,
        ),
        luma: small,
    }
}

/// Detect SIFT features over the whole image.
pub fn detect(image: &GrayImage, limit: Option<usize>) -> Features {
    // SIFT builds a scale-space pyramid; a degenerate image has nothing to
    // build it from and is not worth handing to the detector.
    if image.width() < 8 || image.height() < 8 {
        return Features::default();
    }

    let result = sift_features::sift(image, limit);
    let keypoint_count = result.keypoints.len();

    // Read the (n, 128) descriptor array in logical order and repack. Using
    // the inherent iterator keeps this independent of the array's memory
    // layout, and of ndarray's version.
    let flat: Vec<u8> = result.descriptors.iter().copied().collect();
    let usable = keypoint_count.min(flat.len() / DESCRIPTOR_LEN);

    let mut points = Vec::with_capacity(usable);
    let mut descriptors = Vec::with_capacity(usable);
    for (idx, keypoint) in result.keypoints.iter().take(usable).enumerate() {
        let start = idx * DESCRIPTOR_LEN;
        let mut descriptor = [0u8; DESCRIPTOR_LEN];
        descriptor.copy_from_slice(&flat[start..start + DESCRIPTOR_LEN]);
        points.push((keypoint.x as f64, keypoint.y as f64));
        descriptors.push(descriptor);
    }

    Features {
        points,
        descriptors,
    }
}

/// Read detection-space keypoints as full-resolution ones, in place.
pub fn rescale(features: &mut Features, scale: (f64, f64)) {
    if scale == (1.0, 1.0) {
        return;
    }
    for point in &mut features.points {
        point.0 *= scale.0;
        point.1 *= scale.1;
    }
}

/// Keep only the features landing on a set pixel of `mask` (row-major,
/// `width` x `height`, in the same coordinates as `features`).
///
/// This is what keeps platen artifacts out of the alignment. Bed edges, dust,
/// and lid shadow sit at *fixed* positions in scanner coordinates, so they are
/// identical in every pass and would match each other perfectly at zero
/// displacement — a strong, entirely wrong correspondence. Restricting
/// keypoints to the extracted object removes them before they can vote.
///
/// A mask whose length disagrees with the image is ignored rather than trusted.
pub fn apply_mask(features: Features, mask: &[bool], width: u32, height: u32) -> Features {
    let (w, h) = (width as usize, height as usize);
    if mask.len() != w * h {
        return features;
    }

    let mut points = Vec::with_capacity(features.points.len());
    let mut descriptors = Vec::with_capacity(features.descriptors.len());
    for (point, descriptor) in features.points.iter().zip(&features.descriptors) {
        let (x, y) = (point.0.round(), point.1.round());
        if x < 0.0 || y < 0.0 {
            continue;
        }
        let (x, y) = (x as usize, y as usize);
        if x < w && y < h && mask[y * w + x] {
            points.push(*point);
            descriptors.push(*descriptor);
        }
    }

    Features {
        points,
        descriptors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merge::geometry::ScaleModel;
    use crate::merge::matching::{self, DEFAULT_MATCH_CONF, RansacParams};
    use crate::merge::test_support::{crop, master_texture};
    use image::Luma;

    #[test]
    fn detects_features_with_descriptors_in_matching_order() {
        let img = master_texture(200, 200);
        let features = detect(&img, Some(DEFAULT_FEATURE_LIMIT));

        assert!(
            features.len() > 20,
            "expected a textured image to yield features, got {}",
            features.len()
        );
        assert_eq!(features.points.len(), features.descriptors.len());
        assert!(features.len() <= DEFAULT_FEATURE_LIMIT);
    }

    /// Zero, and any target the pass already meets, must leave it alone —
    /// that is how `--merge-detect-mp 0` turns the reduction off.
    #[test]
    fn a_target_of_zero_or_larger_than_the_pass_detects_at_full_size() {
        let pass = DynamicImage::ImageLuma8(master_texture(400, 300));
        for target in [0.0, -1.0, 5.0] {
            let detection = detection_target(&pass, target);
            assert!(!detection.reduced(), "target {target} should not reduce");
            assert_eq!(detection.luma.dimensions(), (400, 300));
            assert_eq!(detection.scale, (1.0, 1.0));
        }
    }

    /// The reduced copy lands near the requested size, and its scale factors
    /// map coordinates back exactly — they come from the rounded dimensions,
    /// not from the ratio that produced them.
    #[test]
    fn reduction_hits_the_target_and_maps_coordinates_back_exactly() {
        let pass = DynamicImage::ImageLuma8(master_texture(4000, 3000));
        let detection = detection_target(&pass, 1.0);

        let (w, h) = detection.luma.dimensions();
        let megapixels = w as f64 * h as f64 / 1e6;
        assert!(
            (0.9..1.1).contains(&megapixels),
            "expected about 1 MP, got {megapixels:.3} ({w}x{h})"
        );
        assert!(detection.reduced());

        // A corner of the reduced copy must map to the corner of the original.
        assert!((w as f64 * detection.scale.0 - 4000.0).abs() < 1e-9);
        assert!((h as f64 * detection.scale.1 - 3000.0).abs() < 1e-9);
        assert!(detection.error_factor() > 1.0);
    }

    /// Keypoints found on the reduced copy have to come back as
    /// full-resolution coordinates, or every stage downstream is wrong by the
    /// reduction factor.
    #[test]
    fn rescaling_moves_keypoints_into_full_resolution_coordinates() {
        let mut features = Features {
            points: vec![(10.0, 20.0), (0.0, 0.0)],
            descriptors: vec![[0u8; DESCRIPTOR_LEN]; 2],
        };
        rescale(&mut features, (4.0, 3.0));
        assert_eq!(features.points, vec![(40.0, 60.0), (0.0, 0.0)]);

        // An identity scale must not perturb anything.
        let mut untouched = features.clone();
        rescale(&mut untouched, (1.0, 1.0));
        assert_eq!(untouched.points, features.points);
    }

    #[test]
    fn tiny_images_are_rejected_rather_than_detected_on() {
        assert!(detect(&GrayImage::new(4, 4), None).is_empty());
        assert!(detect(&GrayImage::new(0, 0), None).is_empty());
    }

    /// The end-to-end reason all three modules exist: two overlapping windows
    /// onto one master must recover the offset between them.
    #[test]
    fn recovers_known_offset_between_overlapping_crops() {
        let master = master_texture(420, 340);
        // `b` is the same scene sampled 60px right and 35px down of `a`, so a
        // point at b-local (x, y) is a-local (x + 60, y + 35).
        let a = crop(&master, 0, 0, 300, 260);
        let b = crop(&master, 60, 35, 300, 260);

        let fa = detect(&a, Some(DEFAULT_FEATURE_LIMIT));
        let fb = detect(&b, Some(DEFAULT_FEATURE_LIMIT));
        let matches = matching::match_descriptors(&fb.descriptors, &fa.descriptors, DEFAULT_MATCH_CONF);
        assert!(
            matches.len() >= 4,
            "too few matches to fit a transform: {}",
            matches.len()
        );

        let pairs = matching::correspondences(&matches, &fb.points, &fa.points);
        let fit = matching::robust_estimate(&pairs, RansacParams::default())
            .expect("overlapping crops should yield a transform");

        assert!(
            (fit.transform.tx - 60.0).abs() < 1.5,
            "tx was {}",
            fit.transform.tx
        );
        assert!(
            (fit.transform.ty - 35.0).abs() < 1.5,
            "ty was {}",
            fit.transform.ty
        );
        assert!(
            fit.transform.theta.abs() < 0.01,
            "theta was {}",
            fit.transform.theta
        );
        assert!(
            fit.confidence > 0.5,
            "confidence {} below the subsetter threshold",
            fit.confidence
        );
    }

    #[test]
    fn mask_restricts_keypoints_to_set_pixels() {
        let img = master_texture(200, 200);
        let (w, h) = (200usize, 200usize);

        // Admit only the left half.
        let mut mask = vec![false; w * h];
        for y in 0..h {
            for x in 0..w / 2 {
                mask[y * w + x] = true;
            }
        }

        let all = detect(&img, None);
        let masked = apply_mask(detect(&img, None), &mask, img.width(), img.height());

        assert!(masked.len() < all.len(), "mask did not remove anything");
        assert!(!masked.is_empty(), "mask removed everything");
        assert!(
            masked.points.iter().all(|&(x, _)| x < 100.0),
            "a keypoint survived outside the mask"
        );
        assert_eq!(masked.points.len(), masked.descriptors.len());
    }

    #[test]
    fn wrongly_sized_mask_is_ignored() {
        let img = master_texture(120, 120);
        let all = detect(&img, None);
        let masked = apply_mask(detect(&img, None), &[true; 10], img.width(), img.height());

        assert_eq!(masked.len(), all.len());
    }

    /// A rotated pass must still align, since the sheet can be laid down at a
    /// slight angle between passes.
    #[test]
    fn recovers_offset_with_small_rotation() {
        let master = master_texture(420, 340);
        let a = crop(&master, 0, 0, 300, 260);
        let rotated = imageproc::geometric_transformations::rotate_about_center(
            &crop(&master, 40, 30, 300, 260),
            0.05,
            imageproc::geometric_transformations::Interpolation::Bilinear,
            imageproc::geometric_transformations::Border::Constant(Luma([235])),
        );

        let fa = detect(&a, Some(DEFAULT_FEATURE_LIMIT));
        let fb = detect(&rotated, Some(DEFAULT_FEATURE_LIMIT));
        let matches =
            matching::match_descriptors(&fb.descriptors, &fa.descriptors, DEFAULT_MATCH_CONF);
        let pairs = matching::correspondences(&matches, &fb.points, &fa.points);

        let fit = matching::robust_estimate(
            &pairs,
            RansacParams {
                scale_model: ScaleModel::Fixed,
                ..RansacParams::default()
            },
        )
        .expect("rotated overlap should still fit");

        // The crop was rotated by +0.05 rad about its centre, so recovering it
        // means finding -0.05 in the mapping back onto `a`.
        assert!(
            (fit.transform.theta + 0.05).abs() < 0.02,
            "theta was {}",
            fit.transform.theta
        );

        // And the fit should actually explain the correspondences it kept.
        let inliers: Vec<_> = fit.inliers.iter().map(|&i| pairs[i]).collect();
        assert!(crate::merge::geometry::mean_residual(fit.transform, &inliers) < 3.0);
    }
}
