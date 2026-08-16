//! Multi-band blending over Laplacian pyramids.
//!
//! Feather blending applies one transition width to every spatial frequency at
//! once, so fine detail from two passes offset by even a pixel gets averaged —
//! which is exactly what blur is. Multi-band splits each pass into frequency
//! bands and blends each with a transition proportional to that band's scale:
//! fine detail crosses over in a few pixels (so it comes from one pass, and
//! stays sharp), while broad tone crosses over across the whole overlap (so
//! there is no visible step in brightness).
//!
//! This is why the previous OpenCV implementation looked clean despite
//! estimating its alignment on 0.6-megapixel thumbnails: `MultiBandBlender` was
//! hiding a residual misregistration considerably larger than ours.
//!
//! Everything here is plain `f32` planes so it can be tested without images.

/// The 5-tap binomial kernel standard for image pyramids, normalized.
const KERNEL: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

/// Clamp-to-edge index, so the border does not darken toward zero.
fn clamp_index(value: isize, limit: usize) -> usize {
    value.clamp(0, limit as isize - 1) as usize
}

/// Separable 5-tap blur.
fn blur(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut horizontal = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut total = 0.0;
            for (tap, weight) in KERNEL.iter().enumerate() {
                let sx = clamp_index(x as isize + tap as isize - 2, w);
                total += src[y * w + sx] * weight;
            }
            horizontal[y * w + x] = total;
        }
    }

    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut total = 0.0;
            for (tap, weight) in KERNEL.iter().enumerate() {
                let sy = clamp_index(y as isize + tap as isize - 2, h);
                total += horizontal[sy * w + x] * weight;
            }
            out[y * w + x] = total;
        }
    }
    out
}

/// Half-size dimensions for the next pyramid level.
pub fn half(w: usize, h: usize) -> (usize, usize) {
    (w.div_ceil(2).max(1), h.div_ceil(2).max(1))
}

/// Blur then drop every other sample.
pub fn pyr_down(src: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let smoothed = blur(src, w, h);
    let (dw, dh) = half(w, h);
    let mut out = vec![0.0f32; dw * dh];
    for y in 0..dh {
        for x in 0..dw {
            out[y * dw + x] = smoothed[(y * 2).min(h - 1) * w + (x * 2).min(w - 1)];
        }
    }
    (out, dw, dh)
}

/// Expand back to `dw` x `dh`: scatter the samples, then blur. The factor of 4
/// compensates for the three zeros introduced per sample.
pub fn pyr_up(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut scattered = vec![0.0f32; dw * dh];
    for y in 0..sh {
        for x in 0..sw {
            let (dx, dy) = (x * 2, y * 2);
            if dx < dw && dy < dh {
                scattered[dy * dw + dx] = src[y * sw + x] * 4.0;
            }
        }
    }
    blur(&scattered, dw, dh)
}

/// Level dimensions from the base size down, `levels` entries long.
pub fn level_sizes(width: usize, height: usize, levels: usize) -> Vec<(usize, usize)> {
    let mut sizes = Vec::with_capacity(levels);
    let (mut w, mut h) = (width, height);
    for _ in 0..levels {
        sizes.push((w, h));
        let (nw, nh) = half(w, h);
        w = nw;
        h = nh;
    }
    sizes
}

/// How many bands to use, following OpenCV's rule: a blend width proportional
/// to the canvas diagonal, then enough levels to cover it. Capped so no level
/// collapses below a few pixels.
pub fn level_count(width: usize, height: usize, strength: f64) -> usize {
    let blend_width = ((width * height) as f64).sqrt() * strength / 100.0;
    if blend_width < 1.0 {
        return 1;
    }
    let from_width = (blend_width.log2() - 1.0).floor().max(1.0) as usize;
    // Stop before a level would be smaller than the blur kernel.
    let from_size = {
        let (mut w, mut h, mut levels) = (width, height, 1usize);
        while w > 8 && h > 8 {
            let (nw, nh) = half(w, h);
            w = nw;
            h = nh;
            levels += 1;
        }
        levels
    };
    from_width.min(from_size).max(1)
}

/// Gaussian pyramid of `src`, `levels` deep.
fn gaussian_pyramid(src: &[f32], width: usize, height: usize, levels: usize) -> Vec<Vec<f32>> {
    let mut pyramid = Vec::with_capacity(levels);
    let (mut current, mut w, mut h) = (src.to_vec(), width, height);
    for level in 0..levels {
        if level + 1 == levels {
            pyramid.push(current);
            break;
        }
        let (next, nw, nh) = pyr_down(&current, w, h);
        pyramid.push(current);
        current = next;
        w = nw;
        h = nh;
    }
    pyramid
}

/// Laplacian pyramid: each level is the detail lost when the next level down is
/// expanded back. The coarsest level is the residual, kept as is.
fn laplacian_pyramid(gaussian: &[Vec<f32>], sizes: &[(usize, usize)]) -> Vec<Vec<f32>> {
    let mut pyramid = Vec::with_capacity(gaussian.len());
    for level in 0..gaussian.len() {
        if level + 1 == gaussian.len() {
            pyramid.push(gaussian[level].clone());
            break;
        }
        let (w, h) = sizes[level];
        let (nw, nh) = sizes[level + 1];
        let expanded = pyr_up(&gaussian[level + 1], nw, nh, w, h);
        let detail: Vec<f32> = gaussian[level]
            .iter()
            .zip(&expanded)
            .map(|(a, b)| a - b)
            .collect();
        pyramid.push(detail);
    }
    pyramid
}

/// Multi-band blend of one channel.
///
/// `planes` holds each pass's canvas-sized channel; `masks` holds each pass's
/// selection weight, which must sum to 1 at every covered pixel (a binary
/// argmax mask satisfies this). Because blurring is linear, masks that sum to 1
/// keep summing to 1 at every level, so no per-level renormalization is needed.
///
/// Returns the blended channel at canvas size.
pub fn blend_channel(
    planes: &[Vec<f32>],
    masks: &[Vec<f32>],
    width: usize,
    height: usize,
    levels: usize,
) -> Vec<f32> {
    let sizes = level_sizes(width, height, levels);
    let mut accumulated: Vec<Vec<f32>> = sizes.iter().map(|&(w, h)| vec![0.0f32; w * h]).collect();

    for (plane, mask) in planes.iter().zip(masks) {
        let plane_gaussian = gaussian_pyramid(plane, width, height, levels);
        let plane_laplacian = laplacian_pyramid(&plane_gaussian, &sizes);
        drop(plane_gaussian);
        let mask_gaussian = gaussian_pyramid(mask, width, height, levels);

        for level in 0..levels {
            let target = &mut accumulated[level];
            for (slot, (detail, weight)) in target
                .iter_mut()
                .zip(plane_laplacian[level].iter().zip(&mask_gaussian[level]))
            {
                *slot += detail * weight;
            }
        }
    }

    // Collapse from the coarsest level upward.
    let mut result = accumulated.pop().expect("at least one level");
    for level in (0..levels - 1).rev() {
        let (w, h) = sizes[level];
        let (nw, nh) = sizes[level + 1];
        let expanded = pyr_up(&result, nw, nh, w, h);
        result = accumulated[level]
            .iter()
            .zip(&expanded)
            .map(|(a, b)| a + b)
            .collect();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blur_preserves_a_constant_field() {
        let flat = vec![7.0f32; 64];
        let out = blur(&flat, 8, 8);
        // Clamp-to-edge means even border pixels keep the constant.
        for value in out {
            assert!((value - 7.0).abs() < 1e-5, "border darkened to {value}");
        }
    }

    #[test]
    fn pyramid_levels_halve_and_stop_at_one() {
        assert_eq!(half(8, 8), (4, 4));
        assert_eq!(half(7, 5), (4, 3));
        assert_eq!(half(1, 1), (1, 1));
    }

    #[test]
    fn level_sizes_follow_the_halving() {
        assert_eq!(level_sizes(16, 16, 3), vec![(16, 16), (8, 8), (4, 4)]);
    }

    #[test]
    fn level_count_grows_with_canvas_and_stays_positive() {
        assert!(level_count(20, 20, 5.0) >= 1);
        assert!(level_count(4000, 4000, 5.0) > level_count(400, 400, 5.0));
    }

    /// A Laplacian pyramid must reconstruct exactly what it decomposed, or
    /// every blend built on it is wrong.
    #[test]
    fn laplacian_pyramid_reconstructs_the_original() {
        let (w, h) = (32usize, 24usize);
        let src: Vec<f32> = (0..w * h)
            .map(|i| ((i % 17) as f32) * 3.0 + ((i / w) as f32))
            .collect();
        let levels = 4;
        let sizes = level_sizes(w, h, levels);

        let gaussian = gaussian_pyramid(&src, w, h, levels);
        let laplacian = laplacian_pyramid(&gaussian, &sizes);

        let mut result = laplacian[levels - 1].clone();
        for level in (0..levels - 1).rev() {
            let (lw, lh) = sizes[level];
            let (nw, nh) = sizes[level + 1];
            let expanded = pyr_up(&result, nw, nh, lw, lh);
            result = laplacian[level]
                .iter()
                .zip(&expanded)
                .map(|(a, b)| a + b)
                .collect();
        }

        for (original, rebuilt) in src.iter().zip(&result) {
            assert!(
                (original - rebuilt).abs() < 1e-3,
                "reconstruction drifted: {original} vs {rebuilt}"
            );
        }
    }

    /// A single pass owning the whole canvas must come back untouched — no
    /// softening introduced by the pyramid round trip.
    #[test]
    fn one_pass_with_a_full_mask_passes_through() {
        let (w, h) = (32usize, 32usize);
        let plane: Vec<f32> = (0..w * h).map(|i| (i % 23) as f32 * 5.0).collect();
        let mask = vec![1.0f32; w * h];

        let blended = blend_channel(
            std::slice::from_ref(&plane),
            std::slice::from_ref(&mask),
            w,
            h,
            4,
        );

        for (original, result) in plane.iter().zip(&blended) {
            assert!(
                (original - result).abs() < 1e-2,
                "pass-through altered a pixel: {original} vs {result}"
            );
        }
    }

    /// The property that makes this worth building: with two misaligned passes,
    /// multi-band must keep high-frequency detail sharp where a plain weighted
    /// average would smear it.
    #[test]
    fn keeps_detail_sharper_than_a_linear_average() {
        let (w, h) = (64usize, 16usize);
        // A fine vertical stripe pattern, and the same pattern shifted by one
        // pixel — the ghosting case.
        let stripes = |offset: usize| -> Vec<f32> {
            (0..w * h)
                .map(|i| if ((i % w) + offset).is_multiple_of(2) { 255.0 } else { 0.0 })
                .collect()
        };
        let left = stripes(0);
        let right = stripes(1);

        // Binary masks meeting at the middle, summing to 1 everywhere.
        let mut mask_left = vec![0.0f32; w * h];
        let mut mask_right = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let left_owns = x < w / 2;
                mask_left[y * w + x] = if left_owns { 1.0 } else { 0.0 };
                mask_right[y * w + x] = if left_owns { 0.0 } else { 1.0 };
            }
        }

        let planes = vec![left.clone(), right.clone()];
        let masks = vec![mask_left, mask_right];
        let blended = blend_channel(&planes, &masks, w, h, 4);

        // Contrast across the middle rows, measured as mean absolute
        // neighbour difference. A 50/50 average of the two offset patterns is
        // flat grey, so its contrast collapses.
        let contrast = |data: &[f32]| -> f32 {
            let mut total = 0.0;
            let mut count = 0;
            for y in 4..h - 4 {
                for x in 8..w - 8 {
                    total += (data[y * w + x] - data[y * w + x + 1]).abs();
                    count += 1;
                }
            }
            total / count as f32
        };

        let averaged: Vec<f32> = left
            .iter()
            .zip(&right)
            .map(|(a, b)| (a + b) / 2.0)
            .collect();

        let blended_contrast = contrast(&blended);
        let averaged_contrast = contrast(&averaged);
        assert!(
            blended_contrast > averaged_contrast * 5.0,
            "multi-band kept contrast {blended_contrast} vs averaged {averaged_contrast}; \
             detail is being smeared"
        );
    }
}
