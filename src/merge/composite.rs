//! Warp the aligned passes onto one canvas and blend them together.
//!
//! Each canvas pixel is claimed outright by whichever pass is furthest from its
//! own border, which puts a hard seam down the centre of the overlap — no
//! max-flow, no graph cut. `blend` then softens that seam per frequency band,
//! so the selection stays sharp where sharpness matters.
//!
//! Three properties this stage is built around:
//!
//! * **One resample.** A single inverse-mapped bilinear fetch per output pixel,
//!   from the original pass. The alignment transform is already composed, so
//!   nothing is resampled twice.
//! * **Multi-band blending.** A plain weighted average across the overlap
//!   ghosts whenever the alignment is off by even a pixel, and on real scans it
//!   always is. Blending per frequency band keeps fine detail coming from one
//!   pass while tone crosses over smoothly.
//! * **Full bit depth.** 16-bit passes stay 16-bit. OpenCV's `MultiBandBlender`
//!   feeds through signed 16-bit and clips everything above 32767; there is a
//!   test here that pins the difference.

use anyhow::{Result, bail};
use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};

use super::blend;
use super::geometry::Similarity;

/// Fetches an RGB sample from one pass at a source coordinate, or `None`
/// outside it. Boxed so both depths share the compositing core.
type Sampler<'a> = Box<dyn Fn(f64, f64) -> Option<[f32; 3]> + 'a>;

/// Width in source pixels over which a pass fades in from its border. Ramping
/// over a wide band is what hides a residual step at the join.
pub const DEFAULT_FEATHER: f64 = 64.0;

/// Uncovered canvas is filled with white, matching the padding the deskew stage
/// already uses, so a gap reads as background to the extractor downstream.
const UNCOVERED_8: u8 = 255;
const UNCOVERED_16: u16 = u16::MAX;

pub type Rgb16Image = ImageBuffer<Rgb<u16>, Vec<u16>>;

/// The merged canvas, plus which of its pixels any pass actually reached.
pub struct Composite {
    pub image: DynamicImage,
    /// Which pass won each canvas pixel: 0 for uncovered, otherwise the index
    /// into `members` plus one. Rendered by `ownership_overlay` for `--debug`.
    pub ownership: Vec<u16>,
    /// Pyramid bands used. 1 means multi-band was skipped and the merge fell
    /// back to feathering.
    pub levels: usize,
    /// Row-major, `canvas_width * canvas_height`. False where no pass
    /// contributed; the final crop uses this to avoid trimming to empty corners.
    pub coverage: Vec<bool>,
}

/// How strongly a pass claims a point, for deciding *ownership*.
///
/// The product of the distances to its four borders, normalized so the centre
/// scores 1 and every border scores 0. Crucially this **never saturates**: it
/// keeps decreasing all the way from the centre outward, so two overlapping
/// passes can always be told apart.
///
/// `feather_weight` cannot be used here. It ramps to 1 over a fixed 64 pixels
/// and then flattens, so on a 5100x7019 pass every pixel more than 64 px inside
/// the border scores exactly 1.0. Across a multi-thousand-pixel overlap every
/// pass then ties, the tie-break awards the whole region to the lowest-indexed
/// pass, and the seam collapses onto a pass boundary — rectangular regions,
/// regardless of content.
///
/// Multiplying rather than taking a minimum still matters, for the reason it
/// always did: two passes offset along one axis share their extent on the
/// other, so that shared factor appears identically on both sides and cancels
/// out of the comparison, leaving the differing axis to place the seam.
fn selection_weight(x: f64, y: f64, width: f64, height: f64) -> f32 {
    if x < 0.0 || y < 0.0 || x > width - 1.0 || y > height - 1.0 {
        return 0.0;
    }
    // `+ 1` keeps a border pixel strictly positive, so a pass covering a point
    // alone still claims it.
    let left = x + 1.0;
    let right = width - x;
    let top = y + 1.0;
    let bottom = height - y;
    let normalizer = (width * 0.5).powi(2) * (height * 0.5).powi(2);
    ((left * right * top * bottom) / normalizer) as f32
}

/// Feather weight for a point inside a `width` x `height` pass: the **product**
/// of a ramp from each of its four borders.
///
/// Multiplying rather than taking the minimum matters for seam placement. Two
/// passes offset horizontally share the same vertical extent, so their
/// distances to the top and bottom borders are identical. Under a `min`, that
/// shared vertical term is the smallest one near the top and bottom of the
/// overlap, both passes tie exactly, and the tie-break hands whole rectangular
/// bands to one pass — a blocky seam instead of a straight one.
///
/// With a product, the shared vertical ramp appears as an equal factor on both
/// sides and cancels out of the comparison, so the horizontal ramps decide the
/// seam at *every* row. A diagonal offset makes both axes contribute, which is
/// what you want there too.
///
/// Never returns exactly zero strictly inside the pass, so a pixel covered by
/// only one pass still resolves to it rather than to background.
fn feather_weight(x: f64, y: f64, width: f64, height: f64, feather: f64) -> f32 {
    if x < 0.0 || y < 0.0 || x > width - 1.0 || y > height - 1.0 {
        return 0.0;
    }
    let span = feather.max(1.0);
    // `distance + 1` keeps every factor strictly positive, so a border row is
    // still decided by the other axis. Were the ramp to hit zero there, both
    // passes would score zero on the very first and last rows, tie, and band
    // exactly as the `min` form did.
    let ramp = |distance: f64| ((distance + 1.0) / span).clamp(0.0, 1.0);
    let weight = ramp(x) * ramp(width - 1.0 - x) * ramp(y) * ramp(height - 1.0 - y);
    (weight as f32).max(f32::MIN_POSITIVE)
}

/// Bilinear fetch of an RGB sample, as floats. Returns `None` outside the
/// image, which is what excludes a pass from a canvas pixel it does not cover.
fn sample_rgb8(img: &RgbImage, x: f64, y: f64) -> Option<[f32; 3]> {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let (x0, y0) = (x.floor() as i64, y.floor() as i64);
    if x0 < 0 || y0 < 0 || x0 >= w || y0 >= h {
        return None;
    }
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);

    let mut out = [0.0f32; 3];
    for (channel, slot) in out.iter_mut().enumerate() {
        let p = |px: i64, py: i64| img.get_pixel(px as u32, py as u32).0[channel] as f32;
        let top = p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx;
        let bottom = p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx;
        *slot = top * (1.0 - fy) + bottom * fy;
    }
    Some(out)
}

fn sample_rgb16(img: &Rgb16Image, x: f64, y: f64) -> Option<[f32; 3]> {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let (x0, y0) = (x.floor() as i64, y.floor() as i64);
    if x0 < 0 || y0 < 0 || x0 >= w || y0 >= h {
        return None;
    }
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);

    let mut out = [0.0f32; 3];
    for (channel, slot) in out.iter_mut().enumerate() {
        let p = |px: i64, py: i64| img.get_pixel(px as u32, py as u32).0[channel] as f32;
        let top = p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx;
        let bottom = p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx;
        *slot = top * (1.0 - fy) + bottom * fy;
    }
    Some(out)
}

/// Shared geometry: for each pass, the inverse of its canvas transform and its
/// dimensions. Inverting once per pass rather than per pixel.
struct Placement {
    inverse: Similarity,
    width: f64,
    height: f64,
}

fn placements(transforms: &[Similarity], sizes: &[(u32, u32)]) -> Vec<Placement> {
    transforms
        .iter()
        .zip(sizes)
        .map(|(&transform, &(w, h))| Placement {
            inverse: transform.inverse(),
            width: w as f64,
            height: h as f64,
        })
        .collect()
}

fn validate(
    pass_count: usize,
    transforms: &[Similarity],
    canvas_width: u32,
    canvas_height: u32,
) -> Result<()> {
    if pass_count == 0 {
        bail!("nothing to composite");
    }
    if pass_count != transforms.len() {
        bail!("expected one transform per pass");
    }
    if canvas_width == 0 || canvas_height == 0 {
        bail!("canvas has zero area");
    }
    Ok(())
}

/// Blend strength, in OpenCV's units: the transition width as a percentage of
/// the canvas diagonal, which sets how many pyramid bands are used.
pub const DEFAULT_BLEND_STRENGTH: f64 = 5.0;

/// Above this canvas size the pyramids stop being worth their memory (roughly
/// 8 bytes per pixel per pass, live at once), so the merge falls back to
/// feathering and says so.
const MAX_MULTIBAND_PIXELS: u64 = 120_000_000;

/// Per-pass selection masks: at each canvas pixel the pass whose feather weight
/// is highest takes it outright. Ties go to the lower index so the result is
/// deterministic.
///
/// Binary rather than proportional on purpose. Multi-band turns this into a
/// transition whose width is proportional to each frequency band, so the
/// selection itself should be a hard seam — the pyramid supplies the softness,
/// and only where softness does not cost sharpness.
///
/// The masks sum to 1 at every covered pixel, which is what lets `blend_channel`
/// skip per-level renormalization.
fn selection_masks(
    places: &[Placement],
    canvas_width: u32,
    canvas_height: u32,
    covered_by: &dyn Fn(usize, f64, f64) -> bool,
) -> (Vec<Vec<f32>>, Vec<bool>, Vec<u16>) {
    let (width, height) = (canvas_width as usize, canvas_height as usize);
    let mut masks: Vec<Vec<f32>> = places.iter().map(|_| vec![0.0f32; width * height]).collect();
    let mut coverage = vec![false; width * height];
    // 0 = uncovered, otherwise the owning pass index + 1.
    let mut ownership = vec![0u16; width * height];

    for y in 0..height {
        for x in 0..width {
            let flat = y * width + x;
            let (mut best, mut best_weight) = (usize::MAX, 0.0f32);
            for (index, place) in places.iter().enumerate() {
                let (sx, sy) = place.inverse.apply((x as f64, y as f64));
                if !covered_by(index, sx, sy) {
                    continue;
                }
                let weight = selection_weight(sx, sy, place.width, place.height);
                if weight > best_weight {
                    best_weight = weight;
                    best = index;
                }
            }
            if best != usize::MAX {
                masks[best][flat] = 1.0;
                coverage[flat] = true;
                ownership[flat] = best as u16 + 1;
            }
        }
    }
    (masks, coverage, ownership)
}

/// Warp one channel of one pass onto the canvas. Uncovered pixels are filled
/// with the pass's nearest edge value rather than a constant, so the pyramid
/// does not manufacture a huge false edge at the pass boundary and bleed it
/// back into the seam.
fn warp_channel(
    place: &Placement,
    canvas_width: u32,
    canvas_height: u32,
    channel: usize,
    sample: &dyn Fn(f64, f64) -> Option<[f32; 3]>,
) -> Vec<f32> {
    let (width, height) = (canvas_width as usize, canvas_height as usize);
    let mut plane = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let (sx, sy) = place.inverse.apply((x as f64, y as f64));
            let clamped = (
                sx.clamp(0.0, place.width - 1.0),
                sy.clamp(0.0, place.height - 1.0),
            );
            if let Some(values) = sample(clamped.0, clamped.1) {
                plane[y * width + x] = values[channel];
            }
        }
    }
    plane
}

/// Shared compositing core. Returns the blended canvas as three f32 planes plus
/// coverage, leaving quantization to the caller.
fn composite_planes(
    places: &[Placement],
    canvas_width: u32,
    canvas_height: u32,
    feather: f64,
    samplers: &[&dyn Fn(f64, f64) -> Option<[f32; 3]>],
) -> (Vec<Vec<f32>>, Vec<bool>, usize, Vec<u16>) {
    let (width, height) = (canvas_width as usize, canvas_height as usize);

    let covered_by = |index: usize, sx: f64, sy: f64| samplers[index](sx, sy).is_some();
    let (masks, coverage, ownership) =
        selection_masks(places, canvas_width, canvas_height, &covered_by);

    let pixels = width as u64 * height as u64;
    let levels = if pixels > MAX_MULTIBAND_PIXELS {
        elog!(
            "warning: {width}x{height} canvas is too large for multi-band blending; \
             falling back to feathering, which may show ghosting if the passes \
             are not perfectly aligned"
        );
        1
    } else {
        blend::level_count(width, height, DEFAULT_BLEND_STRENGTH)
    };

    // A single band is a plain masked composite, so soften the hard selection
    // edge with the feather weights instead of leaving a visible cut.
    let masks = if levels == 1 {
        feathered_masks(places, canvas_width, canvas_height, feather, &covered_by)
    } else {
        masks
    };

    let mut planes = Vec::with_capacity(3);
    for channel in 0..3 {
        let warped: Vec<Vec<f32>> = places
            .iter()
            .zip(samplers)
            .map(|(place, sample)| {
                warp_channel(place, canvas_width, canvas_height, channel, *sample)
            })
            .collect();
        planes.push(blend::blend_channel(&warped, &masks, width, height, levels));
    }

    (planes, coverage, levels, ownership)
}

/// Proportional feather weights, normalized to sum to 1 per pixel. Only used
/// for the single-band fallback.
fn feathered_masks(
    places: &[Placement],
    canvas_width: u32,
    canvas_height: u32,
    feather: f64,
    covered_by: &dyn Fn(usize, f64, f64) -> bool,
) -> Vec<Vec<f32>> {
    let (width, height) = (canvas_width as usize, canvas_height as usize);
    let mut masks: Vec<Vec<f32>> = places.iter().map(|_| vec![0.0f32; width * height]).collect();

    for y in 0..height {
        for x in 0..width {
            let flat = y * width + x;
            let mut total = 0.0f32;
            for (index, place) in places.iter().enumerate() {
                let (sx, sy) = place.inverse.apply((x as f64, y as f64));
                if !covered_by(index, sx, sy) {
                    continue;
                }
                let weight = feather_weight(sx, sy, place.width, place.height, feather);
                masks[index][flat] = weight;
                total += weight;
            }
            if total > 0.0 {
                for mask in masks.iter_mut() {
                    mask[flat] /= total;
                }
            }
        }
    }
    masks
}

/// Composite 8-bit RGB passes.
pub fn composite_rgb8(
    passes: &[RgbImage],
    transforms: &[Similarity],
    canvas_width: u32,
    canvas_height: u32,
    feather: f64,
) -> Result<(RgbImage, Vec<bool>, usize, Vec<u16>)> {
    validate(passes.len(), transforms, canvas_width, canvas_height)?;
    let sizes: Vec<(u32, u32)> = passes.iter().map(|p| (p.width(), p.height())).collect();
    let places = placements(transforms, &sizes);

    let samplers: Vec<Sampler<'_>> = passes
        .iter()
        .map(|pass| {
            Box::new(move |x: f64, y: f64| sample_rgb8(pass, x, y)) as Sampler<'_>
        })
        .collect();
    let sampler_refs: Vec<&dyn Fn(f64, f64) -> Option<[f32; 3]>> =
        samplers.iter().map(|s| s.as_ref()).collect();

    let (planes, coverage, levels, ownership) = composite_planes(
        &places,
        canvas_width,
        canvas_height,
        feather,
        &sampler_refs,
    );

    let width = canvas_width as usize;
    let mut out = RgbImage::from_pixel(canvas_width, canvas_height, Rgb([UNCOVERED_8; 3]));
    for y in 0..canvas_height {
        for x in 0..canvas_width {
            let flat = y as usize * width + x as usize;
            if !coverage[flat] {
                continue;
            }
            out.put_pixel(
                x,
                y,
                Rgb([
                    planes[0][flat].round().clamp(0.0, 255.0) as u8,
                    planes[1][flat].round().clamp(0.0, 255.0) as u8,
                    planes[2][flat].round().clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }

    Ok((out, coverage, levels, ownership))
}

/// Composite 16-bit RGB passes, at full depth.
pub fn composite_rgb16(
    passes: &[Rgb16Image],
    transforms: &[Similarity],
    canvas_width: u32,
    canvas_height: u32,
    feather: f64,
) -> Result<(Rgb16Image, Vec<bool>, usize, Vec<u16>)> {
    validate(passes.len(), transforms, canvas_width, canvas_height)?;
    let sizes: Vec<(u32, u32)> = passes.iter().map(|p| (p.width(), p.height())).collect();
    let places = placements(transforms, &sizes);

    let samplers: Vec<Sampler<'_>> = passes
        .iter()
        .map(|pass| {
            Box::new(move |x: f64, y: f64| sample_rgb16(pass, x, y)) as Sampler<'_>
        })
        .collect();
    let sampler_refs: Vec<&dyn Fn(f64, f64) -> Option<[f32; 3]>> =
        samplers.iter().map(|s| s.as_ref()).collect();

    let (planes, coverage, levels, ownership) = composite_planes(
        &places,
        canvas_width,
        canvas_height,
        feather,
        &sampler_refs,
    );

    let width = canvas_width as usize;
    let mut out = Rgb16Image::from_pixel(canvas_width, canvas_height, Rgb([UNCOVERED_16; 3]));
    for y in 0..canvas_height {
        for x in 0..canvas_width {
            let flat = y as usize * width + x as usize;
            if !coverage[flat] {
                continue;
            }
            out.put_pixel(
                x,
                y,
                Rgb([
                    planes[0][flat].round().clamp(0.0, 65535.0) as u16,
                    planes[1][flat].round().clamp(0.0, 65535.0) as u16,
                    planes[2][flat].round().clamp(0.0, 65535.0) as u16,
                ]),
            );
        }
    }

    Ok((out, coverage, levels, ownership))
}

/// Composite passes at whichever depth they carry. 16-bit inputs stay 16-bit;
/// everything else is handled as 8-bit RGB.
pub fn composite(
    passes: &[&DynamicImage],
    transforms: &[Similarity],
    canvas_width: u32,
    canvas_height: u32,
    feather: f64,
) -> Result<Composite> {
    validate(passes.len(), transforms, canvas_width, canvas_height)?;

    // Any 16-bit pass promotes the whole merge, so depth is never silently lost
    // because one pass in the group happened to be 8-bit.
    let deep = passes.iter().any(|p| {
        matches!(
            p.color(),
            image::ColorType::L16
                | image::ColorType::La16
                | image::ColorType::Rgb16
                | image::ColorType::Rgba16
        )
    });

    if deep {
        let converted: Vec<Rgb16Image> = passes.iter().map(|p| p.to_rgb16()).collect();
        let (image, coverage, levels, ownership) = composite_rgb16(
            &converted,
            transforms,
            canvas_width,
            canvas_height,
            feather,
        )?;
        Ok(Composite {
            image: DynamicImage::ImageRgb16(image),
            levels,
            ownership,
            coverage,
        })
    } else {
        let converted: Vec<RgbImage> = passes.iter().map(|p| p.to_rgb8()).collect();
        let (image, coverage, levels, ownership) =
            composite_rgb8(&converted, transforms, canvas_width, canvas_height, feather)?;
        Ok(Composite {
            image: DynamicImage::ImageRgb8(image),
            levels,
            ownership,
            coverage,
        })
    }
}

/// Distinct hues for the provenance overlay. Chosen to stay separable in the
/// tinted, half-strength form the overlay uses, and to read differently in
/// grayscale for anyone printing it.
const OWNER_COLOURS: [[u8; 3]; 6] = [
    [220, 60, 60],   // red
    [60, 140, 220],  // blue
    [70, 180, 90],   // green
    [230, 170, 40],  // amber
    [160, 90, 200],  // violet
    [40, 190, 190],  // teal
];

/// Render which pass supplied each region of the merged canvas.
///
/// The merged image is flattened to grey and tinted by its owning pass, so both
/// the content and the seam between passes are visible at once — a solid-colour
/// map would show the seams but not whether they land somewhere sensible.
/// Uncovered canvas is left magenta, a colour no scan produces.
pub fn ownership_overlay(merged: &DynamicImage, ownership: &[u16]) -> Result<RgbImage> {
    let (width, height) = (merged.width(), merged.height());
    if ownership.len() != (width as usize) * (height as usize) {
        bail!("ownership map does not match the merged image");
    }
    let grey = merged.to_luma8();
    let mut out = RgbImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let owner = ownership[y as usize * width as usize + x as usize];
            let pixel = if owner == 0 {
                Rgb([255, 0, 255])
            } else {
                let tint = OWNER_COLOURS[(owner as usize - 1) % OWNER_COLOURS.len()];
                let luma = grey.get_pixel(x, y).0[0] as u16;
                // Half tint, half content: enough colour to read the region,
                // enough detail to judge whether the seam sits well.
                Rgb([
                    ((luma + tint[0] as u16) / 2) as u8,
                    ((luma + tint[1] as u16) / 2) as u8,
                    ((luma + tint[2] as u16) / 2) as u8,
                ])
            };
            out.put_pixel(x, y, pixel);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merge::geometry::Similarity;

    fn solid8(w: u32, h: u32, colour: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(w, h, Rgb(colour))
    }

    #[test]
    fn feather_is_zero_outside_and_ramps_inward() {
        // Outside the pass entirely.
        assert_eq!(feather_weight(-1.0, 5.0, 100.0, 100.0, 10.0), 0.0);
        // On the border: minimal but not zero, so a lone pass still wins.
        assert!(feather_weight(0.0, 50.0, 100.0, 100.0, 10.0) > 0.0);
        // Deep inside, every ramp saturates.
        assert_eq!(feather_weight(50.0, 50.0, 100.0, 100.0, 10.0), 1.0);
        // Approaching one border lowers the weight monotonically.
        let near = feather_weight(3.0, 50.0, 100.0, 100.0, 10.0);
        let nearer = feather_weight(1.0, 50.0, 100.0, 100.0, 10.0);
        assert!(nearer < near && near < 1.0, "{nearer} < {near} < 1");
    }

    /// The failure that produced rectangular regions on real scans.
    ///
    /// `feather_weight` saturates at 1.0 beyond `DEFAULT_FEATHER` pixels from
    /// any border, so on passes far larger than that every pixel deep inside
    /// scores identically and every pass ties. `selection_weight` must keep
    /// discriminating all the way to the centre.
    #[test]
    fn selection_weight_does_not_saturate_away_from_borders() {
        // Dimensions in the range real scans use.
        let (w, h) = (5100.0f64, 7019.0f64);

        // The old feather field is flat across essentially the whole pass.
        let a = feather_weight(1000.0, 3000.0, w, h, DEFAULT_FEATHER);
        let b = feather_weight(2000.0, 3000.0, w, h, DEFAULT_FEATHER);
        assert_eq!(a, b, "precondition: the feather ramp saturates");

        // The selection field still separates those same two points, and rises
        // toward the centre.
        let a = selection_weight(1000.0, 3000.0, w, h);
        let b = selection_weight(2000.0, 3000.0, w, h);
        let centre = selection_weight(w / 2.0, h / 2.0, w, h);
        assert!(a < b, "{a} should rank below {b}, nearer the centre");
        assert!(b < centre, "{b} should rank below the centre {centre}");
        assert!((centre - 1.0).abs() < 1e-3, "centre should score ~1, got {centre}");
        assert_eq!(selection_weight(-1.0, 10.0, w, h), 0.0, "outside must score 0");
    }

    /// With overlaps far wider than the feather span — as on real passes — the
    /// seam must still land mid-overlap rather than collapsing onto a border.
    #[test]
    fn seam_stays_mid_overlap_on_large_passes() {
        // 1200-wide passes overlapping by 900px: nearly 20x the feather span.
        let left = solid8(1200, 400, [0, 0, 0]);
        let right = solid8(1200, 400, [255, 255, 255]);
        let transforms = [Similarity::IDENTITY, Similarity::translation(300.0, 0.0)];

        let (_, _, _, ownership) =
            composite_rgb8(&[left, right], &transforms, 1500, 400, DEFAULT_FEATHER).unwrap();

        let width = 1500usize;
        let boundary = (0..width)
            .find(|&x| ownership[200 * width + x] == 2)
            .unwrap_or(width);

        // Overlap spans x 300..1200, so the seam belongs near x = 750.
        assert!(
            (700..800).contains(&boundary),
            "seam at x={boundary} is not mid-overlap (300..1200); it has              collapsed onto a pass border"
        );
    }

    /// The seam between two horizontally offset passes must be a straight
    /// vertical line at every row.
    ///
    /// The earlier `min`-of-distances weighting tied exactly near the top and
    /// bottom of the overlap, where both passes shared the same vertical
    /// distance, and handed whole bands to one pass. Sampling only the middle
    /// row — as the crossover test does — never saw it.
    #[test]
    fn seam_between_offset_passes_is_vertical_at_every_row() {
        let left = solid8(100, 60, [0, 0, 0]);
        let right = solid8(100, 60, [255, 255, 255]);
        let transforms = [Similarity::IDENTITY, Similarity::translation(60.0, 0.0)];

        let (_, _, _, ownership) =
            composite_rgb8(&[left, right], &transforms, 160, 60, 30.0).unwrap();

        // For each row, the first column owned by the second pass.
        let width = 160usize;
        let boundaries: Vec<usize> = (0..60)
            .map(|y| {
                (0..width)
                    .find(|&x| ownership[y * width + x] == 2)
                    .unwrap_or(width)
            })
            .collect();

        let first = boundaries[2];
        for (y, &boundary) in boundaries.iter().enumerate() {
            assert!(
                boundary.abs_diff(first) <= 1,
                "seam jumps to x={boundary} at row {y} (expected ~{first});                  the selection is banding instead of cutting straight"
            );
        }
        // And it should sit inside the overlap, not hard against an edge.
        assert!(
            (65..95).contains(&first),
            "seam at x={first} is not near the middle of the 60..100 overlap"
        );
    }

    #[test]
    fn single_pass_at_identity_reproduces_itself() {
        let pass = solid8(40, 30, [10, 120, 250]);
        let (out, coverage, _, _) =
            composite_rgb8(std::slice::from_ref(&pass), &[Similarity::IDENTITY], 40, 30, 8.0).unwrap();

        assert_eq!(out.dimensions(), (40, 30));
        assert!(coverage.iter().all(|&c| c), "everything should be covered");
        for (a, b) in out.pixels().zip(pass.pixels()) {
            assert_eq!(a, b, "a lone pass must pass through unchanged");
        }
    }

    #[test]
    fn uncovered_canvas_is_white_and_marked_uncovered() {
        // A 20x20 pass on a 60x20 canvas leaves two thirds empty.
        let pass = solid8(20, 20, [0, 0, 0]);
        let (out, coverage, _, _) =
            composite_rgb8(&[pass], &[Similarity::IDENTITY], 60, 20, 4.0).unwrap();

        assert_eq!(*out.get_pixel(50, 10), Rgb([255, 255, 255]));
        assert!(!coverage[10 * 60 + 50], "empty area should read as uncovered");
        assert!(coverage[10 * 60 + 5], "the pass area should read as covered");
    }

    /// Two passes of different flat colours: the blend must be monotonic across
    /// the overlap, ending at each pass's own colour outside it.
    #[test]
    fn overlap_blends_between_the_two_passes() {
        let left = solid8(100, 40, [0, 0, 0]);
        let right = solid8(100, 40, [255, 255, 255]);
        let transforms = [Similarity::IDENTITY, Similarity::translation(60.0, 0.0)];

        let (out, _, _, _) = composite_rgb8(&[left, right], &transforms, 160, 40, 30.0).unwrap();

        // Outside the overlap each pass owns its colour.
        assert_eq!(*out.get_pixel(5, 20), Rgb([0, 0, 0]));
        assert_eq!(*out.get_pixel(155, 20), Rgb([255, 255, 255]));

        // Across the overlap (x 60..100) the value rises without reversing.
        let mut previous = 0i32;
        for x in 60..100 {
            let value = out.get_pixel(x, 20).0[0] as i32;
            assert!(value >= previous, "blend reversed at x={x}");
            previous = value;
        }
        // And it genuinely traverses, rather than stepping at one edge.
        let midpoint = out.get_pixel(80, 20).0[0];
        assert!(
            (40..=215).contains(&midpoint),
            "midpoint {midpoint} is not a blend"
        );
    }

    /// The crossover should sit at the middle of the overlap, which is the
    /// "straight seam down the centre" this design is meant to produce.
    #[test]
    fn crossover_lands_mid_overlap() {
        let left = solid8(100, 40, [0, 0, 0]);
        let right = solid8(100, 40, [255, 255, 255]);
        let transforms = [Similarity::IDENTITY, Similarity::translation(60.0, 0.0)];

        let (out, _, _, _) = composite_rgb8(&[left, right], &transforms, 160, 40, 200.0).unwrap();

        // Overlap spans x 60..100, so the halfway mix belongs near x = 80.
        let midpoint = out.get_pixel(80, 20).0[0] as i32;
        assert!(
            (midpoint - 128).abs() <= 12,
            "expected a near-even mix at the overlap centre, got {midpoint}"
        );
    }

    /// The reason this stage is written in Rust at all: OpenCV's multi-band
    /// blender composites through signed 16-bit and clips at 32767. Values
    /// above that must survive here.
    #[test]
    fn sixteen_bit_values_above_int16_max_survive() {
        let bright = Rgb16Image::from_pixel(40, 30, Rgb([60000, 65535, 40000]));
        let (out, _, _, _) =
            composite_rgb16(&[bright], &[Similarity::IDENTITY], 40, 30, 8.0).unwrap();

        assert_eq!(*out.get_pixel(20, 15), Rgb([60000, 65535, 40000]));
        assert!(
            out.get_pixel(20, 15).0[0] > 32767,
            "value was clipped to signed 16-bit range"
        );
    }

    #[test]
    fn dispatch_preserves_sixteen_bit_inputs() {
        let deep = DynamicImage::ImageRgb16(Rgb16Image::from_pixel(20, 20, Rgb([50000; 3])));
        let merged = composite(&[&deep], &[Similarity::IDENTITY], 20, 20, 4.0).unwrap();

        assert_eq!(merged.image.color(), image::ColorType::Rgb16);
        assert_eq!(merged.image.to_rgb16().get_pixel(10, 10).0[0], 50000);
    }

    /// A single 8-bit pass in the group must not drag a 16-bit one down.
    #[test]
    fn any_sixteen_bit_pass_promotes_the_merge() {
        let shallow = DynamicImage::ImageRgb8(solid8(20, 20, [200, 200, 200]));
        let deep = DynamicImage::ImageRgb16(Rgb16Image::from_pixel(20, 20, Rgb([50000; 3])));
        let transforms = [Similarity::IDENTITY, Similarity::translation(10.0, 0.0)];

        let merged = composite(&[&shallow, &deep], &transforms, 30, 20, 4.0).unwrap();
        assert_eq!(merged.image.color(), image::ColorType::Rgb16);
    }

    /// A tall single pass must survive the pyramid round trip unchanged — no
    /// horizontal banding, no drift down the image.
    #[test]
    fn tall_single_pass_survives_the_pyramid_round_trip() {
        let height = 400u32;
        let tall = RgbImage::from_fn(16, height, |_, y| Rgb([(y % 256) as u8; 3]));
        let (out, coverage, _, _) = composite_rgb8(
            std::slice::from_ref(&tall),
            &[Similarity::IDENTITY],
            16,
            height,
            4.0,
        )
        .unwrap();

        assert!(coverage.iter().all(|&c| c));
        for y in 0..height {
            assert_eq!(out.get_pixel(8, y), tall.get_pixel(8, y), "drift at row {y}");
        }
    }

    #[test]
    fn ownership_overlay_tints_each_pass_and_flags_gaps() {
        let merged = DynamicImage::ImageRgb8(solid8(6, 2, [128, 128, 128]));
        // Left pair from pass 1, middle from pass 2, right uncovered.
        let ownership = vec![1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0];

        let overlay = ownership_overlay(&merged, &ownership).unwrap();

        let first = *overlay.get_pixel(0, 0);
        let second = *overlay.get_pixel(2, 0);
        assert_ne!(first, second, "passes should be tinted differently");
        assert_eq!(
            *overlay.get_pixel(4, 0),
            Rgb([255, 0, 255]),
            "uncovered canvas should be magenta"
        );
        // The tint keeps content visible rather than flooding it.
        assert!(first.0.iter().any(|&c| c > 0 && c < 255));
    }

    #[test]
    fn ownership_overlay_rejects_a_mismatched_map() {
        let merged = DynamicImage::ImageRgb8(solid8(4, 4, [0, 0, 0]));
        assert!(ownership_overlay(&merged, &[1, 1]).is_err());
    }

    /// More passes than palette entries must still render, wrapping rather
    /// than panicking.
    #[test]
    fn ownership_overlay_wraps_beyond_the_palette() {
        let merged = DynamicImage::ImageRgb8(solid8(2, 1, [100, 100, 100]));
        let beyond = OWNER_COLOURS.len() as u16 + 1;
        let overlay = ownership_overlay(&merged, &[beyond, beyond]).unwrap();
        assert_ne!(*overlay.get_pixel(0, 0), Rgb([255, 0, 255]));
    }

    #[test]
    fn rejects_malformed_requests() {
        let pass = solid8(10, 10, [0, 0, 0]);
        assert!(composite_rgb8(&[], &[], 10, 10, 4.0).is_err());
        assert!(composite_rgb8(std::slice::from_ref(&pass), &[], 10, 10, 4.0).is_err());
        assert!(composite_rgb8(&[pass], &[Similarity::IDENTITY], 0, 10, 4.0).is_err());
    }
}
