//! Warp the aligned passes onto one canvas and blend them together.
//!
//! Each canvas pixel starts out claimed by whichever pass is furthest from its
//! own border; `seam` then moves the boundary between every overlapping pair
//! onto pixels where the two already agree, and `blend` softens what is left
//! per frequency band, so the selection stays sharp where sharpness matters.
//!
//! Three properties this stage is built around:
//!
//! * **One resample.** A single inverse-mapped bilinear fetch per pass per
//!   output pixel, from the original pass, yielding all three channels at once.
//!   The alignment transform is already composed, so nothing is resampled
//!   twice.
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
use super::seam;

/// One input pass, as the compositing core sees it: something that yields an
/// RGB sample at a source coordinate, or `None` outside itself.
///
/// A trait rather than a boxed closure so the core monomorphizes per depth.
/// Sampling happens hundreds of millions of times per merge, and through
/// `dyn Fn` every one of those was an indirect call the bilinear arithmetic
/// could not be inlined into.
pub trait Pass {
    fn sample(&self, x: f64, y: f64) -> Option<[f32; 3]>;
}

impl Pass for RgbImage {
    #[inline]
    fn sample(&self, x: f64, y: f64) -> Option<[f32; 3]> {
        sample_rgb8(self, x, y)
    }
}

impl Pass for Rgb16Image {
    #[inline]
    fn sample(&self, x: f64, y: f64) -> Option<[f32; 3]> {
        sample_rgb16(self, x, y)
    }
}

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

/// The four corner offsets and fractional position of a bilinear fetch, or
/// `None` outside the image.
///
/// Shared by both depths, and expressed as flat offsets into the raw buffer
/// rather than as `get_pixel` calls: this runs once per pass per canvas pixel,
/// and `get_pixel` would re-derive the same offset with a bounds check for
/// every corner of every channel — twelve times over for one fetch.
#[inline]
fn bilinear_taps(width: u32, height: u32, x: f64, y: f64) -> Option<([usize; 4], f32, f32)> {
    let (w, h) = (width as i64, height as i64);
    let (x0, y0) = (x.floor() as i64, y.floor() as i64);
    if x0 < 0 || y0 < 0 || x0 >= w || y0 >= h {
        return None;
    }
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);

    // Offsets are in channels, so a caller adds its channel index directly.
    let stride = width as usize * 3;
    let (x0, x1) = (x0 as usize * 3, x1 as usize * 3);
    let (row0, row1) = (y0 as usize * stride, y1 as usize * stride);
    Some(([row0 + x0, row0 + x1, row1 + x0, row1 + x1], fx, fy))
}

/// Bilinear fetch of an RGB sample, as floats. Returns `None` outside the
/// image, which is what excludes a pass from a canvas pixel it does not cover.
fn sample_rgb8(img: &RgbImage, x: f64, y: f64) -> Option<[f32; 3]> {
    let ([tl, tr, bl, br], fx, fy) = bilinear_taps(img.width(), img.height(), x, y)?;
    let data: &[u8] = img;

    let mut out = [0.0f32; 3];
    for (channel, slot) in out.iter_mut().enumerate() {
        let top = data[tl + channel] as f32 * (1.0 - fx) + data[tr + channel] as f32 * fx;
        let bottom = data[bl + channel] as f32 * (1.0 - fx) + data[br + channel] as f32 * fx;
        *slot = top * (1.0 - fy) + bottom * fy;
    }
    Some(out)
}

fn sample_rgb16(img: &Rgb16Image, x: f64, y: f64) -> Option<[f32; 3]> {
    let ([tl, tr, bl, br], fx, fy) = bilinear_taps(img.width(), img.height(), x, y)?;
    let data: &[u16] = img;

    let mut out = [0.0f32; 3];
    for (channel, slot) in out.iter_mut().enumerate() {
        let top = data[tl + channel] as f32 * (1.0 - fx) + data[tr + channel] as f32 * fx;
        let bottom = data[bl + channel] as f32 * (1.0 - fx) + data[br + channel] as f32 * fx;
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
    /// Axis-aligned bounds of this pass's footprint on the canvas, as
    /// `(x0, y0, x1, y1)` inclusive. A rotated pass does not fill its own
    /// bounding box, so this is a superset of where `covers` holds — which is
    /// all it is used for: skipping canvas the pass cannot possibly reach.
    bounds: (f64, f64, f64, f64),
}

fn placements(transforms: &[Similarity], sizes: &[(u32, u32)]) -> Vec<Placement> {
    transforms
        .iter()
        .zip(sizes)
        .map(|(&transform, &(w, h))| {
            // `covers` accepts source coordinates in [0, w-1] x [0, h-1], so
            // those corners are the ones whose image bounds the footprint.
            let (last_x, last_y) = (w as f64 - 1.0, h as f64 - 1.0);
            let corners = [
                transform.apply((0.0, 0.0)),
                transform.apply((last_x, 0.0)),
                transform.apply((last_x, last_y)),
                transform.apply((0.0, last_y)),
            ];
            let bounds = corners.iter().fold(
                (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
                |(x0, y0, x1, y1), &(x, y)| (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
            );
            Placement {
                inverse: transform.inverse(),
                width: w as f64,
                height: h as f64,
                bounds,
            }
        })
        .collect()
}

/// Canvas rows and columns where two passes could both reach, as half-open
/// ranges already clipped to the canvas. Empty when their footprints are
/// disjoint.
fn overlap_window(
    a: &Placement,
    b: &Placement,
    width: usize,
    height: usize,
) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
    let x0 = a.bounds.0.max(b.bounds.0).floor().max(0.0) as usize;
    let y0 = a.bounds.1.max(b.bounds.1).floor().max(0.0) as usize;
    // `+ 1` for the half-open end, then clamp; `ceil` so a fractional edge is
    // included rather than trimmed.
    let x1 = (a.bounds.2.min(b.bounds.2).ceil().max(0.0) as usize + 1).min(width);
    let y1 = (a.bounds.3.min(b.bounds.3).ceil().max(0.0) as usize + 1).min(height);
    (x0..x1.max(x0), y0..y1.max(y0))
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
/// One bit per canvas pixel, recording whether a given pass reaches it.
///
/// Deciding that costs an inverse transform and a bounds test, and later stages
/// need the same answer for the same pixels. A bit each is 8 MB for a
/// 60-megapixel canvas, against re-deriving it tens of millions of times.
pub struct Coverage {
    words: Vec<u64>,
}

impl Coverage {
    fn new(pixels: usize) -> Self {
        Self {
            words: vec![0u64; pixels.div_ceil(64)],
        }
    }

    #[inline]
    fn set(&mut self, pixel: usize) {
        self.words[pixel / 64] |= 1u64 << (pixel % 64);
    }

    #[inline]
    pub fn get(&self, pixel: usize) -> bool {
        self.words[pixel / 64] & (1u64 << (pixel % 64)) != 0
    }
}

fn selection_masks<P: Pass>(
    places: &[Placement],
    passes: &[P],
    canvas_width: u32,
    canvas_height: u32,
) -> (Vec<Vec<f32>>, Vec<bool>, Vec<u16>, Vec<Coverage>) {
    let (width, height) = (canvas_width as usize, canvas_height as usize);
    let mut masks: Vec<Vec<f32>> = places.iter().map(|_| vec![0.0f32; width * height]).collect();
    let mut coverage = vec![false; width * height];
    // 0 = uncovered, otherwise the owning pass index + 1.
    let mut ownership = vec![0u16; width * height];
    let mut per_pass: Vec<Coverage> = places
        .iter()
        .map(|_| Coverage::new(width * height))
        .collect();

    for y in 0..height {
        for x in 0..width {
            let flat = y * width + x;
            let (mut best, mut best_weight) = (usize::MAX, 0.0f32);
            for (index, place) in places.iter().enumerate() {
                let (sx, sy) = place.inverse.apply((x as f64, y as f64));
                // Recorded against `covers`, deliberately not against
                // `covered_by`. The two disagree on the outermost half-pixel —
                // a sampler still interpolates at `w - 0.5`, where `covers`
                // has already stopped — and the consumers of this bitmap are
                // the ones that ask `covers`. Both are derived from the same
                // inverse transform, so keeping them apart is free.
                if in_bounds(sx, sy, place) {
                    per_pass[index].set(flat);
                }
                if passes[index].sample(sx, sy).is_none() {
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
    (masks, coverage, ownership, per_pass)
}

/// Nodes the seam solve may use. Seam *placement* does not need full
/// resolution — only the blend does — and BK holds roughly 40 bytes per node
/// plus its arcs, so a full-resolution overlap would run to gigabytes across
/// several pairs. OpenCV finds seams at 0.1 MP for the same reason; this is
/// more generous than that and still small.
const SEAM_MAX_NODES: u64 = 600_000;

/// Whether a source coordinate lies within a pass, given the pass it came from.
#[inline]
fn in_bounds(sx: f64, sy: f64, place: &Placement) -> bool {
    sx >= 0.0 && sy >= 0.0 && sx <= place.width - 1.0 && sy <= place.height - 1.0
}

/// Whether a pass reaches a canvas point, without paying for a sample.
fn covers(place: &Placement, x: f64, y: f64) -> bool {
    let (sx, sy) = place.inverse.apply((x, y));
    in_bounds(sx, sy, place)
}

/// Re-place the boundary between each overlapping pair of passes so it follows
/// pixels where they agree, instead of the geometric locus.
///
/// Each pair is solved independently at reduced resolution and written back
/// over the full-resolution ownership map. A pair that cannot produce a cut —
/// no shared area, or nothing anchored to one side — is left as the geometry
/// had it.
fn refine_seams<P: Pass>(
    ownership: &mut [u16],
    places: &[Placement],
    passes: &[P],
    canvas_width: u32,
    canvas_height: u32,
) -> Option<usize> {
    let (width, height) = (canvas_width as usize, canvas_height as usize);
    let area = width as u64 * height as u64;
    // One stride for every pair, so the solves stay comparable.
    let stride = ((area as f64 / SEAM_MAX_NODES as f64).sqrt().ceil() as usize).max(1);
    let (grid_w, grid_h) = (width.div_ceil(stride), height.div_ceil(stride));
    if grid_w < 3 || grid_h < 3 {
        return None;
    }

    for a in 0..places.len() {
        for b in (a + 1)..places.len() {
            let (owner_a, owner_b) = (a as u16 + 1, b as u16 + 1);
            let mut state = vec![seam::Cell::Absent; grid_w * grid_h];
            let mut cost = vec![0.0f32; grid_w * grid_h];
            let mut shared = 0usize;

            for gy in 0..grid_h {
                for gx in 0..grid_w {
                    let (x, y) = ((gx * stride) as f64, (gy * stride) as f64);
                    let (in_a, in_b) = (covers(&places[a], x, y), covers(&places[b], x, y));
                    let slot = gy * grid_w + gx;
                    if !in_a && !in_b {
                        continue;
                    }
                    if in_a && !in_b {
                        state[slot] = seam::Cell::OnlyA;
                        continue;
                    }
                    if in_b && !in_a {
                        state[slot] = seam::Cell::OnlyB;
                        continue;
                    }
                    // Shared: only contest it if one of the two currently holds
                    // it, so a third pass's territory is left alone.
                    let held = ownership[(gy * stride).min(height - 1) * width
                        + (gx * stride).min(width - 1)];
                    if held != owner_a && held != owner_b {
                        continue;
                    }
                    state[slot] = seam::Cell::Both;
                    shared += 1;

                    // Disagreement between the passes here: OpenCV's COST_COLOR.
                    let (ax, ay) = places[a].inverse.apply((x, y));
                    let (bx, by) = places[b].inverse.apply((x, y));
                    let sample_a = passes[a].sample(ax, ay);
                    let sample_b = passes[b].sample(bx, by);
                    if let (Some(sa), Some(sb)) = (sample_a, sample_b) {
                        cost[slot] = sa
                            .iter()
                            .zip(&sb)
                            .map(|(p, q)| (p - q).abs())
                            .sum::<f32>();
                    }
                }
            }
            if shared == 0 {
                continue;
            }

            let Some(owned_by_a) = seam::cut_ownership(&state, &cost, grid_w, grid_h) else {
                continue;
            };

            // Write the decision back at full resolution, touching only the
            // pixels these two were contesting.
            //
            // Confined to where their footprints actually meet. A pixel outside
            // that window cannot be covered by both, so the full-canvas scan
            // this replaces spent most of its time proving the same negative
            // over and over — twice per pixel, per pair.
            let (xs, ys) = overlap_window(&places[a], &places[b], width, height);
            for y in ys {
                let gy = (y / stride).min(grid_h - 1);
                for x in xs.clone() {
                    let index = y * width + x;
                    let held = ownership[index];
                    if held != owner_a && held != owner_b {
                        continue;
                    }
                    let (fx, fy) = (x as f64, y as f64);
                    if !covers(&places[a], fx, fy) || !covers(&places[b], fx, fy) {
                        continue;
                    }
                    let gx = (x / stride).min(grid_w - 1);
                    ownership[index] = if owned_by_a[gy * grid_w + gx] {
                        owner_a
                    } else {
                        owner_b
                    };
                }
            }
        }
    }
    Some(stride)
}

/// Soften the ownership boundary, so the seam hands over gradually instead of
/// as a hard step.
///
/// Two reasons it needs this. The cut is solved on a coarse grid and upscaled,
/// so its boundary arrives carrying that grid's staircase; and even a perfect
/// boundary is a one-pixel switch, which the finest pyramid band reproduces
/// faithfully as an edge. Smoothing over roughly one coarse cell removes the
/// staircase and gives the blend a gentle handover, at a cost of only the
/// sharpness within that narrow band.
///
/// The smoothing runs down and back up an image pyramid rather than as a wide
/// convolution: reaching a ten-pixel radius directly would need a very large
/// kernel over a canvas of tens of megapixels.
fn soften_masks(
    masks: &mut [Vec<f32>],
    per_pass: &[Coverage],
    canvas_width: u32,
    canvas_height: u32,
    stride: usize,
) {
    let (width, height) = (canvas_width as usize, canvas_height as usize);
    // Each level halves, so this reaches about one coarse cell.
    let levels = (stride as f64).log2().ceil().max(1.0) as usize;
    let levels = levels.clamp(1, 5);

    for mask in masks.iter_mut() {
        let mut sizes = vec![(width, height)];
        let mut current = mask.clone();
        let (mut w, mut h) = (width, height);
        for _ in 0..levels {
            let (next, nw, nh) = blend::pyr_down(&current, w, h);
            current = next;
            w = nw;
            h = nh;
            sizes.push((w, h));
        }
        for level in (0..levels).rev() {
            let (tw, th) = sizes[level];
            let (sw, sh) = sizes[level + 1];
            current = blend::pyr_up(&current, sw, sh, tw, th);
        }
        *mask = current;
    }

    // Smoothing spreads a mask past its own pass, where the warp only has
    // clamped edge pixels to offer. Clip each mask back to its coverage, then
    // restore the sum-to-one that the blender relies on.
    //
    // Coverage comes from the bitmap `selection_masks` already built. Asking
    // `covers` again would repeat an inverse transform per pass per pixel, for
    // an answer that has not changed.
    for slot in 0..width * height {
        let mut total = 0.0f32;
        for (index, coverage) in per_pass.iter().enumerate() {
            if !coverage.get(slot) {
                masks[index][slot] = 0.0;
            } else {
                masks[index][slot] = masks[index][slot].max(0.0);
                total += masks[index][slot];
            }
        }
        if total > 0.0 {
            for mask in masks.iter_mut() {
                mask[slot] /= total;
            }
        }
    }
}

/// Warp one pass onto the canvas, all three channels at once. Uncovered pixels
/// are filled with the pass's nearest edge value rather than a constant, so the
/// pyramid does not manufacture a huge false edge at the pass boundary and
/// bleed it back into the seam.
///
/// All three channels together, not one at a time, because the sampler computes
/// every channel of a bilinear fetch whether or not the caller wants them.
/// Warping channel by channel therefore did the inverse transform and the fetch
/// three times over and discarded two thirds of each result.
fn warp_pass<P: Pass>(
    place: &Placement,
    canvas_width: u32,
    canvas_height: u32,
    pass: &P,
) -> [Vec<f32>; 3] {
    let (width, height) = (canvas_width as usize, canvas_height as usize);
    let mut planes: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; width * height]);
    for y in 0..height {
        for x in 0..width {
            let (sx, sy) = place.inverse.apply((x as f64, y as f64));
            let clamped = (
                sx.clamp(0.0, place.width - 1.0),
                sy.clamp(0.0, place.height - 1.0),
            );
            if let Some(values) = pass.sample(clamped.0, clamped.1) {
                let slot = y * width + x;
                planes[0][slot] = values[0];
                planes[1][slot] = values[1];
                planes[2][slot] = values[2];
            }
        }
    }
    planes
}

/// Shared compositing core. Returns the blended canvas as three f32 planes plus
/// coverage, leaving quantization to the caller.
fn composite_planes<P: Pass>(
    places: &[Placement],
    passes: &[P],
    canvas_width: u32,
    canvas_height: u32,
    feather: f64,
    timings: &mut Vec<(&'static str, std::time::Duration)>,
) -> (Vec<Vec<f32>>, Vec<bool>, usize, Vec<u16>) {
    let (width, height) = (canvas_width as usize, canvas_height as usize);

    let (mut masks, coverage, mut ownership, per_pass) = super::timed(timings, "select", || {
        selection_masks(places, passes, canvas_width, canvas_height)
    });

    // Geometry only decides where the boundary *starts*; the min cut moves it
    // onto pixels where the passes agree, which is what makes the regions
    // follow content instead of the layout.
    let seam_stride = super::timed(timings, "seam", || {
        refine_seams(&mut ownership, places, passes, canvas_width, canvas_height)
    });
    for (index, mask) in masks.iter_mut().enumerate() {
        let owner = index as u16 + 1;
        for (slot, value) in mask.iter_mut().enumerate() {
            *value = if ownership[slot] == owner { 1.0 } else { 0.0 };
        }
    }
    if let Some(stride) = seam_stride {
        super::timed(timings, "soften", || {
            soften_masks(&mut masks, &per_pass, canvas_width, canvas_height, stride)
        });
    }

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
        feathered_masks(places, passes, canvas_width, canvas_height, feather)
    } else {
        masks
    };

    // Stream pass by pass: warp one, fold it into the bands, drop it. Only one
    // pass's planes and pyramids are live at a time.
    let mut blender = blend::Blender::<3>::new(width, height, levels);
    for ((place, pass), mask) in places.iter().zip(passes).zip(&masks) {
        let warped = super::timed(timings, "warp", || {
            warp_pass(place, canvas_width, canvas_height, pass)
        });
        super::timed(timings, "blend", || blender.add_pass(warped, mask));
    }
    let planes = super::timed(timings, "collapse", || blender.finish()).to_vec();

    (planes, coverage, levels, ownership)
}

/// Proportional feather weights, normalized to sum to 1 per pixel. Only used
/// for the single-band fallback.
fn feathered_masks<P: Pass>(
    places: &[Placement],
    passes: &[P],
    canvas_width: u32,
    canvas_height: u32,
    feather: f64,
) -> Vec<Vec<f32>> {
    let (width, height) = (canvas_width as usize, canvas_height as usize);
    let mut masks: Vec<Vec<f32>> = places.iter().map(|_| vec![0.0f32; width * height]).collect();

    for y in 0..height {
        for x in 0..width {
            let flat = y * width + x;
            let mut total = 0.0f32;
            for (index, place) in places.iter().enumerate() {
                let (sx, sy) = place.inverse.apply((x as f64, y as f64));
                if passes[index].sample(sx, sy).is_none() {
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
    timings: &mut Vec<(&'static str, std::time::Duration)>,
) -> Result<(RgbImage, Vec<bool>, usize, Vec<u16>)> {
    validate(passes.len(), transforms, canvas_width, canvas_height)?;
    let sizes: Vec<(u32, u32)> = passes.iter().map(|p| (p.width(), p.height())).collect();
    let places = placements(transforms, &sizes);

    let (planes, coverage, levels, ownership) =
        composite_planes(&places, passes, canvas_width, canvas_height, feather, timings);

    let mut out = RgbImage::from_pixel(canvas_width, canvas_height, Rgb([UNCOVERED_8; 3]));
    let data: &mut [u8] = &mut out;
    for (flat, pixel) in data.chunks_exact_mut(3).enumerate() {
        if !coverage[flat] {
            continue;
        }
        for (channel, slot) in pixel.iter_mut().enumerate() {
            *slot = planes[channel][flat].round().clamp(0.0, 255.0) as u8;
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
    timings: &mut Vec<(&'static str, std::time::Duration)>,
) -> Result<(Rgb16Image, Vec<bool>, usize, Vec<u16>)> {
    validate(passes.len(), transforms, canvas_width, canvas_height)?;
    let sizes: Vec<(u32, u32)> = passes.iter().map(|p| (p.width(), p.height())).collect();
    let places = placements(transforms, &sizes);

    let (planes, coverage, levels, ownership) =
        composite_planes(&places, passes, canvas_width, canvas_height, feather, timings);

    let mut out = Rgb16Image::from_pixel(canvas_width, canvas_height, Rgb([UNCOVERED_16; 3]));
    let data: &mut [u16] = &mut out;
    for (flat, pixel) in data.chunks_exact_mut(3).enumerate() {
        if !coverage[flat] {
            continue;
        }
        for (channel, slot) in pixel.iter_mut().enumerate() {
            *slot = planes[channel][flat].round().clamp(0.0, 65535.0) as u16;
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
    timings: &mut Vec<(&'static str, std::time::Duration)>,
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
            timings,
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
            composite_rgb8(&converted, transforms, canvas_width, canvas_height, feather, timings)?;
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
    let luma: &[u8] = &grey;
    let mut out = RgbImage::new(width, height);
    let data: &mut [u8] = &mut out;

    for (flat, pixel) in data.chunks_exact_mut(3).enumerate() {
        let owner = ownership[flat];
        if owner == 0 {
            pixel.copy_from_slice(&[255, 0, 255]);
            continue;
        }
        let tint = OWNER_COLOURS[(owner as usize - 1) % OWNER_COLOURS.len()];
        let value = luma[flat] as u16;
        // Half tint, half content: enough colour to read the region, enough
        // detail to judge whether the seam sits well.
        for (channel, slot) in pixel.iter_mut().enumerate() {
            *slot = ((value + tint[channel] as u16) / 2) as u8;
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

    // The stage timings are a `--debug` diagnostic, not something any test
    // asserts on, so these shadow the real entry points with the sink supplied.
    fn composite_rgb8(
        passes: &[RgbImage],
        transforms: &[Similarity],
        w: u32,
        h: u32,
        feather: f64,
    ) -> Result<(RgbImage, Vec<bool>, usize, Vec<u16>)> {
        super::composite_rgb8(passes, transforms, w, h, feather, &mut Vec::new())
    }

    fn composite_rgb16(
        passes: &[Rgb16Image],
        transforms: &[Similarity],
        w: u32,
        h: u32,
        feather: f64,
    ) -> Result<(Rgb16Image, Vec<bool>, usize, Vec<u16>)> {
        super::composite_rgb16(passes, transforms, w, h, feather, &mut Vec::new())
    }

    fn composite(
        passes: &[&DynamicImage],
        transforms: &[Similarity],
        w: u32,
        h: u32,
        feather: f64,
    ) -> Result<Composite> {
        super::composite(passes, transforms, w, h, feather, &mut Vec::new())
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

    /// Softening must produce a gradual handover, not leave a one-pixel step,
    /// and must still leave the masks summing to one so the blend stays
    /// energy-preserving.
    #[test]
    fn softening_spreads_the_boundary_and_preserves_the_sum() {
        let (w, h) = (64usize, 32usize);
        let places = placements(
            &[Similarity::IDENTITY, Similarity::IDENTITY],
            &[(w as u32, h as u32), (w as u32, h as u32)],
        );

        // A hard split down the middle, as the cut hands over.
        let mut masks = vec![vec![0.0f32; w * h], vec![0.0f32; w * h]];
        for y in 0..h {
            for x in 0..w {
                let left = x < w / 2;
                masks[0][y * w + x] = if left { 1.0 } else { 0.0 };
                masks[1][y * w + x] = if left { 0.0 } else { 1.0 };
            }
        }

        // Both passes are identity-placed over the whole canvas, so both cover
        // every pixel.
        let per_pass: Vec<Coverage> = places
            .iter()
            .map(|_| {
                let mut coverage = Coverage::new(w * h);
                for slot in 0..w * h {
                    coverage.set(slot);
                }
                coverage
            })
            .collect();

        soften_masks(&mut masks, &per_pass, w as u32, h as u32, 8);

        // Sum-to-one holds everywhere both passes cover.
        for (slot, (first, second)) in masks[0].iter().zip(&masks[1]).enumerate() {
            let total = first + second;
            assert!(
                (total - 1.0).abs() < 1e-3,
                "masks sum to {total} at {slot}, not 1"
            );
        }

        // The handover is now gradual rather than a single-pixel step.
        let row = h / 2;
        let mid = w / 2;
        let intermediate = (mid - 6..mid + 6)
            .filter(|&x| {
                let v = masks[0][row * w + x];
                v > 0.05 && v < 0.95
            })
            .count();
        assert!(
            intermediate >= 4,
            "only {intermediate} pixels are partially blended; the seam is              still effectively a hard step"
        );

        // And it stays a handover: monotonically decreasing left to right.
        let mut previous = f32::INFINITY;
        for x in mid - 6..mid + 6 {
            let value = masks[0][row * w + x];
            assert!(value <= previous + 1e-4, "handover reversed at x={x}");
            previous = value;
        }
    }

    /// Integration guard for the saturation bug, at the level it actually
    /// broke: one pass swallowing the entire overlap.
    ///
    /// This deliberately does *not* assert where the seam lands. Ownership is
    /// decided by minimum cut now, so on featureless input every boundary
    /// position costs the same and any of them is a correct answer. What must
    /// stay true is that both passes end up with a real share — the saturation
    /// bug gave one pass everything it covered.
    #[test]
    fn neither_pass_swallows_a_large_overlap() {
        // 1200-wide passes overlapping by 900px, far beyond the feather span
        // that used to saturate.
        let left = solid8(1200, 400, [40, 40, 40]);
        let right = solid8(1200, 400, [210, 210, 210]);
        let transforms = [Similarity::IDENTITY, Similarity::translation(300.0, 0.0)];

        let (_, _, _, ownership) =
            composite_rgb8(&[left, right], &transforms, 1500, 400, DEFAULT_FEATHER).unwrap();

        let first = ownership.iter().filter(|&&o| o == 1).count();
        let second = ownership.iter().filter(|&&o| o == 2).count();
        let total = (first + second) as f64;
        assert!(total > 0.0, "nothing was assigned");

        let smaller = first.min(second) as f64 / total;
        assert!(
            smaller > 0.15,
            "one pass took {:.0}% of the assigned canvas; the selection has              collapsed onto a pass border",
            (1.0 - smaller) * 100.0
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
