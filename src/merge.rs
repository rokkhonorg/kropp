//! Photomerge: align and composite several overlapping flatbed passes of one
//! oversized original into a single image.
//!
//! The pipeline mirrors OpenCV's stitching module, but restricted to what a
//! flatbed actually needs. Passes are coplanar, share a DPI, and are lit
//! identically, so the alignment is a similarity rather than a homography,
//! there is no exposure to compensate, and no wave correction to apply.
//! Everything runs in Rust so the merge keeps its input bit depth end to end —
//! OpenCV's multi-band blender composites through signed 16-bit and would clip
//! the top half of a 16-bit scan's range.
//!
//! Stages, in order:
//!
//! 1. `geometry` — the similarity transform and its closed-form estimator.
//! 2. `matching` — descriptor matching and seeded RANSAC for one pair.
//! 3. `features` — the SIFT adapter, and the mask that keeps platen artifacts
//!    out of the alignment.
//! 4. `align` — pairwise fits, largest confident component, and one shared
//!    coordinate frame with the canvas that holds it.
//! 5. `seam` — content-aware boundary between overlapping passes.
//! 6. `blend` — Laplacian-pyramid multi-band blending.
//! 7. `composite` — one inverse-mapped resample per output pixel, multi-band
//!    blended across the overlaps, at the input's own bit depth.

// The stages land one at a time and each is fully exercised by its own tests
// before the one above it exists to call it. Drop this once the compositor
// wires the whole chain into the pipeline.
#![allow(dead_code)]

pub mod align;
pub mod blend;
pub mod bundle;
pub mod composite;
pub mod features;
pub mod geometry;
pub mod matching;
pub mod maxflow;
pub mod seam;

use anyhow::{Result, bail};
use image::DynamicImage;

/// How a merge was arrived at, for `--debug`. The number that matters is
/// `PairFit::residual`: it is the mean pixel error of each accepted fit, and
/// therefore how much ghosting the blender has to hide.
pub struct MergeReport {
    /// Per input pass: dimensions, and how many keypoints survived masking.
    pub passes: Vec<((u32, u32), usize)>,
    pub fits: Vec<align::PairFit>,
    pub reference: usize,
    /// Canvas placement per member, in `members` order.
    pub transforms: Vec<geometry::Similarity>,
    pub canvas: (u32, u32),
    /// Pyramid bands used; 1 means it fell back to feathering.
    pub levels: usize,
    /// Whether keypoints were confined to the foreground mask.
    pub masked_features: bool,
}

/// A completed merge.
pub struct Merged {
    pub image: DynamicImage,
    pub report: MergeReport,
    /// Which pass supplied each canvas pixel: 0 uncovered, else the index into
    /// `members` plus one. Feed to `composite::ownership_overlay` to see it.
    pub ownership: Vec<u16>,
    /// Indices of the input passes that made it into the canvas. Anything
    /// missing here did not overlap the rest confidently enough — usually a
    /// stray file in the group rather than a geometry problem, so the caller
    /// reports it rather than failing.
    pub members: Vec<usize>,
}

/// Merge decoded passes into one image.
///
/// `masks`, when given, confines keypoints to foreground pixels so platen
/// artifacts — bed edges, dust, lid shadow, all fixed in scanner coordinates
/// and therefore identical across passes — cannot match each other into a false
/// zero-displacement alignment.
///
/// It defaults to off, because the previous implementation masked nothing and
/// aligned well, while masking discards every keypoint the segmenter did not
/// call foreground. On a tight or imperfect mask that can starve the fit of
/// exactly the correspondences it needs. Taking masks as an argument keeps this
/// independent of which extractor produced them.
pub fn merge(passes: &[&DynamicImage], masks: Option<&[Vec<bool>]>) -> Result<Merged> {
    if passes.len() < 2 {
        bail!("a merge needs at least two passes");
    }
    if let Some(masks) = masks
        && passes.len() != masks.len()
    {
        bail!("expected one mask per pass");
    }

    let sizes: Vec<(u32, u32)> = passes.iter().map(|p| (p.width(), p.height())).collect();
    let features: Vec<features::Features> = passes
        .iter()
        .enumerate()
        .map(|(index, pass)| {
            let luma = features::detection_luma(pass);
            let limit = Some(features::feature_limit(pass.width(), pass.height()));
            match masks {
                Some(masks) => features::detect_masked(&luma, &masks[index], limit),
                None => features::detect(&luma, limit),
            }
        })
        .collect();

    let alignment = align::align_with_defaults(&features, &sizes)?;
    let members: Vec<&DynamicImage> = alignment.members.iter().map(|&m| passes[m]).collect();
    let composed = composite::composite(
        &members,
        &alignment.transforms,
        alignment.canvas_width,
        alignment.canvas_height,
        composite::DEFAULT_FEATHER,
    )?;

    Ok(Merged {
        image: composed.image,
        ownership: composed.ownership,
        report: MergeReport {
            passes: sizes
                .iter()
                .copied()
                .zip(features.iter().map(|f| f.len()))
                .collect(),
            fits: alignment.fits,
            reference: alignment.reference,
            transforms: alignment.transforms,
            canvas: (alignment.canvas_width, alignment.canvas_height),
            levels: composed.levels,
            masked_features: masks.is_some(),
        },
        members: alignment.members,
    })
}

/// Synthetic passes shared by the merge tests. Kept here rather than in one
/// module's test block so alignment and detection exercise the same fixture.
#[cfg(test)]
pub mod test_support {
    use image::{GenericImageView, GrayImage, Luma};

    /// Deterministic LCG, so the texture is identical on every run and any
    /// failure is reproducible.
    pub struct Lcg(pub u64);

    impl Lcg {
        pub fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }

        pub fn range(&mut self, lo: u32, hi: u32) -> u32 {
            lo + (self.next() % (hi - lo) as u64) as u32
        }
    }

    /// A textured "scan" of blobs at varied positions, sizes, and gray levels.
    /// A flat or repetitive field gives SIFT nothing distinctive to key on, and
    /// the strict 0.35 ratio test would reject the ambiguous matches that
    /// result — so the variation is load-bearing, not decoration.
    pub fn master_texture(width: u32, height: u32) -> GrayImage {
        let mut img = GrayImage::from_pixel(width, height, Luma([235]));
        let mut rng = Lcg(0xC0FFEE);
        for _ in 0..90 {
            let bw = rng.range(6, 26);
            let bh = rng.range(6, 26);
            let x0 = rng.range(0, width - bw);
            let y0 = rng.range(0, height - bh);
            let level = rng.range(10, 190) as u8;
            for y in y0..y0 + bh {
                for x in x0..x0 + bw {
                    img.put_pixel(x, y, Luma([level]));
                }
            }
        }
        img
    }

    pub fn crop(img: &GrayImage, x: u32, y: u32, w: u32, h: u32) -> GrayImage {
        img.view(x, y, w, h).to_image()
    }
}
