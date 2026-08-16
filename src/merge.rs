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
    /// How many passes were detected concurrently.
    pub detection_jobs: usize,
    /// Wall-clock per stage, in the order they ran. Printed under `--debug`;
    /// the only way to tell which stage a slow merge is actually spending its
    /// time in, since the costs are wildly uneven and not where you would guess.
    pub timings: Vec<(&'static str, std::time::Duration)>,
}

/// Time `body`, push the result onto `into` under `label`, and return it.
pub(crate) fn timed<T>(
    into: &mut Vec<(&'static str, std::time::Duration)>,
    label: &'static str,
    body: impl FnOnce() -> T,
) -> T {
    let start = std::time::Instant::now();
    let value = body();
    into.push((label, start.elapsed()));
    value
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

/// Transient bytes one SIFT detection holds, per pixel of the pass it runs on.
///
/// Measured at roughly 8 GB for a 35.8-megapixel pass, which is about what the
/// algorithm implies: `sift-features` doubles the image before building its
/// scale space, so the base octave alone covers four times the input area, held
/// as `f32` across the Gaussian and difference-of-Gaussian layers, and the
/// octaves above it add most of another third.
///
/// This is why detection is not simply run on every pass at once. It is the
/// memory high-water mark of the whole merge — larger than the compositor's —
/// and it multiplies by the number of detections in flight.
const SIFT_BYTES_PER_PIXEL: u64 = 240;

/// Budget to assume when the platform will not say how much memory is free.
/// Low enough that a full-platen pass gets one detection at a time, which is
/// what the merge did before it could run them in parallel at all.
const FALLBACK_MEMORY_BUDGET: u64 = 8 << 30;

/// Share of free memory detection may plan around, leaving the rest for the
/// decoded passes, the compositor's float planes, and everything else running.
const MEMORY_HEADROOM: u64 = 4;

/// Free physical memory in bytes, or `None` on a platform this cannot ask.
///
/// Worth asking rather than assuming a constant: the safe fixed budget is one
/// detection at a time, which gives up the parallelism entirely on the large
/// machines where it is free, while any budget generous enough to help them
/// would push a 16 GB laptop into swap — where it would be far slower than the
/// serial version it replaced.
fn free_memory() -> Option<u64> {
    #[cfg(windows)]
    {
        // GlobalMemoryStatusEx, declared by hand rather than pulled in as a
        // dependency: the struct layout and the call have been stable since
        // Windows 2000, and this is the only OS query kropp makes.
        #[repr(C)]
        struct MemoryStatusEx {
            length: u32,
            memory_load: u32,
            total_phys: u64,
            avail_phys: u64,
            total_page_file: u64,
            avail_page_file: u64,
            total_virtual: u64,
            avail_virtual: u64,
            avail_extended_virtual: u64,
        }

        unsafe extern "system" {
            fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
        }

        let mut status = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        // SAFETY: `status` is a live, correctly sized MEMORYSTATUSEX with
        // `length` set as the API requires; the call only writes into it.
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
        (ok != 0).then_some(status.avail_phys)
    }

    #[cfg(target_os = "linux")]
    {
        // MemAvailable is the kernel's own estimate of what a new allocation
        // can have without swapping, which is exactly the question here.
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        meminfo
            .lines()
            .find_map(|line| line.strip_prefix("MemAvailable:"))
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|kb| kb.parse::<u64>().ok())
            .map(|kb| kb * 1024)
    }

    #[cfg(not(any(windows, target_os = "linux")))]
    {
        None
    }
}

/// How many detections to run at once.
///
/// Bounded by cores, by the number of passes, and by memory. The last is the
/// binding constraint on real scans and the reason this is not just
/// `available_parallelism`: detection is cheap in threads and expensive in
/// gigabytes, so on a full-platen group the memory ceiling is reached long
/// before the core count is.
fn detection_jobs(sizes: &[(u32, u32)], requested: usize) -> usize {
    if requested > 0 {
        return requested.min(sizes.len()).max(1);
    }
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let ceiling = sizes.len().min(cores).max(1);

    // Budget against the largest pass, since any of them could be in flight.
    let largest = sizes
        .iter()
        .map(|&(w, h)| w as u64 * h as u64 * SIFT_BYTES_PER_PIXEL)
        .max()
        .unwrap_or(u64::MAX)
        .max(1);
    let budget = free_memory()
        .map(|free| free / MEMORY_HEADROOM * (MEMORY_HEADROOM - 1))
        .unwrap_or(FALLBACK_MEMORY_BUDGET);
    let affordable = (budget / largest) as usize;
    affordable.clamp(1, ceiling)
}

/// Detect features on every pass, up to `jobs` at a time.
///
/// Results are written to their own pass's slot rather than collected in
/// completion order, so the feature set — and therefore the whole merge — is
/// identical however the threads interleave.
fn detect_all(
    passes: &[&DynamicImage],
    masks: Option<&[Vec<bool>]>,
    jobs: usize,
) -> Vec<features::Features> {
    let detect_one = |index: usize| {
        let pass = passes[index];
        let luma = features::detection_luma(pass);
        let limit = Some(features::feature_limit(pass.width(), pass.height()));
        match masks {
            Some(masks) => features::detect_masked(&luma, &masks[index], limit),
            None => features::detect(&luma, limit),
        }
    };

    if jobs <= 1 {
        return (0..passes.len()).map(detect_one).collect();
    }

    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: std::sync::Mutex<Vec<Option<features::Features>>> =
        std::sync::Mutex::new((0..passes.len()).map(|_| None).collect());

    std::thread::scope(|scope| {
        for _ in 0..jobs.min(passes.len()) {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if index >= passes.len() {
                        break;
                    }
                    let found = detect_one(index);
                    // Held only to hand over a finished result, never across
                    // the detection itself.
                    slots.lock().expect("detection slots")[index] = Some(found);
                }
            });
        }
    });

    slots
        .into_inner()
        .expect("detection slots")
        .into_iter()
        .map(|found| found.expect("every pass detected"))
        .collect()
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
///
/// `jobs` caps how many passes are detected concurrently; 0 picks a count from
/// the cores available and the memory the passes imply.
pub fn merge(
    passes: &[&DynamicImage],
    masks: Option<&[Vec<bool>]>,
    jobs: usize,
) -> Result<Merged> {
    if passes.len() < 2 {
        bail!("a merge needs at least two passes");
    }
    if let Some(masks) = masks
        && passes.len() != masks.len()
    {
        bail!("expected one mask per pass");
    }

    let mut timings = Vec::new();
    let sizes: Vec<(u32, u32)> = passes.iter().map(|p| (p.width(), p.height())).collect();
    let jobs = detection_jobs(&sizes, jobs);
    let features: Vec<features::Features> =
        timed(&mut timings, "detect", || detect_all(passes, masks, jobs));

    let alignment = timed(&mut timings, "align", || {
        align::align_with_defaults(&features, &sizes)
    })?;
    let members: Vec<&DynamicImage> = alignment.members.iter().map(|&m| passes[m]).collect();
    let composed = composite::composite(
        &members,
        &alignment.transforms,
        alignment.canvas_width,
        alignment.canvas_height,
        composite::DEFAULT_FEATHER,
        &mut timings,
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
            detection_jobs: jobs,
            timings,
        },
        members: alignment.members,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Full-platen passes: whatever the core count, memory is what decides, and
    /// there must always be at least one job.
    #[test]
    fn detection_jobs_never_reports_less_than_one() {
        let huge = vec![(20_000, 20_000); 4];
        assert!(detection_jobs(&huge, 0) >= 1);
        assert!(detection_jobs(&[], 0) >= 1);
    }

    /// Never more jobs than there are passes to detect — spare threads would
    /// only sit on the queue.
    #[test]
    fn detection_jobs_is_capped_by_the_pass_count() {
        let small = vec![(64, 64); 2];
        assert!(detection_jobs(&small, 0) <= 2);
        assert_eq!(detection_jobs(&small, 16), 2);
    }

    /// An explicit request wins over the memory estimate, so a user who knows
    /// their machine can have the cores. It is still clamped to something
    /// runnable.
    #[test]
    fn an_explicit_request_overrides_the_estimate() {
        let passes = vec![(5100, 7019); 4];
        assert_eq!(detection_jobs(&passes, 3), 3);
        assert_eq!(detection_jobs(&passes, 99), 4);
    }

    /// Bigger passes must never buy *more* concurrency than smaller ones: the
    /// budget divides by the per-pass estimate, and a sign or overflow slip
    /// there would invert it.
    #[test]
    fn larger_passes_never_get_more_concurrency() {
        let small = detection_jobs(&[(512, 512); 8], 0);
        let large = detection_jobs(&[(9000, 12000); 8], 0);
        assert!(
            large <= small,
            "{large} jobs for large passes beats {small} for small ones"
        );
    }
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
