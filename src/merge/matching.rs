//! Descriptor matching and robust transform estimation for a single pair of
//! passes.
//!
//! Deliberately dependency-free and generic over plain descriptor rows, so the
//! SIFT crate stays at the edge of the module and the robust-estimation core
//! can be tested against synthetic data. The RANSAC driver is seeded and its
//! sampling is fully determined by that seed, which is what lets a re-run of
//! the same merge produce a byte-identical result — something OpenCV's
//! stitcher does not guarantee.
//!
//! Numeric conventions follow OpenCV's `BestOf2NearestMatcher` so results stay
//! comparable with the previous Python implementation: the ratio test uses
//! `(1 - match_conf)`, matching is two-way with a union of both directions, and
//! pair confidence is `inliers / (8 + 0.3 * matches)`.

use super::geometry::{self, Point, ScaleModel, Similarity};

/// A SIFT descriptor row. `sift-features` produces these as rows of an
/// `Array2<u8>` with 128 columns; the adapter converts, this module never sees
/// ndarray.
pub type Descriptor = [u8; 128];

/// OpenCV's default `match_conf` for SIFT (0.3 is the ORB default). The ratio
/// test threshold is `1 - match_conf`, so this is a strict 0.35.
pub const DEFAULT_MATCH_CONF: f32 = 0.65;

/// OpenCV's default `ransacReprojThreshold` for partial-affine estimation.
pub const DEFAULT_INLIER_THRESHOLD: f64 = 3.0;

/// A descriptor correspondence: indices into the two keypoint lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Match {
    pub from: usize,
    pub to: usize,
}

#[derive(Debug, Clone)]
pub struct RobustFit {
    pub transform: Similarity,
    /// Indices into the `matches` slice that agreed with `transform`.
    pub inliers: Vec<usize>,
    /// OpenCV's pairwise confidence. The subsetter compares this against its
    /// threshold (0.5 in the previous implementation) to decide whether two
    /// passes overlap at all.
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct RansacParams {
    pub inlier_threshold: f64,
    pub max_iters: u32,
    pub scale_model: ScaleModel,
    /// Fixed seed; the same inputs and seed always yield the same fit.
    pub seed: u64,
}

impl Default for RansacParams {
    fn default() -> Self {
        Self {
            inlier_threshold: DEFAULT_INLIER_THRESHOLD,
            max_iters: 2000,
            // Four degrees of freedom, as OpenCV's partial-affine model uses.
            // A flatbed's passes ought to share a DPI exactly, which argues for
            // pinning scale to 1 — but the previous implementation solved for
            // it and aligned well, and a pinned scale cannot absorb even a tiny
            // magnification difference: it spreads the error across the whole
            // overlap instead.
            scale_model: ScaleModel::Free,
            seed: 0x5EED_5C4Fu64,
        }
    }
}

/// Squared L2 distance between descriptors. Each of the 128 lanes contributes
/// at most 255² so the total cannot exceed 8_323_200 and stays inside u32.
fn descriptor_distance_sq(a: &Descriptor, b: &Descriptor) -> u32 {
    let mut total = 0u32;
    for i in 0..128 {
        let diff = a[i] as i32 - b[i] as i32;
        total += (diff * diff) as u32;
    }
    total
}

/// Best and second-best match for `query` over `haystack`, by squared
/// distance. Returns `None` when there are fewer than two candidates, since the
/// ratio test needs a runner-up to compare against.
fn two_nearest(query: &Descriptor, haystack: &[Descriptor]) -> Option<(usize, u32, u32)> {
    if haystack.len() < 2 {
        return None;
    }
    let (mut best_idx, mut best, mut second) = (0usize, u32::MAX, u32::MAX);
    for (idx, candidate) in haystack.iter().enumerate() {
        let dist = descriptor_distance_sq(query, candidate);
        if dist < best {
            second = best;
            best = dist;
            best_idx = idx;
        } else if dist < second {
            second = dist;
        }
    }
    Some((best_idx, best, second))
}

/// One direction of the ratio test. `d0 < (1 - match_conf) * d1`, on squared
/// distances, so the ratio is squared too.
fn ratio_matches(from: &[Descriptor], to: &[Descriptor], match_conf: f32) -> Vec<Match> {
    let ratio = (1.0 - match_conf as f64).max(0.0);
    let ratio_sq = ratio * ratio;
    let mut matches = Vec::new();
    for (idx, query) in from.iter().enumerate() {
        let Some((best_idx, best, second)) = two_nearest(query, to) else {
            continue;
        };
        if (best as f64) < ratio_sq * second as f64 {
            matches.push(Match {
                from: idx,
                to: best_idx,
            });
        }
    }
    matches
}

/// Two-way descriptor matching, mirroring `BestOf2NearestMatcher`: run the
/// ratio test in both directions and take the **union**, dropping any reversed
/// pair already present. Brute force is intentional — a few hundred features
/// over a handful of passes does not justify an index.
pub fn match_descriptors(from: &[Descriptor], to: &[Descriptor], match_conf: f32) -> Vec<Match> {
    let mut matches = ratio_matches(from, to, match_conf);
    for reverse in ratio_matches(to, from, match_conf) {
        let flipped = Match {
            from: reverse.to,
            to: reverse.from,
        };
        if !matches.contains(&flipped) {
            matches.push(flipped);
        }
    }
    matches
}

/// xorshift64*, inlined to keep the sampling deterministic without depending on
/// a particular `rand` version's generator or its seeding behaviour.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // A zero state is a fixed point for xorshift; nudge it off.
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

/// OpenCV's pairwise confidence, including its cap: a score above 3 means the
/// images are near-duplicates rather than genuine neighbours, and is zeroed.
fn confidence(inliers: usize, total_matches: usize) -> f64 {
    let raw = inliers as f64 / (8.0 + 0.3 * total_matches as f64);
    if raw > 3.0 { 0.0 } else { raw }
}

/// Fit a similarity to `correspondences` by RANSAC, then refit on the consensus
/// set. Returns `None` when fewer than two correspondences exist or no sample
/// produced a usable transform.
pub fn robust_estimate(
    correspondences: &[(Point, Point)],
    params: RansacParams,
) -> Option<RobustFit> {
    if correspondences.len() < 2 {
        return None;
    }
    // Two correspondences determine the transform exactly, so a 2-point sample
    // is both the minimum and the right RANSAC sample size.
    if correspondences.len() == 2 {
        let transform = geometry::estimate(correspondences, params.scale_model)?;
        return Some(RobustFit {
            transform,
            inliers: vec![0, 1],
            confidence: confidence(2, 2),
        });
    }

    let mut rng = Rng::new(params.seed);
    let threshold_sq = params.inlier_threshold * params.inlier_threshold;
    let mut best_inliers: Vec<usize> = Vec::new();

    for _ in 0..params.max_iters {
        let a = rng.below(correspondences.len());
        let mut b = rng.below(correspondences.len());
        if a == b {
            b = (b + 1) % correspondences.len();
        }
        let sample = [correspondences[a], correspondences[b]];
        let Some(candidate) = geometry::estimate(&sample, params.scale_model) else {
            continue;
        };

        let inliers: Vec<usize> = correspondences
            .iter()
            .enumerate()
            .filter(|(_, pair)| {
                let (from, to) = **pair;
                let (mx, my) = candidate.apply(from);
                let (dx, dy) = (mx - to.0, my - to.1);
                dx * dx + dy * dy <= threshold_sq
            })
            .map(|(idx, _)| idx)
            .collect();

        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
            // Everything agrees; no later sample can beat this.
            if best_inliers.len() == correspondences.len() {
                break;
            }
        }
    }

    if best_inliers.len() < 2 {
        return None;
    }

    // Refit over the whole consensus set — the 2-point sample only located it.
    let consensus: Vec<(Point, Point)> = best_inliers.iter().map(|&i| correspondences[i]).collect();
    let transform = geometry::estimate(&consensus, params.scale_model)?;
    Some(RobustFit {
        confidence: confidence(best_inliers.len(), correspondences.len()),
        transform,
        inliers: best_inliers,
    })
}

/// Pair matched keypoint indices with their coordinates, the input
/// `robust_estimate` wants. Matches indexing outside either keypoint list are
/// dropped rather than panicking.
pub fn correspondences(
    matches: &[Match],
    from_points: &[Point],
    to_points: &[Point],
) -> Vec<(Point, Point)> {
    matches
        .iter()
        .filter_map(|m| Some((*from_points.get(m.from)?, *to_points.get(m.to)?)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A descriptor that is `value` in every lane except lane 0, which carries
    /// `tag` — enough to make each one distinctly nearest to its twin.
    fn descriptor(tag: u8, value: u8) -> Descriptor {
        let mut d = [value; 128];
        d[0] = tag;
        d
    }

    fn distinct_descriptors(n: usize) -> Vec<Descriptor> {
        (0..n).map(|i| descriptor(i as u8 * 20, 0)).collect()
    }

    #[test]
    fn descriptor_distance_is_zero_for_identical_and_symmetric() {
        let a = descriptor(10, 3);
        let b = descriptor(200, 90);
        assert_eq!(descriptor_distance_sq(&a, &a), 0);
        assert_eq!(descriptor_distance_sq(&a, &b), descriptor_distance_sq(&b, &a));
    }

    /// The maximum possible distance must not overflow the u32 accumulator.
    #[test]
    fn descriptor_distance_does_not_overflow_at_extremes() {
        let low = [0u8; 128];
        let high = [255u8; 128];
        assert_eq!(descriptor_distance_sq(&low, &high), 255 * 255 * 128);
    }

    #[test]
    fn matches_identical_descriptor_sets_pairwise() {
        let descriptors = distinct_descriptors(5);
        let matches = match_descriptors(&descriptors, &descriptors, DEFAULT_MATCH_CONF);

        assert_eq!(matches.len(), 5);
        for m in &matches {
            assert_eq!(m.from, m.to, "each descriptor should match its twin");
        }
    }

    /// The union of both directions must not contain the same pair twice.
    #[test]
    fn two_way_matching_deduplicates_reversed_pairs() {
        let descriptors = distinct_descriptors(4);
        let matches = match_descriptors(&descriptors, &descriptors, DEFAULT_MATCH_CONF);

        let mut seen = matches.clone();
        seen.sort_by_key(|m| (m.from, m.to));
        seen.dedup();
        assert_eq!(seen.len(), matches.len(), "duplicate pairs in union");
    }

    #[test]
    fn ratio_test_rejects_ambiguous_matches() {
        // Two near-identical candidates make the best/second ratio ~1, which
        // the strict SIFT threshold must reject.
        let query = vec![descriptor(100, 0)];
        let ambiguous = vec![descriptor(101, 0), descriptor(102, 0)];
        assert!(ratio_matches(&query, &ambiguous, DEFAULT_MATCH_CONF).is_empty());

        // A clear winner passes.
        let distinct = vec![descriptor(100, 0), descriptor(255, 200)];
        assert_eq!(ratio_matches(&query, &distinct, DEFAULT_MATCH_CONF).len(), 1);
    }

    #[test]
    fn single_candidate_cannot_pass_ratio_test() {
        let query = vec![descriptor(10, 0)];
        let only_one = vec![descriptor(10, 0)];
        assert!(ratio_matches(&query, &only_one, DEFAULT_MATCH_CONF).is_empty());
    }

    fn shifted(points: &[Point], t: Similarity) -> Vec<(Point, Point)> {
        points.iter().map(|&p| (p, t.apply(p))).collect()
    }

    const GRID: [Point; 8] = [
        (0.0, 0.0),
        (50.0, 0.0),
        (100.0, 0.0),
        (0.0, 50.0),
        (50.0, 50.0),
        (100.0, 50.0),
        (25.0, 90.0),
        (75.0, 90.0),
    ];

    #[test]
    fn ransac_recovers_transform_from_clean_correspondences() {
        let truth = Similarity {
            scale: 1.0,
            theta: 0.05,
            tx: 40.0,
            ty: -15.0,
        };
        let fit = robust_estimate(&shifted(&GRID, truth), RansacParams::default()).unwrap();

        assert_eq!(fit.inliers.len(), GRID.len());
        assert!((fit.transform.tx - truth.tx).abs() < 1e-6);
        assert!((fit.transform.ty - truth.ty).abs() < 1e-6);
        assert!((fit.transform.theta - truth.theta).abs() < 1e-6);
    }

    /// The point of RANSAC: a minority of wild correspondences must not drag
    /// the fit, the way a plain least-squares solve would.
    #[test]
    fn ransac_rejects_outliers() {
        let truth = Similarity::translation(30.0, 20.0);
        let mut correspondences = shifted(&GRID, truth);
        correspondences.push(((10.0, 10.0), (900.0, -700.0)));
        correspondences.push(((20.0, 20.0), (-500.0, 400.0)));

        let fit = robust_estimate(&correspondences, RansacParams::default()).unwrap();

        assert_eq!(fit.inliers.len(), GRID.len(), "outliers were absorbed");
        assert!((fit.transform.tx - 30.0).abs() < 1e-6);
        assert!((fit.transform.ty - 20.0).abs() < 1e-6);

        // A plain fit over everything would be visibly dragged off.
        let naive = geometry::estimate(&correspondences, ScaleModel::Fixed).unwrap();
        assert!(
            (naive.tx - 30.0).abs() > 10.0,
            "test is not exercising outlier rejection"
        );
    }

    /// Determinism is the reason for the explicit seed; same input, same seed,
    /// same answer, every time.
    #[test]
    fn ransac_is_deterministic_for_a_given_seed() {
        let truth = Similarity {
            scale: 1.0,
            theta: 0.02,
            tx: 12.0,
            ty: 7.0,
        };
        let mut correspondences = shifted(&GRID, truth);
        correspondences.push(((5.0, 5.0), (400.0, 400.0)));

        let a = robust_estimate(&correspondences, RansacParams::default()).unwrap();
        let b = robust_estimate(&correspondences, RansacParams::default()).unwrap();

        assert_eq!(a.transform, b.transform);
        assert_eq!(a.inliers, b.inliers);
    }

    #[test]
    fn ransac_needs_at_least_two_correspondences() {
        assert!(robust_estimate(&[], RansacParams::default()).is_none());
        assert!(robust_estimate(&[((0.0, 0.0), (1.0, 1.0))], RansacParams::default()).is_none());
    }

    /// Confidence follows OpenCV's formula, and its near-duplicate cap fires
    /// above 3 so an image never matches itself into the graph.
    #[test]
    fn confidence_matches_opencv_formula_and_cap() {
        assert!((confidence(20, 40) - 20.0 / (8.0 + 12.0)).abs() < 1e-12);
        assert_eq!(confidence(1000, 1), 0.0, "near-duplicate cap should fire");
    }

    #[test]
    fn correspondences_drop_out_of_range_matches() {
        let from_points = vec![(0.0, 0.0), (1.0, 1.0)];
        let to_points = vec![(5.0, 5.0)];
        let matches = vec![
            Match { from: 0, to: 0 },
            Match { from: 1, to: 9 },
            Match { from: 9, to: 0 },
        ];

        let paired = correspondences(&matches, &from_points, &to_points);
        assert_eq!(paired, vec![((0.0, 0.0), (5.0, 5.0))]);
    }
}
