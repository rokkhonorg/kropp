//! Global alignment: fit every overlapping pair, discard passes that don't
//! belong, and resolve the survivors into one shared coordinate frame.
//!
//! This is the Rust stand-in for OpenCV's `leaveBiggestComponent` plus
//! `AffineBasedEstimator`. Both reduce to graph work at the scale kropp
//! operates on — a handful of passes, not a photo library — so it is a
//! union-find and a Prim's walk rather than a dependency.
//!
//! Every tie is broken by index, so the resulting frame assignment is a
//! deterministic function of the inputs.

use anyhow::{Result, bail};

use super::features::Features;
use super::geometry::{Point, Similarity};
use super::matching::{self, DEFAULT_MATCH_CONF, RansacParams};

/// OpenCV's `confidence_threshold`, and the value the previous Python
/// implementation ran with. Pairs scoring below this are treated as
/// non-overlapping.
pub const DEFAULT_CONFIDENCE_THRESHOLD: f64 = 0.5;

/// A refusal valve: a wildly wrong fit can imply an enormous canvas, and
/// allocating it would take the process down. Expressed as a multiple of the
/// summed input area.
const MAX_CANVAS_AREA_FACTOR: f64 = 12.0;

/// A fitted relationship between two passes. `transform` maps points in
/// `from`'s pixel frame into `to`'s.
#[derive(Debug, Clone, PartialEq)]
pub struct PairFit {
    pub from: usize,
    pub to: usize,
    pub transform: Similarity,
    pub confidence: f64,
    /// Correspondences the descriptor matcher proposed for this pair.
    pub matches: usize,
    /// How many of those the fitted transform explained.
    pub inliers: usize,
    /// Mean distance, in pixels, between the inliers and where the transform
    /// puts them. This is the number that says whether a merge will ghost:
    /// under about a pixel is a clean fit, several pixels will show.
    pub residual: f64,
    /// The agreeing correspondences, in each pass's own frame. Bundle
    /// adjustment needs these to reconcile all the fits at once.
    pub inlier_pairs: Vec<(Point, Point)>,
}

/// The resolved layout: which passes take part, where each one lands on the
/// canvas, and how big that canvas is.
#[derive(Debug, Clone)]
pub struct Alignment {
    /// Indices of the passes in the merged component, ascending.
    pub members: Vec<usize>,
    /// The pass whose frame everything else was expressed in.
    pub reference: usize,
    /// Every pairwise fit that was attempted, for diagnostics.
    pub fits: Vec<PairFit>,
    /// For each entry of `members`, the transform from that pass's pixel frame
    /// into canvas coordinates. Composed once here so the compositor resamples
    /// exactly once.
    pub transforms: Vec<Similarity>,
    pub canvas_width: u32,
    pub canvas_height: u32,
}

/// Fit every pass against every other. Order is fixed (`i < j`) and the seeded
/// RANSAC makes each fit reproducible, so the whole edge set is deterministic.
pub fn pairwise_fits(
    features: &[Features],
    match_conf: f32,
    ransac: RansacParams,
) -> Vec<PairFit> {
    let mut fits = Vec::new();
    for i in 0..features.len() {
        for j in (i + 1)..features.len() {
            let matches = matching::match_descriptors(
                &features[j].descriptors,
                &features[i].descriptors,
                match_conf,
            );
            let pairs = matching::correspondences(&matches, &features[j].points, &features[i].points);
            let Some(fit) = matching::robust_estimate(&pairs, ransac) else {
                continue;
            };
            // `robust_estimate` mapped j's points onto i's, so the transform
            // carries j-local coordinates into i-local ones.
            let inlier_pairs: Vec<_> = fit.inliers.iter().map(|&k| pairs[k]).collect();
            fits.push(PairFit {
                from: j,
                to: i,
                transform: fit.transform,
                confidence: fit.confidence,
                matches: pairs.len(),
                inliers: fit.inliers.len(),
                residual: super::geometry::mean_residual(fit.transform, &inlier_pairs),
                inlier_pairs,
            });
        }
    }
    fits
}

/// Union-find over the confident edges, returning the largest connected set of
/// passes. Ties in size are broken toward the group containing the lowest
/// index, so the choice is stable.
pub fn largest_component(
    pass_count: usize,
    fits: &[PairFit],
    min_confidence: f64,
) -> Vec<usize> {
    let mut parent: Vec<usize> = (0..pass_count).collect();

    fn find(parent: &mut [usize], mut node: usize) -> usize {
        while parent[node] != node {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        node
    }

    for fit in fits.iter().filter(|f| f.confidence >= min_confidence) {
        let (a, b) = (find(&mut parent, fit.from), find(&mut parent, fit.to));
        if a != b {
            // Attach to the lower root so grouping is index-stable.
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            parent[hi] = lo;
        }
    }

    let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for node in 0..pass_count {
        let root = find(&mut parent, node);
        groups.entry(root).or_default().push(node);
    }

    groups
        .into_values()
        .max_by_key(|members| (members.len(), std::cmp::Reverse(members[0])))
        .unwrap_or_default()
}

/// The member best connected to the rest, used as the frame everything else is
/// expressed in. Highest summed confidence wins; ties go to the lowest index.
fn reference_member(members: &[usize], fits: &[PairFit], min_confidence: f64) -> usize {
    let mut best = (f64::NEG_INFINITY, usize::MAX);
    for &member in members {
        let total: f64 = fits
            .iter()
            .filter(|f| f.confidence >= min_confidence && (f.from == member || f.to == member))
            .map(|f| f.confidence)
            .sum();
        if total > best.0 {
            best = (total, member);
        }
    }
    best.1
}

/// Walk a maximum-confidence spanning tree out from `reference`, composing
/// each edge onto its parent's transform. The result maps every member's pixel
/// frame into the reference's.
///
/// Returns `None` if the members are not actually all reachable, which would
/// mean `largest_component` and this disagree.
pub fn spanning_transforms(
    members: &[usize],
    fits: &[PairFit],
    reference: usize,
    min_confidence: f64,
) -> Option<Vec<Similarity>> {
    let index_of = |pass: usize| members.iter().position(|&m| m == pass);
    let reference_slot = index_of(reference)?;

    let mut resolved: Vec<Option<Similarity>> = vec![None; members.len()];
    resolved[reference_slot] = Some(Similarity::IDENTITY);
    let mut settled = 1usize;

    while settled < members.len() {
        // Highest-confidence edge joining a settled pass to an unsettled one.
        // Sorting the key by (confidence, from, to) keeps the choice stable
        // when two edges score identically.
        let mut best: Option<(f64, usize, usize, Similarity)> = None;
        for fit in fits.iter().filter(|f| f.confidence >= min_confidence) {
            for (known, unknown, transform) in [
                (fit.to, fit.from, fit.transform),
                (fit.from, fit.to, fit.transform.inverse()),
            ] {
                let (Some(known_slot), Some(unknown_slot)) = (index_of(known), index_of(unknown))
                else {
                    continue;
                };
                if resolved[known_slot].is_none() || resolved[unknown_slot].is_some() {
                    continue;
                }
                // `transform` maps unknown-local into known-local here.
                // Highest confidence wins; an exact tie goes to the lowest
                // index pair, so the tree is the same on every run.
                let better = match best {
                    None => true,
                    Some((best_conf, best_known, best_unknown, _)) => {
                        if fit.confidence != best_conf {
                            fit.confidence > best_conf
                        } else {
                            (known, unknown) < (best_known, best_unknown)
                        }
                    }
                };
                if better {
                    best = Some((fit.confidence, known, unknown, transform));
                }
            }
        }

        let (_, known, unknown, transform) = best?;
        let known_slot = index_of(known)?;
        let unknown_slot = index_of(unknown)?;
        let parent = resolved[known_slot]?;
        resolved[unknown_slot] = Some(parent.compose(transform));
        settled += 1;
    }

    resolved.into_iter().collect()
}

/// Reconcile the whole layout at once, using every confident pair rather than
/// only the spanning-tree edges. Falls back to the input when there is nothing
/// to solve.
fn bundle_adjust(
    members: &[usize],
    transforms: &[Similarity],
    fits: &[PairFit],
    reference: usize,
    min_confidence: f64,
) -> Vec<Similarity> {
    let slot_of = |pass: usize| members.iter().position(|&m| m == pass);
    let Some(reference_slot) = slot_of(reference) else {
        return transforms.to_vec();
    };

    let mut observations = Vec::new();
    for fit in fits.iter().filter(|f| f.confidence >= min_confidence) {
        let (Some(from), Some(to)) = (slot_of(fit.from), slot_of(fit.to)) else {
            continue;
        };
        for &(from_point, to_point) in &fit.inlier_pairs {
            observations.push(super::bundle::Observation {
                from,
                to,
                from_point,
                to_point,
            });
        }
    }
    if observations.is_empty() {
        return transforms.to_vec();
    }
    super::bundle::refine(transforms, &observations, reference_slot)
}

/// The four corners of a `width` x `height` image, mapped through `transform`.
fn warped_corners(transform: Similarity, width: u32, height: u32) -> [Point; 4] {
    let (w, h) = (width as f64, height as f64);
    [
        transform.apply((0.0, 0.0)),
        transform.apply((w, 0.0)),
        transform.apply((w, h)),
        transform.apply((0.0, h)),
    ]
}

/// Shift `transforms` so the union of all warped passes starts at the origin,
/// and report the canvas that union needs.
///
/// `sizes` is indexed in step with `transforms`.
pub fn fit_canvas(
    transforms: &[Similarity],
    sizes: &[(u32, u32)],
) -> Result<(Vec<Similarity>, u32, u32)> {
    if transforms.is_empty() || transforms.len() != sizes.len() {
        bail!("canvas fit needs one size per transform");
    }

    let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for (&transform, &(w, h)) in transforms.iter().zip(sizes) {
        for (x, y) in warped_corners(transform, w, h) {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if !(min_x.is_finite() && min_y.is_finite() && max_x.is_finite() && max_y.is_finite()) {
        bail!("alignment produced a non-finite canvas");
    }

    let width = (max_x - min_x).ceil();
    let height = (max_y - min_y).ceil();
    if width < 1.0 || height < 1.0 {
        bail!("alignment produced an empty canvas");
    }

    // A bad fit can imply an absurd canvas; refuse rather than try to allocate
    // it. Legitimate merges grow roughly with the number of passes.
    let input_area: f64 = sizes.iter().map(|&(w, h)| w as f64 * h as f64).sum();
    if width * height > input_area * MAX_CANVAS_AREA_FACTOR {
        bail!(
            "alignment implies a {width}x{height} canvas, far larger than the inputs; \
             the passes probably do not overlap"
        );
    }

    let shift = Similarity::translation(-min_x, -min_y);
    let shifted = transforms.iter().map(|&t| shift.compose(t)).collect();
    Ok((shifted, width as u32, height as u32))
}

/// Full alignment for a group of passes: pairwise fits, largest confident
/// component, spanning-tree frames, canvas.
pub fn align(
    features: &[Features],
    sizes: &[(u32, u32)],
    min_confidence: f64,
    match_conf: f32,
    ransac: RansacParams,
) -> Result<Alignment> {
    if features.len() < 2 {
        bail!("a merge needs at least two passes");
    }
    if features.len() != sizes.len() {
        bail!("expected one size per pass");
    }

    let fits = pairwise_fits(features, match_conf, ransac);
    let members = largest_component(features.len(), &fits, min_confidence);
    if members.len() < 2 {
        bail!(
            "no two passes overlap confidently enough to merge \
             (needed confidence {min_confidence})"
        );
    }

    let reference = reference_member(&members, &fits, min_confidence);
    let transforms = spanning_transforms(&members, &fits, reference, min_confidence)
        .ok_or_else(|| anyhow::anyhow!("passes did not resolve into one connected frame"))?;

    // The spanning tree places each pass by chaining pairwise fits, so error
    // accumulates along the chain and pairs the tree skipped are never
    // reconciled. Refine every placement against every correspondence at once.
    let transforms = bundle_adjust(&members, &transforms, &fits, reference, min_confidence);

    let member_sizes: Vec<(u32, u32)> = members.iter().map(|&m| sizes[m]).collect();
    let (transforms, canvas_width, canvas_height) = fit_canvas(&transforms, &member_sizes)?;

    Ok(Alignment {
        members,
        reference,
        fits,
        transforms,
        canvas_width,
        canvas_height,
    })
}

/// Convenience default matching the previous implementation's settings.
pub fn align_with_defaults(features: &[Features], sizes: &[(u32, u32)]) -> Result<Alignment> {
    align(
        features,
        sizes,
        DEFAULT_CONFIDENCE_THRESHOLD,
        DEFAULT_MATCH_CONF,
        RansacParams::default(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fit(from: usize, to: usize, tx: f64, ty: f64, confidence: f64) -> PairFit {
        PairFit {
            from,
            to,
            transform: Similarity::translation(tx, ty),
            confidence,
            matches: 0,
            inliers: 0,
            residual: 0.0,
            inlier_pairs: Vec::new(),
        }
    }

    #[test]
    fn largest_component_keeps_the_biggest_confident_group() {
        // 0-1-2 joined confidently; 3-4 joined as a smaller island.
        let fits = vec![
            fit(1, 0, 10.0, 0.0, 0.9),
            fit(2, 1, 10.0, 0.0, 0.8),
            fit(4, 3, 10.0, 0.0, 0.9),
        ];
        assert_eq!(largest_component(5, &fits, 0.5), vec![0, 1, 2]);
    }

    #[test]
    fn largest_component_drops_edges_below_threshold() {
        let fits = vec![fit(1, 0, 10.0, 0.0, 0.9), fit(2, 1, 10.0, 0.0, 0.2)];
        // The 1-2 edge is too weak, so pass 2 is left out.
        assert_eq!(largest_component(3, &fits, 0.5), vec![0, 1]);
    }

    #[test]
    fn largest_component_is_empty_of_pairs_when_nothing_is_confident() {
        let fits = vec![fit(1, 0, 10.0, 0.0, 0.1)];
        assert_eq!(largest_component(2, &fits, 0.5).len(), 1);
    }

    /// Three passes laid left to right; resolving them into one frame must
    /// reproduce the chain of offsets, including the transitive one.
    #[test]
    fn spanning_transforms_compose_along_a_chain() {
        // 1 sits 100 right of 0; 2 sits 100 right of 1.
        let fits = vec![fit(1, 0, 100.0, 0.0, 0.9), fit(2, 1, 100.0, 0.0, 0.9)];
        let members = vec![0, 1, 2];

        let transforms = spanning_transforms(&members, &fits, 0, 0.5).unwrap();

        assert_eq!(transforms[0], Similarity::IDENTITY);
        assert!((transforms[1].tx - 100.0).abs() < 1e-9);
        assert!((transforms[2].tx - 200.0).abs() < 1e-9, "transitive offset");
        assert!(transforms.iter().all(|t| t.ty.abs() < 1e-9));
    }

    /// An edge stored as (from -> to) must still resolve when the tree reaches
    /// it from the other end, via the inverse.
    #[test]
    fn spanning_transforms_traverse_edges_in_reverse() {
        let fits = vec![fit(1, 0, 100.0, 25.0, 0.9)];
        let members = vec![0, 1];

        // Reference is pass 1, so the 1->0 edge must be inverted to place 0.
        let transforms = spanning_transforms(&members, &fits, 1, 0.5).unwrap();

        assert_eq!(transforms[1], Similarity::IDENTITY);
        assert!((transforms[0].tx + 100.0).abs() < 1e-9);
        assert!((transforms[0].ty + 25.0).abs() < 1e-9);
    }

    #[test]
    fn spanning_transforms_reject_unreachable_members() {
        let fits = vec![fit(1, 0, 10.0, 0.0, 0.9)];
        // Pass 2 has no confident edge, so the walk cannot settle it.
        assert!(spanning_transforms(&[0, 1, 2], &fits, 0, 0.5).is_none());
    }

    #[test]
    fn reference_is_the_best_connected_member() {
        // Pass 1 touches both others, so it should anchor the frame.
        let fits = vec![fit(1, 0, 10.0, 0.0, 0.9), fit(2, 1, 10.0, 0.0, 0.9)];
        assert_eq!(reference_member(&[0, 1, 2], &fits, 0.5), 1);
    }

    #[test]
    fn canvas_covers_the_union_and_shifts_to_the_origin() {
        let transforms = vec![
            Similarity::IDENTITY,
            Similarity::translation(100.0, 50.0),
        ];
        let sizes = vec![(200, 200), (200, 200)];

        let (shifted, w, h) = fit_canvas(&transforms, &sizes).unwrap();

        // Union spans x 0..300 and y 0..250.
        assert_eq!((w, h), (300, 250));
        // Nothing was left of or above the origin, so no shift was needed.
        assert!((shifted[0].tx).abs() < 1e-9 && (shifted[0].ty).abs() < 1e-9);
        assert!((shifted[1].tx - 100.0).abs() < 1e-9);
    }

    /// A pass placed up and to the left of the reference must pull everything
    /// right and down rather than being clipped off the canvas.
    #[test]
    fn canvas_shifts_negative_placements_into_view() {
        let transforms = vec![
            Similarity::IDENTITY,
            Similarity::translation(-80.0, -60.0),
        ];
        let sizes = vec![(200, 200), (200, 200)];

        let (shifted, w, h) = fit_canvas(&transforms, &sizes).unwrap();

        assert_eq!((w, h), (280, 260));
        assert!((shifted[0].tx - 80.0).abs() < 1e-9, "reference shifted right");
        assert!((shifted[0].ty - 60.0).abs() < 1e-9, "reference shifted down");
        assert!(shifted[1].tx.abs() < 1e-9 && shifted[1].ty.abs() < 1e-9);
    }

    #[test]
    fn canvas_accounts_for_rotation() {
        // A 45-degree rotation of a square needs a canvas of side w*sqrt(2).
        let transforms = vec![Similarity {
            scale: 1.0,
            theta: std::f64::consts::FRAC_PI_4,
            tx: 0.0,
            ty: 0.0,
        }];
        let (_, w, h) = fit_canvas(&transforms, &[(100, 100)]).unwrap();

        let expected = (100.0 * std::f64::consts::SQRT_2).ceil() as u32;
        assert_eq!((w, h), (expected, expected));
    }

    /// The guard that stops a bad fit from requesting a ruinous allocation.
    #[test]
    fn canvas_refuses_an_implausibly_large_union() {
        let transforms = vec![
            Similarity::IDENTITY,
            Similarity::translation(50_000.0, 50_000.0),
        ];
        let sizes = vec![(200, 200), (200, 200)];

        let err = fit_canvas(&transforms, &sizes).unwrap_err().to_string();
        assert!(err.contains("do not overlap"), "unexpected error: {err}");
    }

    #[test]
    fn canvas_rejects_mismatched_inputs() {
        assert!(fit_canvas(&[], &[]).is_err());
        assert!(fit_canvas(&[Similarity::IDENTITY], &[]).is_err());
    }

    #[test]
    fn align_requires_at_least_two_passes() {
        let err = align_with_defaults(&[Features::default()], &[(10, 10)])
            .unwrap_err()
            .to_string();
        assert!(err.contains("at least two passes"), "got: {err}");
    }

    /// Two edges of equal confidence must resolve the same way every run.
    #[test]
    fn spanning_transforms_break_confidence_ties_deterministically() {
        // 1 and 2 are both reachable from 0 at identical confidence.
        let fits = vec![
            fit(1, 0, 100.0, 0.0, 0.75),
            fit(2, 0, 0.0, 100.0, 0.75),
            fit(2, 1, 7.0, 7.0, 0.75),
        ];
        let members = vec![0, 1, 2];

        let first = spanning_transforms(&members, &fits, 0, 0.5).unwrap();
        for _ in 0..8 {
            assert_eq!(
                spanning_transforms(&members, &fits, 0, 0.5).unwrap(),
                first,
                "tie-break is not stable across runs"
            );
        }
    }

    /// The whole module against real detections: three overlapping windows
    /// onto one master must land in the right places on one canvas.
    #[test]
    fn aligns_three_real_overlapping_passes() {
        use crate::merge::features::{DEFAULT_FEATURE_LIMIT, detect};
        use crate::merge::test_support::{crop, master_texture};

        let master = master_texture(560, 300);
        // Three windows striding right, overlapping by 140px each time.
        let passes = [
            crop(&master, 0, 0, 300, 300),
            crop(&master, 160, 0, 300, 300),
            crop(&master, 260, 0, 300, 300),
        ];

        let features: Vec<_> = passes
            .iter()
            .map(|p| detect(p, Some(DEFAULT_FEATURE_LIMIT)))
            .collect();
        let sizes: Vec<(u32, u32)> = passes.iter().map(|p| (p.width(), p.height())).collect();

        let alignment = align_with_defaults(&features, &sizes).unwrap();

        assert_eq!(alignment.members, vec![0, 1, 2], "all passes should join");

        // Whatever frame was chosen as reference, the *relative* placements
        // must reproduce the 0/160/260 strides.
        let x = |i: usize| alignment.transforms[i].tx;
        assert!((x(1) - x(0) - 160.0).abs() < 2.0, "0->1 stride: {}", x(1) - x(0));
        assert!((x(2) - x(0) - 260.0).abs() < 2.0, "0->2 stride: {}", x(2) - x(0));
        assert!(
            alignment.transforms.iter().all(|t| t.ty.abs() < 2.0),
            "passes should stay on one row"
        );

        // The union spans 560 wide and 300 tall, give or take rounding.
        assert!(
            (alignment.canvas_width as i64 - 560).abs() <= 3,
            "canvas width {}",
            alignment.canvas_width
        );
        assert!(
            (alignment.canvas_height as i64 - 300).abs() <= 3,
            "canvas height {}",
            alignment.canvas_height
        );
    }

    #[test]
    fn align_reports_when_nothing_overlaps() {
        // Featureless passes produce no matches, so no confident component.
        let features = vec![Features::default(), Features::default()];
        let err = align_with_defaults(&features, &[(10, 10), (10, 10)])
            .unwrap_err()
            .to_string();
        assert!(err.contains("no two passes overlap"), "got: {err}");
    }
}
