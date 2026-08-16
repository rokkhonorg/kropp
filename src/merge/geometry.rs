//! The 2D transform used to align overlapping flatbed passes, and its
//! closed-form estimator.
//!
//! Passes of one oversized original are coplanar and come off the same platen,
//! so the mapping between two of them is a similarity — rotation, translation,
//! and (at most) a uniform scale. At a fixed scanner DPI the scale is known to
//! be exactly 1, which drops the problem to three degrees of freedom and makes
//! the least-squares fit a handful of dot products. That is why nothing here
//! needs an SVD, a matrix library, or an iterative solver.

/// A point in image coordinates. Matches the `(f64, f64)` corner convention
/// the rest of kropp already uses.
pub type Point = (f64, f64);

/// Whether the estimator solves for a uniform scale or pins it to 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleModel {
    /// Scanner DPI is identical across passes, so scale is exactly 1 (3 DOF).
    /// The right default for a flatbed.
    Fixed,
    /// Solve for a uniform scale as well (4 DOF), matching OpenCV's partial
    /// affine model. Use when the passes might not share a resolution.
    Free,
}

/// A uniform-scale 2D similarity: scale, then rotate about the origin, then
/// translate. Composing these is closed, so a chain of pairwise alignments
/// collapses into one transform per image and the compositor resamples once.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Similarity {
    pub scale: f64,
    /// Rotation in radians, counter-clockwise in a y-down image frame.
    pub theta: f64,
    pub tx: f64,
    pub ty: f64,
}

impl Similarity {
    pub const IDENTITY: Self = Self {
        scale: 1.0,
        theta: 0.0,
        tx: 0.0,
        ty: 0.0,
    };

    pub fn translation(tx: f64, ty: f64) -> Self {
        Self {
            scale: 1.0,
            theta: 0.0,
            tx,
            ty,
        }
    }

    /// Map a point through this transform.
    pub fn apply(self, (x, y): Point) -> Point {
        let (sin, cos) = self.theta.sin_cos();
        let (sx, sy) = (self.scale * x, self.scale * y);
        (sx * cos - sy * sin + self.tx, sx * sin + sy * cos + self.ty)
    }

    /// The transform that applies `self` after `inner` — i.e. `self ∘ inner`.
    /// Walking a spanning tree of pairwise alignments is repeated composition.
    pub fn compose(self, inner: Self) -> Self {
        let (tx, ty) = self.apply((inner.tx, inner.ty));
        Self {
            scale: self.scale * inner.scale,
            theta: self.theta + inner.theta,
            tx,
            ty,
        }
    }

    /// The inverse mapping. A spanning-tree edge traversed against its
    /// recorded direction needs this.
    pub fn inverse(self) -> Self {
        let inv_scale = 1.0 / self.scale;
        let inv_theta = -self.theta;
        let (sin, cos) = inv_theta.sin_cos();
        // -R(-theta) * (1/s) * t
        let (sx, sy) = (inv_scale * self.tx, inv_scale * self.ty);
        Self {
            scale: inv_scale,
            theta: inv_theta,
            tx: -(sx * cos - sy * sin),
            ty: -(sx * sin + sy * cos),
        }
    }

    /// The 6 coefficients of the equivalent affine matrix, row-major:
    /// `[a, b, tx, c, d, ty]` for `x' = a·x + b·y + tx`, `y' = c·x + d·y + ty`.
    /// This is the form `imageproc`'s `Projection` wants.
    pub fn to_affine_coeffs(self) -> [f64; 6] {
        let (sin, cos) = self.theta.sin_cos();
        let (a, b) = (self.scale * cos, -self.scale * sin);
        let (c, d) = (self.scale * sin, self.scale * cos);
        [a, b, self.tx, c, d, self.ty]
    }
}

/// Least-squares fit of the similarity mapping `from` onto `to`, over point
/// correspondences. This is the 2D Umeyama solution, which for a uniform-scale
/// similarity reduces to a closed form in two accumulated sums:
///
/// - `dot` — Σ centred_from · centred_to, which the rotation's cosine follows
/// - `cross` — Σ centred_from × centred_to, which its sine follows
///
/// Returns `None` when the correspondences cannot determine a transform: fewer
/// than two of them, or all `from` points coincident (zero spread, so rotation
/// is unconstrained).
pub fn estimate(correspondences: &[(Point, Point)], scale_model: ScaleModel) -> Option<Similarity> {
    // 4 DOF (or 3 with scale fixed); each correspondence supplies 2 equations.
    if correspondences.len() < 2 {
        return None;
    }
    let n = correspondences.len() as f64;

    let (mut from_cx, mut from_cy, mut to_cx, mut to_cy) = (0.0, 0.0, 0.0, 0.0);
    for &((fx, fy), (tx, ty)) in correspondences {
        from_cx += fx;
        from_cy += fy;
        to_cx += tx;
        to_cy += ty;
    }
    from_cx /= n;
    from_cy /= n;
    to_cx /= n;
    to_cy /= n;

    let (mut dot, mut cross, mut from_var) = (0.0, 0.0, 0.0);
    for &((fx, fy), (tx, ty)) in correspondences {
        let (fx, fy) = (fx - from_cx, fy - from_cy);
        let (tx, ty) = (tx - to_cx, ty - to_cy);
        dot += fx * tx + fy * ty;
        cross += fx * ty - fy * tx;
        from_var += fx * fx + fy * fy;
    }

    // Coincident source points leave the rotation unconstrained.
    if from_var <= f64::EPSILON {
        return None;
    }
    // A degenerate target (all points coincident) leaves atan2 at (0, 0).
    if dot == 0.0 && cross == 0.0 {
        return None;
    }

    let theta = cross.atan2(dot);
    let scale = match scale_model {
        ScaleModel::Fixed => 1.0,
        // The scale minimizing residual is the ratio of the aligned spread to
        // the source spread; hypot(dot, cross) is that aligned spread.
        ScaleModel::Free => {
            let s = dot.hypot(cross) / from_var;
            if s <= f64::EPSILON || !s.is_finite() {
                return None;
            }
            s
        }
    };

    // Place the source centroid onto the target centroid under the fitted
    // rotation and scale.
    let (sin, cos) = theta.sin_cos();
    let (sx, sy) = (scale * from_cx, scale * from_cy);
    Some(Similarity {
        scale,
        theta,
        tx: to_cx - (sx * cos - sy * sin),
        ty: to_cy - (sx * sin + sy * cos),
    })
}

/// Mean euclidean distance between `to` and the mapped `from` points. The
/// RANSAC driver scores candidate transforms with this, and it is the natural
/// residual to report in debug output.
pub fn mean_residual(t: Similarity, correspondences: &[(Point, Point)]) -> f64 {
    if correspondences.is_empty() {
        return 0.0;
    }
    let total: f64 = correspondences
        .iter()
        .map(|&(from, to)| {
            let (mx, my) = t.apply(from);
            (mx - to.0).hypot(my - to.1)
        })
        .sum();
    total / correspondences.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn assert_close(a: f64, b: f64, what: &str) {
        assert!((a - b).abs() < 1e-6, "{what}: {a} != {b}");
    }

    fn assert_point_close(a: Point, b: Point) {
        assert!(
            (a.0 - b.0).abs() < 1e-6 && (a.1 - b.1).abs() < 1e-6,
            "{a:?} != {b:?}"
        );
    }

    /// Map `points` through `t` and pair each with its image, the input shape
    /// the estimator is supposed to invert.
    fn correspondences_through(t: &Similarity, points: &[Point]) -> Vec<(Point, Point)> {
        points.iter().map(|&p| (p, t.apply(p))).collect()
    }

    const SAMPLE: [Point; 4] = [(0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (30.0, 80.0)];

    #[test]
    fn identity_maps_points_unchanged() {
        for &p in &SAMPLE {
            assert_point_close(Similarity::IDENTITY.apply(p), p);
        }
    }

    #[test]
    fn recovers_pure_translation() {
        let truth = Similarity::translation(37.5, -12.25);
        let t = estimate(
            &correspondences_through(&truth, &SAMPLE),
            ScaleModel::Fixed,
        )
        .unwrap();

        assert_close(t.theta, 0.0, "theta");
        assert_close(t.tx, 37.5, "tx");
        assert_close(t.ty, -12.25, "ty");
        assert_close(t.scale, 1.0, "scale");
    }

    #[test]
    fn recovers_rotation_and_translation() {
        let truth = Similarity {
            scale: 1.0,
            theta: 0.15,
            tx: -8.0,
            ty: 21.0,
        };
        let t = estimate(
            &correspondences_through(&truth, &SAMPLE),
            ScaleModel::Fixed,
        )
        .unwrap();

        assert_close(t.theta, truth.theta, "theta");
        assert_close(t.tx, truth.tx, "tx");
        assert_close(t.ty, truth.ty, "ty");
        assert!(mean_residual(t, &correspondences_through(&truth, &SAMPLE)) < 1e-9);
    }

    #[test]
    fn free_scale_model_recovers_scale() {
        let truth = Similarity {
            scale: 1.25,
            theta: -0.3,
            tx: 4.0,
            ty: 9.0,
        };
        let t = estimate(&correspondences_through(&truth, &SAMPLE), ScaleModel::Free).unwrap();

        assert_close(t.scale, 1.25, "scale");
        assert_close(t.theta, truth.theta, "theta");
        assert_close(t.tx, truth.tx, "tx");
        assert_close(t.ty, truth.ty, "ty");
    }

    /// The flatbed case: even when the data really is scaled, Fixed must pin
    /// scale to 1 rather than absorbing the difference.
    #[test]
    fn fixed_scale_model_ignores_scaled_data() {
        let truth = Similarity {
            scale: 1.25,
            theta: 0.0,
            tx: 0.0,
            ty: 0.0,
        };
        let t = estimate(
            &correspondences_through(&truth, &SAMPLE),
            ScaleModel::Fixed,
        )
        .unwrap();

        assert_close(t.scale, 1.0, "scale");
    }

    #[test]
    fn rejects_too_few_correspondences() {
        let truth = Similarity::translation(1.0, 2.0);
        let one = correspondences_through(&truth, &SAMPLE[..1]);
        assert!(estimate(&one, ScaleModel::Fixed).is_none());
        assert!(estimate(&[], ScaleModel::Fixed).is_none());
    }

    #[test]
    fn rejects_coincident_source_points() {
        let coincident = vec![((5.0, 5.0), (1.0, 1.0)); 4];
        assert!(estimate(&coincident, ScaleModel::Fixed).is_none());
    }

    #[test]
    fn compose_matches_sequential_application() {
        let inner = Similarity {
            scale: 1.1,
            theta: 0.2,
            tx: 5.0,
            ty: -3.0,
        };
        let outer = Similarity {
            scale: 0.9,
            theta: -0.4,
            tx: -11.0,
            ty: 7.5,
        };
        let composed = outer.compose(inner);

        for &p in &SAMPLE {
            assert_point_close(composed.apply(p), outer.apply(inner.apply(p)));
        }
    }

    #[test]
    fn inverse_round_trips() {
        let t = Similarity {
            scale: 1.3,
            theta: 0.7,
            tx: 14.0,
            ty: -6.0,
        };
        let identity = t.compose(t.inverse());

        assert_close(identity.scale, 1.0, "scale");
        assert_close(identity.theta, 0.0, "theta");
        assert_close(identity.tx, 0.0, "tx");
        assert_close(identity.ty, 0.0, "ty");
        for &p in &SAMPLE {
            assert_point_close(t.inverse().apply(t.apply(p)), p);
        }
    }

    #[test]
    fn affine_coeffs_agree_with_apply() {
        let t = Similarity {
            scale: 1.4,
            theta: 0.35,
            tx: 3.0,
            ty: -9.0,
        };
        let [a, b, tx, c, d, ty] = t.to_affine_coeffs();

        for &(x, y) in &SAMPLE {
            assert_point_close(t.apply((x, y)), (a * x + b * y + tx, c * x + d * y + ty));
        }
    }

    /// Correspondences are never exact in practice; the fit should be the
    /// least-squares compromise rather than latching onto one outlier-free pair.
    #[test]
    fn least_squares_fit_absorbs_symmetric_noise() {
        let truth = Similarity {
            scale: 1.0,
            theta: 0.1,
            tx: 20.0,
            ty: -5.0,
        };
        let mut correspondences = correspondences_through(&truth, &SAMPLE);
        // Equal and opposite perturbations cancel in the least-squares sums.
        correspondences[0].1.0 += 0.5;
        correspondences[1].1.0 -= 0.5;

        let t = estimate(&correspondences, ScaleModel::Fixed).unwrap();
        assert!(
            (t.theta - truth.theta).abs() < 0.02,
            "theta drifted: {} vs {}",
            t.theta,
            truth.theta
        );
    }

    #[test]
    fn mean_residual_is_zero_for_exact_fit_and_positive_otherwise() {
        let truth = Similarity::translation(3.0, 4.0);
        let exact = correspondences_through(&truth, &SAMPLE);
        assert!(mean_residual(truth, &exact) < EPS);

        // A transform off by (3, 4) puts every point 5 units away.
        assert_close(
            mean_residual(Similarity::IDENTITY, &exact),
            5.0,
            "residual",
        );
    }
}
