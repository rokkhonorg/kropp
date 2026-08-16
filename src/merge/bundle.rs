//! Joint refinement of every pass's placement.
//!
//! The spanning-tree walk in `align` gives each pass a transform by composing
//! pairwise fits from the reference outward. That is a *chain*: a pass three
//! edges away carries the accumulated error of all three, and nothing ever
//! reconciles a pair whose edge the tree did not use. On a two-pass merge it
//! barely matters; across four it shows as a merge that is clean at one end and
//! progressively worse at the other.
//!
//! Bundle adjustment fixes that by optimizing every placement *simultaneously*
//! against every correspondence, not just the ones on the tree. This is what
//! OpenCV's `BundleAdjusterAffinePartial` does, and skipping it was one of the
//! deviations from the previous implementation.
//!
//! The problem is small — four parameters per non-reference pass — so this is
//! a hand-rolled Levenberg-Marquardt over dense normal equations rather than a
//! dependency. Scale is solved for alongside rotation and translation, matching
//! OpenCV's partial-affine model.

use super::geometry::{Point, Similarity};

/// A correspondence between two passes, in their own pixel frames.
#[derive(Debug, Clone, Copy)]
pub struct Observation {
    /// Index into the transforms being solved for.
    pub from: usize,
    pub to: usize,
    /// The same physical point, as seen by each pass.
    pub from_point: Point,
    pub to_point: Point,
}

/// Iterations to attempt. The problem is tiny and well conditioned; if it has
/// not settled by now it is not going to.
const MAX_ITERATIONS: usize = 60;

/// Stop once a step improves the cost by less than this fraction.
const CONVERGENCE: f64 = 1e-9;

/// Parameters per pass: rotation, translation, and uniform scale.
const PARAMS_PER_PASS: usize = 4;

/// Scale is physically near 1 for passes off one flatbed. Bounding it keeps a
/// poorly constrained solve from collapsing or exploding the layout.
const MIN_SCALE: f64 = 0.5;
const MAX_SCALE: f64 = 2.0;

/// Refine `transforms` so that all `observations` agree as closely as possible.
///
/// `reference` is held fixed, which anchors the solution — without it the whole
/// layout could drift or spin freely and the normal equations would be
/// singular. Returns the refined transforms, or the originals when there is
/// nothing to solve.
pub fn refine(
    transforms: &[Similarity],
    observations: &[Observation],
    reference: usize,
) -> Vec<Similarity> {
    let count = transforms.len();
    if count < 2 || observations.is_empty() || reference >= count {
        return transforms.to_vec();
    }

    // Map pass index -> slot in the parameter vector, skipping the reference.
    let mut slot_of = vec![usize::MAX; count];
    let mut slot = 0usize;
    for (index, slot_entry) in slot_of.iter_mut().enumerate() {
        if index != reference {
            *slot_entry = slot;
            slot += 1;
        }
    }
    let unknowns = slot * PARAMS_PER_PASS;
    if unknowns == 0 {
        return transforms.to_vec();
    }

    let mut current = transforms.to_vec();
    let mut lambda = 1e-3f64;
    let mut cost = total_cost(&current, observations);

    for _ in 0..MAX_ITERATIONS {
        // Accumulate the normal equations J^T J and J^T r.
        let mut normal = vec![0.0f64; unknowns * unknowns];
        let mut gradient = vec![0.0f64; unknowns];

        for observation in observations {
            let (from, to) = (observation.from, observation.to);
            if from >= count || to >= count {
                continue;
            }
            let mapped_from = current[from].apply(observation.from_point);
            let mapped_to = current[to].apply(observation.to_point);
            let residual = [mapped_from.0 - mapped_to.0, mapped_from.1 - mapped_to.1];

            // Each observation contributes two rows, one per coordinate.
            let mut row_x = vec![0.0f64; unknowns];
            let mut row_y = vec![0.0f64; unknowns];
            accumulate_jacobian(
                &mut row_x,
                &mut row_y,
                slot_of[from],
                current[from],
                observation.from_point,
                1.0,
            );
            accumulate_jacobian(
                &mut row_x,
                &mut row_y,
                slot_of[to],
                current[to],
                observation.to_point,
                -1.0,
            );

            for row in [(&row_x, residual[0]), (&row_y, residual[1])] {
                let (jacobian, value) = row;
                for i in 0..unknowns {
                    if jacobian[i] == 0.0 {
                        continue;
                    }
                    gradient[i] += jacobian[i] * value;
                    for j in 0..unknowns {
                        if jacobian[j] != 0.0 {
                            normal[i * unknowns + j] += jacobian[i] * jacobian[j];
                        }
                    }
                }
            }
        }

        // Levenberg damping, then solve for the step.
        let mut damped = normal.clone();
        for i in 0..unknowns {
            damped[i * unknowns + i] *= 1.0 + lambda;
            // Keep a strictly positive diagonal so a parameter no observation
            // constrains cannot make the system singular.
            damped[i * unknowns + i] += 1e-12;
        }
        let Some(step) = solve(&mut damped, &gradient, unknowns) else {
            break;
        };

        let mut candidate = current.clone();
        for index in 0..count {
            let slot = slot_of[index];
            if slot == usize::MAX {
                continue;
            }
            let base = slot * PARAMS_PER_PASS;
            candidate[index].theta -= step[base];
            candidate[index].tx -= step[base + 1];
            candidate[index].ty -= step[base + 2];
            candidate[index].scale =
                (candidate[index].scale - step[base + 3]).clamp(MIN_SCALE, MAX_SCALE);
        }

        let candidate_cost = total_cost(&candidate, observations);
        if candidate_cost < cost {
            let improvement = (cost - candidate_cost) / cost.max(f64::MIN_POSITIVE);
            current = candidate;
            cost = candidate_cost;
            lambda = (lambda * 0.5).max(1e-12);
            if improvement < CONVERGENCE {
                break;
            }
        } else {
            // Step made things worse: trust the linearization less and retry.
            lambda *= 4.0;
            if lambda > 1e12 {
                break;
            }
        }
    }

    current
}

/// Partial derivatives of `transform.apply(point)` with respect to that pass's
/// three parameters, added into the two Jacobian rows with the given sign.
///
/// A reference pass has `slot == usize::MAX` and contributes nothing, which is
/// exactly what holding it fixed means.
fn accumulate_jacobian(
    row_x: &mut [f64],
    row_y: &mut [f64],
    slot: usize,
    transform: Similarity,
    point: Point,
    sign: f64,
) {
    if slot == usize::MAX {
        return;
    }
    let base = slot * PARAMS_PER_PASS;
    let (sin, cos) = transform.theta.sin_cos();
    let (px, py) = point;
    let (x, y) = (transform.scale * px, transform.scale * py);

    // d/dtheta of (x cos - y sin, x sin + y cos), with x, y already scaled.
    row_x[base] += sign * (-x * sin - y * cos);
    row_y[base] += sign * (x * cos - y * sin);
    // d/dtx and d/dty
    row_x[base + 1] += sign;
    row_y[base + 2] += sign;
    // d/dscale is the rotated, unscaled point.
    row_x[base + 3] += sign * (px * cos - py * sin);
    row_y[base + 3] += sign * (px * sin + py * cos);
}

/// Sum of squared residuals over every observation.
fn total_cost(transforms: &[Similarity], observations: &[Observation]) -> f64 {
    let mut total = 0.0;
    for observation in observations {
        if observation.from >= transforms.len() || observation.to >= transforms.len() {
            continue;
        }
        let a = transforms[observation.from].apply(observation.from_point);
        let b = transforms[observation.to].apply(observation.to_point);
        total += (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2);
    }
    total
}

/// Solve `matrix * x = rhs` by Gauss-Jordan with partial pivoting. `matrix` is
/// consumed. Returns `None` if the system is singular.
fn solve(matrix: &mut [f64], rhs: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut solution = rhs.to_vec();

    for column in 0..n {
        // Partial pivoting keeps the elimination stable.
        let mut pivot = column;
        let mut best = matrix[column * n + column].abs();
        for row in (column + 1)..n {
            let value = matrix[row * n + column].abs();
            if value > best {
                best = value;
                pivot = row;
            }
        }
        if best < 1e-14 {
            return None;
        }
        if pivot != column {
            for k in 0..n {
                matrix.swap(column * n + k, pivot * n + k);
            }
            solution.swap(column, pivot);
        }

        let diagonal = matrix[column * n + column];
        for row in 0..n {
            if row == column {
                continue;
            }
            let factor = matrix[row * n + column] / diagonal;
            if factor == 0.0 {
                continue;
            }
            for k in column..n {
                matrix[row * n + k] -= factor * matrix[column * n + k];
            }
            solution[row] -= factor * solution[column];
        }
    }

    for i in 0..n {
        solution[i] /= matrix[i * n + i];
    }
    Some(solution)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(from: usize, to: usize, from_point: Point, to_point: Point) -> Observation {
        Observation {
            from,
            to,
            from_point,
            to_point,
        }
    }

    const GRID: [Point; 6] = [
        (0.0, 0.0),
        (80.0, 0.0),
        (160.0, 0.0),
        (0.0, 60.0),
        (80.0, 60.0),
        (160.0, 60.0),
    ];

    /// Observations generated from placements that already agree must be left
    /// alone rather than wandered off.
    #[test]
    fn an_exact_solution_is_left_untouched() {
        let truth = vec![Similarity::IDENTITY, Similarity::translation(100.0, 20.0)];
        let observations: Vec<Observation> = GRID
            .iter()
            .map(|&p| {
                // The same physical point seen by pass 1 at p, and by pass 0
                // wherever truth[1] places it.
                let canvas = truth[1].apply(p);
                observation(1, 0, p, canvas)
            })
            .collect();

        let refined = refine(&truth, &observations, 0);

        assert!((refined[1].tx - 100.0).abs() < 1e-6, "tx {}", refined[1].tx);
        assert!((refined[1].ty - 20.0).abs() < 1e-6, "ty {}", refined[1].ty);
    }

    /// The point of the module: a placement started off its true position must
    /// be pulled back onto it by the observations.
    #[test]
    fn recovers_a_placement_from_a_poor_starting_guess() {
        let truth = Similarity {
            scale: 1.0,
            theta: 0.03,
            tx: 120.0,
            ty: -18.0,
        };
        let observations: Vec<Observation> = GRID
            .iter()
            .map(|&p| observation(1, 0, p, truth.apply(p)))
            .collect();

        // Start 15px and 0.02 rad away from the answer.
        let start = vec![
            Similarity::IDENTITY,
            Similarity {
                scale: 1.0,
                theta: 0.01,
                tx: 135.0,
                ty: -5.0,
            },
        ];
        let refined = refine(&start, &observations, 0);

        assert!((refined[1].tx - truth.tx).abs() < 0.05, "tx {}", refined[1].tx);
        assert!((refined[1].ty - truth.ty).abs() < 0.05, "ty {}", refined[1].ty);
        assert!(
            (refined[1].theta - truth.theta).abs() < 1e-4,
            "theta {}",
            refined[1].theta
        );
    }

    /// Scale is now a free parameter, so a magnification difference between
    /// passes must be absorbed by it rather than smeared across the fit as
    /// position error.
    #[test]
    fn recovers_a_scale_difference() {
        let truth = Similarity {
            scale: 1.004,
            theta: 0.01,
            tx: 90.0,
            ty: 12.0,
        };
        let observations: Vec<Observation> = GRID
            .iter()
            .map(|&p| observation(1, 0, p, truth.apply(p)))
            .collect();

        let start = vec![
            Similarity::IDENTITY,
            Similarity {
                scale: 1.0,
                theta: 0.0,
                tx: 90.0,
                ty: 12.0,
            },
        ];
        let refined = refine(&start, &observations, 0);

        assert!(
            (refined[1].scale - truth.scale).abs() < 1e-4,
            "scale {} should have converged on {}",
            refined[1].scale,
            truth.scale
        );
        // And with scale free the residual should collapse to nothing.
        assert!(total_cost(&refined, &observations) < 1e-6);
    }

    /// The reference must not move, or the whole layout is free to drift.
    #[test]
    fn the_reference_pass_is_held_fixed() {
        let start = vec![
            Similarity::translation(7.0, -3.0),
            Similarity::translation(90.0, 0.0),
        ];
        let observations: Vec<Observation> = GRID
            .iter()
            .map(|&p| observation(1, 0, p, (p.0 + 100.0, p.1)))
            .collect();

        let refined = refine(&start, &observations, 0);

        assert_eq!(refined[0], start[0], "reference placement moved");
    }

    /// A chain of three passes where the middle link is right but the end one
    /// is not: the joint solve should correct the end without breaking the
    /// middle, which a sequential composition cannot do.
    #[test]
    fn corrects_accumulated_error_along_a_chain() {
        let truth = [Similarity::IDENTITY,
            Similarity::translation(100.0, 0.0),
            Similarity::translation(200.0, 0.0)];
        let mut observations = Vec::new();
        for &p in &GRID {
            // 0 <-> 1 and 1 <-> 2 both observed.
            observations.push(observation(1, 0, p, truth[1].apply(p)));
            observations.push(observation(2, 1, p, truth[1].inverse().apply(truth[2].apply(p))));
        }

        // Pass 2 starts 12px off, as chained pairwise error would leave it.
        let start = vec![
            truth[0],
            truth[1],
            Similarity::translation(212.0, 4.0),
        ];
        let refined = refine(&start, &observations, 0);

        assert!(
            (refined[2].tx - 200.0).abs() < 0.1,
            "chained pass not corrected: tx {}",
            refined[2].tx
        );
        assert!((refined[2].ty).abs() < 0.1, "ty {}", refined[2].ty);
        // And the already-correct middle pass stayed put.
        assert!((refined[1].tx - 100.0).abs() < 0.1, "middle moved");
    }

    /// Refinement must reduce total disagreement, never increase it.
    #[test]
    fn never_makes_the_fit_worse() {
        let start = vec![
            Similarity::IDENTITY,
            Similarity::translation(80.0, 10.0),
            Similarity::translation(150.0, -20.0),
        ];
        let mut observations = Vec::new();
        for (index, &p) in GRID.iter().enumerate() {
            let jitter = (index as f64) * 0.3;
            observations.push(observation(1, 0, p, (p.0 + 100.0 + jitter, p.1)));
            observations.push(observation(2, 0, p, (p.0 + 200.0, p.1 + jitter)));
        }

        let before = total_cost(&start, &observations);
        let refined = refine(&start, &observations, 0);
        let after = total_cost(&refined, &observations);

        assert!(after <= before, "cost rose from {before} to {after}");
    }

    #[test]
    fn degenerate_inputs_are_returned_unchanged() {
        let single = vec![Similarity::IDENTITY];
        assert_eq!(refine(&single, &[], 0).len(), 1);

        let pair = vec![Similarity::IDENTITY, Similarity::translation(5.0, 5.0)];
        // No observations to fit.
        assert_eq!(refine(&pair, &[], 0), pair);
        // Reference out of range.
        assert_eq!(refine(&pair, &[observation(1, 0, (0.0, 0.0), (1.0, 1.0))], 9), pair);
    }

    #[test]
    fn solve_handles_a_small_system() {
        // 2x + y = 5 ; x + 3y = 10  ->  x = 1, y = 3
        let mut matrix = vec![2.0, 1.0, 1.0, 3.0];
        let solution = solve(&mut matrix, &[5.0, 10.0], 2).unwrap();
        assert!((solution[0] - 1.0).abs() < 1e-9, "{solution:?}");
        assert!((solution[1] - 3.0).abs() < 1e-9, "{solution:?}");
    }

    #[test]
    fn solve_reports_a_singular_system() {
        let mut matrix = vec![1.0, 2.0, 2.0, 4.0];
        assert!(solve(&mut matrix, &[1.0, 2.0], 2).is_none());
    }
}
