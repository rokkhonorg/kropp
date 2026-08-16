//! Content-aware seam placement by minimum cut.
//!
//! A geometric boundary — the locus where two passes are equally deep inside
//! their own borders — is a smooth curve fixed by the layout alone. It lands
//! wherever the geometry happens to put it, including straight through
//! high-contrast detail, where any residual misalignment shows as a broken edge
//! that blending cannot hide.
//!
//! A minimum cut instead threads the boundary through pixels where the two
//! passes already *agree*, so it dodges around detail and follows whatever
//! path is cheapest. That is what makes the regions in Photoshop's Photomerge —
//! and in the previous implementation's `gc_color` finder — irregular rather
//! than geometric.
//!
//! The construction is Kwatra et al.'s, and matches OpenCV's
//! `GraphCutSeamFinder` with `COST_COLOR`: one node per candidate pixel, an
//! edge between neighbours whose capacity is how much the passes disagree
//! there, and the regions only one pass covers wired to the terminals so the
//! cut has something to separate. Cutting is then cheap exactly where the two
//! passes look the same.

use super::maxflow::Graph;

/// What a pixel in the region under consideration is available for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    /// Only the first pass reaches here, so it must stay with that pass. These
    /// anchor the cut — without them there is nothing to separate and the
    /// minimum cut is trivially empty.
    OnlyA,
    /// Likewise for the second pass.
    OnlyB,
    /// Both passes reach here; the cut decides.
    Both,
    /// Outside both passes, or already owned by a third. Gets no node.
    Absent,
}

/// Decide ownership between two passes over a region, by minimum cut.
///
/// `cost` holds each pixel's disagreement between the passes, in any consistent
/// unit — it is normalized internally. `true` in the result means the first
/// pass owns that pixel.
///
/// Returns `None` when the region cannot produce a meaningful cut: no pixels,
/// mismatched inputs, or nothing anchored to one of the two terminals.
pub fn cut_ownership(
    state: &[Cell],
    cost: &[f32],
    width: usize,
    height: usize,
) -> Option<Vec<bool>> {
    if width == 0 || height == 0 || state.len() != width * height || cost.len() != state.len() {
        return None;
    }

    // One node per participating pixel.
    let mut node_of = vec![usize::MAX; state.len()];
    let mut nodes = 0usize;
    let (mut has_a, mut has_b) = (false, false);
    for (index, cell) in state.iter().enumerate() {
        match cell {
            Cell::Absent => continue,
            Cell::OnlyA => has_a = true,
            Cell::OnlyB => has_b = true,
            Cell::Both => {}
        }
        node_of[index] = nodes;
        nodes += 1;
    }
    // With only one terminal represented there is nothing to separate, and the
    // caller should keep whatever it already had.
    if nodes == 0 || !has_a || !has_b {
        return None;
    }

    // Normalize so capacities land in a predictable range, which lets the
    // hard-link weight below be a safe finite number rather than an infinity
    // that could swamp the arithmetic.
    let peak = cost
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(0.0f32, f32::max)
        .max(f32::MIN_POSITIVE);

    // A neighbour edge costs at most 2 after normalization, and the cut can
    // sever at most two edges per node, so this exceeds any possible cut and
    // therefore cannot itself be cut.
    let anchor = 4.0 * nodes as f32 + 16.0;

    let mut graph = Graph::new(nodes, nodes * 2);
    for (index, cell) in state.iter().enumerate() {
        let node = node_of[index];
        if node == usize::MAX {
            continue;
        }
        match cell {
            Cell::OnlyA => graph.add_terminal_weights(node, anchor, 0.0),
            Cell::OnlyB => graph.add_terminal_weights(node, 0.0, anchor),
            _ => {}
        }
    }

    // 4-connected edges. Capacity is the summed disagreement at both ends, so
    // the cut prefers to run where the passes already look alike.
    let normalized = |index: usize| {
        let value = cost[index];
        if value.is_finite() {
            (value / peak).clamp(0.0, 1.0)
        } else {
            1.0
        }
    };
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let node = node_of[index];
            if node == usize::MAX {
                continue;
            }
            for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                if nx >= width || ny >= height {
                    continue;
                }
                let neighbour_index = ny * width + nx;
                let neighbour = node_of[neighbour_index];
                if neighbour == usize::MAX {
                    continue;
                }
                // A floor keeps every edge cuttable but non-free, so the cut
                // stays connected rather than wandering through zero-cost voids.
                let capacity = normalized(index) + normalized(neighbour_index) + 1e-3;
                graph.add_edge(node, neighbour, capacity, capacity);
            }
        }
    }

    graph.maxflow();

    let mut owned_by_a = vec![false; state.len()];
    for (index, slot) in owned_by_a.iter_mut().enumerate() {
        let node = node_of[index];
        if node != usize::MAX {
            *slot = graph.is_source_side(node);
        }
    }
    Some(owned_by_a)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a region that is `OnlyA` down the left edge, `OnlyB` down the
    /// right, and `Both` in between — the shape a real overlap has.
    fn strip(width: usize, height: usize, margin: usize) -> Vec<Cell> {
        let mut state = vec![Cell::Both; width * height];
        for y in 0..height {
            for x in 0..margin {
                state[y * width + x] = Cell::OnlyA;
                state[y * width + (width - 1 - x)] = Cell::OnlyB;
            }
        }
        state
    }

    /// Where in each row the ownership flips from A to B.
    fn boundary_per_row(owned: &[bool], width: usize, height: usize) -> Vec<usize> {
        (0..height)
            .map(|y| {
                (0..width)
                    .find(|&x| !owned[y * width + x])
                    .unwrap_or(width)
            })
            .collect()
    }

    #[test]
    fn anchors_are_respected() {
        let (w, h) = (24usize, 16usize);
        let state = strip(w, h, 4);
        let cost = vec![1.0f32; w * h];

        let owned = cut_ownership(&state, &cost, w, h).unwrap();

        for y in 0..h {
            assert!(owned[y * w], "left edge should stay with the first pass");
            assert!(
                !owned[y * w + (w - 1)],
                "right edge should stay with the second pass"
            );
        }
    }

    /// The whole point: given a cheap path that bends, the seam follows it
    /// rather than cutting straight across.
    #[test]
    fn seam_follows_a_bent_low_cost_path() {
        let (w, h) = (40usize, 24usize);
        let state = strip(w, h, 4);

        // Expensive everywhere, except a connected corridor that runs down
        // x=10, jogs right at the midpoint, then continues down x=28.
        //
        // The corridor has to be two pixels wide and actually joined up. An
        // edge's capacity is the cost at *both* its endpoints, so a one-pixel
        // channel leaves every crossing edge costing the same as one through
        // open ground — and then bending only adds the step edges, making the
        // straight cut correctly cheaper. That is a badly built test, not a
        // seam finder that ignores content.
        let mut cost = vec![10.0f32; w * h];
        let mid = h / 2;
        let mut make_cheap = |x: usize, y: usize| cost[y * w + x] = 0.0;
        for y in 0..mid {
            make_cheap(10, y);
            make_cheap(11, y);
        }
        for y in mid..h {
            make_cheap(28, y);
            make_cheap(29, y);
        }
        for x in 10..=29 {
            make_cheap(x, mid);
            make_cheap(x, mid.saturating_sub(1));
        }

        let owned = cut_ownership(&state, &cost, w, h).unwrap();
        let boundary = boundary_per_row(&owned, w, h);

        let top = boundary[2];
        let bottom = boundary[h - 3];
        assert!(
            (9..=12).contains(&top),
            "top of the seam at {top}, expected near the x=10 channel"
        );
        assert!(
            (27..=30).contains(&bottom),
            "bottom of the seam at {bottom}, expected near the x=28 channel"
        );
        assert!(
            bottom > top + 10,
            "seam ran straight ({top} -> {bottom}); it is not following content"
        );
    }

    /// With uniform cost the cut is free to sit anywhere, but it must still be
    /// a single connected boundary rather than salt and pepper.
    #[test]
    fn uniform_cost_still_yields_one_contiguous_region_per_pass() {
        let (w, h) = (24usize, 16usize);
        let state = strip(w, h, 4);
        let cost = vec![1.0f32; w * h];

        let owned = cut_ownership(&state, &cost, w, h).unwrap();

        for y in 0..h {
            // Once ownership flips to B along a row it must not flip back.
            let mut seen_b = false;
            for x in 0..w {
                if !owned[y * w + x] {
                    seen_b = true;
                } else {
                    assert!(!seen_b, "row {y} flipped back to A at x={x}");
                }
            }
        }
    }

    #[test]
    fn absent_pixels_get_no_ownership() {
        let (w, h) = (12usize, 8usize);
        let mut state = strip(w, h, 3);
        // Punch a hole a third pass owns.
        for y in 2..5 {
            for x in 5..8 {
                state[y * w + x] = Cell::Absent;
            }
        }
        let cost = vec![1.0f32; w * h];

        let owned = cut_ownership(&state, &cost, w, h).unwrap();

        for y in 2..5 {
            for x in 5..8 {
                assert!(!owned[y * w + x], "absent pixel was claimed");
            }
        }
    }

    /// Without both anchors there is nothing to separate, and the caller should
    /// be told so rather than handed a meaningless all-one-side answer.
    #[test]
    fn regions_missing_an_anchor_are_refused() {
        let (w, h) = (8usize, 8usize);
        let cost = vec![1.0f32; w * h];

        let all_both = vec![Cell::Both; w * h];
        assert!(cut_ownership(&all_both, &cost, w, h).is_none());

        let mut only_a = vec![Cell::Both; w * h];
        only_a[0] = Cell::OnlyA;
        assert!(cut_ownership(&only_a, &cost, w, h).is_none());
    }

    #[test]
    fn malformed_input_is_refused() {
        assert!(cut_ownership(&[], &[], 0, 0).is_none());
        assert!(cut_ownership(&[Cell::Both], &[1.0, 2.0], 1, 1).is_none());
        assert!(cut_ownership(&[Cell::Both], &[1.0], 2, 3).is_none());
    }

    /// Costs that are not finite must not poison the capacities.
    #[test]
    fn non_finite_costs_are_tolerated() {
        let (w, h) = (16usize, 10usize);
        let state = strip(w, h, 3);
        let mut cost = vec![1.0f32; w * h];
        cost[5 * w + 8] = f32::NAN;
        cost[6 * w + 8] = f32::INFINITY;

        let owned = cut_ownership(&state, &cost, w, h).unwrap();
        assert!(owned[5 * w], "anchors should still hold");
    }
}
