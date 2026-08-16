//! Minimum-cost seam finding.
//!
//! A geometric boundary — the locus where two passes are equally far from their
//! own borders — is always a straight line for axis-aligned passes. That puts
//! the seam wherever the geometry happens to fall, including straight through
//! high-contrast detail, where any residual misalignment shows as a broken
//! edge that no amount of blending hides.
//!
//! A content-aware seam instead threads through pixels where the two passes
//! already agree, dodging around detail. That is what makes the regions in
//! Photoshop's Photomerge — and in the previous implementation's `gc_color`
//! finder — irregular rather than rectangular.
//!
//! This uses dynamic programming rather than graph cut. For the strip-shaped
//! overlaps a flatbed produces, a single monotone path across the strip is the
//! right model, and DP finds the global optimum for it in one pass without
//! needing a max-flow solver. (OpenCV offers the same choice as
//! `DpSeamFinder`.) The tradeoff is that the path must advance one row or
//! column at a time, so it cannot double back — fine for a strip, wrong for
//! seams that must wrap around an obstacle.

/// Which way the seam runs across the overlap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    /// One x per row, advancing down. For a tall overlap between passes offset
    /// horizontally.
    Vertical,
    /// One y per column, advancing right. For a wide overlap between passes
    /// offset vertically.
    Horizontal,
}

/// How far the seam may step sideways between adjacent rows. Allowing more than
/// one keeps the path from being forced through expensive pixels just because
/// it could not reach around them.
const MAX_STEP: isize = 2;

/// Cost of stepping sideways, in units of pixel cost. Small, so the seam
/// prefers agreement but does not wander pointlessly.
const STEP_PENALTY: f32 = 0.05;

/// Find the cheapest monotone path across a `width` x `height` cost field.
///
/// `cost` is row-major disagreement between the two passes; `valid` marks the
/// pixels the seam may pass through (the overlap). Returns one coordinate per
/// row (`Vertical`) or per column (`Horizontal`), or `None` when some line has
/// no valid pixel at all and no path can cross.
pub fn min_cost_path(
    cost: &[f32],
    valid: &[bool],
    width: usize,
    height: usize,
    orientation: Orientation,
) -> Option<Vec<usize>> {
    if width == 0 || height == 0 || cost.len() != width * height || valid.len() != cost.len() {
        return None;
    }

    // Work in "line" x "position" terms so one implementation covers both
    // orientations: a vertical seam has one position per row, a horizontal one
    // has one position per column.
    let (lines, positions) = match orientation {
        Orientation::Vertical => (height, width),
        Orientation::Horizontal => (width, height),
    };
    let at = |line: usize, position: usize| match orientation {
        Orientation::Vertical => line * width + position,
        Orientation::Horizontal => position * width + line,
    };

    let infinity = f32::INFINITY;
    let mut totals = vec![infinity; lines * positions];
    let mut came_from = vec![usize::MAX; lines * positions];

    let mut any_valid = false;
    for position in 0..positions {
        if valid[at(0, position)] {
            totals[position] = cost[at(0, position)];
            any_valid = true;
        }
    }
    if !any_valid {
        return None;
    }

    for line in 1..lines {
        let mut line_reachable = false;
        for position in 0..positions {
            if !valid[at(line, position)] {
                continue;
            }
            let mut best = infinity;
            let mut best_previous = usize::MAX;
            for step in -MAX_STEP..=MAX_STEP {
                let previous = position as isize + step;
                if previous < 0 || previous >= positions as isize {
                    continue;
                }
                let previous = previous as usize;
                let carried = totals[(line - 1) * positions + previous];
                if carried == infinity {
                    continue;
                }
                let candidate = carried + STEP_PENALTY * step.unsigned_abs() as f32;
                if candidate < best {
                    best = candidate;
                    best_previous = previous;
                }
            }
            if best_previous != usize::MAX {
                totals[line * positions + position] = best + cost[at(line, position)];
                came_from[line * positions + position] = best_previous;
                line_reachable = true;
            }
        }
        // A line the seam cannot reach means the overlap is broken in two.
        if !line_reachable {
            return None;
        }
    }

    let last = lines - 1;
    let mut end = usize::MAX;
    let mut best = infinity;
    for position in 0..positions {
        let total = totals[last * positions + position];
        if total < best {
            best = total;
            end = position;
        }
    }
    if end == usize::MAX {
        return None;
    }

    let mut path = vec![0usize; lines];
    let mut position = end;
    for line in (0..lines).rev() {
        path[line] = position;
        if line > 0 {
            position = came_from[line * positions + position];
            if position == usize::MAX {
                return None;
            }
        }
    }
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cheap channel down one column, everything else expensive: the seam
    /// should follow the channel exactly.
    #[test]
    fn follows_a_straight_low_cost_channel() {
        let (w, h) = (9usize, 12usize);
        let mut cost = vec![100.0f32; w * h];
        for y in 0..h {
            cost[y * w + 4] = 0.0;
        }

        let path = min_cost_path(&cost, &vec![true; w * h], w, h, Orientation::Vertical).unwrap();

        assert_eq!(path.len(), h);
        assert!(path.iter().all(|&x| x == 4), "path strayed: {path:?}");
    }

    /// The point of the whole module: when the cheap route bends, the seam
    /// bends with it instead of cutting straight.
    #[test]
    fn follows_a_diagonal_channel_rather_than_going_straight() {
        let (w, h) = (16usize, 12usize);
        let mut cost = vec![100.0f32; w * h];
        for y in 0..h {
            // A channel drifting right one pixel per row.
            cost[y * w + (2 + y)] = 0.0;
        }

        let path = min_cost_path(&cost, &vec![true; w * h], w, h, Orientation::Vertical).unwrap();

        for (y, &x) in path.iter().enumerate() {
            assert_eq!(x, 2 + y, "row {y} left the channel");
        }
        assert!(
            path.first() != path.last(),
            "a straight path means the seam is not content-aware"
        );
    }

    /// Sideways movement is capped, so a channel that jumps further than
    /// MAX_STEP in one row cannot be followed exactly — the seam should still
    /// return a usable path rather than failing.
    #[test]
    fn survives_a_channel_that_jumps_further_than_one_step() {
        let (w, h) = (20usize, 6usize);
        let mut cost = vec![10.0f32; w * h];
        for y in 0..h {
            cost[y * w + if y < 3 { 2 } else { 17 }] = 0.0;
        }

        let path = min_cost_path(&cost, &vec![true; w * h], w, h, Orientation::Vertical).unwrap();
        assert_eq!(path.len(), h);
        for window in path.windows(2) {
            assert!(
                window[1].abs_diff(window[0]) <= MAX_STEP as usize,
                "path stepped further than allowed: {path:?}"
            );
        }
    }

    #[test]
    fn horizontal_orientation_walks_columns() {
        let (w, h) = (12usize, 9usize);
        let mut cost = vec![100.0f32; w * h];
        for x in 0..w {
            cost[3 * w + x] = 0.0;
        }

        let path = min_cost_path(&cost, &vec![true; w * h], w, h, Orientation::Horizontal).unwrap();

        assert_eq!(path.len(), w);
        assert!(path.iter().all(|&y| y == 3), "path strayed: {path:?}");
    }

    #[test]
    fn respects_the_valid_mask() {
        let (w, h) = (8usize, 8usize);
        let cost = vec![1.0f32; w * h];
        // Only the right half may be crossed.
        let valid: Vec<bool> = (0..w * h).map(|i| i % w >= 4).collect();

        let path = min_cost_path(&cost, &valid, w, h, Orientation::Vertical).unwrap();
        assert!(path.iter().all(|&x| x >= 4), "seam left the overlap: {path:?}");
    }

    #[test]
    fn reports_failure_when_a_line_is_unreachable() {
        let (w, h) = (8usize, 8usize);
        let cost = vec![1.0f32; w * h];
        let mut valid = vec![true; w * h];
        // Block an entire row: nothing can cross it.
        for x in 0..w {
            valid[4 * w + x] = false;
        }

        assert!(min_cost_path(&cost, &valid, w, h, Orientation::Vertical).is_none());
    }

    #[test]
    fn rejects_malformed_input() {
        assert!(min_cost_path(&[], &[], 0, 0, Orientation::Vertical).is_none());
        assert!(min_cost_path(&[1.0], &[true, true], 1, 1, Orientation::Vertical).is_none());
    }
}
