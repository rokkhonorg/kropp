//! Splitting row-major plane work across threads.
//!
//! The compositor's inner loops are all of one shape: an output plane whose
//! every row is computed independently of the others. That is exactly what
//! parallelizes without changing a single result — no accumulator spans rows,
//! so nothing is reassociated and the arithmetic each pixel sees is identical
//! to the serial version's. The merge stays byte-for-byte reproducible.
//!
//! Deliberately not a work-stealing pool. The bands are equal-sized and the
//! work per band is uniform, so a fixed split with one thread each has nothing
//! to steal, and `std::thread::scope` borrows the output plane directly instead
//! of demanding `'static`.

use std::sync::OnceLock;

/// Below this, a plane is not worth splitting: spawning threads costs more
/// than the work saved.
///
/// This matters more than it looks. `blur` runs on every level of every
/// pyramid, and the coarse levels are a few hundred pixels — without a floor
/// the merge would spawn a thread per band for an 8x8 plane, hundreds of times
/// over, and end up slower than when it was serial.
const MIN_PARALLEL_PIXELS: usize = 1 << 16;

/// Rows small enough that splitting finer just adds overhead.
const MIN_ROWS_PER_BAND: usize = 8;

fn threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

/// How many bands to split `height` rows into.
fn band_count(width: usize, height: usize) -> usize {
    let pixels = width * height;
    if width == 0 || height == 0 || pixels < MIN_PARALLEL_PIXELS {
        return 1;
    }
    // Bands are sized by work, not just by thread count. A pyramid's coarse
    // levels are a fraction of the canvas, and splitting a quarter-megapixel
    // plane across every core spends more on starting the threads than the
    // few thousand pixels each would then do.
    let by_work = pixels / MIN_PARALLEL_PIXELS;
    let by_rows = height.div_ceil(MIN_ROWS_PER_BAND);
    threads().min(by_work).min(by_rows).max(1)
}

/// Run `body` over `plane` split into horizontal bands, one thread each.
///
/// `body` is handed the index of the band's first row and the rows themselves,
/// so it can locate itself in the source it is reading from.
pub fn bands<T: Send>(
    plane: &mut [T],
    width: usize,
    height: usize,
    body: impl Fn(usize, &mut [T]) + Sync,
) {
    let count = band_count(width, height);
    if count <= 1 {
        body(0, plane);
        return;
    }

    let rows_per_band = height.div_ceil(count);
    std::thread::scope(|scope| {
        for (index, band) in plane.chunks_mut(rows_per_band * width).enumerate() {
            let body = &body;
            scope.spawn(move || body(index * rows_per_band, band));
        }
    });
}

/// [`bands`], over two planes of the same shape but different element types.
pub fn bands2<A: Send, B: Send>(
    first: &mut [A],
    second: &mut [B],
    width: usize,
    height: usize,
    body: impl Fn(usize, &mut [A], &mut [B]) + Sync,
) {
    let count = band_count(width, height);
    if count <= 1 {
        body(0, first, second);
        return;
    }

    let rows_per_band = height.div_ceil(count);
    let step = rows_per_band * width;
    std::thread::scope(|scope| {
        let paired = first.chunks_mut(step).zip(second.chunks_mut(step));
        for (index, (a, b)) in paired.enumerate() {
            let body = &body;
            scope.spawn(move || body(index * rows_per_band, a, b));
        }
    });
}

/// [`bands`], over a fixed number of planes of the same type — the channels of
/// one warped pass, say.
pub fn bands_n<const N: usize, T: Send>(
    planes: &mut [Vec<T>; N],
    width: usize,
    height: usize,
    body: impl Fn(usize, [&mut [T]; N]) + Sync,
) {
    let count = band_count(width, height);
    if count <= 1 {
        body(0, planes.each_mut().map(|plane| plane.as_mut_slice()));
        return;
    }

    let rows_per_band = height.div_ceil(count);
    let step = (rows_per_band * width).max(1);
    let mut chunks = planes.each_mut().map(|plane| plane.chunks_mut(step));
    std::thread::scope(|scope| {
        let mut index = 0;
        loop {
            let next = chunks.each_mut().map(|chunk| chunk.next());
            if next.iter().any(Option::is_none) {
                break;
            }
            let bands = next.map(|band| band.expect("all planes are the same length"));
            let body = &body;
            scope.spawn(move || body(index * rows_per_band, bands));
            index += 1;
        }
    });
}

/// [`bands`], over however many planes of the same type the caller has — the
/// selection masks, whose count is the number of passes.
pub fn bands_many<T: Send>(
    planes: &mut [Vec<T>],
    width: usize,
    height: usize,
    body: impl Fn(usize, &mut [&mut [T]]) + Sync,
) {
    let count = band_count(width, height);
    let mut rest: Vec<&mut [T]> = planes.iter_mut().map(|plane| plane.as_mut_slice()).collect();
    if count <= 1 {
        body(0, &mut rest);
        return;
    }

    let rows_per_band = height.div_ceil(count);
    let step = rows_per_band * width;
    std::thread::scope(|scope| {
        let mut first = 0;
        while first < height * width {
            let take = step.min(height * width - first);
            let mut heads = Vec::with_capacity(rest.len());
            let mut tails = Vec::with_capacity(rest.len());
            for plane in rest {
                let (head, tail) = plane.split_at_mut(take);
                heads.push(head);
                tails.push(tail);
            }
            rest = tails;
            let body = &body;
            let start = first;
            scope.spawn(move || body(start, &mut heads));
            first += take;
        }
    });
}

/// Run `body` over each of `count` items, on up to one thread each.
///
/// For work that is already split into independent pieces — the channels of a
/// blend, the cells of a coarse grid — rather than into rows of one plane.
pub fn each(count: usize, body: impl Fn(usize) + Sync) {
    if count <= 1 || threads() <= 1 {
        for index in 0..count {
            body(index);
        }
        return;
    }
    std::thread::scope(|scope| {
        for index in 0..count {
            let body = &body;
            scope.spawn(move || body(index));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row must be visited exactly once, and handed the right first-row
    /// index — the whole correctness of a banded loop rests on that mapping.
    #[test]
    fn bands_cover_every_row_exactly_once_with_correct_offsets() {
        for (width, height) in [(1usize, 1usize), (317, 1), (640, 480), (6556, 9166)] {
            let mut plane = vec![0u32; width * height];
            bands(&mut plane, width, height, |first, band| {
                for (offset, row) in band.chunks_mut(width).enumerate() {
                    let y = first + offset;
                    for slot in row {
                        // Stamp the row index this cell was told it belonged to.
                        *slot = y as u32 + 1;
                    }
                }
            });
            for y in 0..height {
                for x in 0..width {
                    assert_eq!(
                        plane[y * width + x],
                        y as u32 + 1,
                        "{width}x{height} at ({x},{y})"
                    );
                }
            }
        }
    }

    /// Pyramids blur planes down to a handful of pixels; those must not each
    /// spawn a fleet of threads.
    #[test]
    fn tiny_planes_are_not_split() {
        assert_eq!(band_count(8, 8), 1);
        assert_eq!(band_count(64, 64), 1);
        assert_eq!(band_count(0, 0), 1);
    }

    /// A band is never thinner than the floor, so a tall thin plane does not
    /// turn into one thread per row.
    #[test]
    fn bands_stay_at_least_the_minimum_thickness() {
        let height = 40;
        let count = band_count(4096, height);
        assert!(count <= height.div_ceil(MIN_ROWS_PER_BAND), "{count} bands");
    }

    #[test]
    fn each_runs_every_index() {
        let seen: Vec<std::sync::atomic::AtomicBool> =
            (0..17).map(|_| std::sync::atomic::AtomicBool::new(false)).collect();
        each(seen.len(), |index| {
            seen[index].store(true, std::sync::atomic::Ordering::Relaxed);
        });
        assert!(seen.iter().all(|s| s.load(std::sync::atomic::Ordering::Relaxed)));
    }
}
