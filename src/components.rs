//! Connected-component labelling and per-object cropping.

use crate::output::Rgba16Image;

/// A labelled connected component: its bounding box and pixel count.
pub struct Object {
    pub label: usize,
    pub area: usize,
    pub min_x: usize,
    pub min_y: usize,
    pub max_x: usize,
    pub max_y: usize,
}

/// Whether a component is large enough to keep, given a noise threshold of
/// `min_side_percent` of the smaller image dimension. A component survives when
/// its longer bounding-box side reaches that fraction of `min(w, h)`; this drops
/// specks that no single side makes large relative to the image. `0` keeps all.
pub fn meets_min_side(obj: &Object, w: usize, h: usize, min_side_percent: f64) -> bool {
    if min_side_percent <= 0.0 {
        return true;
    }
    let longer_side = (obj.max_x - obj.min_x + 1).max(obj.max_y - obj.min_y + 1);
    longer_side as f64 >= min_side_percent / 100.0 * w.min(h) as f64
}

/// Label 8-connected foreground regions via iterative flood fill, keeping only
/// those with at least `min_area` pixels. Returns the per-pixel label map (0 =
/// background) and the surviving objects sorted largest-first. Each object's
/// `label` indexes into the label map.
pub fn connected_components(
    fg: &[bool],
    w: usize,
    h: usize,
    min_area: usize,
) -> (Vec<usize>, Vec<Object>) {
    let mut labels = vec![0usize; fg.len()]; // 0 = unvisited/background
    let mut objects: Vec<Object> = Vec::new();
    let mut stack: Vec<(usize, usize)> = Vec::new();

    for sy in 0..h {
        for sx in 0..w {
            let start = sy * w + sx;
            if !fg[start] || labels[start] != 0 {
                continue;
            }

            let label = objects.len() + 1;
            let mut obj = Object {
                label,
                area: 0,
                min_x: sx,
                min_y: sy,
                max_x: sx,
                max_y: sy,
            };

            labels[start] = label;
            stack.push((sx, sy));
            while let Some((x, y)) = stack.pop() {
                obj.area += 1;
                obj.min_x = obj.min_x.min(x);
                obj.min_y = obj.min_y.min(y);
                obj.max_x = obj.max_x.max(x);
                obj.max_y = obj.max_y.max(y);

                let x0 = x.saturating_sub(1);
                let y0 = y.saturating_sub(1);
                let x1 = (x + 1).min(w - 1);
                let y1 = (y + 1).min(h - 1);
                for ny in y0..=y1 {
                    for nx in x0..=x1 {
                        let n = ny * w + nx;
                        if fg[n] && labels[n] == 0 {
                            labels[n] = label;
                            stack.push((nx, ny));
                        }
                    }
                }
            }

            if obj.area >= min_area.max(1) {
                objects.push(obj);
            }
        }
    }

    // Sorting only reorders the metadata; each object keeps its `label`, so the
    // label map stays valid for membership tests.
    objects.sort_by_key(|o| std::cmp::Reverse(o.area));
    (labels, objects)
}

/// Build an RGBA image cropped to `obj`'s bounding box, keeping opaque only the
/// pixels whose label matches this object (so overlapping bounding boxes from
/// other objects don't leak in); everything else is transparent.
pub fn crop_object(
    rgb: &image::RgbaImage,
    w: usize,
    labels: &[usize],
    obj: &Object,
) -> image::RgbaImage {
    let ow = (obj.max_x - obj.min_x + 1) as u32;
    let oh = (obj.max_y - obj.min_y + 1) as u32;
    let mut out = image::RgbaImage::new(ow, oh);

    for y in obj.min_y..=obj.max_y {
        for x in obj.min_x..=obj.max_x {
            if labels[y * w + x] == obj.label {
                let mut px = *rgb.get_pixel(x as u32, y as u32);
                px[3] = 255;
                out.put_pixel((x - obj.min_x) as u32, (y - obj.min_y) as u32, px);
            }
        }
    }
    out
}

pub fn crop_object_rgba16(
    rgb: &Rgba16Image,
    w: usize,
    labels: &[usize],
    obj: &Object,
) -> Rgba16Image {
    let ow = (obj.max_x - obj.min_x + 1) as u32;
    let oh = (obj.max_y - obj.min_y + 1) as u32;
    let mut out = Rgba16Image::new(ow, oh);

    for y in obj.min_y..=obj.max_y {
        for x in obj.min_x..=obj.max_x {
            if labels[y * w + x] == obj.label {
                let mut px = *rgb.get_pixel(x as u32, y as u32);
                px[3] = u16::MAX;
                out.put_pixel((x - obj.min_x) as u32, (y - obj.min_y) as u32, px);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_side_filter_drops_specks_relative_to_smaller_image_dim() {
        let obj = |min_x, min_y, max_x, max_y| Object {
            label: 1,
            area: 1,
            min_x,
            min_y,
            max_x,
            max_y,
        };
        // Image 200x100 -> threshold is 10% of min(200,100) = 10 px.
        let (w, h) = (200, 100);

        // 9x9 box: longer side 9 < 10 -> dropped.
        assert!(!meets_min_side(&obj(0, 0, 8, 8), w, h, 10.0));
        // 3x40 sliver: longer side 40 >= 10 -> kept (one side suffices).
        assert!(meets_min_side(&obj(0, 0, 2, 39), w, h, 10.0));
        // Exactly at the threshold (10 px) is kept.
        assert!(meets_min_side(&obj(0, 0, 9, 0), w, h, 10.0));
        // Disabled: the tiny box survives.
        assert!(meets_min_side(&obj(0, 0, 1, 1), w, h, 0.0));
    }
}
