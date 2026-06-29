//! Foreground extraction: turning a source image into a label map of objects,
//! before the crop/deskew stage. This is the swappable front of the pipeline —
//! RMBG by default, or SAM3 (background-inversion / multi-prompt) behind the
//! `sam3` feature.

use anyhow::{Context, Result};
use usls::Image;

use crate::cli::{Args, ExtractorKind};
use crate::components::{Object, connected_components};
use crate::model::build_rmbg;

/// The output of an extractor: the source pixels at model resolution plus a
/// per-pixel label map (0 = background) and its objects, sorted largest-first.
/// This is exactly what the rest of the pipeline (crop, deskew) consumes.
pub struct Extraction {
    pub src: image::RgbaImage,
    pub width: u32,
    pub height: u32,
    pub labels: Vec<usize>,
    pub objects: Vec<Object>,
}

/// A swappable foreground extractor.
pub trait Extractor {
    fn extract(&mut self, input: &str) -> Result<Extraction>;
}

/// Build the extractor selected by `--extractor`.
pub fn build_extractor(args: &Args, cutoff: u8) -> Result<Box<dyn Extractor>> {
    match args.extractor {
        ExtractorKind::Rmbg => Ok(Box::new(RmbgExtractor::new(cutoff, args.min_area)?)),
        #[cfg(feature = "sam3")]
        ExtractorKind::Sam3Bg => Ok(Box::new(sam3::Sam3BgExtractor::new(
            args.sam_background.clone(),
            args.min_area,
        )?)),
        #[cfg(feature = "sam3")]
        ExtractorKind::Sam3Fg => Ok(Box::new(sam3::Sam3FgExtractor::new(
            args.sam_fg_prompts(),
            args.sam_overlap,
            args.min_area,
        )?)),
        #[cfg(not(feature = "sam3"))]
        ExtractorKind::Sam3Bg | ExtractorKind::Sam3Fg => anyhow::bail!(
            "this build has no SAM3 support; rebuild with the `sam3` feature (on by default)"
        ),
    }
}

/// RMBG: a single alpha mask thresholded to a binary foreground, split into
/// objects by connected components. The default extractor.
struct RmbgExtractor {
    model: usls::models::RMBG,
    cutoff: u8,
    min_area: usize,
}

impl RmbgExtractor {
    fn new(cutoff: u8, min_area: usize) -> Result<Self> {
        Ok(Self {
            model: build_rmbg()?,
            cutoff,
            min_area,
        })
    }
}

impl Extractor for RmbgExtractor {
    fn extract(&mut self, input: &str) -> Result<Extraction> {
        let image = Image::try_read(input).with_context(|| format!("failed to read image: {input}"))?;
        let ys = self.model.forward(std::slice::from_ref(&image))?;
        let mask = ys
            .first()
            .and_then(|y| y.masks.first())
            .context("model returned no mask")?;
        let alpha = mask.to_vec();

        let src = image.to_rgba8();
        let (w, h) = src.dimensions();
        if (w * h) as usize != alpha.len() {
            anyhow::bail!(
                "mask size ({}) does not match image size ({w}x{h})",
                alpha.len()
            );
        }

        let fg: Vec<bool> = alpha.iter().map(|&a| a >= self.cutoff).collect();
        let (labels, objects) = connected_components(&fg, w as usize, h as usize, self.min_area);
        Ok(Extraction {
            src,
            width: w,
            height: h,
            labels,
            objects,
        })
    }
}

#[cfg(feature = "sam3")]
mod sam3 {
    use anyhow::{Context, Result};
    use usls::Image;
    use usls::models::{SAM3, Sam3Prompt};

    use super::Extraction;
    use crate::components::{connected_components, objects_from_instances};
    use crate::model::build_sam3;

    /// SAM3 background inversion: segment the background by text prompt, union
    /// those masks, invert to get the foreground, then split by connected
    /// components.
    pub struct Sam3BgExtractor {
        model: SAM3,
        background: String,
        min_area: usize,
    }

    impl Sam3BgExtractor {
        pub fn new(background: String, min_area: usize) -> Result<Self> {
            Ok(Self {
                model: build_sam3()?,
                background,
                min_area,
            })
        }
    }

    impl super::Extractor for Sam3BgExtractor {
        fn extract(&mut self, input: &str) -> Result<Extraction> {
            let image =
                Image::try_read(input).with_context(|| format!("failed to read image: {input}"))?;
            let prompt = Sam3Prompt::new(&self.background);
            let ys = self
                .model
                .forward(std::slice::from_ref(&image), std::slice::from_ref(&prompt))?;

            let src = image.to_rgba8();
            let (w, h) = src.dimensions();

            // Union every background mask, then invert to foreground.
            let mut background = vec![false; (w * h) as usize];
            if let Some(y) = ys.first() {
                for mask in &y.masks {
                    for (b, v) in background.iter_mut().zip(mask_to_fg(mask, w, h)) {
                        *b |= v;
                    }
                }
            }
            let fg: Vec<bool> = background.iter().map(|&b| !b).collect();
            let (labels, objects) = connected_components(&fg, w as usize, h as usize, self.min_area);
            Ok(Extraction {
                src,
                width: w,
                height: h,
                labels,
                objects,
            })
        }
    }

    /// SAM3 multi-prompt foreground: run several object prompts, then merge
    /// overlapping detections (IoU above a threshold) into distinct objects.
    pub struct Sam3FgExtractor {
        model: SAM3,
        prompts: Vec<String>,
        overlap: f64,
        min_area: usize,
    }

    impl Sam3FgExtractor {
        pub fn new(prompts: Vec<String>, overlap: f64, min_area: usize) -> Result<Self> {
            Ok(Self {
                model: build_sam3()?,
                prompts,
                overlap,
                min_area,
            })
        }
    }

    impl super::Extractor for Sam3FgExtractor {
        fn extract(&mut self, input: &str) -> Result<Extraction> {
            let image =
                Image::try_read(input).with_context(|| format!("failed to read image: {input}"))?;
            let prompts: Vec<Sam3Prompt> =
                self.prompts.iter().map(|t| Sam3Prompt::new(t)).collect();
            let ys = self.model.forward(std::slice::from_ref(&image), &prompts)?;

            let src = image.to_rgba8();
            let (w, h) = src.dimensions();

            let mut instances: Vec<Vec<bool>> = Vec::new();
            if let Some(y) = ys.first() {
                for mask in &y.masks {
                    instances.push(mask_to_fg(mask, w, h));
                }
            }
            let merged = iou_merge(&instances, self.overlap);
            let (labels, objects) =
                objects_from_instances(&merged, w as usize, h as usize, self.min_area);
            Ok(Extraction {
                src,
                width: w,
                height: h,
                labels,
                objects,
            })
        }
    }

    /// Rasterize a SAM3 mask to a `w*h` foreground bitmap, resizing
    /// (nearest-neighbour) if the model returned it at a different resolution.
    fn mask_to_fg(mask: &usls::Mask, w: u32, h: u32) -> Vec<bool> {
        let (mw, mh) = mask.dimensions();
        let bytes = mask.to_vec();
        if (mw, mh) == (w, h) {
            return bytes.iter().map(|&b| b >= 128).collect();
        }
        let buf: image::GrayImage = image::ImageBuffer::from_raw(mw, mh, bytes)
            .expect("mask raster matches its dimensions");
        let resized = image::imageops::resize(&buf, w, h, image::imageops::FilterType::Nearest);
        resized.iter().map(|&b| b >= 128).collect()
    }

    /// Merge masks whose IoU exceeds `threshold` by unioning each connected group
    /// (Python `merge_overlapping_masks`, ported). Returns one mask per group.
    pub fn iou_merge(masks: &[Vec<bool>], threshold: f64) -> Vec<Vec<bool>> {
        let n = masks.len();
        if n <= 1 {
            return masks.to_vec();
        }
        let len = masks[0].len();

        // Adjacency: an edge between masks that overlap beyond the threshold.
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..n {
            for j in (i + 1)..n {
                if iou(&masks[i], &masks[j]) > threshold {
                    adjacency[i].push(j);
                    adjacency[j].push(i);
                }
            }
        }

        // Union each connected component of that graph into one mask.
        let mut visited = vec![false; n];
        let mut out: Vec<Vec<bool>> = Vec::new();
        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut merged = vec![false; len];
            let mut stack = vec![start];
            while let Some(k) = stack.pop() {
                if visited[k] {
                    continue;
                }
                visited[k] = true;
                for (m, v) in merged.iter_mut().zip(&masks[k]) {
                    *m |= *v;
                }
                for &next in &adjacency[k] {
                    if !visited[next] {
                        stack.push(next);
                    }
                }
            }
            out.push(merged);
        }
        out
    }

    /// Intersection-over-union of two equal-length bitmaps.
    fn iou(a: &[bool], b: &[bool]) -> f64 {
        let mut intersection = 0usize;
        let mut union = 0usize;
        for (&x, &y) in a.iter().zip(b) {
            if x && y {
                intersection += 1;
            }
            if x || y {
                union += 1;
            }
        }
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn iou_merge_unions_overlapping_keeps_disjoint() {
            // Two overlapping masks (share a pixel) and one disjoint mask.
            let a = vec![true, true, false, false];
            let b = vec![false, true, true, false];
            let c = vec![false, false, false, true];

            let merged = iou_merge(&[a, b, c], 0.0);
            assert_eq!(merged.len(), 2);
            // The union of a and b covers indices 0,1,2; the disjoint mask stays.
            assert!(merged.iter().any(|m| *m == vec![true, true, true, false]));
            assert!(merged.iter().any(|m| *m == vec![false, false, false, true]));
        }

        #[test]
        fn iou_merge_keeps_low_overlap_separate_above_threshold() {
            let a = vec![true, true, true, false];
            let b = vec![false, false, true, true]; // IoU with a = 1/4 = 0.25
            let merged = iou_merge(&[a, b], 0.5);
            assert_eq!(merged.len(), 2);
        }
    }
}
