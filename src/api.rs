//! The library façade: the same stages the CLI runs, separated so an
//! interactive caller can stop between them.
//!
//! [`Pipeline::process_image`] is the CLI's unit of work — one scan in, every
//! crop written — and it is the right unit when nobody is watching. It is the
//! wrong one when somebody is: the agent shows a contributor every object found
//! on a platen, lets them drop the ones that aren't wanted and group the scans
//! that are really overlapping passes of a single original, and only then
//! decides what to commit. That review has to happen *between* extraction and
//! output, which is exactly where `process_image` has no seam.
//!
//! So this module re-cuts the same work three ways:
//!
//! * [`Kropp::extract`] — find the objects, and hand the result back to be held.
//! * [`Kropp::raw_crop`] — a loose, padded, unrotated crop of one object, in
//!   memory. Cheap: no model runs, it is a rectangle out of pixels already read.
//! * [`Kropp::finish_object`] — the full treatment (deskew, upright, tight
//!   crop) for one object that survived the review, written to a named path.
//!
//! **Extraction is the whole cost.** RMBG or SAM3 over a full-platen scan
//! dwarfs every other stage here, and both renderings read the same
//! [`Extraction`], so a caller holds one per source rather than extracting
//! twice. That is also what makes the review coherent without any matching
//! step: object *k* of a scan is the same `Object` in both passes because it is
//! literally the same struct, not a re-detection that has to be paired up.

use std::path::Path;

use anyhow::{Context, Result, bail};
use image::DynamicImage;

use crate::cli::Args;
use crate::extract::Extraction;
use crate::merge;
use crate::model::read_original_image;
use crate::output::{OutputFormat, encode_rgb, render_rgb, rotated_crop_bounds};
use crate::pipeline::{ObjectSink, Pipeline};

/// A crop handed back in memory, with the size it came out at.
pub struct ObjectRender {
    pub image: DynamicImage,
    pub width: u32,
    pub height: u32,
}

/// The result of merging several overlapping passes.
pub struct MergeOutcome {
    pub image: DynamicImage,
    /// Indices into the slice passed to [`Kropp::merge_sources`] that actually
    /// made it into the merge. A pass that overlapped nothing is left out
    /// rather than failing the whole group, so a caller that cares about the
    /// difference has to look.
    pub members: Vec<usize>,
}

/// A loaded extractor and the settings it was built with.
///
/// Building this loads a segmentation model, so keep one and reuse it across
/// every scan in a batch — the same reason the CLI builds a single `Pipeline`
/// for a whole directory.
pub struct Kropp {
    pipeline: Pipeline,
    args: Args,
}

impl Kropp {
    /// Build the extractor described by `args`. Only the extraction and
    /// geometry settings matter here; the input/output fields are ignored,
    /// since this API is told what to read and where to write per call.
    pub fn new(args: Args) -> Result<Self> {
        let cutoff = args.validate()?;
        let pipeline = Pipeline::new(&args, cutoff)?;
        Ok(Self { pipeline, args })
    }

    pub fn args(&self) -> &Args {
        &self.args
    }

    /// Read a source image and the format it was stored in, the same way the
    /// CLI does — 16-bit stays 16-bit.
    pub fn read_source(path: &Path) -> Result<(DynamicImage, image::ImageFormat)> {
        read_original_image(path)
    }

    /// Find the objects on one scan. This is the expensive call; hold the
    /// result rather than making it twice for the same path.
    ///
    /// Objects are filtered by `--min-side-percent` here so that the indices a
    /// caller reviews are the same indices [`Self::finish_object`] expects.
    pub fn extract(&mut self, path: &Path) -> Result<Extraction> {
        let mut extraction = self
            .pipeline
            .extract_image(&path.to_string_lossy())
            .with_context(|| format!("failed to find objects in {}", path.display()))?;

        let (w, h) = (extraction.width as usize, extraction.height as usize);
        let min_side = self.args.min_side_percent;
        extraction
            .objects
            .retain(|obj| crate::components::meets_min_side(obj, w, h, min_side));
        Ok(extraction)
    }

    /// A loose, axis-aligned, padded crop of one object — the review rendering.
    ///
    /// Deliberately *not* deskewed or uprighted. This is both the cheap thing
    /// to produce (no classifier runs, so a platen of twelve objects costs one
    /// extraction and twelve rectangle copies) and the honest thing to archive:
    /// the object as it sat on the glass, with margin, before any decision was
    /// made about it. Straightening is [`Self::finish_object`]'s job, and it
    /// works from the source rather than from this.
    ///
    /// `pad_percent` is per side, as a percentage of the crop's own size.
    pub fn raw_crop(
        &self,
        source: &DynamicImage,
        extraction: &Extraction,
        index: usize,
        pad_percent: f64,
    ) -> Result<ObjectRender> {
        let obj = extraction
            .objects
            .get(index)
            .with_context(|| format!("object {index} is out of range"))?;

        let mut bounds = rotated_crop_bounds(
            None,
            &extraction.labels,
            extraction.width as usize,
            obj,
            0.0,
        )
        .context("object has no foreground pixels")?;
        bounds.expand_percent(pad_percent);

        let image = render_rgb(source, bounds, 0.0, 0)?;
        let (width, height) = (image.width(), image.height());
        Ok(ObjectRender {
            image,
            width,
            height,
        })
    }

    /// Encode an already-rendered crop. Split from [`Self::raw_crop`] so the
    /// same pixels can be shown to a contributor and written to disk without
    /// being cropped twice.
    pub fn encode(&self, render: &ObjectRender, path: &Path, format: OutputFormat) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        encode_rgb(&render.image, path, format)?;
        Ok(())
    }

    /// The finished crop of one object: deskewed, uprighted by the orientation
    /// classifier, cropped to its fitted rectangle, and written to `path`.
    ///
    /// Takes the same `Extraction` the review used, so `index` means what the
    /// contributor saw. Reads pixels from `source`, not from any earlier crop —
    /// deskew resamples the neighbourhood around the object, and it can only do
    /// that where the neighbourhood still exists.
    pub fn finish_object(
        &mut self,
        source: &DynamicImage,
        extraction: &Extraction,
        index: usize,
        path: &Path,
        format: OutputFormat,
        extra_turns: u8,
    ) -> Result<()> {
        let obj = extraction
            .objects
            .get(index)
            .with_context(|| format!("object {index} is out of range"))?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }

        self.pipeline.process_object(
            index,
            obj,
            &extraction.src,
            &extraction.labels,
            extraction.width,
            source,
            &self.args,
            ObjectSink::File(path, format),
            extra_turns,
        )?;
        Ok(())
    }

    /// The same finished crop [`Self::finish_object`] would write, returned
    /// instead of written.
    ///
    /// This is the review rendering for a caller that reviews *outputs* rather
    /// than candidates: what the contributor judges is the crop as it will
    /// exist, deskewed and uprighted and cut to its fitted rectangle, not the
    /// padded rectangle it was cut from. `extra_turns` lets the same call show
    /// a rotation the contributor asked for.
    ///
    /// It costs what `finish_object` costs — the orientation classifier runs —
    /// so it is a per-object price, not a per-scan one.
    pub fn render_object(
        &mut self,
        source: &DynamicImage,
        extraction: &Extraction,
        index: usize,
        extra_turns: u8,
    ) -> Result<ObjectRender> {
        let obj = extraction
            .objects
            .get(index)
            .with_context(|| format!("object {index} is out of range"))?;

        let image = self
            .pipeline
            .process_object(
                index,
                obj,
                &extraction.src,
                &extraction.labels,
                extraction.width,
                source,
                &self.args,
                ObjectSink::Memory,
                extra_turns,
            )?
            .context("the render produced no image")?;
        let (width, height) = (image.width(), image.height());
        Ok(ObjectRender {
            image,
            width,
            height,
        })
    }

    /// Merge several overlapping passes of one oversized original.
    ///
    /// Unlike the CLI's merge path this never touches disk: the passes come in
    /// as decoded images and the result goes back the same way, so a caller can
    /// extract objects from it directly. `--merge-mask-features` is not
    /// supported here, since masking would mean an extra extraction per pass
    /// and the caller has usually already paid for those.
    pub fn merge_sources(&mut self, passes: &[&DynamicImage]) -> Result<MergeOutcome> {
        if passes.len() < 2 {
            bail!("a merge needs at least two passes, got {}", passes.len());
        }
        let merged = merge::merge(
            passes,
            None,
            self.args.merge_jobs,
            self.args.merge_detect_mp,
        )?;
        Ok(MergeOutcome {
            image: merged.image,
            members: merged.members,
        })
    }
}

/// Rotate a written crop in place by `turns` quarter-turns clockwise, keeping
/// its format and bit depth.
///
/// This is the contributor's last word on a crop: the orientation classifier is
/// right most of the time and wrong often enough to need overriding, and it is
/// cheaper to re-encode one file than to hold every finished crop in memory
/// waiting to hear whether it was correct. Lossless throughout, so a rotation
/// costs nothing but the encode.
pub fn rotate_in_place(path: &Path, turns: u8) -> Result<()> {
    if turns.is_multiple_of(4) {
        return Ok(());
    }
    let (image, format) = read_original_image(path)?;
    let rotated = match turns % 4 {
        1 => image.rotate90(),
        2 => image.rotate180(),
        _ => image.rotate270(),
    };
    // `render_rgb`'s output shape is what `encode_rgb` accepts; normalise to it
    // so a greyscale or alpha-carrying source doesn't fall through the match.
    let rotated = match rotated {
        rgb @ (DynamicImage::ImageRgb8(_) | DynamicImage::ImageRgb16(_)) => rgb,
        other => DynamicImage::ImageRgb8(other.to_rgb8()),
    };
    encode_rgb(&rotated, path, OutputFormat::Image(format))?;
    Ok(())
}
