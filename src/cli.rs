//! Command-line interface: argument parsing and validation.

use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Parser, ValueEnum};
use image::ImageFormat;

use crate::output::OutputFormat;

/// Output format for `--format`, overriding the format kropp would otherwise
/// infer. Only the formats that make sense for scan crops are offered; `jxl` is
/// lossless JPEG XL, which `image` cannot write and kropp encodes itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum OutputFormatArg {
    Png,
    Jpeg,
    Tiff,
    Webp,
    Bmp,
    Qoi,
    Jxl,
}

impl OutputFormatArg {
    pub fn to_output_format(self) -> OutputFormat {
        match self {
            Self::Png => OutputFormat::Image(ImageFormat::Png),
            Self::Jpeg => OutputFormat::Image(ImageFormat::Jpeg),
            Self::Tiff => OutputFormat::Image(ImageFormat::Tiff),
            Self::Webp => OutputFormat::Image(ImageFormat::WebP),
            Self::Bmp => OutputFormat::Image(ImageFormat::Bmp),
            Self::Qoi => OutputFormat::Image(ImageFormat::Qoi),
            Self::Jxl => OutputFormat::Jxl,
        }
    }
}

/// How the foreground objects are extracted from the scan before the crop and
/// deskew stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ExtractorKind {
    /// RMBG background-removal model (a single alpha mask, split into objects by
    /// connected components). The default.
    Rmbg,
    /// SAM3 background inversion: prompt the background, invert it to one
    /// foreground mask, then split by connected components.
    Sam3Bg,
    /// SAM3 multi-prompt foreground: run each object prompt, merge overlapping
    /// detections by IoU into distinct objects.
    Sam3Fg,
}

/// Crop and deskew objects from a flatbed scan. Each foreground object is
/// detected, straightened upright, and written out as its own cropped file.
/// Foreground detection currently uses the RMBG background-removal model.
#[derive(Parser, Debug)]
#[command(
    name = "kropp",
    about = "Crop and deskew objects from flatbed scans"
)]
pub struct Args {
    /// Input image or directory (explicit form). Usually you can pass paths
    /// positionally instead — `kropp photo.jpg` — or, on Windows, drag files
    /// onto the executable.
    #[arg(short, long)]
    pub input: Option<String>,

    /// Input images or directories given positionally. Accepts multiple paths;
    /// this is what Windows passes when you drag files onto the executable. By
    /// default each object is written next to its input (a folder's crops go
    /// into that folder), unless --output/--output-dir overrides it.
    #[arg(value_name = "PATHS")]
    pub paths: Vec<PathBuf>,

    /// Foreground extraction mode: `rmbg` (default), `sam3-bg` (SAM3 background
    /// inversion), or `sam3-fg` (SAM3 multi-prompt). The SAM3 modes require a
    /// build with the `sam3` feature (on by default).
    #[arg(long, value_enum, default_value_t = ExtractorKind::Rmbg)]
    pub extractor: ExtractorKind,

    /// Background text prompt for `--extractor sam3-bg`. The mask of this prompt
    /// is inverted to obtain the foreground.
    #[arg(long, default_value = "background")]
    pub sam_background: String,

    /// Object text prompts for `--extractor sam3-fg`. Repeat to pass several;
    /// detections are merged across prompts by IoU. Defaults to a generic set.
    #[arg(long = "sam-prompt", value_name = "TEXT")]
    pub sam_prompts: Vec<String>,

    /// IoU threshold above which `sam3-fg` detections merge into one object.
    #[arg(long, default_value_t = 0.001)]
    pub sam_overlap: f64,

    /// Output path for a single input file; a `_<index>` suffix is added per
    /// object. Defaults to the input path's stem and format, e.g.
    /// `photo.jpg` -> `photo_0.jpg`.
    #[arg(short, long)]
    pub output: Option<String>,

    /// Output directory when the input is a directory. Each input file keeps
    /// its own name and format under this directory.
    #[arg(long)]
    pub output_dir: Option<String>,

    /// Merge several overlapping passes of one oversized original into a single
    /// image before cropping. Accepts a list of files, or directories:
    ///
    ///   --merge-input a1.tif a2.tif      one group from these files
    ///   --merge-input passes/            one group from that directory
    ///   --merge-input left/ right/       two groups, one per directory
    ///
    /// When every argument is a directory each one is merged as its own group;
    /// otherwise the whole list is a single group. The flag may be repeated and
    /// its values accumulate. Passes must overlap; the merged result is
    /// deskewed, uprighted, and cropped as one object.
    #[arg(long = "merge-input", value_name = "PATHS", num_args = 1.., verbatim_doc_comment)]
    pub merge_input: Vec<String>,

    /// Keep the merged image alongside the crops, named `<stem>_merged.<ext>`.
    /// Without this the merge is written to a temporary file and removed once
    /// its crops have been produced.
    #[arg(long, default_value_t = false)]
    pub keep_merged: bool,

    /// Confine merge keypoints to the extracted foreground, keeping platen
    /// artifacts (bed edges, dust, lid shadow) out of the alignment. Off by
    /// default: it also discards every keypoint the segmenter did not call
    /// foreground, which on a tight mask starves the fit. Costs an extra
    /// extraction pass per input.
    #[arg(long, default_value_t = false)]
    pub merge_mask_features: bool,

    /// How many passes to detect features on at once. Feature detection is the
    /// slowest stage of a merge and also its memory high-water mark — several
    /// gigabytes per pass at full platen resolution — so the default balances
    /// the two rather than using every core. 0 picks it automatically.
    #[arg(long, value_name = "N", default_value_t = 0)]
    pub merge_jobs: usize,

    /// Write the merged image to this exact path instead of the default
    /// `<stem>_merged.<ext>`, and keep it. Only valid with a single merge
    /// group. Crops are still written next to it with `_<index>` suffixes.
    #[arg(long, value_name = "PATH")]
    pub merge_output: Option<String>,

    /// Write every crop in this format instead of the one kropp infers, and
    /// rename the output extension to match. Overrides both the input's format
    /// and `-o`'s extension. `jxl` is lossless JPEG XL. A lossy choice
    /// (`jpeg`) still needs --allow-lossy-conversion. Unset by default, which
    /// leaves format selection exactly as it was.
    #[arg(long, value_enum, value_name = "FORMAT")]
    pub format: Option<OutputFormatArg>,

    /// Alpha threshold as a percentage: pixels below this become fully
    /// transparent, at or above become fully opaque.
    #[arg(short, long, default_value_t = 95.0)]
    pub threshold: f32,

    /// Minimum object area in pixels; connected components smaller than this
    /// are discarded as noise.
    #[arg(short, long, default_value_t = 0)]
    pub min_area: usize,

    /// Minimum component size as a percentage of the smaller image dimension:
    /// a component is dropped as noise unless its longer bounding-box side
    /// reaches at least this fraction of `min(width, height)`. Set to 0 to
    /// disable.
    #[arg(long, default_value_t = 10.0)]
    pub min_side_percent: f64,

    /// Agreement tolerance in degrees: text boxes within this of the longest
    /// line are inliers; the rest are discarded as outliers.
    #[arg(short = 'a', long, default_value_t = 10.0)]
    pub angle_tol: f64,

    /// Force text-based rotation for every object: run the text detector and
    /// prefer its angle when text is found. Without this flag, text rotation is
    /// only auto-enabled for sufficiently non-rectangular masks.
    #[arg(long, default_value_t = false)]
    pub text: bool,

    /// In default mode, run text rotation for masks with rectangularity below
    /// this value. Rectangularity scores how strongly the mask boundary and fill
    /// support its fitted rectangle. Set to 0 to disable auto text. --text
    /// ignores this threshold and always tries text rotation.
    #[arg(long, default_value_t = 0.30)]
    pub auto_text_rectangularity_threshold: f64,

    /// Force every object through the rectangular pipeline, ignoring its
    /// rectangularity score (equivalent to
    /// `--auto-text-rectangularity-threshold 0`). Cannot be combined with
    /// `--text`, which forces the opposite.
    #[arg(long, default_value_t = false, conflicts_with = "text")]
    pub force_rectangular: bool,

    /// Padding percentage to add around non-rectangular objects after the final
    /// tight crop. Applied per side; 2 means 2% of the crop width on left/right
    /// and 5% of the crop height on top/bottom.
    #[arg(long, default_value_t = 2.0)]
    pub non_rectangular_padding: f64,

    /// Cut the object out using the mask as an alpha channel (transparent
    /// background, RGBA output). By default crops keep their original
    /// background as a plain rectangular RGB image.
    #[arg(long, default_value_t = false)]
    pub alpha: bool,

    /// Write crops in the input's lossy format instead of converting to PNG.
    /// By default a lossy input (e.g. JPEG) is written as lossless PNG so the
    /// crop isn't recompressed; lossless inputs always keep their format.
    #[arg(long, default_value_t = false)]
    pub allow_lossy_conversion: bool,

    /// Report the detected angle but skip rotating crops to upright.
    #[arg(long, default_value_t = false)]
    pub no_deskew: bool,

    /// Skip the document-orientation model for rectangular crops, leaving their
    /// 0/90/180/270 orientation uncorrected. Text-driven crops still use the
    /// textline 0/180 vote.
    #[arg(long, default_value_t = false)]
    pub no_doc_orient: bool,

    /// Path to the document-orientation ONNX used to upright rectangular crops,
    /// loaded from disk. Defaults to the bundled `onnx/bkori.onnx`; a relative
    /// path is resolved against the executable's directory (so it works under
    /// drag-and-drop). Any model is assumed to share the default's preprocessing
    /// and 0/90/180/270 output.
    #[arg(long, default_value = "onnx/bkori.onnx")]
    pub doc_orient_model: Option<String>,

    /// Reprocess directory inputs even when matching outputs already exist.
    /// Without this, an input whose `<stem>_` outputs are already present is
    /// skipped.
    #[arg(long, default_value_t = false)]
    pub overwrite: bool,

    /// Fit every mask rectangle to the maximal extent (the "safezone"),
    /// enclosing object extrusions fully. By default only non-rectangular
    /// objects use the safezone; rectangular objects use the tight body fit.
    #[arg(long, default_value_t = false)]
    pub safezone: bool,

    /// Print detailed per-object angle diagnostics (text boxes, mask vote,
    /// consensus inliers/outliers) to stderr.
    #[arg(short = 'v', long, default_value_t = false)]
    pub debug: bool,

    /// Mirror diagnostic output (everything normally printed to stderr) to this
    /// file as well. Defaults to `kropp-debug.log` when --debug is set and this
    /// is not given.
    #[arg(long)]
    pub log_file: Option<String>,
}

impl Args {
    /// The effective auto-text rectangularity threshold. `--force-rectangular`
    /// pins it to 0 so no object scores as non-rectangular and every object
    /// takes the rectangular pipeline.
    pub fn rectangularity_threshold(&self) -> f64 {
        if self.force_rectangular {
            0.0
        } else {
            self.auto_text_rectangularity_threshold
        }
    }

    /// Resolve the document-orientation model path. A relative path is resolved
    /// against the executable's directory (not the working directory), so a model
    /// bundled next to the exe is found regardless of where it's launched from —
    /// e.g. under drag-and-drop. Absolute paths are used as given.
    pub fn doc_orient_model_path(&self) -> Option<PathBuf> {
        let path = PathBuf::from(self.doc_orient_model.as_ref()?);
        if path.is_absolute() {
            return Some(path);
        }
        if let Ok(exe) = std::env::current_exe()
            && let Some(dir) = exe.parent()
        {
            return Some(dir.join(path));
        }
        // No executable path available; fall back to the path as given.
        Some(path)
    }

    /// Object prompts for SAM3 multi-prompt (`sam3-fg`) mode, falling back to a
    /// generic flatbed-scan set when none are given on the command line.
    #[cfg(feature = "sam3")]
    pub fn sam_fg_prompts(&self) -> Vec<String> {
        if self.sam_prompts.is_empty() {
            ["photo", "document", "card", "booklet", "leaflet", "disc"]
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            self.sam_prompts.clone()
        }
    }

    /// Validate the numeric ranges and return the alpha cutoff on the 0..=255
    /// scale derived from `threshold`.
    pub fn validate(&self) -> Result<u8> {
        if !(0.0..=1.0).contains(&self.sam_overlap) {
            bail!("sam overlap must be between 0 and 1, got {}", self.sam_overlap);
        }
        if !(0.0..=100.0).contains(&self.threshold) {
            bail!("threshold must be between 0 and 100, got {}", self.threshold);
        }
        if !(0.0..=1.0).contains(&self.auto_text_rectangularity_threshold) {
            bail!(
                "auto text rectangularity threshold must be between 0 and 1, got {}",
                self.auto_text_rectangularity_threshold
            );
        }
        if !(0.0..=100.0).contains(&self.non_rectangular_padding) {
            bail!(
                "non-rectangular padding must be between 0 and 100, got {}",
                self.non_rectangular_padding
            );
        }
        if !(0.0..=100.0).contains(&self.min_side_percent) {
            bail!(
                "min side percent must be between 0 and 100, got {}",
                self.min_side_percent
            );
        }
        // Convert the percentage cutoff to the 0..=255 alpha scale.
        Ok((self.threshold / 100.0 * 255.0).round() as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn force_rectangular_pins_rectangularity_threshold_to_zero() {
        let forced = Args::parse_from(["kropp", "-i", "x", "--force-rectangular"]);
        assert_eq!(forced.rectangularity_threshold(), 0.0);

        let default = Args::parse_from(["kropp", "-i", "x"]);
        assert_eq!(default.rectangularity_threshold(), 0.30);
    }

    #[test]
    fn force_rectangular_conflicts_with_text() {
        let result = Args::try_parse_from(["kropp", "-i", "x", "--text", "--force-rectangular"]);
        assert!(result.is_err());
    }

    #[test]
    fn positional_paths_collect_multiple_inputs() {
        let args = Args::parse_from(["kropp", "a.jpg", "b.png", "folder"]);
        assert_eq!(
            args.paths,
            vec![
                PathBuf::from("a.jpg"),
                PathBuf::from("b.png"),
                PathBuf::from("folder")
            ]
        );
        assert!(args.input.is_none());
    }

    #[test]
    fn merge_input_collects_several_paths() {
        let args = Args::parse_from(["kropp", "--merge-input", "a1.tif", "a2.tif", "a3.tif"]);
        assert_eq!(args.merge_input, vec!["a1.tif", "a2.tif", "a3.tif"]);
    }

    /// Repeating the flag accumulates rather than replacing, so a long group
    /// can be split across several flags.
    #[test]
    fn merge_input_accumulates_across_occurrences() {
        let args = Args::parse_from([
            "kropp",
            "--merge-input",
            "a1.tif",
            "--merge-input",
            "a2.tif",
        ]);
        assert_eq!(args.merge_input, vec!["a1.tif", "a2.tif"]);
    }

    /// A merge-only run is a complete command; it must not require -i as well.
    #[test]
    fn merge_input_alone_parses_without_other_inputs() {
        let args = Args::parse_from(["kropp", "--merge-input", "a.tif", "b.tif"]);
        assert!(args.input.is_none());
        assert!(args.paths.is_empty());
        assert_eq!(args.merge_input.len(), 2);
    }

    #[test]
    fn merged_output_is_discarded_unless_requested() {
        assert!(!Args::parse_from(["kropp", "-i", "x"]).keep_merged);
        assert!(Args::parse_from(["kropp", "-i", "x", "--keep-merged"]).keep_merged);
    }

    #[test]
    fn input_flag_still_parses() {
        let args = Args::parse_from(["kropp", "-i", "photo.jpg"]);
        assert_eq!(args.input.as_deref(), Some("photo.jpg"));
        assert!(args.paths.is_empty());
    }
}
