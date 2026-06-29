//! Command-line interface: argument parsing and validation.

use anyhow::{Result, bail};
use clap::Parser;

/// Crop and deskew objects from a flatbed scan. Each foreground object is
/// detected, straightened upright, and written out as its own cropped file.
/// Foreground detection currently uses the RMBG background-removal model.
#[derive(Parser, Debug)]
#[command(
    name = "kropp",
    about = "Crop and deskew objects from flatbed scans"
)]
pub struct Args {
    /// Path to the input image or directory of images.
    #[arg(short, long)]
    pub input: String,

    /// Output path for a single input file; a `_<index>` suffix is added per
    /// object. Defaults to the input path's stem and format, e.g.
    /// `photo.jpg` -> `photo_0.jpg`.
    #[arg(short, long)]
    pub output: Option<String>,

    /// Output directory when the input is a directory. Each input file keeps
    /// its own name and format under this directory.
    #[arg(long)]
    pub output_dir: Option<String>,

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

    /// Path to a custom (e.g. finetuned) document-orientation ONNX, loaded from
    /// disk instead of downloading the default. Assumed to share the default's
    /// preprocessing and 0/90/180/270 output.
    #[arg(long)]
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

    /// Validate the numeric ranges and return the alpha cutoff on the 0..=255
    /// scale derived from `threshold`.
    pub fn validate(&self) -> Result<u8> {
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
}
