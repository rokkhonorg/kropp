#[macro_use]
mod logging;
mod angle;
mod cli;
mod components;
mod debug_overlay;
mod extract;
mod merge;
mod model;
mod output;
mod pipeline;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;

use crate::cli::Args;
use crate::model::read_original_image;
use crate::output::{OutputPlan, suffixed_path};
use crate::pipeline::{Pipeline, collect_input_paths, output_dir_has_prefix};

fn main() {
    // When launched from Explorer (double-click or drag-and-drop) the process
    // gets its own console that vanishes on exit, so pause at the end to let the
    // user read the output. A run from an existing terminal must not pause.
    let standalone = launched_standalone();

    let code = match run() {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("error: {e:#}");
            1
        }
    };

    if standalone {
        wait_for_enter();
    }
    std::process::exit(code);
}

fn run() -> Result<()> {
    let args = Args::parse();

    // Install the log-file sink before any diagnostics are emitted. An explicit
    // --log-file wins; otherwise --debug writes to a default file.
    let log_path = args
        .log_file
        .clone()
        .map(PathBuf::from)
        .or_else(|| args.debug.then(|| PathBuf::from("kropp-debug.log")));
    if let Some(path) = &log_path {
        logging::init_log_file(path)?;
    }

    let cutoff = args.validate()?;
    // A run may merge groups, process ordinary inputs, or both. Only demand
    // ordinary inputs when no merge group was given, so `--merge-input` alone
    // is a complete command.
    let merge_groups = resolve_merge_groups(&args.merge_input)?;
    let merging = !merge_groups.is_empty();
    let inputs = if merging {
        resolve_optional_inputs(&args)?
    } else {
        resolve_inputs(&args)?
    };

    // --output renames a single file's output; it can't address many inputs.
    if args.output.is_some() {
        if args.output_dir.is_some() {
            bail!("--output and --output-dir cannot be combined");
        }
        if merge_groups.len() > 1 {
            bail!("--output cannot address several --merge-input groups; use --output-dir");
        }
        if !merging && (inputs.len() != 1 || !inputs[0].is_file()) {
            bail!(
                "--output is only valid with a single image file; \
                 use --output-dir, or omit both to write next to each input"
            );
        }
    }
    if args.merge_output.is_some() {
        if merge_groups.is_empty() {
            bail!("--merge-output is only meaningful with --merge-input");
        }
        if merge_groups.len() > 1 {
            bail!(
                "--merge-output names one file but {} merge groups were given; \
                 use --output-dir instead",
                merge_groups.len()
            );
        }
    }

    // Build the extractor (and lazy classifiers) once and reuse across every input.
    let mut pipeline = Pipeline::new(&args, cutoff)?;

    // Several units of work in one run: one failure shouldn't abort the rest.
    // A lone unit stays fatal so its error surfaces directly.
    let units = inputs.len() + merge_groups.len();
    let multi = units > 1;

    for group in &merge_groups {
        if let Err(e) = process_merge_group(&mut pipeline, group, &args) {
            if multi {
                elog!("skipping merge group ({e:#})");
            } else {
                return Err(e);
            }
        }
    }

    for input in &inputs {
        if let Err(e) = process_input(&mut pipeline, input, &args, multi) {
            if multi {
                elog!("skipping {}: {e:#}", input.display());
            } else {
                return Err(e);
            }
        }
    }
    Ok(())
}

/// Inputs when a merge group is also present: the same resolution as
/// `resolve_inputs`, except that having none is fine.
fn resolve_optional_inputs(args: &Args) -> Result<Vec<PathBuf>> {
    if args.paths.is_empty() && args.input.is_none() {
        return Ok(Vec::new());
    }
    resolve_inputs(args)
}

/// Split `--merge-input` into the groups to merge.
///
/// Every argument being a directory means each directory is its own group,
/// which is how a batch of separate originals is expressed. Any file present
/// makes the whole list one group. Mixing the two is rejected rather than
/// guessed at, since either reading would silently discard work.
fn resolve_merge_groups(entries: &[String]) -> Result<Vec<Vec<PathBuf>>> {
    if entries.is_empty() {
        return Ok(Vec::new());
    }
    let paths: Vec<PathBuf> = entries.iter().map(PathBuf::from).collect();
    let directories = paths.iter().filter(|p| p.is_dir()).count();

    if directories == paths.len() {
        let mut groups = Vec::with_capacity(paths.len());
        for directory in &paths {
            let passes = collect_input_paths(directory)?;
            if passes.len() < 2 {
                bail!(
                    "--merge-input {} holds {} image(s); a merge needs at least two",
                    directory.display(),
                    passes.len()
                );
            }
            groups.push(passes);
        }
        return Ok(groups);
    }

    if directories > 0 {
        bail!(
            "--merge-input takes either a list of files (one group) or only \
             directories (one group each), not a mix"
        );
    }
    if paths.len() < 2 {
        bail!("--merge-input needs at least two files, or a directory of them");
    }
    Ok(vec![paths])
}

/// Merge one group into a single image, then run the normal crop pipeline over
/// it. The merged image is written to disk first because the extractor reads by
/// path; `--keep-merged` decides whether that file outlives the run.
fn process_merge_group(pipeline: &mut Pipeline, passes: &[PathBuf], args: &Args) -> Result<()> {
    println!("merging {} passes", passes.len());

    let mut images = Vec::with_capacity(passes.len());
    let mut masks = Vec::with_capacity(passes.len());
    let mut format = image::ImageFormat::Tiff;
    for (idx, path) in passes.iter().enumerate() {
        let (image, image_format) = read_original_image(path)?;
        if idx == 0 {
            format = image_format;
        }
        // Extraction is only needed when masking features, and it is a full
        // model pass per input, so skip it otherwise.
        if args.merge_mask_features {
            let path_str = path.to_string_lossy();
            masks.push(pipeline.extract_mask(&path_str)?);
        }
        images.push(image);
    }

    let refs: Vec<&image::DynamicImage> = images.iter().collect();
    let merged = merge::merge(
        &refs,
        args.merge_mask_features.then_some(masks.as_slice()),
    )?;
    if args.debug {
        log_merge_report(passes, &merged);
    }

    // A pass left out did not overlap the others confidently. That is nearly
    // always a stray file in the group, so say so rather than failing.
    if merged.members.len() != passes.len() {
        for (idx, path) in passes.iter().enumerate() {
            if !merged.members.contains(&idx) {
                elog!(
                    "warning: {} did not overlap the rest and was left out of the merge",
                    path.display()
                );
            }
        }
    }

    // The merge is lossless work; never hand it to a lossy encoder.
    let merged_path = merged_output_path(&passes[0], args, format)?;
    if let Some(parent) = merged_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    merged
        .image
        .save(&merged_path)
        .with_context(|| format!("failed to write merged image: {}", merged_path.display()))?;
    println!(
        "merged -> {} ({}x{})",
        merged_path.display(),
        merged.image.width(),
        merged.image.height()
    );

    // A picture of which pass supplied which region — far easier to judge than
    // the numeric report, and the fastest way to spot a seam in a bad place.
    if args.debug {
        let overlay_path = suffixed_path(&merged_path, ".regions");
        match merge::composite::ownership_overlay(&merged.image, &merged.ownership) {
            Ok(overlay) => match overlay.save(&overlay_path) {
                Ok(()) => println!("regions -> {}", overlay_path.display()),
                Err(e) => elog!("could not write region overlay: {e}"),
            },
            Err(e) => elog!("could not build region overlay: {e:#}"),
        }
    }

    let result = process_file(pipeline, &merged_path, args, false);
    // Naming the merge explicitly is a request to keep it.
    if !args.keep_merged && args.merge_output.is_none() {
        let _ = std::fs::remove_file(&merged_path);
    }
    result
}

/// Report how the merge was arrived at, under `--debug`.
///
/// The column to read first is `resid`, the mean pixel error of each accepted
/// fit. Under about a pixel the passes genuinely line up; several pixels means
/// the alignment is a compromise and the blender is hiding it, which is where
/// visible ghosting comes from if it shows at all.
fn log_merge_report(passes: &[PathBuf], merged: &merge::Merged) {
    let report = &merged.report;
    let name = |index: usize| {
        passes
            .get(index)
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| format!("#{index}"))
    };

    elog!("merge report");
    elog!("  passes:");
    for (index, ((w, h), features)) in report.passes.iter().enumerate() {
        elog!(
            "    {:<28} {w}x{h}  {features} keypoints on the object",
            name(index)
        );
    }

    elog!("  pairwise fits:");
    if report.fits.is_empty() {
        elog!("    (none: no pair produced a transform)");
    }
    for fit in &report.fits {
        let used = if fit.confidence >= merge::align::DEFAULT_CONFIDENCE_THRESHOLD {
            "accepted"
        } else {
            "rejected"
        };
        elog!(
            "    {} -> {}  {}/{} inliers  conf {:.2}  resid {:.2}px  \
             dx {:+.1} dy {:+.1} rot {:+.3} deg  [{used}]",
            name(fit.from),
            name(fit.to),
            fit.inliers,
            fit.matches,
            fit.confidence,
            fit.residual,
            fit.transform.tx,
            fit.transform.ty,
            fit.transform.theta.to_degrees(),
        );
    }

    let masked = if report.masked_features {
        " (confined to the extracted foreground)"
    } else {
        ""
    };
    elog!("  features{masked}");
    elog!("  frame: reference {} (bundle adjusted)", name(report.reference));
    for (slot, &member) in merged.members.iter().enumerate() {
        let placement = report.transforms[slot];
        elog!(
            "    {:<28} at ({:+.1}, {:+.1}) rot {:+.3} deg scale {:.5}",
            name(member),
            placement.tx,
            placement.ty,
            placement.theta.to_degrees(),
            placement.scale,
        );
    }

    let (canvas_w, canvas_h) = report.canvas;
    let input_area: u64 = report
        .passes
        .iter()
        .map(|((w, h), _)| *w as u64 * *h as u64)
        .sum();
    elog!(
        "  canvas: {canvas_w}x{canvas_h} ({:.0}% of the summed pass area — \
         near 100% means little overlap was found)",
        (canvas_w as f64 * canvas_h as f64) / input_area as f64 * 100.0
    );
    if report.levels <= 1 {
        elog!("  blending: feather only (canvas too large for multi-band; expect ghosting)");
    } else {
        elog!("  blending: multi-band, {} bands", report.levels);
    }
}

/// Where the merged image lands: next to the first pass (or under
/// `--output-dir`), named after it with a `_merged` suffix. Lossy input formats
/// are written as PNG so the merge is not recompressed.
fn merged_output_path(
    first_pass: &Path,
    args: &Args,
    format: image::ImageFormat,
) -> Result<PathBuf> {
    // An explicit --merge-output wins outright, extension and all.
    if let Some(explicit) = &args.merge_output {
        return Ok(PathBuf::from(explicit));
    }

    let stem = first_pass
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("merged");
    let extension = match format {
        image::ImageFormat::Jpeg | image::ImageFormat::Avif => "png",
        other => other.extensions_str().first().copied().unwrap_or("png"),
    };
    let directory = match &args.output_dir {
        Some(dir) => PathBuf::from(dir),
        None => first_pass
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
    };
    Ok(directory.join(format!("{stem}_merged.{extension}")))
}

/// Resolve the inputs to process: positional paths if any (the drag-and-drop
/// form), otherwise the single `--input`. Errors when neither is given or both.
fn resolve_inputs(args: &Args) -> Result<Vec<PathBuf>> {
    if !args.paths.is_empty() {
        if args.input.is_some() {
            bail!("provide inputs with --input or as positional paths, not both");
        }
        return Ok(args.paths.clone());
    }
    if let Some(input) = &args.input {
        return Ok(vec![PathBuf::from(input)]);
    }
    bail!(
        "no input given; pass an image or folder (e.g. `kropp photo.jpg`), \
         or drag files onto the executable"
    );
}

/// Dispatch one top-level input. A directory is processed in place (or into
/// `--output-dir`); a file is written next to itself (or per `--output` /
/// `--output-dir`). `label` adds the input path to the summary line when more
/// than one input is being processed.
fn process_input(pipeline: &mut Pipeline, input: &Path, args: &Args, label: bool) -> Result<()> {
    if input.is_dir() {
        // Default a folder's output to the folder itself, matching the
        // drag-a-folder-on-the-exe expectation.
        let output_dir = args
            .output_dir
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or_else(|| input.to_path_buf());
        process_directory(pipeline, input, &output_dir, args)
    } else if input.is_file() {
        process_file(pipeline, input, args, label)
    } else {
        bail!(
            "input path is neither a file nor a directory: {}",
            input.display()
        );
    }
}

/// Process every image file in `dir`, writing crops into `output_dir` and
/// skipping inputs whose outputs are already present unless `--overwrite` is set.
fn process_directory(
    pipeline: &mut Pipeline,
    dir: &Path,
    output_dir: &Path,
    args: &Args,
) -> Result<()> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create output dir: {}", output_dir.display()))?;
    let output_dir_str = output_dir
        .to_str()
        .context("output directory path is not valid UTF-8")?;

    for input_file in collect_input_paths(dir)? {
        if !args.overwrite && output_dir_has_prefix(output_dir, &input_file)? {
            if args.debug {
                elog!("skipping {} (already processed)", input_file.display());
            }
            continue;
        }

        // A batch shouldn't abort because one scan has no objects (or fails to
        // decode): report the file and move on.
        if let Err(e) = process_dir_file(pipeline, &input_file, output_dir_str, args) {
            elog!("skipping {}: {e:#}", input_file.display());
        }
    }
    Ok(())
}

/// Process one file within a directory batch, writing its crops into `output_dir`.
fn process_dir_file(
    pipeline: &mut Pipeline,
    input_file: &Path,
    output_dir: &str,
    args: &Args,
) -> Result<()> {
    let (src_original, input_format) = read_original_image(input_file)?;
    let output_plan = OutputPlan::new(
        input_file,
        None,
        Some(output_dir),
        args.alpha,
        args.allow_lossy_conversion,
        args.format.map(|f| f.to_output_format()),
        input_format,
        true,
    )?;
    let model_input = input_file.to_string_lossy();
    pipeline.process_image(
        &src_original,
        &model_input,
        args,
        &output_plan,
        Some(input_file),
    )
}

/// Process a single input file. Output goes to `--output-dir` if set, else the
/// explicit `--output`, else next to the input.
fn process_file(pipeline: &mut Pipeline, input: &Path, args: &Args, label: bool) -> Result<()> {
    let (src_original, input_format) = read_original_image(input)?;
    let output_plan = match &args.output_dir {
        Some(output_dir) => {
            std::fs::create_dir_all(output_dir)
                .with_context(|| format!("failed to create output dir: {output_dir}"))?;
            OutputPlan::new(
                input,
                None,
                Some(output_dir.as_str()),
                args.alpha,
                args.allow_lossy_conversion,
                args.format.map(|f| f.to_output_format()),
                input_format,
                true,
            )?
        }
        None => OutputPlan::new(
            input,
            args.output.as_deref(),
            None,
            args.alpha,
            args.allow_lossy_conversion,
            args.format.map(|f| f.to_output_format()),
            input_format,
            false,
        )?,
    };

    let model_input = input.to_string_lossy();
    let summary = label.then_some(input);
    pipeline.process_image(&src_original, &model_input, args, &output_plan, summary)
}

/// Whether this process owns a freshly allocated console — the signature of an
/// Explorer launch (double-click or drag-and-drop) rather than a run from an
/// existing terminal.
#[cfg(windows)]
fn launched_standalone() -> bool {
    // GetConsoleProcessList reports how many processes share our console. A
    // console spun up just for us has only this process attached; a terminal we
    // were launched from is also attached, giving a count of two or more.
    unsafe extern "system" {
        fn GetConsoleProcessList(process_list: *mut u32, count: u32) -> u32;
    }
    let mut pids = [0u32; 2];
    let count = unsafe { GetConsoleProcessList(pids.as_mut_ptr(), pids.len() as u32) };
    count == 1
}

#[cfg(not(windows))]
fn launched_standalone() -> bool {
    false
}

fn wait_for_enter() {
    use std::io::{Read, Write};
    let mut stdout = std::io::stdout();
    let _ = write!(stdout, "\nPress Enter to exit...");
    let _ = stdout.flush();
    let _ = std::io::stdin().read(&mut [0u8]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "kropp-merge-{tag}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_png(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        image::RgbImage::new(4, 4).save(&path).unwrap();
        path
    }

    #[test]
    fn no_merge_input_yields_no_groups() {
        assert!(resolve_merge_groups(&[]).unwrap().is_empty());
    }

    #[test]
    fn a_list_of_files_is_one_group() {
        let groups = resolve_merge_groups(&["a.tif".into(), "b.tif".into(), "c.tif".into()])
            .unwrap();

        assert_eq!(groups.len(), 1, "files should form a single group");
        assert_eq!(groups[0].len(), 3);
    }

    #[test]
    fn a_single_file_is_rejected() {
        let err = resolve_merge_groups(&["only.tif".into()])
            .unwrap_err()
            .to_string();
        assert!(err.contains("at least two files"), "got: {err}");
    }

    /// Directories are how a batch of separate originals is expressed, so each
    /// becomes its own group rather than being pooled.
    #[test]
    fn each_directory_is_its_own_group() {
        let root = scratch_dir("dirs");
        let (left, right) = (root.join("left"), root.join("right"));
        std::fs::create_dir_all(&left).unwrap();
        std::fs::create_dir_all(&right).unwrap();
        write_png(&left, "a.png");
        write_png(&left, "b.png");
        write_png(&right, "c.png");
        write_png(&right, "d.png");

        let groups = resolve_merge_groups(&[
            left.to_string_lossy().into_owned(),
            right.to_string_lossy().into_owned(),
        ])
        .unwrap();

        assert_eq!(groups.len(), 2, "each directory should be its own group");
        assert_eq!(groups[0].len(), 2);
        assert_eq!(groups[1].len(), 2);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_directory_holding_one_image_is_rejected() {
        let root = scratch_dir("lonely");
        write_png(&root, "only.png");

        let err = resolve_merge_groups(&[root.to_string_lossy().into_owned()])
            .unwrap_err()
            .to_string();
        assert!(err.contains("needs at least two"), "got: {err}");
        std::fs::remove_dir_all(&root).ok();
    }

    /// Mixing a directory with loose files has two plausible readings and
    /// either would quietly drop work, so it is refused instead of guessed.
    #[test]
    fn mixing_directories_and_files_is_refused() {
        let root = scratch_dir("mixed");
        write_png(&root, "a.png");
        write_png(&root, "b.png");

        let err = resolve_merge_groups(&[
            root.to_string_lossy().into_owned(),
            "loose.tif".into(),
        ])
        .unwrap_err()
        .to_string();
        assert!(err.contains("not a mix"), "got: {err}");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn merged_path_sits_beside_the_first_pass_with_a_merged_suffix() {
        let args = Args::parse_from(["kropp", "--merge-input", "a.tif", "b.tif"]);
        let path =
            merged_output_path(Path::new("scans/poster_0.tiff"), &args, image::ImageFormat::Tiff)
                .unwrap();

        assert_eq!(path, PathBuf::from("scans/poster_0_merged.tiff"));
    }

    /// The merge is lossless work; a JPEG source must not send it back through
    /// a lossy encoder.
    #[test]
    fn merged_path_avoids_lossy_formats() {
        let args = Args::parse_from(["kropp", "--merge-input", "a.jpg", "b.jpg"]);
        let path = merged_output_path(Path::new("poster.jpg"), &args, image::ImageFormat::Jpeg)
            .unwrap();

        assert_eq!(path, PathBuf::from("poster_merged.png"));
    }

    #[test]
    fn merged_path_honours_output_dir() {
        let args = Args::parse_from([
            "kropp",
            "--merge-input",
            "a.tif",
            "b.tif",
            "--output-dir",
            "out",
        ]);
        let path = merged_output_path(Path::new("scans/a.tif"), &args, image::ImageFormat::Tiff)
            .unwrap();

        assert_eq!(path, PathBuf::from("out/a_merged.tiff"));
    }
}
