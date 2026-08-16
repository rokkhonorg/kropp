# kropp

Crop and deskew objects from a flatbed scan. kropp finds each object in a scan,
straightens it upright, and writes one cropped file per object — handy for
digitizing a stack of photos, receipts, or documents laid on the glass.

Foreground detection runs through a swappable extractor — RMBG by default, or
SAM3 — see [Extraction modes](#extraction-modes).

## Build

```sh
cargo build --release
```

Model weights download automatically on first run. kropp tries CUDA, then
DirectML, then CPU, and uses whichever is available.

SAM3 support is behind the `sam3` cargo feature, which is **on by default**. To
build an RMBG-only binary (no SAM3, lighter dependencies):

```sh
cargo build --release --no-default-features
```

## Usage

Single image (pass the path positionally, or with `-i`):

```sh
kropp photo.jpg
```

This writes `photo_0.png`, `photo_1.png`, … next to the input — one file per
object found. A lossy input like JPEG is saved as lossless PNG by default so the
crop isn't recompressed (see [Output format](#output-format)).

On Windows you can **drag image files (or a folder) onto `kropp.exe`** — the
crops are written next to each input, and the window stays open at the end so
you can read the output.

Several inputs at once (each is written next to itself):

```sh
kropp a.jpg b.png ./scans
```

Pick the output name for a single file (the `_<index>` suffix is still added per
object):

```sh
kropp photo.jpg -o cropped.png
```

Collect everything into one directory instead of writing in place:

```sh
kropp ./inputs --output-dir ./outputs
```

Each input keeps its own name and format under the output directory. Files that
already have outputs are skipped unless you pass `--overwrite`.

Cut the object out on a transparent background instead of a plain rectangle:

```sh
kropp -i photo.jpg --alpha
```

## Common options

| Option | Default | What it does |
| --- | --- | --- |
| `-i, --input` | — | Input image or directory (or pass paths positionally) |
| `--extractor` | `rmbg` | Foreground extractor: `rmbg`, `sam3-bg`, or `sam3-fg` (see [Extraction modes](#extraction-modes)) |
| `-o, --output` | input name | Output path for a single input file |
| `--output-dir` | — | Output directory (required when input is a directory) |
| `--format` | inferred | Force the output format: `png`, `jpeg`, `tiff`, `webp`, `bmp`, `qoi`, `jxl` (see [Output format](#output-format)) |
| `--merge-input` | — | Merge overlapping passes of one oversized original before cropping (see [Merging oversized originals](#merging-oversized-originals)) |
| `--keep-merged` | off | Keep the merged image next to the crops |
| `-t, --threshold` | `95` | Alpha cutoff %: below this is transparent, at/above is kept |
| `-m, --min-area` | `0` | Drop objects smaller than this many pixels |
| `--min-side-percent` | `10` | Drop objects whose longer side is under this % of the smaller image dimension |
| `--text` | off | Force text-based rotation for every object |
| `--force-rectangular` | off | Force the rectangular pipeline for every object (ignores the rectangularity score) |
| `--alpha` | off | Transparent (RGBA) cutout instead of a rectangular RGB crop |
| `--allow-lossy-conversion` | off | Keep the input's lossy format instead of converting to PNG |
| `--no-deskew` | off | Report the angle but don't rotate crops upright |
| `--overwrite` | off | Reprocess directory inputs even if outputs already exist |
| `-v, --debug` | off | Print per-object diagnostics and write overlay images |

## Output format

By default the output format follows the input, with one exception: a **lossy**
input (JPEG, AVIF) is written as lossless **PNG** so cropping and deskewing don't
recompress it. **Lossless** inputs (PNG, TIFF, BMP, …) keep their own format.

Pass `--allow-lossy-conversion` to write the lossy format anyway:

```sh
kropp -i photo.jpg --allow-lossy-conversion   # -> photo_0.jpg
```

An explicit `-o name.ext` still picks the format from its extension (and is
itself redirected to PNG if lossy, unless `--allow-lossy-conversion` is set).

### Forcing a format

`--format` overrides all of the above and writes every crop in one format,
renaming the output extension to match:

```sh
kropp ./inputs --output-dir ./outputs --format jxl   # -> outputs/<name>_0.jxl
kropp -i scan.tiff -o crop.png --format jxl          # -> crop_0.jxl
```

Accepted values are `png`, `jpeg`, `tiff`, `webp`, `bmp`, `qoi`, and `jxl`.
`jxl` is **lossless JPEG XL**, encoded with
[`zune-jpegxl`](https://crates.io/crates/zune-jpegxl) in both 8- and 16-bit,
with or without `--alpha`. It is a fast, simple modular encoder rather than a
size-optimizing one, so it does not always beat PNG on file size — check
against your own scans before switching a whole workflow to it. The lossy guard
still applies to the override, so `--format jpeg` is redirected to PNG unless
you also pass `--allow-lossy-conversion`.

Note that JPEG XL is write-only here: kropp encodes `.jxl` but cannot read it,
so `.jxl` files are not accepted as inputs.

## Merging oversized originals

An original too large for the platen can be scanned in several overlapping
passes and merged before cropping:

```sh
kropp --merge-input poster_a.tif poster_b.tif        # one group from these files
kropp --merge-input passes/                          # one group from a directory
kropp --merge-input left/ right/ --output-dir out    # two groups, one per directory
```

When every argument is a directory each one is merged as its own group;
otherwise the whole list is a single group. Add `--keep-merged` to keep the
merged image (`<stem>_merged.<ext>`) alongside the crops.

Passes must **overlap** — the alignment is driven by features found in the
shared region, and a group whose passes don't overlap is reported rather than
merged. A pass that overlaps nothing is left out with a warning while the rest
proceed.

How it works, and why it isn't OpenCV's stitcher:

- **Alignment is a similarity, not a homography.** Flatbed passes are coplanar
  and share a DPI, so scale is fixed at 1 and only rotation and translation are
  solved. That makes the fit closed-form rather than an iterative bundle
  adjustment.
- **Keypoints are confined to the extracted object.** Platen artifacts — bed
  edges, dust, lid shadow — sit at fixed scanner coordinates and are therefore
  identical in every pass, which would match them together at zero
  displacement. Masking removes them before they can vote.
- **Passes are never individually deskewed.** They stay in a common frame
  through the merge; deskew, orientation, and the final tight crop run once, on
  the completed original.
- **Bit depth is preserved.** 16-bit passes merge at 16 bits. OpenCV's
  multi-band blender composites through *signed* 16-bit and clips everything
  above 32767.
- **The result is reproducible.** Feature matching and RANSAC are seeded, so
  the same inputs always produce the same merge.

## Extraction modes

How kropp finds the foreground objects (before any cropping or deskewing) is
chosen with `--extractor`:

| Mode | What it does |
| --- | --- |
| `rmbg` (default) | RMBG background-removal model: one alpha mask, thresholded and split into objects by connected components. |
| `sam3-bg` | SAM3 "background inversion": segment the background by text prompt (`--sam-background`, default `"background"`), invert it to one foreground mask, then split by connected components. |
| `sam3-fg` | SAM3 multi-prompt: run each object prompt (`--sam-prompt`, repeatable), then merge overlapping detections by IoU (`--sam-overlap`) into distinct objects. |

```sh
# SAM3 background inversion
kropp scan.tif --extractor sam3-bg

# SAM3 multi-prompt, several object types at once
kropp scan.tif --extractor sam3-fg --sam-prompt photo --sam-prompt postcard
```

The `sam3-*` modes need a build with the `sam3` feature (the default). SAM3
downloads its own ONNX models on first use. `--threshold`/`-t` applies to RMBG
only; the SAM3 masks are already binary.

## Deskew and orientation

By default kropp straightens each crop:

- **Rectangular objects** are rotated upright using a document-orientation model
  (0/90/180/270), loaded from the bundled `onnx/bkori.onnx` (kept next to the
  executable; override with `--doc-orient-model`). Disable with `--no-doc-orient`.
- **Non-rectangular objects** are straightened using detected text lines, with a
  0/180 flip vote. Pass `--text` to force text-based rotation on every object.

Which path an object takes is decided by its rectangularity score against
`--auto-text-rectangularity-threshold`. To force one path for every object:

- `--text` — always use the text path.
- `--force-rectangular` — always use the rectangular path, ignoring the score
  (the same as `--auto-text-rectangularity-threshold 0`). The two are mutually
  exclusive.

Run `kropp --help` for the full list of options.

## License

kropp is released under the [MIT License](LICENSE).

Note that kropp downloads the RMBG-2.0 model at runtime, which is licensed by
BRIA for non-commercial use; commercial use of the model requires a separate
agreement with BRIA.
