# kropp

Crop and deskew objects from a flatbed scan. kropp finds each object in a scan,
straightens it upright, and writes one cropped file per object — handy for
digitizing a stack of photos, receipts, or documents laid on the glass.

Foreground detection currently uses the RMBG background-removal model; other
methods may be added later.

## Build

```sh
cargo build --release
```

The model weights download automatically on first run. kropp tries CUDA, then
DirectML, then CPU, and uses whichever is available.

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
| `-i, --input` | — | Input image or directory (required) |
| `-o, --output` | input name | Output path for a single input file |
| `--output-dir` | — | Output directory (required when input is a directory) |
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
