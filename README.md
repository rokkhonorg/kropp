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

Single image:

```sh
kropp -i photo.jpg
```

This writes `photo_0.jpg`, `photo_1.jpg`, … — one file per object found.

Pick the output name (the `_<index>` suffix is still added per object):

```sh
kropp -i photo.jpg -o cropped.png
```

A whole directory at once:

```sh
kropp -i ./inputs --output-dir ./outputs
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
| `--alpha` | off | Transparent (RGBA) cutout instead of a rectangular RGB crop |
| `--no-deskew` | off | Report the angle but don't rotate crops upright |
| `--overwrite` | off | Reprocess directory inputs even if outputs already exist |
| `-v, --debug` | off | Print per-object diagnostics and write overlay images |

## Deskew and orientation

By default kropp straightens each crop:

- **Rectangular objects** are rotated upright using a document-orientation model
  (0/90/180/270). Disable with `--no-doc-orient`.
- **Non-rectangular objects** are straightened using detected text lines, with a
  0/180 flip vote. Pass `--text` to force text-based rotation on every object.

Run `kropp --help` for the full list of options.

## License

kropp is released under the [MIT License](LICENSE).

Note that kropp downloads the RMBG-2.0 model at runtime, which is licensed by
BRIA for non-commercial use; commercial use of the model requires a separate
agreement with BRIA.
