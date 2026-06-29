//! Output planning, crop geometry, and image writing (including deskew, padding,
//! quarter-turn correction, and TIFF encoding).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use image::{DynamicImage, ImageBuffer, ImageFormat, Pixel, Rgb, Rgba};
use imageproc::definitions::Clamp;
use imageproc::geometric_transformations::{
    Border, Interpolation, Projection, rotate_about_center_no_crop, warp_into,
};
use tiff::encoder::{
    Compression as TiffCompression, DeflateLevel, TiffEncoder as TiffImageEncoder,
    colortype::{RGB8 as TiffRgb8, RGB16 as TiffRgb16, RGBA8 as TiffRgba8, RGBA16 as TiffRgba16},
};

use crate::components::{Object, crop_object, crop_object_rgba16};

pub type Rgba16Image = ImageBuffer<Rgba<u16>, Vec<u16>>;

#[derive(Debug, Clone)]
pub struct OutputPlan {
    pub base: PathBuf,
    pub format: ImageFormat,
}

impl OutputPlan {
    pub fn new(
        input: &Path,
        output: Option<&str>,
        output_dir: Option<&str>,
        alpha: bool,
        allow_lossy: bool,
        input_format: ImageFormat,
        input_is_dir: bool,
    ) -> Result<Self> {
        if input_is_dir {
            let output_dir =
                output_dir.context("output-dir is required when input is a directory")?;
            if output.is_some() {
                bail!("--output cannot be used when the input is a directory; use --output-dir");
            }
            let mut base = PathBuf::from(output_dir);
            base.push(
                input
                    .file_name()
                    .context("directory inputs must have a file name")?,
            );
            let format = resolve_format(&mut base, input_format, alpha, allow_lossy);

            if !format.writing_enabled() {
                bail!("output format {format:?} is not enabled for writing");
            }

            return Ok(Self { base, format });
        }

        if output_dir.is_some() {
            bail!("--output-dir can only be used when the input is a directory");
        }

        let mut base = output
            .map(PathBuf::from)
            .unwrap_or_else(|| input.to_path_buf());
        let requested_format = output
            .and_then(|p| ImageFormat::from_path(p).ok())
            .unwrap_or(input_format);
        let format = resolve_format(&mut base, requested_format, alpha, allow_lossy);

        if !format.writing_enabled() {
            bail!("output format {format:?} is not enabled for writing");
        }

        Ok(Self { base, format })
    }

    pub fn object_path(&self, idx: usize) -> PathBuf {
        suffixed_path(&self.base, &format!("_{idx}"))
    }
}

/// Resolve the final output format and align `base`'s extension to it. A lossy
/// target is redirected to lossless PNG (unless `allow_lossy` is set) so crops
/// aren't silently recompressed; lossless targets are kept as is. Alpha output
/// likewise falls back to PNG when the chosen format has no alpha channel.
fn resolve_format(
    base: &mut PathBuf,
    mut format: ImageFormat,
    alpha: bool,
    allow_lossy: bool,
) -> ImageFormat {
    // Redirect to PNG when alpha needs a channel the format lacks, or when a
    // lossy target would recompress the crop and the user hasn't opted in.
    let needs_png =
        (alpha && !format_supports_alpha(format)) || (!allow_lossy && is_lossy_format(format));
    if needs_png {
        format = ImageFormat::Png;
        base.set_extension("png");
    } else if base.extension().is_none() {
        base.set_extension(preferred_extension(format));
    }
    format
}

fn preferred_extension(format: ImageFormat) -> &'static str {
    format.extensions_str().first().copied().unwrap_or("png")
}

/// Whether re-encoding to `format` discards image data. These are the lossy
/// encoders `image` can write; everything else it supports is lossless.
fn is_lossy_format(format: ImageFormat) -> bool {
    matches!(format, ImageFormat::Jpeg | ImageFormat::Avif)
}

fn format_supports_alpha(format: ImageFormat) -> bool {
    matches!(
        format,
        ImageFormat::Png
            | ImageFormat::Ico
            | ImageFormat::Tiff
            | ImageFormat::Tga
            | ImageFormat::Pnm
            | ImageFormat::Farbfeld
            | ImageFormat::Avif
            | ImageFormat::WebP
            | ImageFormat::OpenExr
            | ImageFormat::Qoi
    )
}

/// Insert `suffix` before the file extension of `path`.
pub fn suffixed_path(path: &Path, suffix: &str) -> PathBuf {
    let mut out = path.to_path_buf();
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let file_name = match path.extension().and_then(|s| s.to_str()) {
        Some(ext) => format!("{stem}{suffix}.{ext}"),
        None => format!("{stem}{suffix}"),
    };
    out.set_file_name(file_name);
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChannelDepth {
    U8,
    U16,
}

fn channel_depth(img: &DynamicImage) -> Result<ChannelDepth> {
    match img.color() {
        image::ColorType::L8
        | image::ColorType::La8
        | image::ColorType::Rgb8
        | image::ColorType::Rgba8 => Ok(ChannelDepth::U8),
        image::ColorType::L16
        | image::ColorType::La16
        | image::ColorType::Rgb16
        | image::ColorType::Rgba16 => Ok(ChannelDepth::U16),
        other => bail!("unsupported source color type {other:?}; refusing to lose precision"),
    }
}

/// Rotate an image by `turns` quarter-turns clockwise (0..=3), the orientation
/// correction the classifier asked for. `turns % 4 == 0` is a no-op.
fn apply_quarter_turns<P>(
    img: &ImageBuffer<P, Vec<P::Subpixel>>,
    turns: u8,
) -> ImageBuffer<P, Vec<P::Subpixel>>
where
    P: Pixel + 'static,
{
    match turns % 4 {
        1 => image::imageops::rotate90(img),
        2 => image::imageops::rotate180(img),
        3 => image::imageops::rotate270(img),
        _ => img.clone(),
    }
}

pub fn write_rgb_output(
    src: &DynamicImage,
    bounds: RotatedBounds,
    theta: f32,
    quarter_turns: u8,
    path: &Path,
    format: ImageFormat,
) -> Result<(u32, u32)> {
    match channel_depth(src)? {
        ChannelDepth::U8 => {
            let src_rgb = src.to_rgb8();
            let mut img = deskew_from_src(&src_rgb, bounds, theta, Rgb([255, 255, 255]));
            if !quarter_turns.is_multiple_of(4) {
                img = apply_quarter_turns(&img, quarter_turns);
            }
            let dims = img.dimensions();
            match format {
                ImageFormat::Tiff => save_tiff_rgb8(path, &img)?,
                _ => img
                    .save_with_format(path, format)
                    .with_context(|| format!("failed to write image: {}", path.display()))?,
            }
            Ok(dims)
        }
        ChannelDepth::U16 => {
            let src_rgb = src.to_rgb16();
            let mut img =
                deskew_from_src(&src_rgb, bounds, theta, Rgb([u16::MAX, u16::MAX, u16::MAX]));
            if !quarter_turns.is_multiple_of(4) {
                img = apply_quarter_turns(&img, quarter_turns);
            }
            let dims = img.dimensions();
            match format {
                ImageFormat::Tiff => save_tiff_rgb16(path, &img)?,
                _ => img
                    .save_with_format(path, format)
                    .with_context(|| format!("failed to write image: {}", path.display()))?,
            }
            Ok(dims)
        }
    }
}

pub struct AlphaOutputRequest<'a> {
    pub labels: &'a [usize],
    pub w: usize,
    pub obj: &'a Object,
    pub theta: f32,
    pub quarter_turns: u8,
    pub crop_padding: f64,
    pub path: &'a Path,
    pub format: ImageFormat,
}

pub fn write_alpha_output(src: &DynamicImage, req: AlphaOutputRequest<'_>) -> Result<(u32, u32)> {
    match channel_depth(src)? {
        ChannelDepth::U8 => {
            let src_rgba = src.to_rgba8();
            let mut img = crop_object(&src_rgba, req.w, req.labels, req.obj);
            if req.theta != 0.0 {
                img = rotate_about_center_no_crop(
                    &img,
                    req.theta,
                    Interpolation::Bilinear,
                    Border::Constant(Rgba([0, 0, 0, 0])),
                );
            }
            img = trim_transparent(&img).unwrap_or(img);
            if req.crop_padding > 0.0 {
                img = pad_rgba(&img, req.crop_padding);
            }
            if !req.quarter_turns.is_multiple_of(4) {
                img = apply_quarter_turns(&img, req.quarter_turns);
            }
            let dims = img.dimensions();
            match req.format {
                ImageFormat::Tiff => save_tiff_rgba8(req.path, &img)?,
                _ => img
                    .save_with_format(req.path, req.format)
                    .with_context(|| format!("failed to write image: {}", req.path.display()))?,
            }
            Ok(dims)
        }
        ChannelDepth::U16 => {
            let src_rgba = src.to_rgba16();
            let mut img = crop_object_rgba16(&src_rgba, req.w, req.labels, req.obj);
            if req.theta != 0.0 {
                img = rotate_about_center_no_crop(
                    &img,
                    req.theta,
                    Interpolation::Bilinear,
                    Border::Constant(Rgba([0, 0, 0, 0])),
                );
            }
            img = trim_transparent16(&img).unwrap_or(img);
            if req.crop_padding > 0.0 {
                img = pad_rgba16(&img, req.crop_padding);
            }
            if !req.quarter_turns.is_multiple_of(4) {
                img = apply_quarter_turns(&img, req.quarter_turns);
            }
            let dims = img.dimensions();
            match req.format {
                ImageFormat::Tiff => save_tiff_rgba16(req.path, &img)?,
                _ => img
                    .save_with_format(req.path, req.format)
                    .with_context(|| format!("failed to write image: {}", req.path.display()))?,
            }
            Ok(dims)
        }
    }
}

fn save_tiff_rgb8(path: &Path, img: &image::RgbImage) -> Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("failed to create image: {}", path.display()))?;
    let mut encoder = TiffImageEncoder::new(file)?
        .with_compression(TiffCompression::Deflate(DeflateLevel::default()));
    encoder
        .write_image::<TiffRgb8>(img.width(), img.height(), img.as_raw())
        .with_context(|| format!("failed to write image: {}", path.display()))?;
    Ok(())
}

fn save_tiff_rgb16(path: &Path, img: &ImageBuffer<image::Rgb<u16>, Vec<u16>>) -> Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("failed to create image: {}", path.display()))?;
    let mut encoder = TiffImageEncoder::new(file)?
        .with_compression(TiffCompression::Deflate(DeflateLevel::default()));
    encoder
        .write_image::<TiffRgb16>(img.width(), img.height(), img.as_raw())
        .with_context(|| format!("failed to write image: {}", path.display()))?;
    Ok(())
}

fn save_tiff_rgba8(path: &Path, img: &image::RgbaImage) -> Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("failed to create image: {}", path.display()))?;
    let mut encoder = TiffImageEncoder::new(file)?
        .with_compression(TiffCompression::Deflate(DeflateLevel::default()));
    encoder
        .write_image::<TiffRgba8>(img.width(), img.height(), img.as_raw())
        .with_context(|| format!("failed to write image: {}", path.display()))?;
    Ok(())
}

fn save_tiff_rgba16(path: &Path, img: &Rgba16Image) -> Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("failed to create image: {}", path.display()))?;
    let mut encoder = TiffImageEncoder::new(file)?
        .with_compression(TiffCompression::Deflate(DeflateLevel::default()));
    encoder
        .write_image::<TiffRgba16>(img.width(), img.height(), img.as_raw())
        .with_context(|| format!("failed to write image: {}", path.display()))?;
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct RotatedBounds {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl RotatedBounds {
    fn include(&mut self, x: f64, y: f64) {
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
    }

    pub fn expand_percent(&mut self, percent: f64) {
        if percent <= 0.0 {
            return;
        }
        let pad_x = (self.max_x - self.min_x).max(1.0) * percent / 100.0;
        let pad_y = (self.max_y - self.min_y).max(1.0) * percent / 100.0;
        self.min_x -= pad_x;
        self.min_y -= pad_y;
        self.max_x += pad_x;
        self.max_y += pad_y;
    }
}

/// Bounds of the actual component pixels after applying the deskew transform.
fn rotated_component_bounds(
    labels: &[usize],
    w: usize,
    obj: &Object,
    theta: f32,
) -> Option<RotatedBounds> {
    let (sin, cos) = (theta as f64).sin_cos();
    let mut bounds: Option<RotatedBounds> = None;

    for y in obj.min_y..=obj.max_y {
        for x in obj.min_x..=obj.max_x {
            if labels[y * w + x] != obj.label {
                continue;
            }

            let corners = [
                (x as f64, y as f64),
                ((x + 1) as f64, y as f64),
                ((x + 1) as f64, (y + 1) as f64),
                (x as f64, (y + 1) as f64),
            ];
            for (px, py) in corners {
                let rx = px * cos - py * sin;
                let ry = px * sin + py * cos;
                match bounds.as_mut() {
                    Some(b) => b.include(rx, ry),
                    None => {
                        bounds = Some(RotatedBounds {
                            min_x: rx,
                            min_y: ry,
                            max_x: rx,
                            max_y: ry,
                        });
                    }
                }
            }
        }
    }

    bounds
}

/// Bounding box, in the theta-rotated frame, of a mask-local rectangle's four
/// corners (offset into full-image coordinates by the object's top-left). This
/// lets the crop follow the fitted mask rectangle instead of every component
/// pixel, so body-fit exclusions are cropped away too.
fn rotated_corner_bounds(
    corners: &[(f64, f64); 4],
    off_x: usize,
    off_y: usize,
    theta: f32,
) -> RotatedBounds {
    let (sin, cos) = (theta as f64).sin_cos();
    let rotate = |&(lx, ly): &(f64, f64)| {
        let (x, y) = (lx + off_x as f64, ly + off_y as f64);
        (x * cos - y * sin, x * sin + y * cos)
    };
    let (fx, fy) = rotate(&corners[0]);
    let mut bounds = RotatedBounds {
        min_x: fx,
        min_y: fy,
        max_x: fx,
        max_y: fy,
    };
    for c in &corners[1..] {
        let (rx, ry) = rotate(c);
        bounds.include(rx, ry);
    }
    bounds
}

pub fn rotated_crop_bounds(
    fit_corners: Option<[(f64, f64); 4]>,
    labels: &[usize],
    w: usize,
    obj: &Object,
    theta: f32,
) -> Option<RotatedBounds> {
    match fit_corners {
        Some(corners) => Some(rotated_corner_bounds(&corners, obj.min_x, obj.min_y, theta)),
        None => rotated_component_bounds(labels, w, obj, theta),
    }
}

fn trim_transparent(img: &image::RgbaImage) -> Option<image::RgbaImage> {
    let (w, h) = img.dimensions();
    let mut min_x = w;
    let mut min_y = h;
    let mut max_x = 0;
    let mut max_y = 0;
    let mut found = false;

    for (x, y, px) in img.enumerate_pixels() {
        if px[3] == 0 {
            continue;
        }
        found = true;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    found.then(|| {
        image::imageops::crop_imm(img, min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
            .to_image()
    })
}

fn trim_transparent16(img: &Rgba16Image) -> Option<Rgba16Image> {
    let (w, h) = img.dimensions();
    let mut min_x = w;
    let mut min_y = h;
    let mut max_x = 0;
    let mut max_y = 0;
    let mut found = false;

    for (x, y, px) in img.enumerate_pixels() {
        if px[3] == 0 {
            continue;
        }
        found = true;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    found.then(|| {
        image::imageops::crop_imm(img, min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
            .to_image()
    })
}

fn pad_rgba(img: &image::RgbaImage, percent: f64) -> image::RgbaImage {
    if percent <= 0.0 {
        return img.clone();
    }

    let (w, h) = img.dimensions();
    let pad_x = (w as f64 * percent / 100.0).ceil() as u32;
    let pad_y = (h as f64 * percent / 100.0).ceil() as u32;
    if pad_x == 0 && pad_y == 0 {
        return img.clone();
    }

    let mut out = image::RgbaImage::from_pixel(
        w + pad_x.saturating_mul(2),
        h + pad_y.saturating_mul(2),
        Rgba([0, 0, 0, 0]),
    );
    image::imageops::overlay(&mut out, img, i64::from(pad_x), i64::from(pad_y));
    out
}

fn pad_rgba16(img: &Rgba16Image, percent: f64) -> Rgba16Image {
    if percent <= 0.0 {
        return img.clone();
    }

    let (w, h) = img.dimensions();
    let pad_x = (w as f64 * percent / 100.0).ceil() as u32;
    let pad_y = (h as f64 * percent / 100.0).ceil() as u32;
    if pad_x == 0 && pad_y == 0 {
        return img.clone();
    }

    let mut out = Rgba16Image::from_pixel(
        w + pad_x.saturating_mul(2),
        h + pad_y.saturating_mul(2),
        Rgba([0, 0, 0, 0]),
    );
    image::imageops::overlay(&mut out, img, i64::from(pad_x), i64::from(pad_y));
    out
}

/// Deskew an object by warping straight out of the full RGB source. The output
/// is cropped tight to the transformed component pixels. Area inside the crop
/// but outside the object is filled from the real surrounding pixels; the white
/// border is used only where the sample falls outside the source image.
fn deskew_from_src<P>(
    src_rgb: &ImageBuffer<P, Vec<P::Subpixel>>,
    bounds: RotatedBounds,
    theta: f32,
    border: P,
) -> ImageBuffer<P, Vec<P::Subpixel>>
where
    P: Pixel + Send + Sync + 'static,
    P::Subpixel: Send + Sync + Into<f32> + Clamp<f32> + 'static,
{
    let min_x = bounds.min_x;
    let min_y = bounds.min_y;
    let max_x = bounds.max_x;
    let max_y = bounds.max_y;

    let new_w = (max_x - min_x).ceil().max(1.0) as u32;
    let new_h = (max_y - min_y).ceil().max(1.0) as u32;

    // Forward map (source -> output): rotate, then shift the crop to the origin.
    let proj = Projection::translate(-min_x as f32, -min_y as f32) * Projection::rotate(theta);

    let mut out = ImageBuffer::from_pixel(new_w, new_h, border);
    warp_into(
        src_rgb,
        proj,
        Interpolation::Bilinear,
        Border::Constant(border),
        &mut out,
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Object;
    use std::fs::File;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tiff::{decoder::Decoder as TiffDecoder, tags::Tag};

    fn temp_image_path(ext: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("kropp-test-{}-{nanos}.{ext}", std::process::id()))
    }

    #[test]
    fn lossy_input_converts_to_png_by_default() {
        let plan = OutputPlan::new(
            Path::new("fixtures/photo.jpeg"),
            None,
            None,
            false,
            false,
            ImageFormat::Jpeg,
            false,
        )
        .unwrap();

        assert_eq!(plan.object_path(3), PathBuf::from("fixtures/photo_3.png"));
        assert_eq!(plan.format, ImageFormat::Png);
    }

    #[test]
    fn allow_lossy_conversion_keeps_lossy_format() {
        let plan = OutputPlan::new(
            Path::new("fixtures/photo.jpeg"),
            None,
            None,
            false,
            true,
            ImageFormat::Jpeg,
            false,
        )
        .unwrap();

        assert_eq!(plan.object_path(3), PathBuf::from("fixtures/photo_3.jpeg"));
        assert_eq!(plan.format, ImageFormat::Jpeg);
    }

    #[test]
    fn lossless_input_keeps_its_format() {
        let plan = OutputPlan::new(
            Path::new("scan.tiff"),
            None,
            None,
            false,
            false,
            ImageFormat::Tiff,
            false,
        )
        .unwrap();

        assert_eq!(plan.object_path(0), PathBuf::from("scan_0.tiff"));
        assert_eq!(plan.format, ImageFormat::Tiff);
    }

    #[test]
    fn alpha_output_falls_back_to_png_when_format_lacks_alpha() {
        let plan = OutputPlan::new(
            Path::new("photo.jpg"),
            None,
            None,
            true,
            false,
            ImageFormat::Jpeg,
            false,
        )
        .unwrap();

        assert_eq!(plan.object_path(0), PathBuf::from("photo_0.png"));
        assert_eq!(plan.format, ImageFormat::Png);
    }

    #[test]
    fn alpha_output_keeps_alpha_capable_format() {
        let plan = OutputPlan::new(
            Path::new("photo.tiff"),
            None,
            None,
            true,
            false,
            ImageFormat::Tiff,
            false,
        )
        .unwrap();

        assert_eq!(plan.object_path(0), PathBuf::from("photo_0.tiff"));
        assert_eq!(plan.format, ImageFormat::Tiff);
    }

    #[test]
    fn dir_input_uses_output_dir_for_each_object_path() {
        let plan = OutputPlan::new(
            Path::new("inputs/photo.jpg"),
            None,
            Some("out"),
            false,
            true,
            ImageFormat::Jpeg,
            true,
        )
        .unwrap();

        assert_eq!(plan.object_path(2), PathBuf::from("out/photo_2.jpg"));
    }

    #[test]
    fn dir_input_converts_lossy_to_png_by_default() {
        let plan = OutputPlan::new(
            Path::new("inputs/photo.jpg"),
            None,
            Some("out"),
            false,
            false,
            ImageFormat::Jpeg,
            true,
        )
        .unwrap();

        assert_eq!(plan.object_path(2), PathBuf::from("out/photo_2.png"));
        assert_eq!(plan.format, ImageFormat::Png);
    }

    #[test]
    fn rotated_bounds_follow_component_pixels_not_bbox() {
        let mut labels = vec![0usize; 100];
        labels[5 * 10 + 5] = 1;
        let obj = Object {
            label: 1,
            area: 1,
            min_x: 0,
            min_y: 0,
            max_x: 9,
            max_y: 9,
        };

        let bounds = rotated_component_bounds(&labels, 10, &obj, 45f32.to_radians()).unwrap();
        let width = bounds.max_x - bounds.min_x;
        let height = bounds.max_y - bounds.min_y;

        assert!(width < 1.5, "width was {width}");
        assert!(height < 1.5, "height was {height}");
    }

    #[test]
    fn rotated_crop_bounds_prefers_fitted_rectangle_over_component() {
        let mut labels = vec![0usize; 400];
        for y in 3..=12 {
            for x in 2..=11 {
                labels[y * 20 + x] = 1;
            }
        }
        let obj = Object {
            label: 1,
            area: 100,
            min_x: 2,
            min_y: 3,
            max_x: 11,
            max_y: 12,
        };
        let fit_corners = Some([(1.0, 1.0), (4.0, 1.0), (4.0, 5.0), (1.0, 5.0)]);

        let bounds = rotated_crop_bounds(fit_corners, &labels, 20, &obj, 0.0).unwrap();

        assert!((bounds.min_x - 3.0).abs() < 1e-6);
        assert!((bounds.max_x - 6.0).abs() < 1e-6);
        assert!((bounds.min_y - 4.0).abs() < 1e-6);
        assert!((bounds.max_y - 8.0).abs() < 1e-6);
    }

    #[test]
    fn rotated_crop_bounds_falls_back_to_component_without_fit_rectangle() {
        let mut labels = vec![0usize; 400];
        for y in 3..=12 {
            for x in 2..=11 {
                labels[y * 20 + x] = 1;
            }
        }
        let obj = Object {
            label: 1,
            area: 100,
            min_x: 2,
            min_y: 3,
            max_x: 11,
            max_y: 12,
        };

        let bounds = rotated_crop_bounds(None, &labels, 20, &obj, 0.0).unwrap();

        assert!((bounds.min_x - 2.0).abs() < 1e-6);
        assert!((bounds.max_x - 12.0).abs() < 1e-6);
        assert!((bounds.min_y - 3.0).abs() < 1e-6);
        assert!((bounds.max_y - 13.0).abs() < 1e-6);
    }

    #[test]
    fn rotated_bounds_padding_expands_each_side_by_percent() {
        let mut bounds = RotatedBounds {
            min_x: 10.0,
            min_y: 20.0,
            max_x: 110.0,
            max_y: 60.0,
        };

        bounds.expand_percent(5.0);

        assert!((bounds.min_x - 5.0).abs() < 1e-6);
        assert!((bounds.max_x - 115.0).abs() < 1e-6);
        assert!((bounds.min_y - 18.0).abs() < 1e-6);
        assert!((bounds.max_y - 62.0).abs() < 1e-6);
    }

    #[test]
    fn trim_transparent_crops_to_visible_alpha() {
        let mut img = image::RgbaImage::new(6, 5);
        img.put_pixel(2, 1, Rgba([10, 20, 30, 255]));
        img.put_pixel(4, 3, Rgba([40, 50, 60, 128]));

        let trimmed = trim_transparent(&img).unwrap();
        assert_eq!(trimmed.dimensions(), (3, 3));
        assert_eq!(trimmed.get_pixel(0, 0), &Rgba([10, 20, 30, 255]));
        assert_eq!(trimmed.get_pixel(2, 2), &Rgba([40, 50, 60, 128]));
    }

    #[test]
    fn pad_rgba_adds_transparent_border() {
        let mut img = image::RgbaImage::new(20, 10);
        img.put_pixel(0, 0, Rgba([10, 20, 30, 255]));

        let padded = pad_rgba(&img, 5.0);

        assert_eq!(padded.dimensions(), (22, 12));
        assert_eq!(padded.get_pixel(0, 0), &Rgba([0, 0, 0, 0]));
        assert_eq!(padded.get_pixel(1, 1), &Rgba([10, 20, 30, 255]));
    }

    #[test]
    fn rgb_output_preserves_16_bit_channels() {
        let path = temp_image_path("png");
        let src =
            DynamicImage::ImageRgb16(ImageBuffer::from_pixel(1, 1, Rgb([0x1234, 0x8000, 0xffff])));
        let bounds = RotatedBounds {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 1.0,
            max_y: 1.0,
        };

        write_rgb_output(&src, bounds, 0.0, 0, &path, ImageFormat::Png).unwrap();
        let decoded = image::open(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(decoded.color(), image::ColorType::Rgb16);
        assert_eq!(
            decoded.to_rgb16().get_pixel(0, 0),
            &Rgb([0x1234, 0x8000, 0xffff])
        );
    }

    #[test]
    fn alpha_output_preserves_16_bit_channels_and_writes_16_bit_alpha() {
        let path = temp_image_path("png");
        let src =
            DynamicImage::ImageRgb16(ImageBuffer::from_pixel(1, 1, Rgb([0x1111, 0x8888, 0xeeee])));
        let labels = vec![1usize];
        let obj = Object {
            label: 1,
            area: 1,
            min_x: 0,
            min_y: 0,
            max_x: 0,
            max_y: 0,
        };

        write_alpha_output(
            &src,
            AlphaOutputRequest {
                labels: &labels,
                w: 1,
                obj: &obj,
                theta: 0.0,
                quarter_turns: 0,
                crop_padding: 0.0,
                path: &path,
                format: ImageFormat::Png,
            },
        )
        .unwrap();
        let decoded = image::open(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(decoded.color(), image::ColorType::Rgba16);
        assert_eq!(
            decoded.to_rgba16().get_pixel(0, 0),
            &Rgba([0x1111, 0x8888, 0xeeee, u16::MAX])
        );
    }

    #[test]
    fn tiff_output_uses_deflate_compression_by_default() {
        let path = temp_image_path("tiff");
        let src =
            DynamicImage::ImageRgb16(ImageBuffer::from_pixel(1, 1, Rgb([0x2222, 0x4444, 0x6666])));
        let labels = vec![1usize];
        let obj = Object {
            label: 1,
            area: 1,
            min_x: 0,
            min_y: 0,
            max_x: 0,
            max_y: 0,
        };

        write_alpha_output(
            &src,
            AlphaOutputRequest {
                labels: &labels,
                w: 1,
                obj: &obj,
                theta: 0.0,
                quarter_turns: 0,
                crop_padding: 0.0,
                path: &path,
                format: ImageFormat::Tiff,
            },
        )
        .unwrap();

        let file = File::open(&path).unwrap();
        let mut decoder = TiffDecoder::new(file).unwrap();
        assert_eq!(decoder.get_tag_u32(Tag::Compression).unwrap(), 8);
        assert_eq!(decoder.dimensions().unwrap(), (1, 1));
        let decoded = decoder.read_image().unwrap();
        let _ = std::fs::remove_file(&path);

        assert!(
            matches!(decoded, tiff::decoder::DecodingResult::U16(v) if v == vec![0x2222, 0x4444, 0x6666, u16::MAX])
        );
    }
}
