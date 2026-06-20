//! Image processing operations.
//!
//! Provides fast pixel-level operations for sprite processing,
//! including resize, crop, normalize, and format conversion.

use crate::error::{CoreError, CoreResult};
use image::{DynamicImage, ImageBuffer, Rgba};

/// Resize an image to the target dimensions using LANCZOS3 filtering.
pub fn resize_image(
    pixels: &[u8],
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
) -> CoreResult<Vec<u8>> {
    let img = DynamicImage::ImageRgba8(
        ImageBuffer::from_raw(width, height, pixels.to_vec()).ok_or_else(|| {
            CoreError::InvalidInput("Failed to create image from pixel data".into())
        })?,
    );

    let resized = img.resize_exact(
        target_width,
        target_height,
        image::imageops::FilterType::Lanczos3,
    );

    Ok(resized.to_rgba8().into_raw())
}

/// Crop an image to the specified bounding box.
pub fn crop_image(
    pixels: &[u8],
    width: u32,
    height: u32,
    x: u32,
    y: u32,
    crop_width: u32,
    crop_height: u32,
) -> CoreResult<Vec<u8>> {
    let img = DynamicImage::ImageRgba8(
        ImageBuffer::from_raw(width, height, pixels.to_vec()).ok_or_else(|| {
            CoreError::InvalidInput("Failed to create image from pixel data".into())
        })?,
    );

    let cropped = img.crop_imm(x, y, crop_width, crop_height);
    Ok(cropped.to_rgba8().into_raw())
}

/// Normalize image pixels to [0, 1] range (for ML model input).
pub fn normalize_image(pixels: &[u8]) -> Vec<f32> {
    pixels.iter().map(|&p| p as f32 / 255.0).collect()
}

/// Convert RGBA image to grayscale (luminosity method).
pub fn to_grayscale(pixels: &[u8], width: u32, height: u32) -> CoreResult<Vec<u8>> {
    let img = DynamicImage::ImageRgba8(
        ImageBuffer::from_raw(width, height, pixels.to_vec())
            .ok_or_else(|| CoreError::InvalidInput("Failed to create image".into()))?,
    );

    Ok(img.to_luma8().into_raw())
}

/// Detect non-transparent bounding box of a sprite (trim transparent edges).
pub fn find_sprite_bounds(
    pixels: &[u8],
    width: u32,
    height: u32,
) -> CoreResult<(u32, u32, u32, u32)> {
    let img = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, pixels.to_vec())
        .ok_or_else(|| CoreError::InvalidInput("Failed to create image".into()))?;

    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0u32;
    let mut max_y = 0u32;

    for (x, y, pixel) in img.enumerate_pixels() {
        if pixel[3] > 0 {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }

    if min_x > max_x || min_y > max_y {
        return Err(CoreError::InvalidInput("Image is fully transparent".into()));
    }

    Ok((min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resize_image() {
        // Create a small test image
        let pixels = vec![255u8; 64 * 64 * 4];
        let result = resize_image(&pixels, 64, 64, 32, 32).unwrap();
        assert_eq!(result.len(), 32 * 32 * 4);
    }

    #[test]
    fn test_find_sprite_bounds() {
        // Create a mostly transparent image with a small opaque square
        let mut pixels = vec![0u8; 16 * 16 * 4];
        // Opaque square in the center
        for y in 4..12 {
            for x in 4..12 {
                let idx = (y * 16 + x) * 4;
                pixels[idx + 3] = 255; // Alpha
            }
        }
        let bounds = find_sprite_bounds(&pixels, 16, 16).unwrap();
        assert_eq!(bounds, (4, 4, 8, 8));
    }
}
