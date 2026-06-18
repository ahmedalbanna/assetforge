//! Bitmap vectorization using a Potrace-style algorithm.
//!
//! Converts binary bitmap images to vector paths (Bezier curves).
//! This is a simplified implementation of the Potrace algorithm.

use crate::error::{CoreError, CoreResult};

/// Configuration for the vectorization process.
#[derive(Debug, Clone)]
pub struct VectorizationConfig {
    /// Binarization threshold (0-255)
    pub threshold: u8,
    /// Suppress speckles smaller than this many pixels
    pub turdsize: usize,
    /// Corner rounding parameter (0.0 - 1.34)
    pub alphamax: f64,
    /// Optimize curve paths
    pub opticurve: bool,
    /// Optimization tolerance
    pub opttolerance: f64,
}

impl Default for VectorizationConfig {
    fn default() -> Self {
        Self {
            threshold: 128,
            turdsize: 2,
            alphamax: 1.0,
            opticurve: true,
            opttolerance: 0.2,
        }
    }
}

/// A segment of a vector path.
#[derive(Debug, Clone)]
pub enum PathSegment {
    /// A sharp corner
    Corner { point: (f64, f64) },
    /// A cubic Bezier curve
    Bezier {
        c1: (f64, f64),
        c2: (f64, f64),
        end: (f64, f64),
    },
}

/// A complete vector path composed of segments.
#[derive(Debug, Clone)]
pub struct VectorPath {
    /// Starting point of the path
    pub start_point: (f64, f64),
    /// Segments making up the path
    pub segments: Vec<PathSegment>,
    /// Bounding box: (x, y, width, height)
    pub bounding_box: (f64, f64, f64, f64),
}

/// Trace a binary bitmap and return vector paths.
///
/// # Arguments
/// * `pixels` - Binary pixel data (0 or 255)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `config` - Vectorization configuration
///
/// # Returns
/// A list of vector paths found in the bitmap
pub fn trace_bitmap(
    pixels: &[u8],
    width: u32,
    height: u32,
    config: &VectorizationConfig,
) -> CoreResult<Vec<VectorPath>> {
    if pixels.len() != (width * height) as usize {
        return Err(CoreError::InvalidInput(format!(
            "Pixel buffer size mismatch: expected {}, got {}",
            width * height,
            pixels.len()
        )));
    }

    // Binarize
    let binary: Vec<bool> = pixels
        .iter()
        .map(|&p| p > config.threshold)
        .collect();

    // Find connected components (simplified)
    let paths = find_contours(&binary, width, height, config)?;

    Ok(paths)
}

/// Find contours in a binary image and convert to vector paths.
fn find_contours(
    binary: &[bool],
    width: u32,
    height: u32,
    _config: &VectorizationConfig,
) -> CoreResult<Vec<VectorPath>> {
    let mut visited = vec![false; binary.len()];
    let mut paths = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            if binary[idx] && !visited[idx] {
                // Trace the boundary of this connected component
                if let Some(path) = trace_component(binary, &mut visited, x, y, width, height) {
                    paths.push(path);
                }
            }
        }
    }

    Ok(paths)
}

/// Trace the boundary of a single connected component.
fn trace_component(
    binary: &[bool],
    visited: &mut [bool],
    start_x: u32,
    start_y: u32,
    width: u32,
    height: u32,
) -> Option<VectorPath> {
    // Simplified: return bounding box as a rectangle path
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0u32;
    let mut max_y = 0u32;

    // Flood fill to find all pixels in this component
    let mut stack = vec![(start_x, start_y)];
    let mut pixel_count = 0u32;

    while let Some((cx, cy)) = stack.pop() {
        let idx = (cy * width + cx) as usize;
        if visited[idx] || !binary[idx] {
            continue;
        }
        visited[idx] = true;
        pixel_count += 1;

        min_x = min_x.min(cx);
        min_y = min_y.min(cy);
        max_x = max_x.max(cx);
        max_y = max_y.max(cy);

        // Check 4-connected neighbors
        if cx > 0 { stack.push((cx - 1, cy)); }
        if cx + 1 < width { stack.push((cx + 1, cy)); }
        if cy > 0 { stack.push((cx, cy - 1)); }
        if cy + 1 < height { stack.push((cx, cy + 1)); }
    }

    if pixel_count == 0 {
        return None;
    }

    // Create a rectangle path for this component
    let x0 = min_x as f64;
    let y0 = min_y as f64;
    let x1 = (max_x + 1) as f64;
    let y1 = (max_y + 1) as f64;

    let path = VectorPath {
        start_point: (x0, y0),
        segments: vec![
            PathSegment::Corner { point: (x1, y0) },
            PathSegment::Corner { point: (x1, y1) },
            PathSegment::Corner { point: (x0, y1) },
        ],
        bounding_box: (x0, y0, x1 - x0, y1 - y0),
    };

    Some(path)
}

/// Convert vector paths to SVG format string.
pub fn paths_to_svg(
    paths: &[VectorPath],
    width: u32,
    height: u32,
) -> String {
    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        width, height, width, height
    ));
    svg.push_str(r#"<path d=""#);

    for path in paths {
        svg.push_str(&format!("M {} {}", path.start_point.0, path.start_point.1));
        for segment in &path.segments {
            match segment {
                PathSegment::Corner { point } => {
                    svg.push_str(&format!(" L {} {}", point.0, point.1));
                }
                PathSegment::Bezier { c1, c2, end } => {
                    svg.push_str(&format!(
                        " C {} {} {} {} {} {}",
                        c1.0, c1.1, c2.0, c2.1, end.0, end.1
                    ));
                }
            }
        }
        svg.push_str(" Z");
    }

    svg.push_str(r#"" fill="black" /></svg>"#);
    svg
}

/// Rasterize vector paths back to a bitmap at the target resolution.
pub fn rasterize_paths(
    paths: &[VectorPath],
    output_width: u32,
    output_height: u32,
) -> CoreResult<Vec<u8>> {
    // Use the SVG string as intermediate representation
    let svg = paths_to_svg(paths, output_width, output_height);

    // Create a simple rasterizer using the svg string
    // In production, use resvg or similar for proper SVG rasterization
    let svg_bytes = svg.as_bytes();
    let img = rasterize_svg_bytes(svg_bytes, output_width, output_height)?;
    Ok(img)
}

/// Simple SVG rasterizer (placeholder — use resvg in production).
fn rasterize_svg_bytes(
    _svg_bytes: &[u8],
    width: u32,
    height: u32,
) -> CoreResult<Vec<u8>> {
    // Create a white image as placeholder
    // In production, use resvg or tiny_skia for proper rendering
    let mut pixels = vec![255u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize * 4;
            pixels[idx] = 0;
            pixels[idx + 1] = 0;
            pixels[idx + 2] = 0;
            pixels[idx + 3] = 255;
        }
    }
    Ok(pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_simple_shape() {
        // Create a 4x4 black square in an 8x8 image
        let mut pixels = vec![0u8; 64];
        for y in 2..6 {
            for x in 2..6 {
                pixels[(y * 8 + x) as usize] = 255;
            }
        }

        let config = VectorizationConfig::default();
        let paths = trace_bitmap(&pixels, 8, 8, &config).unwrap();
        assert!(!paths.is_empty());

        let svg = paths_to_svg(&paths, 8, 8);
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
    }

    #[test]
    fn test_invalid_pixel_buffer() {
        let pixels = vec![0u8; 10]; // Wrong size
        let config = VectorizationConfig::default();
        let result = trace_bitmap(&pixels, 8, 8, &config);
        assert!(result.is_err());
    }
}
