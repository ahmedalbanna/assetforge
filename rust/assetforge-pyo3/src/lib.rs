//! # AssetForge PyO3 Bridge
//!
//! Exposes Rust core functions to Python for high-performance operations.
//!
//! Available Python functions:
//! - `vectorize_bitmap` — Convert binary bitmap to SVG vector paths
//! - `compress_splats` — Compress Gaussian splats to SPZ format
//! - `rasterize_sprite` — Rasterize SVG paths to PNG pixels
//! - `optimize_mesh` — Simplify mesh vertices/faces

use assetforge_core::mesh_utils::{self, Face, Mesh, Vertex};
use assetforge_core::splat_compress::{self, CompressionQuality, GaussianSplats};
use assetforge_core::vectorize::{self, VectorizationConfig};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Convert a binary bitmap to SVG vector paths.
///
/// Args:
///     pixels: Binary pixel data (bytes, each is 0 or 255)
///     width: Image width
///     height: Image height
///     threshold: Binarization threshold (0-255)
///     turdsize: Speckle suppression size
///
/// Returns:
///     SVG string representation of the vector paths
#[pyfunction]
fn vectorize_bitmap(
    _py: Python,
    pixels: &[u8],
    width: u32,
    height: u32,
    threshold: u8,
    turdsize: usize,
) -> PyResult<String> {
    let config = VectorizationConfig {
        threshold,
        turdsize,
        ..Default::default()
    };

    let paths = vectorize::trace_bitmap(pixels, width, height, &config)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let svg = vectorize::paths_to_svg(&paths, width, height);
    Ok(svg)
}

/// Compress Gaussian splat data to SPZ format.
///
/// Args:
///     positions: List of [x, y, z] position arrays
///     rotations: List of [w, x, y, z] quaternion arrays
///     scales: List of [sx, sy, sz] scale arrays
///     opacities: List of opacity values
///     colors: List of [r, g, b] color arrays
///     quality: Compression quality (0-100)
///
/// Returns:
///     Compressed SPZ bytes
#[pyfunction]
fn compress_splats(
    _py: Python,
    positions: Vec<[f32; 3]>,
    rotations: Vec<[f32; 4]>,
    scales: Vec<[f32; 3]>,
    opacities: Vec<f32>,
    colors: Vec<[f32; 3]>,
    quality: u8,
) -> PyResult<Vec<u8>> {
    let splats = GaussianSplats {
        positions,
        rotations,
        scales,
        opacities,
        colors,
    };

    let quality_enum = if quality > 66 {
        CompressionQuality::High
    } else if quality > 33 {
        CompressionQuality::Medium
    } else {
        CompressionQuality::Low
    };

    let compressed = splat_compress::compress_to_spz(&splats, quality_enum)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(compressed)
}

/// Rasterize SVG paths to RGBA pixel data.
///
/// Args:
///     svg_content: SVG string content
///     output_width: Target width in pixels
///     output_height: Target height in pixels
///
/// Returns:
///     RGBA pixel bytes
#[pyfunction]
fn rasterize_sprite(
    _py: Python,
    svg_content: &str,
    output_width: u32,
    output_height: u32,
) -> PyResult<Vec<u8>> {
    // Parse SVG paths
    let paths = parse_svg_paths(svg_content).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let pixels = vectorize::rasterize_paths(&paths, output_width, output_height)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(pixels)
}

/// Optimize/simplify a 3D mesh by reducing polygon count.
///
/// Args:
///     vertices: List of [x, y, z] vertex positions
///     indices: Triangle vertex indices (flat list, multiple of 3)
///     target_ratio: Target face ratio (0.0 - 1.0)
///
/// Returns:
///     Tuple of (vertices, indices) for the simplified mesh
#[pyfunction]
fn optimize_mesh(
    _py: Python,
    vertices: Vec<[f32; 3]>,
    indices: Vec<u32>,
    target_ratio: f32,
) -> PyResult<(Vec<[f32; 3]>, Vec<u32>)> {
    if indices.len() % 3 != 0 {
        return Err(PyValueError::new_err(
            "Indices length must be a multiple of 3",
        ));
    }

    let faces: Vec<Face> = indices
        .chunks_exact(3)
        .map(|chunk| Face {
            v0: chunk[0] as usize,
            v1: chunk[1] as usize,
            v2: chunk[2] as usize,
        })
        .collect();

    let mesh = Mesh {
        vertices: vertices
            .iter()
            .map(|v| Vertex {
                x: v[0],
                y: v[1],
                z: v[2],
            })
            .collect(),
        faces,
    };

    let simplified = mesh_utils::simplify_mesh(&mesh, target_ratio)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let out_vertices: Vec<[f32; 3]> = simplified
        .vertices
        .iter()
        .map(|v| [v.x, v.y, v.z])
        .collect();

    let out_indices: Vec<u32> = simplified
        .faces
        .iter()
        .flat_map(|f| vec![f.v0 as u32, f.v1 as u32, f.v2 as u32])
        .collect();

    Ok((out_vertices, out_indices))
}

/// Simple SVG path parser (minimal implementation).
fn parse_svg_paths(svg: &str) -> Result<Vec<vectorize::VectorPath>, String> {
    // For production, use a proper SVG parser like resvg
    // This is a simplified placeholder
    let mut paths = Vec::new();

    for line in svg.lines() {
        let line = line.trim();
        if line.starts_with("<path") {
            // Extract 'd' attribute and parse path data
            if let Some(d_start) = line.find("d=\"") {
                let d_end = line[d_start + 3..]
                    .find('"')
                    .map(|i| d_start + 3 + i)
                    .unwrap_or(line.len());
                let d_data = &line[d_start + 3..d_end];

                if let Some(path) = parse_svg_path_data(d_data) {
                    paths.push(path);
                }
            }
        }
    }

    Ok(paths)
}

/// Parse SVG path data string into a VectorPath.
fn parse_svg_path_data(data: &str) -> Option<vectorize::VectorPath> {
    let data = data.trim();
    if data.is_empty() {
        return None;
    }

    // Simple 'M L Z' path parser
    let mut start_point = (0.0, 0.0);
    let mut current = (0.0, 0.0);
    let mut segments = Vec::new();
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;

    let parts: Vec<&str> = data.split_whitespace().collect();
    let mut i = 0;

    while i < parts.len() {
        match parts[i] {
            "M" | "m" => {
                if i + 2 < parts.len() {
                    let x = parts[i + 1].parse::<f64>().unwrap_or(0.0);
                    let y = parts[i + 2].parse::<f64>().unwrap_or(0.0);
                    if parts[i] == "M" {
                        start_point = (x, y);
                    } else {
                        start_point = (current.0 + x, current.1 + y);
                    }
                    current = start_point;
                    update_bounds(current, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                    i += 3;
                } else {
                    i += 1;
                }
            }
            "L" | "l" => {
                if i + 2 < parts.len() {
                    let x = parts[i + 1].parse::<f64>().unwrap_or(0.0);
                    let y = parts[i + 2].parse::<f64>().unwrap_or(0.0);
                    let point = if parts[i] == "L" {
                        (x, y)
                    } else {
                        (current.0 + x, current.1 + y)
                    };
                    segments.push(vectorize::PathSegment::Corner { point });
                    current = point;
                    update_bounds(current, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                    i += 3;
                } else {
                    i += 1;
                }
            }
            "Z" | "z" => {
                segments.push(vectorize::PathSegment::Corner { point: start_point });
                current = start_point;
                i += 1;
            }
            "C" | "c" => {
                if i + 6 < parts.len() {
                    let parse_coord = |idx: usize, relative: bool| -> (f64, f64) {
                        let x = parts[idx].parse::<f64>().unwrap_or(0.0);
                        let y = parts[idx + 1].parse::<f64>().unwrap_or(0.0);
                        if relative {
                            (current.0 + x, current.1 + y)
                        } else {
                            (x, y)
                        }
                    };

                    let is_rel = parts[i] == "c";
                    let c1 = parse_coord(i + 1, is_rel);
                    let c2 = parse_coord(i + 3, is_rel);
                    let end = parse_coord(i + 5, is_rel);

                    segments.push(vectorize::PathSegment::Bezier { c1, c2, end });
                    current = end;
                    update_bounds(current, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                    i += 7;
                } else {
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }

    Some(vectorize::VectorPath {
        start_point,
        segments,
        bounding_box: (min_x, min_y, max_x - min_x, max_y - min_y),
    })
}

fn update_bounds(
    point: (f64, f64),
    min_x: &mut f64,
    min_y: &mut f64,
    max_x: &mut f64,
    max_y: &mut f64,
) {
    *min_x = (*min_x).min(point.0);
    *min_y = (*min_y).min(point.1);
    *max_x = (*max_x).max(point.0);
    *max_y = (*max_y).max(point.1);
}

/// Export Gaussian splats to GLTF format with KHR_gaussian_splatting extension.
///
/// Args:
///     splat_path: Path to the splat file (.ply or .spz)
///     output_path: Path for the output .glb file
///
/// Returns:
///     Success status
#[pyfunction]
fn export_splat_gltf(_py: Python, splat_path: &str, output_path: &str) -> PyResult<bool> {
    use std::fs;
    use std::path::Path;

    let splat = Path::new(splat_path);
    let output = Path::new(output_path);

    if !splat.exists() {
        return Err(PyValueError::new_err(format!(
            "Splat file not found: {}",
            splat_path
        )));
    }

    // Read splat data
    let data = fs::read(splat)
        .map_err(|e| PyValueError::new_err(format!("Failed to read splat: {}", e)))?;

    let ext = splat.extension().and_then(|e| e.to_str()).unwrap_or("ply");

    match ext {
        "spz" => {
            // Write SPZ data as-is into a minimal GLB container
            // For production, implement the full KHR_gaussian_splatting extension
            let glb_header = vec![
                0x67, 0x6C, 0x54, 0x46, // magic: glTF
                0x02, 0x00, 0x00, 0x00, // version 2
                0x00, 0x00, 0x00, 0x00, // total length (placeholder)
            ];
            let mut glb = glb_header;

            // JSON chunk referencing extension
            let json = br#"{
                "asset": {"version": "2.0", "generator": "AssetForge"},
                "extensionsUsed": ["KHR_gaussian_splatting"],
                "extensions": {
                    "KHR_gaussian_splatting": {
                        "splatUri": "splat.spz"
                    }
                }
            }"#;

            // Write JSON chunk
            let json_padded = pad_to_4(json);
            let json_len = json_padded.len() as u32;
            glb.extend_from_slice(&json_len.to_le_bytes());
            glb.extend_from_slice(b"JSON");
            glb.extend_from_slice(&json_padded);

            // Write binary chunk (SPZ data)
            let bin_padded = pad_to_4(&data);
            let bin_len = data.len() as u32;
            glb.extend_from_slice(&bin_len.to_le_bytes());
            glb.extend_from_slice(b"BIN\x00");
            glb.extend_from_slice(&bin_padded);

            // Update total length in header
            let total_len = glb.len() as u32;
            glb[8..12].copy_from_slice(&total_len.to_le_bytes());

            fs::write(output, &glb)
                .map_err(|e| PyValueError::new_err(format!("Failed to write GLB: {}", e)))?;

            Ok(true)
        }
        "ply" => {
            // For PLY files, create a simple container
            fs::copy(splat, output.with_extension("ply"))
                .map_err(|e| PyValueError::new_err(format!("Failed to copy PLY: {}", e)))?;
            Ok(true)
        }
        _ => Err(PyValueError::new_err(format!(
            "Unsupported splat format: {}",
            ext
        ))),
    }
}

/// Pad bytes to 4-byte alignment (glTF 2.0 requirement).
fn pad_to_4(data: &[u8]) -> Vec<u8> {
    let padding = (4 - (data.len() % 4)) % 4;
    let mut padded = data.to_vec();
    padded.extend(std::iter::repeat(0x20).take(padding));
    padded
}

/// PyO3 module initialization.
#[pymodule]
fn assetforge_pyo3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(vectorize_bitmap, m)?)?;
    m.add_function(wrap_pyfunction!(compress_splats, m)?)?;
    m.add_function(wrap_pyfunction!(rasterize_sprite, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(export_splat_gltf, m)?)?;
    Ok(())
}
