//! Gaussian Splat compression — SPZ format implementation.
//!
//! Compresses 3D Gaussian splat data (positions, rotations, scales,
//! opacities, colors) using quantization and entropy coding.

use crate::error::{CoreError, CoreResult};

/// Quality level for splat compression.
#[derive(Debug, Clone, Copy)]
pub enum CompressionQuality {
    Low = 10,
    Medium = 50,
    High = 90,
}

/// Gaussian splat data structure.
#[derive(Debug, Clone)]
pub struct GaussianSplats {
    /// XYZ positions (Nx3)
    pub positions: Vec<[f32; 3]>,
    /// Quaternion rotations (Nx4)
    pub rotations: Vec<[f32; 4]>,
    /// Scale factors (Nx3)
    pub scales: Vec<[f32; 3]>,
    /// Opacity values (N)
    pub opacities: Vec<f32>,
    /// RGB colors (Nx3)
    pub colors: Vec<[f32; 3]>,
}

impl GaussianSplats {
    /// Number of splats.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Check if the data is empty.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

/// Compress Gaussian splats to SPZ format.
///
/// SPZ is a compressed format developed by Niantic that reduces
/// PLY file sizes by up to 90%.
///
/// # Arguments
/// * `splats` - The Gaussian splat data
/// * `quality` - Compression quality
///
/// # Returns
/// Compressed SPZ bytes
pub fn compress_to_spz(
    splats: &GaussianSplats,
    quality: CompressionQuality,
) -> CoreResult<Vec<u8>> {
    if splats.is_empty() {
        return Err(CoreError::InvalidInput("No splats to compress".into()));
    }

    let quality_level = quality as u32;
    let n = splats.len();

    // Simplified SPZ-like compression using quantization and zstd
    // In production, implement the full SPZ spec

    // Build header
    let mut header = Vec::new();
    header.extend_from_slice(b"SPZ"); // Magic bytes
    header.push(1u8); // Version
    header.extend_from_slice(&(n as u32).to_le_bytes());
    header.extend_from_slice(&quality_level.to_le_bytes());

    // Quantize and compress each channel
    let pos_data = quantize_positions(&splats.positions);
    let rot_data = quantize_rotations(&splats.rotations);
    let scale_data = quantize_scales(&splats.scales);
    let opacity_data = quantize_opacities(&splats.opacities);
    let color_data = quantize_colors(&splats.colors);

    // Compress each channel with zstd
    let mut compressed = Vec::new();
    compressed.extend_from_slice(&header);

    compress_channel(&mut compressed, &pos_data, quality)?;
    compress_channel(&mut compressed, &rot_data, quality)?;
    compress_channel(&mut compressed, &scale_data, quality)?;
    compress_channel(&mut compressed, &opacity_data, quality)?;
    compress_channel(&mut compressed, &color_data, quality)?;

    Ok(compressed)
}

/// Decompress SPZ format back to Gaussian splats.
pub fn decompress_spz(data: &[u8]) -> CoreResult<GaussianSplats> {
    if data.len() < 12 {
        return Err(CoreError::InvalidInput("SPZ data too short".into()));
    }

    // Validate header
    if &data[0..3] != b"SPZ" {
        return Err(CoreError::InvalidInput("Invalid SPZ magic bytes".into()));
    }

    let _version = data[3];
    let n = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    let _quality = u32::from_le_bytes(data[8..12].try_into().unwrap());

    let mut offset = 12;

    let pos_data = decompress_channel(data, &mut offset, n * 3 * 2)?; // f16 quantized = 2 bytes each
    let rot_data = decompress_channel(data, &mut offset, n * 4 * 2)?;
    let scale_data = decompress_channel(data, &mut offset, n * 3 * 2)?;
    let opacity_data = decompress_channel(data, &mut offset, n * 2)?; // u16 quantized
    let color_data = decompress_channel(data, &mut offset, n * 3 * 2)?;

    let positions = dequantize_positions(&pos_data, n);
    let rotations = dequantize_rotations(&rot_data, n);
    let scales = dequantize_scales(&scale_data, n);
    let opacities = dequantize_opacities(&opacity_data, n);
    let colors = dequantize_colors(&color_data, n);

    Ok(GaussianSplats {
        positions,
        rotations,
        scales,
        opacities,
        colors,
    })
}

// ── Quantization helpers ──

fn quantize_positions(positions: &[[f32; 3]]) -> Vec<u16> {
    positions.iter().flat_map(|p| {
        p.iter().map(|&v| {
            // Normalize to [0, 65535] range (assumes positions in unit cube)
            let clamped = v.clamp(-10.0, 10.0);
            let normalized = (clamped + 10.0) / 20.0;
            (normalized * 65535.0) as u16
        })
    }).collect()
}

fn dequantize_positions(data: &[u16], n: usize) -> Vec<[f32; 3]> {
    data.chunks(3).take(n).map(|chunk| {
        [
            (chunk[0] as f32 / 65535.0) * 20.0 - 10.0,
            (chunk[1] as f32 / 65535.0) * 20.0 - 10.0,
            (chunk[2] as f32 / 65535.0) * 20.0 - 10.0,
        ]
    }).collect()
}

fn quantize_rotations(rotations: &[[f32; 4]]) -> Vec<u16> {
    rotations.iter().flat_map(|q| {
        q.iter().map(|&v| {
            // Map from [-1, 1] to [0, 65535]
            let clamped = v.clamp(-1.0, 1.0);
            let normalized = (clamped + 1.0) / 2.0;
            (normalized * 65535.0) as u16
        })
    }).collect()
}

fn dequantize_rotations(data: &[u16], n: usize) -> Vec<[f32; 4]> {
    data.chunks(4).take(n).map(|chunk| {
        let mut q = [
            (chunk[0] as f32 / 65535.0) * 2.0 - 1.0,
            (chunk[1] as f32 / 65535.0) * 2.0 - 1.0,
            (chunk[2] as f32 / 65535.0) * 2.0 - 1.0,
            (chunk[3] as f32 / 65535.0) * 2.0 - 1.0,
        ];
        // Normalize quaternion
        let len = (q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]).sqrt();
        if len > 0.0 {
            for v in &mut q {
                *v /= len;
            }
        }
        q
    }).collect()
}

fn quantize_scales(scales: &[[f32; 3]]) -> Vec<u16> {
    scales.iter().flat_map(|s| {
        s.iter().map(|&v| {
            let clamped = v.clamp(0.0, 100.0);
            let normalized = clamped / 100.0;
            (normalized * 65535.0) as u16
        })
    }).collect()
}

fn dequantize_scales(data: &[u16], n: usize) -> Vec<[f32; 3]> {
    data.chunks(3).take(n).map(|chunk| {
        [
            chunk[0] as f32 / 65535.0 * 100.0,
            chunk[1] as f32 / 65535.0 * 100.0,
            chunk[2] as f32 / 65535.0 * 100.0,
        ]
    }).collect()
}

fn quantize_opacities(opacities: &[f32]) -> Vec<u16> {
    opacities.iter().map(|&v| {
        let clamped = v.clamp(0.0, 1.0);
        (clamped * 65535.0) as u16
    }).collect()
}

fn dequantize_opacities(data: &[u16], n: usize) -> Vec<f32> {
    data.iter().take(n).map(|&v| {
        v as f32 / 65535.0
    }).collect()
}

fn quantize_colors(colors: &[[f32; 3]]) -> Vec<u16> {
    colors.iter().flat_map(|c| {
        c.iter().map(|&v| {
            let clamped = v.clamp(0.0, 1.0);
            (clamped * 65535.0) as u16
        })
    }).collect()
}

fn dequantize_colors(data: &[u16], n: usize) -> Vec<[f32; 3]> {
    data.chunks(3).take(n).map(|chunk| {
        [
            chunk[0] as f32 / 65535.0,
            chunk[1] as f32 / 65535.0,
            chunk[2] as f32 / 65535.0,
        ]
    }).collect()
}

fn compress_channel(
    output: &mut Vec<u8>,
    data: &[u16],
    quality: CompressionQuality,
) -> CoreResult<()> {
    let byte_data: Vec<u8> = data.iter()
        .flat_map(|&v| v.to_le_bytes())
        .collect();

    let level = match quality {
        CompressionQuality::Low => 1,
        CompressionQuality::Medium => 6,
        CompressionQuality::High => 19,
    };

    let compressed = zstd::encode_all(&byte_data[..], level)
        .map_err(|e| CoreError::Compression(format!("ZSTD compression failed: {}", e)))?;

    output.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
    output.extend_from_slice(&compressed);
    Ok(())
}

fn decompress_channel(
    data: &[u8],
    offset: &mut usize,
    _expected_bytes: usize,
) -> CoreResult<Vec<u16>> {
    if *offset + 4 > data.len() {
        return Err(CoreError::InvalidInput("Unexpected end of SPZ data".into()));
    }

    let compressed_len = u32::from_le_bytes(
        data[*offset..*offset + 4].try_into().unwrap()
    ) as usize;
    *offset += 4;

    if *offset + compressed_len > data.len() {
        return Err(CoreError::InvalidInput("Compressed chunk exceeds data length".into()));
    }

    let decompressed = zstd::decode_all(&data[*offset..*offset + compressed_len])
        .map_err(|e| CoreError::Compression(format!("ZSTD decompression failed: {}", e)))?;
    *offset += compressed_len;

    let result: Vec<u16> = decompressed.chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_small_splat() {
        let splats = GaussianSplats {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            rotations: vec![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            scales: vec![[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]],
            opacities: vec![0.5, 0.8],
            colors: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        };

        let compressed = compress_to_spz(&splats, CompressionQuality::High).unwrap();
        let decompressed = decompress_spz(&compressed).unwrap();

        assert_eq!(decompressed.len(), 2);
        assert!((decompressed.opacities[0] - 0.5).abs() < 0.01);
        assert!((decompressed.opacities[1] - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_empty_splats() {
        let splats = GaussianSplats {
            positions: vec![],
            rotations: vec![],
            scales: vec![],
            opacities: vec![],
            colors: vec![],
        };
        let result = compress_to_spz(&splats, CompressionQuality::Medium);
        assert!(result.is_err());
    }
}
