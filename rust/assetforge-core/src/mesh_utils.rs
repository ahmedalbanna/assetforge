//! Mesh utilities — simplification, format conversion, and analysis.
//!
//! Provides fast mesh simplification and format detection/ conversion.

use crate::error::{CoreError, CoreResult};

/// A simple 3D vertex.
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// A triangle face (indices into vertex array).
#[derive(Debug, Clone, Copy)]
pub struct Face {
    pub v0: usize,
    pub v1: usize,
    pub v2: usize,
}

/// A 3D mesh consisting of vertices and faces.
#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub faces: Vec<Face>,
}

impl Mesh {
    /// Create a new empty mesh.
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            faces: Vec::new(),
        }
    }

    /// Total number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.faces.len()
    }

    /// Total number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Compute the bounding box of the mesh.
    pub fn bounding_box(&self) -> (f32, f32, f32, f32, f32, f32) {
        if self.vertices.is_empty() {
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut min_z = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        let mut max_z = f32::MIN;

        for v in &self.vertices {
            min_x = min_x.min(v.x);
            min_y = min_y.min(v.y);
            min_z = min_z.min(v.z);
            max_x = max_x.max(v.x);
            max_y = max_y.max(v.y);
            max_z = max_z.max(v.z);
        }

        (min_x, min_y, min_z, max_x, max_y, max_z)
    }
}

impl Default for Mesh {
    fn default() -> Self {
        Self::new()
    }
}

/// Simplify a mesh by collapsing edges (quadric error metric based).
///
/// # Arguments
/// * `mesh` - Input mesh
/// * `target_ratio` - Target ratio of faces to keep (0.0 - 1.0)
///
/// # Returns
/// Simplified mesh
pub fn simplify_mesh(mesh: &Mesh, target_ratio: f32) -> CoreResult<Mesh> {
    if target_ratio <= 0.0 || target_ratio > 1.0 {
        return Err(CoreError::InvalidInput(format!(
            "Target ratio must be in (0, 1], got {}",
            target_ratio
        )));
    }

    let target_faces = (mesh.faces.len() as f32 * target_ratio) as usize;
    if target_faces >= mesh.faces.len() {
        return Ok(mesh.clone());
    }

    // Simplified edge collapse using vertex clustering
    // In production, use a full QEM (Quadric Error Metric) implementation
    let simplified = cluster_vertices(mesh, target_faces)?;
    Ok(simplified)
}

/// Simple vertex clustering simplification.
fn cluster_vertices(mesh: &Mesh, _target_faces: usize) -> CoreResult<Mesh> {
    // Placeholder: returns a decimated version by sampling
    // In production, implement full edge collapse with QEM
    if mesh.faces.len() <= 100 {
        return Ok(mesh.clone());
    }

    // Simple uniform sampling for placeholder
    let step = (mesh.faces.len() / _target_faces.max(1)).max(1);
    let simplified_faces: Vec<Face> = mesh.faces.iter().step_by(step).copied().collect();

    Ok(Mesh {
        vertices: mesh.vertices.clone(),
        faces: simplified_faces,
    })
}

/// Convert mesh to PLY format (ASCII).
pub fn mesh_to_ply(mesh: &Mesh) -> String {
    let mut ply = String::new();
    ply.push_str("ply\n");
    ply.push_str("format ascii 1.0\n");
    ply.push_str(&format!("element vertex {}\n", mesh.vertices.len()));
    ply.push_str("property float x\n");
    ply.push_str("property float y\n");
    ply.push_str("property float z\n");
    ply.push_str(&format!("element face {}\n", mesh.faces.len()));
    ply.push_str("property list uchar int vertex_indices\n");
    ply.push_str("end_header\n");

    for v in &mesh.vertices {
        ply.push_str(&format!("{} {} {}\n", v.x, v.y, v.z));
    }

    for f in &mesh.faces {
        ply.push_str(&format!("3 {} {} {}\n", f.v0, f.v1, f.v2));
    }

    ply
}

/// Parse a simple OBJ format string into a Mesh.
pub fn parse_obj(obj_data: &str) -> CoreResult<Mesh> {
    let mut mesh = Mesh::new();

    for line in obj_data.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "v" => {
                if parts.len() >= 4 {
                    let x = parts[1].parse::<f32>()
                        .map_err(|e| CoreError::InvalidInput(format!("Invalid vertex x: {}", e)))?;
                    let y = parts[2].parse::<f32>()
                        .map_err(|e| CoreError::InvalidInput(format!("Invalid vertex y: {}", e)))?;
                    let z = parts[3].parse::<f32>()
                        .map_err(|e| CoreError::InvalidInput(format!("Invalid vertex z: {}", e)))?;
                    mesh.vertices.push(Vertex { x, y, z });
                }
            }
            "f" => {
                if parts.len() >= 4 {
                    // Parse face indices (handle v/vt/vn format)
                    let parse_idx = |s: &str| -> CoreResult<usize> {
                        let idx_str = s.split('/').next().unwrap_or(s);
                        let idx: i32 = idx_str.parse()
                            .map_err(|e| CoreError::InvalidInput(format!("Invalid face index: {}", e)))?;
                        if idx > 0 {
                            Ok((idx - 1) as usize)
                        } else {
                            // Negative indices are relative
                            Ok((mesh.vertices.len() as i32 + idx) as usize)
                        }
                    };

                    let v0 = parse_idx(parts[1])?;
                    let v1 = parse_idx(parts[2])?;
                    let v2 = parse_idx(parts[3])?;
                    mesh.faces.push(Face { v0, v1, v2 });
                }
            }
            _ => {}
        }
    }

    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_bounding_box() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex { x: 0.0, y: 0.0, z: 0.0 });
        mesh.vertices.push(Vertex { x: 1.0, y: 1.0, z: 1.0 });
        mesh.faces.push(Face { v0: 0, v1: 1, v2: 0 });

        let bb = mesh.bounding_box();
        assert_eq!(bb, (0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    }

    #[test]
    fn test_parse_obj() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
        let mesh = parse_obj(obj).unwrap();
        assert_eq!(mesh.vertices.len(), 3);
        assert_eq!(mesh.faces.len(), 1);
    }

    #[test]
    fn test_mesh_to_ply() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex { x: 0.0, y: 0.0, z: 0.0 });
        mesh.vertices.push(Vertex { x: 1.0, y: 0.0, z: 0.0 });
        mesh.vertices.push(Vertex { x: 0.0, y: 1.0, z: 0.0 });
        mesh.faces.push(Face { v0: 0, v1: 1, v2: 2 });

        let ply = mesh_to_ply(&mesh);
        assert!(ply.contains("element vertex 3"));
        assert!(ply.contains("element face 1"));
    }
}
