//! Mesh utilities — simplification, format conversion, and analysis.
//!
//! Provides fast mesh simplification using Quadric Error Metrics (QEM)
//! edge collapse, along with format conversion (OBJ/PLY) and analysis.
//!
//! # Algorithm
//! The simplification uses iterative edge collapse with quadric error
//! metrics. For each vertex, a fundamental quadric Q = (A, b, c) encodes
//! the sum of squared distances to its incident face planes. When an edge
//! (v1, v2) is collapsed to a new vertex v, the cost is v^T A v + 2b^T v + c.
//! Edges with the lowest cost are collapsed first.

use crate::error::{CoreError, CoreResult};
use std::collections::HashMap;

/// A simple 3D vertex.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// A triangle face (indices into vertex array).
#[derive(Debug, Clone, Copy, PartialEq)]
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

// ─── Quadric Error Metric types ──────────────────────────────────

/// A symmetric 3x3 matrix + 3-vector + scalar for the quadric error.
#[derive(Debug, Clone, Copy)]
struct Quadric {
    /// 3x3 matrix A (symmetric, stored as [a00, a01, a02, a11, a12, a22])
    a: [f32; 6],
    /// 3-vector b
    b: [f32; 3],
    /// Scalar c
    c: f32,
}

impl Quadric {
    fn zero() -> Self {
        Self {
            a: [0.0; 6],
            b: [0.0; 3],
            c: 0.0,
        }
    }

    /// Create a quadric from a plane equation: n·p + d = 0
    /// where n is unit normal and d is the plane constant.
    fn from_plane(nx: f32, ny: f32, nz: f32, d: f32) -> Self {
        let area_weight = 1.0; // Can weight by triangle area
        Self {
            a: [
                nx * nx * area_weight,
                nx * ny * area_weight,
                nx * nz * area_weight,
                ny * ny * area_weight,
                ny * nz * area_weight,
                nz * nz * area_weight,
            ],
            b: [
                nx * d * area_weight,
                ny * d * area_weight,
                nz * d * area_weight,
            ],
            c: d * d * area_weight,
        }
    }

    /// Add another quadric to this one.
    fn add(&mut self, other: &Quadric) {
        for i in 0..6 {
            self.a[i] += other.a[i];
        }
        for i in 0..3 {
            self.b[i] += other.b[i];
        }
        self.c += other.c;
    }

    /// Evaluate the quadratic error at point (x, y, z).
    fn evaluate(&self, x: f32, y: f32, z: f32) -> f32 {
        // v^T A v + 2*b·v + c
        let v_av = x * (self.a[0] * x + self.a[1] * y + self.a[2] * z)
            + y * (self.a[1] * x + self.a[3] * y + self.a[4] * z)
            + z * (self.a[2] * x + self.a[4] * y + self.a[5] * z);
        let b_dot_v = self.b[0] * x + self.b[1] * y + self.b[2] * z;
        v_av + 2.0 * b_dot_v + self.c
    }

    /// Find the optimal position that minimizes this quadric.
    /// Returns None if the matrix is singular.
    fn optimal_position(&self) -> Option<(f32, f32, f32)> {
        // Solve A·v = -b using Cramer's rule on the 3x3 system
        let a00 = self.a[0];
        let a01 = self.a[1];
        let a02 = self.a[2];
        let a10 = self.a[1];
        let a11 = self.a[3];
        let a12 = self.a[4];
        let a20 = self.a[2];
        let a21 = self.a[4];
        let a22 = self.a[5];

        let det = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20)
            + a02 * (a10 * a21 - a11 * a20);

        if det.abs() < 1e-10 {
            return None; // Singular matrix
        }

        let inv_det = 1.0 / det;
        let bx = -self.b[0];
        let by = -self.b[1];
        let bz = -self.b[2];

        let x = (bx * (a11 * a22 - a12 * a21) - a01 * (by * a22 - a12 * bz)
            + a02 * (by * a21 - a11 * bz))
            * inv_det;
        let y = (a00 * (by * a22 - a12 * bz) - bx * (a10 * a22 - a12 * a20)
            + a02 * (a10 * bz - by * a20))
            * inv_det;
        let z = (a00 * (a11 * bz - by * a21) - a01 * (a10 * bz - by * a20)
            + bx * (a10 * a21 - a11 * a20))
            * inv_det;

        Some((x, y, z))
    }
}

// ─── Edge structure ──────────────────────────────────────────────

/// An edge between two vertices (v0 < v1 for consistency).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Edge {
    v0: usize,
    v1: usize,
}

impl Edge {
    fn new(a: usize, b: usize) -> Self {
        if a < b {
            Self { v0: a, v1: b }
        } else {
            Self { v0: b, v1: a }
        }
    }
}

/// A candidate edge collapse with its cost and optimal position.
#[derive(Debug, Clone)]
struct CollapseCandidate {
    edge: Edge,
    cost: f32,
    optimal_pos: (f32, f32, f32),
    is_valid: bool,
}

// ─── Main simplification function ─────────────────────────────────

/// Simplify a mesh by collapsing edges using quadric error metrics.
///
/// # Arguments
/// * `mesh` - Input mesh
/// * `target_ratio` - Target ratio of faces to keep (0.0 - 1.0)
///
/// # Returns
/// Simplified mesh
///
/// # Algorithm
/// 1. Compute a quadric for each vertex by summing the quadrics of
///    its incident face planes.
/// 2. For each edge (pair of vertices sharing a face), compute the
///    optimal collapse position and the error cost.
/// 3. Sort edges by cost and iteratively collapse the lowest-cost edge,
///    updating affected edges.
pub fn simplify_mesh(mesh: &Mesh, target_ratio: f32) -> CoreResult<Mesh> {
    if target_ratio <= 0.0 || target_ratio > 1.0 {
        return Err(CoreError::InvalidInput(format!(
            "Target ratio must be in (0, 1], got {}",
            target_ratio
        )));
    }

    if mesh.faces.is_empty() || mesh.vertices.is_empty() {
        return Ok(mesh.clone());
    }

    let target_faces = (mesh.faces.len() as f32 * target_ratio) as usize;
    let target_faces = target_faces.max(1).min(mesh.faces.len());

    if target_faces >= mesh.faces.len() {
        return Ok(mesh.clone());
    }

    let num_collapses = mesh.faces.len() - target_faces;

    // Step 1: Compute per-vertex quadrics from incident face planes
    let n_vertices = mesh.vertices.len();
    let mut vertex_quadrics = vec![Quadric::zero(); n_vertices];

    for face in &mesh.faces {
        let v0 = &mesh.vertices[face.v0];
        let v1 = &mesh.vertices[face.v1];
        let v2 = &mesh.vertices[face.v2];

        // Compute face normal
        let e1x = v1.x - v0.x;
        let e1y = v1.y - v0.y;
        let e1z = v1.z - v0.z;
        let e2x = v2.x - v0.x;
        let e2y = v2.y - v0.y;
        let e2z = v2.z - v0.z;

        let nx = e1y * e2z - e1z * e2y;
        let ny = e1z * e2x - e1x * e2z;
        let nz = e1x * e2y - e1y * e2x;
        let len = (nx * nx + ny * ny + nz * nz).sqrt();

        if len < 1e-10 {
            continue; // Degenerate face
        }

        let nx = nx / len;
        let ny = ny / len;
        let nz = nz / len;

        // Plane constant d = -n·p for any point on the plane
        let d = -(nx * v0.x + ny * v0.y + nz * v0.z);

        let face_quadric = Quadric::from_plane(nx, ny, nz, d);

        // Add to each vertex's quadric
        vertex_quadrics[face.v0].add(&face_quadric);
        vertex_quadrics[face.v1].add(&face_quadric);
        vertex_quadrics[face.v2].add(&face_quadric);
    }

    // Step 2: Build edge set and compute initial collapse candidates
    let mut edge_map: HashMap<Edge, CollapseCandidate> = HashMap::new();

    for face in &mesh.faces {
        let edges = [
            Edge::new(face.v0, face.v1),
            Edge::new(face.v1, face.v2),
            Edge::new(face.v2, face.v0),
        ];

        for &edge in &edges {
            if !edge_map.contains_key(&edge) {
                let q = compute_collapse_q(&vertex_quadrics, edge);
                let cost = q.evaluate(
                    mesh.vertices[edge.v0].x,
                    mesh.vertices[edge.v0].y,
                    mesh.vertices[edge.v0].z,
                );

                let opt_pos = q.optimal_position().unwrap_or_else(|| {
                    // Fallback to midpoint
                    let mid_x = (mesh.vertices[edge.v0].x + mesh.vertices[edge.v1].x) * 0.5;
                    let mid_y = (mesh.vertices[edge.v0].y + mesh.vertices[edge.v1].y) * 0.5;
                    let mid_z = (mesh.vertices[edge.v0].z + mesh.vertices[edge.v1].z) * 0.5;
                    (mid_x, mid_y, mid_z)
                });

                edge_map.insert(
                    edge,
                    CollapseCandidate {
                        edge,
                        cost,
                        optimal_pos: opt_pos,
                        is_valid: true,
                    },
                );
            }
        }
    }

    // Step 3: Iteratively collapse edges with lowest cost
    let mut current_vertices = mesh.vertices.clone();
    let mut current_faces = mesh.faces.clone();
    let mut vertex_remap: Vec<usize> = (0..n_vertices).collect();
    let mut valid_vertex: Vec<bool> = vec![true; n_vertices];
    let mut collapses_done = 0;

    while collapses_done < num_collapses {
        // Find the valid edge with the lowest cost
        let best_edge = edge_map
            .values_mut()
            .filter(|c| c.is_valid)
            .min_by(|a, b| {
                a.cost
                    .partial_cmp(&b.cost)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|c| c.edge);

        let edge = match best_edge {
            Some(e) => e,
            None => break, // No more valid edges
        };

        // Get the collapse candidate
        let candidate = edge_map.get(&edge).unwrap().clone();
        if !candidate.is_valid {
            break;
        }

        // Mark edge as invalid (being collapsed)
        if let Some(c) = edge_map.get_mut(&edge) {
            c.is_valid = false;
        }

        // Collapse: move v1 to v0's position, remap v1 -> v0
        let v0 = candidate.edge.v0;
        let v1 = candidate.edge.v1;

        // Set v0 to optimal position
        current_vertices[v0] = Vertex {
            x: candidate.optimal_pos.0,
            y: candidate.optimal_pos.1,
            z: candidate.optimal_pos.2,
        };

        // Remap v1 to v0
        vertex_remap[v1] = v0;
        valid_vertex[v1] = false;

        // Update quadric: add v1's quadric to v0's
        // Use raw pointer approach to avoid borrow checker conflict
        let q_v1 = vertex_quadrics.get(v1).cloned().unwrap_or(Quadric::zero());
        if let Some(q_v0) = vertex_quadrics.get_mut(v0) {
            q_v0.add(&q_v1);
        }

        // Update faces: replace v1 with v0, remove degenerate faces
        let mut new_faces = Vec::with_capacity(current_faces.len());
        for face in &current_faces {
            let mut f = *face;
            // Remap indices
            if f.v0 == v1 {
                f.v0 = v0;
            }
            if f.v1 == v1 {
                f.v1 = v0;
            }
            if f.v2 == v1 {
                f.v2 = v0;
            }
            // Skip degenerate faces (two vertices the same)
            if f.v0 == f.v1 || f.v1 == f.v2 || f.v2 == f.v0 {
                continue;
            }
            new_faces.push(f);
        }
        current_faces = new_faces;

        // Update edges: invalidate edges that involve v1
        // and update edges that involve v0
        let edges_to_update: Vec<Edge> = edge_map
            .keys()
            .filter(|e| e.v0 == v0 || e.v1 == v0 || e.v0 == v1 || e.v1 == v1)
            .copied()
            .collect();

        for old_edge in edges_to_update {
            // Remove old edge
            edge_map.remove(&old_edge);

            // Create new edge (remap v1 -> v0)
            let new_edge = Edge::new(
                if old_edge.v0 == v1 { v0 } else { old_edge.v0 },
                if old_edge.v1 == v1 { v0 } else { old_edge.v1 },
            );

            if new_edge.v0 == new_edge.v1 {
                continue; // Self-loop, skip
            }

            if !edge_map.contains_key(&new_edge) {
                // Compute new cost
                let v0_pos = &current_vertices[new_edge.v0];
                let v1_pos = &current_vertices[new_edge.v1];
                let q = compute_collapse_q(&vertex_quadrics, new_edge);
                let cost = q.evaluate(v0_pos.x, v0_pos.y, v0_pos.z);
                let opt_pos = q.optimal_position().unwrap_or_else(|| {
                    (
                        (v0_pos.x + v1_pos.x) * 0.5,
                        (v0_pos.y + v1_pos.y) * 0.5,
                        (v0_pos.z + v1_pos.z) * 0.5,
                    )
                });

                edge_map.insert(
                    new_edge,
                    CollapseCandidate {
                        edge: new_edge,
                        cost,
                        optimal_pos: opt_pos,
                        is_valid: true,
                    },
                );
            }
        }

        collapses_done += 1;

        // Safety check to prevent infinite loops
        if collapses_done > num_collapses * 2 {
            break;
        }
    }

    // Step 4: Build output mesh
    // Build a mapping from old vertex indices to new contiguous indices
    let mut old_to_new: Vec<Option<usize>> = vec![None; current_vertices.len()];
    let mut new_vertices = Vec::new();
    for (old_idx, &valid) in valid_vertex.iter().enumerate() {
        if valid {
            old_to_new[old_idx] = Some(new_vertices.len());
            new_vertices.push(current_vertices[old_idx]);
        }
    }

    let new_faces: Vec<Face> = current_faces
        .iter()
        .filter_map(|f| {
            let v0 = old_to_new[f.v0]?;
            let v1 = old_to_new[f.v1]?;
            let v2 = old_to_new[f.v2]?;
            Some(Face { v0, v1, v2 })
        })
        .collect();

    Ok(Mesh {
        vertices: new_vertices,
        faces: new_faces,
    })
}

/// Compute the combined quadric for an edge collapse (sum of both vertex quadrics).
fn compute_collapse_q(vertex_quadrics: &[Quadric], edge: Edge) -> Quadric {
    let mut q = vertex_quadrics[edge.v0];
    q.add(&vertex_quadrics[edge.v1]);
    q
}

// ─── Format conversion ───────────────────────────────────────────

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
                    let x = parts[1]
                        .parse::<f32>()
                        .map_err(|e| CoreError::InvalidInput(format!("Invalid vertex x: {}", e)))?;
                    let y = parts[2]
                        .parse::<f32>()
                        .map_err(|e| CoreError::InvalidInput(format!("Invalid vertex y: {}", e)))?;
                    let z = parts[3]
                        .parse::<f32>()
                        .map_err(|e| CoreError::InvalidInput(format!("Invalid vertex z: {}", e)))?;
                    mesh.vertices.push(Vertex { x, y, z });
                }
            }
            "f" => {
                if parts.len() >= 4 {
                    // Parse face indices (handle v/vt/vn format)
                    let parse_idx = |s: &str| -> CoreResult<usize> {
                        let idx_str = s.split('/').next().unwrap_or(s);
                        let idx: i32 = idx_str.parse().map_err(|e| {
                            CoreError::InvalidInput(format!("Invalid face index: {}", e))
                        })?;
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
        mesh.vertices.push(Vertex {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        });
        mesh.vertices.push(Vertex {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        });
        mesh.faces.push(Face {
            v0: 0,
            v1: 1,
            v2: 0,
        });

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
        mesh.vertices.push(Vertex {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        });
        mesh.vertices.push(Vertex {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        });
        mesh.vertices.push(Vertex {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        });
        mesh.faces.push(Face {
            v0: 0,
            v1: 1,
            v2: 2,
        });

        let ply = mesh_to_ply(&mesh);
        assert!(ply.contains("element vertex 3"));
        assert!(ply.contains("element face 1"));
    }

    #[test]
    fn test_simplify_simple_mesh() {
        // Create a simple mesh: 4 vertices forming 2 triangles (a quad)
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        });
        mesh.vertices.push(Vertex {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        });
        mesh.vertices.push(Vertex {
            x: 1.0,
            y: 1.0,
            z: 0.0,
        });
        mesh.vertices.push(Vertex {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        });
        mesh.faces.push(Face {
            v0: 0,
            v1: 1,
            v2: 2,
        });
        mesh.faces.push(Face {
            v0: 0,
            v1: 2,
            v2: 3,
        });

        let simplified = simplify_mesh(&mesh, 0.5).unwrap();
        assert!(
            simplified.faces.len() <= mesh.faces.len(),
            "Simplified mesh should have fewer or equal faces"
        );
        assert!(simplified.faces.len() >= 1, "Should keep at least one face");
    }

    #[test]
    fn test_simplify_no_change() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        });
        mesh.vertices.push(Vertex {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        });
        mesh.vertices.push(Vertex {
            x: 1.0,
            y: 1.0,
            z: 0.0,
        });
        mesh.faces.push(Face {
            v0: 0,
            v1: 1,
            v2: 2,
        });

        // Ratio 1.0 should return the same mesh
        let simplified = simplify_mesh(&mesh, 1.0).unwrap();
        assert_eq!(simplified.faces.len(), mesh.faces.len());
    }

    #[test]
    fn test_simplify_invalid_ratio() {
        let mesh = Mesh::new();
        assert!(simplify_mesh(&mesh, 0.0).is_err());
        assert!(simplify_mesh(&mesh, 1.5).is_err());
    }

    #[test]
    fn test_empty_mesh() {
        let mesh = Mesh::new();
        let simplified = simplify_mesh(&mesh, 0.5).unwrap();
        assert_eq!(simplified.faces.len(), 0);
    }
}
