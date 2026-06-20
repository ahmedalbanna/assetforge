//! # AssetForge Core
//!
//! High-performance Rust library for:
//! - Image processing (resize, crop, normalize)
//! - Bitmap vectorization (Potrace-style)
//! - Mesh simplification and format conversion
//! - Gaussian splat compression (SPZ)
//! - GLTF/OBJ export helpers

/// The version of the assetforge-core crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod error;
pub mod image_ops;
pub mod mesh_utils;
pub mod splat_compress;
pub mod vectorize;

pub use error::{CoreError, CoreResult};
pub use image_ops::*;
pub use mesh_utils::*;
pub use splat_compress::*;
pub use vectorize::*;
