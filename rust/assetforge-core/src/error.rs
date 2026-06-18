//! Error types for the AssetForge core library.

use thiserror::Error;

/// Result type alias for core operations.
pub type CoreResult<T> = Result<T, CoreError>;

/// Errors that can occur in core operations.
#[derive(Error, Debug)]
pub enum CoreError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image processing error: {0}")]
    Image(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Vectorization error: {0}")]
    Vectorization(String),

    #[error("Mesh error: {0}")]
    Mesh(String),

    #[error("Compression error: {0}")]
    Compression(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl From<image::ImageError> for CoreError {
    fn from(err: image::ImageError) -> Self {
        CoreError::Image(err.to_string())
    }
}
