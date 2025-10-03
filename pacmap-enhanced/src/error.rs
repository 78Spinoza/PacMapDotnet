//! Centralized error handling for PacMAP Enhanced
//!
//! This module provides unified error types and better error handling
//! throughout the PacMAP Enhanced library to improve debugging and
//! user experience.

use std::fmt;

/// Main error type for PacMAP Enhanced operations
#[derive(Debug, Clone)]
pub enum PacMapError {
    /// Error during data validation or preprocessing
    DataValidation(String),

    /// Error during HNSW index construction or search
    HnswError(String),

    /// Error during model serialization/deserialization
    Serialization(String),

    /// Error during normalization operations
    Normalization(String),

    /// Memory allocation or management error
    Memory(String),

    /// Configuration error (invalid parameters, etc.)
    Configuration(String),

    /// FFI (Foreign Function Interface) related error
    Ffi(String),

    /// Input/Output error (file operations, etc.)
    Io(String),

    /// General/unknown error
    General(String),
}

impl fmt::Display for PacMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PacMapError::DataValidation(msg) => write!(f, "Data validation error: {}", msg),
            PacMapError::HnswError(msg) => write!(f, "HNSW error: {}", msg),
            PacMapError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            PacMapError::Normalization(msg) => write!(f, "Normalization error: {}", msg),
            PacMapError::Memory(msg) => write!(f, "Memory error: {}", msg),
            PacMapError::Configuration(msg) => write!(f, "Configuration error: {}", msg),
            PacMapError::Ffi(msg) => write!(f, "FFI error: {}", msg),
            PacMapError::Io(msg) => write!(f, "I/O error: {}", msg),
            PacMapError::General(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for PacMapError {}

/// Convenient Result type alias for PacMAP operations
pub type PacMapResult<T> = Result<T, PacMapError>;

/// Error conversion helpers for common error types
impl From<std::io::Error> for PacMapError {
    fn from(err: std::io::Error) -> Self {
        PacMapError::Io(err.to_string())
    }
}


impl From<rmp_serde::encode::Error> for PacMapError {
    fn from(err: rmp_serde::encode::Error) -> Self {
        PacMapError::Serialization(format!("MessagePack serialization error: {}", err))
    }
}

impl From<rmp_serde::decode::Error> for PacMapError {
    fn from(err: rmp_serde::decode::Error) -> Self {
        PacMapError::Serialization(format!("MessagePack deserialization error: {}", err))
    }
}


/// Helper macros for creating consistent error messages
#[macro_export]
macro_rules! pacmap_error {
    ($variant:ident, $msg:expr) => {
        $crate::error::PacMapError::$variant($msg.to_string())
    };
    ($variant:ident, $fmt:expr, $($arg:tt)*) => {
        $crate::error::PacMapError::$variant(format!($fmt, $($arg)*))
    };
}

#[macro_export]
macro_rules! pacmap_result {
    ($expr:expr, $variant:ident, $msg:expr) => {
        $expr.map_err(|e| $crate::error::PacMapError::$variant(format!("{}: {}", $msg, e)))
    };
    ($expr:expr, $variant:ident, $fmt:expr, $($arg:tt)*) => {
        $expr.map_err(|e| $crate::error::PacMapError::$variant(format!($fmt, $($arg)*, e)))
    };
}

/// Safe unwrap with a helpful error message
#[macro_export]
macro_rules! safe_unwrap {
    ($expr:expr, $variant:ident, $msg:expr) => {
        $expr.unwrap_or_else(|e| {
            panic!("{}: {}", $msg, e)
        })
    };
    ($expr:expr, $variant:ident, $fmt:expr, $($arg:tt)*) => {
        $expr.unwrap_or_else(|e| {
            panic!("{}: {}", format!($fmt, $($arg)*), e)
        })
    };
}

/// Validation helpers for common checks
pub fn validate_data_shape(data: &ndarray::ArrayView2<f64>, min_samples: usize, min_features: usize) -> PacMapResult<()> {
    let (n_samples, n_features) = data.dim();

    if n_samples < min_samples {
        return Err(PacMapError::DataValidation(
            format!("Insufficient samples: {} (minimum {})", n_samples, min_samples)
        ));
    }

    if n_features < min_features {
        return Err(PacMapError::DataValidation(
            format!("Insufficient features: {} (minimum {})", n_features, min_features)
        ));
    }

    // Check for NaN or infinite values
    for (i, &val) in data.iter().enumerate() {
        if !val.is_finite() {
            return Err(PacMapError::DataValidation(
                format!("Non-finite value found at position {}: {}", i, val)
            ));
        }
    }

    Ok(())
}

pub fn validate_neighbors(n_samples: usize, n_neighbors: usize) -> PacMapResult<()> {
    if n_neighbors == 0 {
        return Err(PacMapError::Configuration(
            "Number of neighbors must be greater than 0".to_string()
        ));
    }

    if n_neighbors >= n_samples {
        return Err(PacMapError::Configuration(
            format!("Number of neighbors ({}) must be less than number of samples ({})",
                   n_neighbors, n_samples)
        ));
    }

    Ok(())
}

pub fn validate_hnsw_params(m: usize, ef_construction: usize, ef_search: usize) -> PacMapResult<()> {
    if m < 4 || m > 64 {
        return Err(PacMapError::Configuration(
            format!("HNSW M parameter ({}) must be between 4 and 64", m)
        ));
    }

    if ef_construction < m || ef_construction > 1000 {
        return Err(PacMapError::Configuration(
            format!("HNSW ef_construction ({}) must be between M ({}) and 1000", ef_construction, m)
        ));
    }

    if ef_search < m || ef_search > 1000 {
        return Err(PacMapError::Configuration(
            format!("HNSW ef_search ({}) must be between M ({}) and 1000", ef_search, m)
        ));
    }

    Ok(())
}