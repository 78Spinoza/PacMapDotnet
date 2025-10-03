//! Deterministic Enhanced PaCMAP Core Implementation
//!
//! This module provides a fully deterministic implementation of the PaCMAP
//! (Pairwise Controlled Manifold Approximation) algorithm with enhanced
//! progress reporting capabilities.
//!
//! ## Key Features
//!
//! - **Deterministic Behavior**: All operations use seeded random number generation
//!   and deterministic algorithms to ensure reproducible results
//! - **Enhanced Progress Reporting**: Detailed callbacks for monitoring algorithm progress
//! - **Improved Error Handling**: Comprehensive validation and error reporting
//! - **Performance Optimized**: SIMD-accelerated distance calculations and parallel processing
//! - **Modular Design**: Clean separation of concerns with individual modules for each component
//!
//! ## Modules
//!
//! - [`adam`]: Adam optimizer with enhanced progress reporting
//! - [`distance`]: SIMD-accelerated distance calculations
//! - [`gradient`]: Gradient computation for PaCMAP loss function
//! - [`knn`]: K-nearest neighbor computation (deterministic exact algorithm)
//! - [`neighbors`]: Point pair generation for optimization
//! - [`sampling`]: Deterministic sampling of various pair types
//! - [`weights`]: Weight management for optimization phases
//!
//! ## Usage
//!
//! ```rust
//! use pacman_core::*;
//!
//! // Create deterministic configuration
//! let config = PacmapConfig::new()
//!     .with_seed(42)
//!     .with_progress_callback(|stage, current, total, percentage, details| {
//!         println!("Progress: {} - {}/{} ({:.1}%) - {}", stage, current, total, percentage, details);
//!     });
//!
//! // Run deterministic PaCMAP with progress reporting
//! let (embedding, model) = fit_transform_deterministic(data, &config)?;
//! ```

pub mod adam;
pub mod distance;
pub mod gradient;
pub mod knn;
pub mod neighbors;
pub mod sampling;
pub mod weights;

// Re-export key types and functions for easier access
pub use adam::{AdamConfig, AdamStats};
pub use distance::{array_euclidean_distance, simd_euclidean_distance};
pub use gradient::{compute_gradient_stats, GradientConfig, GradientStats};
pub use knn::{find_k_nearest_neighbors, find_k_nearest_neighbors_with_progress, KnnConfig};
pub use neighbors::{generate_pairs, generate_pairs_with_progress, PairConfig, PairStats};
pub use sampling::{sample_fp_pair_deterministic, sample_mn_pair_deterministic, SamplingConfig};
pub use weights::{find_weights, find_weights_with_config, PhaseInfo, WeightConfig, Weights};

use ndarray::{Array1, Array2, ArrayView2};
use std::error::Error;
use std::fmt;

/// Progress callback type for PaCMAP operations
pub type ProgressCallback = Box<dyn Fn(&str, usize, usize, f32, &str) + Send + Sync>;

/// Configuration for PaCMAP algorithm (compatible with external pacmap crate)
#[derive(Debug, Clone)]
pub struct Configuration {
    /// Number of dimensions for the output embedding
    pub embedding_dimensions: usize,
    /// Number of iterations for each optimization phase
    pub num_iters: (usize, usize, usize),
    /// Whether to use random initialization
    pub random_state: Option<u64>,
    /// Number of neighbors for pair generation
    pub override_neighbors: Option<usize>,
    /// Learning rate for optimization
    pub learning_rate: f32,
    /// Ratio for mid-near pair sampling
    pub mid_near_ratio: f32,
    /// Ratio for far pair sampling
    pub far_pair_ratio: f32,
    /// Pair generation configuration
    pub pair_configuration: PairConfiguration,
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            embedding_dimensions: 2,
            num_iters: (100, 100, 250),
            random_state: Some(42),
            override_neighbors: None,
            learning_rate: 1.0,
            mid_near_ratio: 0.1,
            far_pair_ratio: 0.05,
            pair_configuration: PairConfiguration::default(),
        }
    }
}

/// Configuration for pair generation
#[derive(Debug, Clone)]
pub enum PairConfiguration {
    /// Use exact KNN computation
    Default,
    /// Use provided neighbor pairs
    NeighborsProvided { pair_neighbors: ndarray::Array2<u32> },
}

impl Default for PairConfiguration {
    fn default() -> Self {
        PairConfiguration::Default
    }
}

/// Main configuration for deterministic PaCMAP with enhanced progress reporting
#[derive(Debug)]
pub struct PacmapConfig {
    /// Number of dimensions for the output embedding
    pub n_dims: usize,
    /// Number of nearest neighbors to consider
    pub n_neighbors: usize,
    /// Number of mid-near pairs per point
    pub n_mn: usize,
    /// Number of far pairs per point
    pub n_fp: usize,
    /// Random seed for deterministic behavior
    pub seed: Option<u64>,
    /// Number of optimization iterations
    pub n_iters: usize,
    /// Pair generation configuration
    pub pair_config: PairConfig,
    /// KNN computation configuration
    pub knn_config: KnnConfig,
    /// Gradient computation configuration
    pub gradient_config: GradientConfig,
    /// Adam optimizer configuration
    pub adam_config: adam::AdamConfig,
    /// Whether to report progress
    pub report_progress: bool,
    /// Progress callback function
    pub progress_callback: Option<ProgressCallback>,
}

impl Default for PacmapConfig {
    fn default() -> Self {
        Self {
            n_dims: 2,
            n_neighbors: 15,
            n_mn: 5,
            n_fp: 10,
            seed: Some(42),
            n_iters: 500,
            pair_config: PairConfig::default(),
            knn_config: KnnConfig::default(),
            gradient_config: GradientConfig::default(),
            adam_config: adam::AdamConfig::default(),
            report_progress: false,
            progress_callback: None,
        }
    }
}

impl PacmapConfig {
    /// Create new PaCMAP configuration with default parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the output embedding dimensions
    pub fn with_dims(mut self, n_dims: usize) -> Self {
        self.n_dims = n_dims;
        self
    }

    /// Set the number of nearest neighbors
    pub fn with_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self.pair_config.n_neighbors = n_neighbors;
        self.knn_config.k = n_neighbors;
        self
    }

    /// Set optimization parameters
    pub fn with_optimization(mut self, n_mn: usize, n_fp: usize, n_iters: usize) -> Self {
        self.n_mn = n_mn;
        self.n_fp = n_fp;
        self.n_iters = n_iters;
        self.pair_config.n_mn = n_mn;
        self.pair_config.n_fp = n_fp;
        self
    }

    /// Set random seed for deterministic behavior
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self.pair_config.random_state = Some(seed);
        self.pair_config.sampling_config.random_state = seed;
        self
    }

    /// Enable progress reporting with callback
    pub fn with_progress_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&str, usize, usize, f32, &str) + Send + Sync + 'static,
    {
        self.report_progress = true;
        self.progress_callback = Some(Box::new(callback));

        // Note: Callback propagation simplified to avoid cloning issues
        // Individual sub-configurations handle their own callback management

        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.n_dims == 0 {
            return Err("Number of dimensions must be positive".to_string());
        }
        if self.n_neighbors == 0 {
            return Err("Number of neighbors must be positive".to_string());
        }
        if self.n_iters == 0 {
            return Err("Number of iterations must be positive".to_string());
        }

        self.pair_config.validate()?;
        self.knn_config.validate()?;
        self.gradient_config.validate()?;
        self.adam_config.validate()?;

        Ok(())
    }
}

/// Errors that can occur during PaCMAP computation
#[derive(Debug)]
pub enum PacmapError {
    /// Invalid configuration
    InvalidConfiguration(String),
    /// Input data validation error
    InvalidInput(String),
    /// KNN computation error
    KnnError(String),
    /// Gradient computation error
    GradientError(String),
    /// Optimization error
    OptimizationError(String),
}

impl fmt::Display for PacmapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PacmapError::InvalidConfiguration(msg) => write!(f, "Invalid configuration: {}", msg),
            PacmapError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            PacmapError::KnnError(msg) => write!(f, "KNN error: {}", msg),
            PacmapError::GradientError(msg) => write!(f, "Gradient error: {}", msg),
            PacmapError::OptimizationError(msg) => write!(f, "Optimization error: {}", msg),
        }
    }
}

impl Error for PacmapError {}

/// Results from PaCMAP fitting
#[derive(Debug)]
pub struct PacmapResult {
    /// The low-dimensional embedding
    pub embedding: Array2<f32>,
    /// Final optimization statistics
    pub final_stats: OptimizationStats,
}

/// Statistics from the optimization process
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    /// Number of iterations completed
    pub iterations_completed: usize,
    /// Final loss value
    pub final_loss: f32,
    /// Final gradient norm
    pub final_gradient_norm: f32,
    /// Final parameter norm
    pub final_parameter_norm: f32,
    /// Total number of pairs used
    pub total_pairs: usize,
}

impl std::fmt::Display for OptimizationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "OptimizationStats {{ iterations: {}, loss: {:.6}, |grad|: {:.6}, |param|: {:.6}, pairs: {} }}",
            self.iterations_completed, self.final_loss, self.final_gradient_norm, self.final_parameter_norm, self.total_pairs
        )
    }
}

/// Deterministic PaCMAP fitting with enhanced progress reporting
///
/// # Arguments
/// * `x` - Input data matrix of shape (n_samples, n_features)
/// * `config` - PaCMAP configuration
///
/// # Returns
/// Result containing the embedding and optimization statistics
///
/// # Errors
/// Returns error if configuration is invalid or computation fails
pub fn fit_transform_deterministic(
    x: ArrayView2<f32>,
    config: &PacmapConfig,
) -> Result<PacmapResult, PacmapError> {
    // Validate configuration
    config.validate().map_err(PacmapError::InvalidConfiguration)?;

    let (n_samples, n_features) = x.dim();
    if n_samples == 0 {
        return Err(PacmapError::InvalidInput("Input data is empty".to_string()));
    }

    report_progress(
        &config.progress_callback,
        "PaCMAP",
        0,
        config.n_iters + 3, // 3 main stages + optimization iterations
        0.0,
        &format!("Starting deterministic PaCMAP: {} samples, {} features -> {} dimensions",
                n_samples, n_features, config.n_dims),
    );

    // Stage 1: Generate pairs
    report_progress(
        &config.progress_callback,
        "PaCMAP",
        1,
        config.n_iters + 3,
        10.0,
        "Generating point pairs",
    );

    let pairs = generate_pairs_with_progress(x, &config.pair_config)
        .map_err(|e| PacmapError::KnnError(e))?;

    // Stage 2: Initialize embedding
    report_progress(
        &config.progress_callback,
        "PaCMAP",
        2,
        config.n_iters + 3,
        20.0,
        "Initializing embedding",
    );

    let mut y = initialize_embedding(x.nrows(), config.n_dims, config.seed.unwrap_or(42));

    // Initialize Adam state
    let (mut m, mut v) = adam::create_adam_state_random((n_samples, config.n_dims), config.seed.unwrap_or(42));

    // Stage 3: Optimization
    report_progress(
        &config.progress_callback,
        "PaCMAP",
        3,
        config.n_iters + 3,
        30.0,
        &format!("Starting optimization for {} iterations", config.n_iters),
    );

    let mut final_stats = OptimizationStats {
        iterations_completed: 0,
        final_loss: 0.0,
        final_gradient_norm: 0.0,
        final_parameter_norm: 0.0,
        total_pairs: pairs.pair_neighbors.nrows() + pairs.pair_mn.nrows() + pairs.pair_fp.nrows(),
    };

    // Optimization loop
    for itr in 0..config.n_iters {
        // Compute weights for this iteration
        let phase_1_iters = config.n_iters / 3;
        let phase_2_iters = config.n_iters / 3;
        let weight_config = WeightConfig::new(2.0, phase_1_iters, phase_2_iters);
        let weights = find_weights_with_config(&weight_config, itr);

        // Compute gradient
        let grad = gradient::pacmap_grad_with_progress(
            y.view(),
            pairs.pair_neighbors.view(),
            pairs.pair_mn.view(),
            pairs.pair_fp.view(),
            &weights,
            &config.gradient_config,
        );

        // Extract loss from gradient
        let loss = grad[[n_samples, 0]];
        let grad_view = grad.slice(s![..n_samples, ..]);

        // Update embedding with Adam
        let adam_stats = adam::update_embedding_adam_with_stats(
            y.view_mut(),
            grad_view,
            m.view_mut(),
            v.view_mut(),
            &config.adam_config,
            itr,
        ).map_err(|e| PacmapError::OptimizationError(e))?;

        // Update final stats
        final_stats.iterations_completed = itr + 1;
        final_stats.final_loss = loss;
        final_stats.final_gradient_norm = adam_stats.gradient_norm;
        final_stats.final_parameter_norm = adam_stats.parameter_norm;

        // Report progress
        if config.report_progress && (itr % 50 == 0 || itr == config.n_iters - 1) {
            let percentage = 30.0 + ((itr + 1) as f32 / config.n_iters as f32) * 70.0;
            report_progress(
                &config.progress_callback,
                "PaCMAP Optimization",
                itr + 1,
                config.n_iters,
                percentage,
                &format!("Iteration {}: loss={:.6}, |grad|={:.6}", itr + 1, loss, adam_stats.gradient_norm),
            );
        }
    }

    report_progress(
        &config.progress_callback,
        "PaCMAP",
        config.n_iters + 3,
        config.n_iters + 3,
        100.0,
        &format!("Completed PaCMAP: final loss {:.6}", final_stats.final_loss),
    );

    Ok(PacmapResult {
        embedding: y,
        final_stats,
    })
}

/// Initialize embedding with small random values
fn initialize_embedding(n_samples: usize, n_dims: usize, seed: u64) -> Array2<f32> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);

    Array2::from_shape_fn((n_samples, n_dims), |_| {
        rng.gen::<f32>() * 1e-4 - 5e-5 // Small random values around zero
    })
}

/// Reports progress safely with error handling
fn report_progress(
    callback: &Option<ProgressCallback>,
    stage: &str,
    current: usize,
    total: usize,
    percentage: f32,
    details: &str,
) {
    if let Some(ref cb) = callback {
        cb(stage, current, total, percentage, details);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_pacmap_config() {
        let config = PacmapConfig::default();
        assert_eq!(config.n_dims, 2);
        assert_eq!(config.n_neighbors, 15);
        assert_eq!(config.seed, Some(42));
        assert!(!config.report_progress);

        let custom_config = PacmapConfig::new()
            .with_dims(3)
            .with_neighbors(10)
            .with_optimization(3, 5, 200)
            .with_seed(123);

        assert_eq!(custom_config.n_dims, 3);
        assert_eq!(custom_config.n_neighbors, 10);
        assert_eq!(custom_config.n_mn, 3);
        assert_eq!(custom_config.n_fp, 5);
        assert_eq!(custom_config.n_iters, 200);
        assert_eq!(custom_config.seed, Some(123));

        // Test validation
        assert!(custom_config.validate().is_ok());

        let invalid_config = PacmapConfig::new().with_dims(0);
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_fit_transform_deterministic() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let config = PacmapConfig::new()
            .with_dims(2)
            .with_neighbors(2)
            .with_optimization(1, 1, 10) // Small number for testing
            .with_seed(42);

        let result = fit_transform_deterministic(x.view(), &config);
        assert!(result.is_ok());

        let pacmap_result = result.unwrap();
        assert_eq!(pacmap_result.embedding.shape(), &[4, 2]);
        assert_eq!(pacmap_result.final_stats.iterations_completed, 10);
        assert!(pacmap_result.final_stats.final_loss.is_finite());
    }

    #[test]
    fn test_progress_callback() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let progress_calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let progress_calls_clone = progress_calls.clone();

        let config = PacmapConfig::new()
            .with_dims(2)
            .with_neighbors(1)
            .with_optimization(1, 1, 5) // Small number for testing
            .with_seed(42)
            .with_progress_callback(move |stage, current, total, percentage, details| {
                let mut calls = progress_calls_clone.lock().unwrap();
                calls.push((stage.to_string(), current, total, percentage, details.to_string()));
            });

        let result = fit_transform_deterministic(x.view(), &config);
        assert!(result.is_ok());

        let calls = progress_calls.lock().unwrap();
        assert!(!calls.is_empty());
        assert!(calls.iter().any(|(stage, _, _, _, _)| stage.contains("PaCMAP")));
    }

    #[test]
    fn test_deterministic_behavior() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let config = PacmapConfig::new()
            .with_dims(2)
            .with_neighbors(2)
            .with_optimization(1, 1, 10)
            .with_seed(42);

        let result1 = fit_transform_deterministic(x.view(), &config).unwrap();
        let result2 = fit_transform_deterministic(x.view(), &config).unwrap();

        // Results should be identical
        assert_eq!(result1.embedding, result2.embedding);
        assert_eq!(result1.final_stats.final_loss, result2.final_stats.final_loss);
    }

    #[test]
    fn test_optimization_stats_display() {
        let stats = OptimizationStats {
            iterations_completed: 100,
            final_loss: 0.123456,
            final_gradient_norm: 0.789012,
            final_parameter_norm: 3.456789,
            total_pairs: 150,
        };

        let display_str = format!("{}", stats);
        assert!(display_str.contains("OptimizationStats"));
        assert!(display_str.contains("iterations: 100"));
        assert!(display_str.contains("loss: 0.123456"));
        assert!(display_str.contains("|grad|: 0.789012"));
        assert!(display_str.contains("|param|: 3.456789"));
        assert!(display_str.contains("pairs: 150"));
    }

    #[test]
    fn test_error_handling() {
        // Empty input
        let empty_x = array![];
        let config = PacmapConfig::default();
        let result = fit_transform_deterministic(empty_x.view(), &config);
        assert!(result.is_err());

        // Invalid configuration
        let x = array![[1.0, 2.0]];
        let invalid_config = PacmapConfig::new().with_dims(0);
        let result = fit_transform_deterministic(x.view(), &invalid_config);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_display() {
        let error = PacmapError::InvalidConfiguration("Test error".to_string());
        let display_str = format!("{}", error);
        assert!(display_str.contains("Invalid configuration"));
        assert!(display_str.contains("Test error"));
    }
}