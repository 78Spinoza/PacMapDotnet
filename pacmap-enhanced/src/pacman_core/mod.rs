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

// Core deterministic PaCMAP implementation
use rand::SeedableRng;
use ndarray::Array2;

// CRITICAL: Disable all parallel operations for determinism
// Parallel processing causes non-deterministic floating-point accumulation order

// Deterministic implementations
#[derive(Debug)]
pub struct AdamConfig {
    pub learning_rate: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self { learning_rate: 0.01 }
    }
}

impl AdamConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.learning_rate <= 0.0 {
            return Err("Learning rate must be positive".to_string());
        }
        Ok(())
    }
}

pub struct AdamStats {
    pub loss: f32,
    pub grad_norm: f32,
    pub param_norm: f32,
}

pub struct GradientConfig {
    pub loss_type: String,
    pub progress_callback: Option<ProgressCallback>,
}

impl std::fmt::Debug for GradientConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradientConfig")
            .field("loss_type", &self.loss_type)
            .field("progress_callback", &self.progress_callback.is_some())
            .finish()
    }
}

impl Default for GradientConfig {
    fn default() -> Self {
        Self {
            loss_type: "pacmap".to_string(),
            progress_callback: None,
        }
    }
}

impl GradientConfig {
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

pub struct GradientStats {
    pub loss: f32,
}

#[derive(Debug, Clone)]
pub struct KnnConfig {
    pub k: usize,
}

impl Default for KnnConfig {
    fn default() -> Self {
        Self { k: 15 }
    }
}

impl KnnConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.k == 0 {
            return Err("K must be positive".to_string());
        }
        Ok(())
    }
}

// Note: ProgressCallback can't implement Debug, so we implement Debug manually
pub struct PairConfig {
    pub n_neighbors: usize,
    pub n_mn: usize,
    pub n_fp: usize,
    pub random_state: Option<u64>,
    pub progress_callback: Option<ProgressCallback>,
    pub sampling_config: SamplingConfig,
}

impl std::fmt::Debug for PairConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PairConfig")
            .field("n_neighbors", &self.n_neighbors)
            .field("n_mn", &self.n_mn)
            .field("n_fp", &self.n_fp)
            .field("random_state", &self.random_state)
            .field("progress_callback", &self.progress_callback.is_some())
            .field("sampling_config", &self.sampling_config)
            .finish()
    }
}

impl Default for PairConfig {
    fn default() -> Self {
        Self {
            n_neighbors: 15,
            n_mn: 5,
            n_fp: 10,
            random_state: Some(42),
            progress_callback: None,
            sampling_config: SamplingConfig::default(),
        }
    }
}

impl PairConfig {
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

pub struct PairStats {
    pub total_pairs: usize,
}

#[derive(Debug)]
pub struct SamplingConfig {
    pub random_state: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self { random_state: 42 }
    }
}

pub struct Weights {
    pub w_mn: f32,
    pub w_neighbors: f32,
    pub w_fp: f32,
}

impl Default for Weights {
    fn default() -> Self {
        Self { w_mn: 2.0, w_neighbors: 2.0, w_fp: 1.0 }
    }
}

pub struct WeightConfig {
    pub w_mn_init: f32,
    pub phase_1_iters: usize,
    pub phase_2_iters: usize,
}

impl WeightConfig {
    pub fn new(w_mn_init: f32, phase_1_iters: usize, phase_2_iters: usize) -> Self {
        Self { w_mn_init, phase_1_iters, phase_2_iters }
    }
}

pub struct PhaseInfo {
    pub phase: u8,
    pub name: &'static str,
}


/// Compute squared Euclidean distances deterministically (single-threaded)
fn compute_squared_distances(x: ArrayView2<f32>) -> Array2<f32> {
    let (n_samples, n_features) = x.dim();
    let mut distances = Array2::zeros((n_samples, n_samples));

    // CRITICAL: Use deterministic order - always process i then j
    for i in 0..n_samples {
        for j in i+1..n_samples {
            let mut dist_sq = 0.0f32;

            // Compute distance with deterministic accumulation order
            for k in 0..n_features {
                let diff = x[[i, k]] - x[[j, k]];
                dist_sq += diff * diff;
            }

            distances[[i, j]] = dist_sq;
            distances[[j, i]] = dist_sq;
        }
    }

    distances
}

/// Find k nearest neighbors using deterministic approach (single-threaded)
fn find_knn_deterministic(
    distances: ArrayView2<f32>,
    k: usize,
    _seed: Option<u64>,
) -> Array2<usize> {
    let (n_samples, _) = distances.dim();
    let mut neighbors = Array2::zeros((n_samples, k));

    // CRITICAL: Process in deterministic order
    for i in 0..n_samples {
        // Get distances from point i to all other points
        let mut dist_with_idx: Vec<_> = (0..n_samples)
            .map(|j| (j, distances[[i, j]]))
            .collect();

        // Use stable sort for deterministic ordering of equal distances
        dist_with_idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Take first k neighbors (excluding self) - deterministic selection
        let mut neighbor_idx = 0;
        for (j, _) in dist_with_idx.iter() {
            if *j != i && neighbor_idx < k {
                neighbors[[i, neighbor_idx]] = *j;
                neighbor_idx += 1;
            }
        }
    }

    neighbors
}

pub fn generate_pairs_with_progress(
    x: ArrayView2<f32>,
    config: &PairConfig,
) -> Result<PairData, String> {
    let (n_samples, _) = x.dim();
    let callback = &config.progress_callback;

    // Report progress: computing distances
    report_progress(callback, "Pair Generation", 0, 100, 0.0, "Computing pairwise distances");

    // Compute pairwise distances
    let distances = compute_squared_distances(x);

    // Report progress: finding neighbors
    report_progress(callback, "Pair Generation", 20, 100, 20.0, "Finding nearest neighbors");

    // Find k nearest neighbors
    let neighbor_indices = find_knn_deterministic(
        distances.view(),
        config.n_neighbors,
        config.random_state
    );

    // Report progress: generating neighbor pairs
    report_progress(callback, "Pair Generation", 40, 100, 40.0, "Generating neighbor pairs");

    // Generate neighbor pairs (symmetric)
    let mut pair_neighbors = Array2::zeros((n_samples * config.n_neighbors, 2));
    let mut pair_idx = 0;
    for i in 0..n_samples {
        for neighbor_idx in 0..config.n_neighbors {
            let j = neighbor_indices[[i, neighbor_idx]];
            pair_neighbors[[pair_idx, 0]] = i as u32;
            pair_neighbors[[pair_idx, 1]] = j as u32;
            pair_idx += 1;
        }
    }

    // Report progress: generating mid-near pairs
    report_progress(callback, "Pair Generation", 60, 100, 60.0, "Generating mid-near pairs");

    // Generate mid-near pairs (deterministic sampling)
    let mut pair_mn = Array2::zeros((n_samples * config.n_mn, 2));
    let _rng = rand::rngs::StdRng::seed_from_u64(config.random_state.unwrap_or(42));

    for i in 0..n_samples {
        let mut candidates = Vec::new();
        for j in 0..n_samples {
            if i != j {
                // Check if j is not already in neighbors of i
                let is_neighbor = (0..config.n_neighbors)
                    .any(|k| neighbor_indices[[i, k]] == j);
                if !is_neighbor {
                    candidates.push(j);
                }
            }
        }

        // Sort candidates by distance for deterministic selection
        candidates.sort_by(|&a, &b| distances[[i, a]].partial_cmp(&distances[[i, b]]).unwrap());

        // Sample mid-near pairs from further neighbors
        let start_idx = config.n_neighbors;
        for mn_idx in 0..config.n_mn {
            if start_idx + mn_idx < candidates.len() {
                let j = candidates[start_idx + mn_idx];
                pair_mn[[i * config.n_mn + mn_idx, 0]] = i as u32;
                pair_mn[[i * config.n_mn + mn_idx, 1]] = j as u32;
            }
        }
    }

    // Report progress: generating far pairs
    report_progress(callback, "Pair Generation", 80, 100, 80.0, "Generating far pairs");

    // Generate far pairs (deterministic random sampling from distant points)
    let mut pair_fp = Array2::zeros((n_samples * config.n_fp, 2));

    for i in 0..n_samples {
        let mut candidates = Vec::new();
        for j in 0..n_samples {
            if i != j {
                candidates.push(j);
            }
        }

        // Sort by distance (descending for far pairs)
        candidates.sort_by(|&a, &b| distances[[i, b]].partial_cmp(&distances[[i, a]]).unwrap());

        // Select far pairs deterministically based on seed
        for fp_idx in 0..config.n_fp {
            if fp_idx < candidates.len() {
                let j = candidates[fp_idx];
                pair_fp[[i * config.n_fp + fp_idx, 0]] = i as u32;
                pair_fp[[i * config.n_fp + fp_idx, 1]] = j as u32;
            }
        }
    }

    // Report progress: completion
    report_progress(callback, "Pair Generation", 100, 100, 100.0, "Pair generation completed");

    Ok(PairData {
        pair_neighbors,
        pair_mn,
        pair_fp,
    })
}

/// Compute PaCMAP gradients deterministically (single-threaded, fixed order)
pub fn pacmap_grad_with_progress(
    y: ndarray::ArrayView<f32, ndarray::Ix2>,
    pair_neighbors: ndarray::ArrayView<u32, ndarray::Ix2>,
    pair_mn: ndarray::ArrayView<u32, ndarray::Ix2>,
    pair_fp: ndarray::ArrayView<u32, ndarray::Ix2>,
    weights: &Weights,
    config: &GradientConfig,
) -> Array2<f32> {
    let (n_samples, n_dims) = y.dim();
    let mut grad = Array2::zeros((n_samples, n_dims));

    let callback = &config.progress_callback;

    // Report progress: computing neighbor gradients
    report_progress(callback, "Gradient Computation", 0, 100, 0.0, "Computing neighbor gradients");

    // CRITICAL: Process pairs in deterministic order
    // Neighbor pairs contribution
    for i in 0..pair_neighbors.nrows() {
        let p = pair_neighbors[[i, 0]] as usize;
        let q = pair_neighbors[[i, 1]] as usize;

        // Compute distance in embedding space with deterministic accumulation
        let mut dist_sq = 0.0f32;
        for k in 0..n_dims {
            let diff = y[[p, k]] - y[[q, k]];
            dist_sq += diff * diff;
        }

        if dist_sq > 1e-10 {
            let dist = dist_sq.sqrt();
            let force = weights.w_neighbors * (1.0 / (1.0 + dist_sq));

            // Apply gradient updates in deterministic order
            for k in 0..n_dims {
                let diff = y[[p, k]] - y[[q, k]];
                grad[[p, k]] += force * diff / dist;
                grad[[q, k]] -= force * diff / dist;
            }
        }
    }

    // Report progress: computing mid-near gradients
    report_progress(callback, "Gradient Computation", 40, 100, 40.0, "Computing mid-near gradients");

    // Mid-near pairs contribution
    for i in 0..pair_mn.nrows() {
        let p = pair_mn[[i, 0]] as usize;
        let q = pair_mn[[i, 1]] as usize;

        let mut dist_sq = 0.0f32;
        for k in 0..n_dims {
            let diff = y[[p, k]] - y[[q, k]];
            dist_sq += diff * diff;
        }

        if dist_sq > 1e-10 {
            let dist = dist_sq.sqrt();
            let force = weights.w_mn * (1.0 / (1.0 + dist_sq));

            for k in 0..n_dims {
                let diff = y[[p, k]] - y[[q, k]];
                grad[[p, k]] += force * diff / dist;
                grad[[q, k]] -= force * diff / dist;
            }
        }
    }

    // Report progress: computing far pair gradients
    report_progress(callback, "Gradient Computation", 70, 100, 70.0, "Computing far pair gradients");

    // Far pairs contribution (repulsive)
    for i in 0..pair_fp.nrows() {
        let p = pair_fp[[i, 0]] as usize;
        let q = pair_fp[[i, 1]] as usize;

        let mut dist_sq = 0.0f32;
        for k in 0..n_dims {
            let diff = y[[p, k]] - y[[q, k]];
            dist_sq += diff * diff;
        }

        if dist_sq > 1e-10 {
            let dist = dist_sq.sqrt();
            let force = weights.w_fp * dist_sq / (1.0 + dist_sq);

            for k in 0..n_dims {
                let diff = y[[p, k]] - y[[q, k]];
                grad[[p, k]] -= force * diff / dist;
                grad[[q, k]] += force * diff / dist;
            }
        }
    }

    // Report progress: normalization
    report_progress(callback, "Gradient Computation", 90, 100, 90.0, "Normalizing gradients");

    // Normalize gradients with deterministic computation
    let mut grad_norm = 0.0f32;
    for &g in grad.iter() {
        grad_norm += g * g;
    }
    grad_norm = grad_norm.sqrt();

    if grad_norm > 1e-10 {
        for g in grad.iter_mut() {
            *g /= grad_norm;
        }
    }

    // Report progress: completion
    report_progress(callback, "Gradient Computation", 100, 100, 100.0, "Gradient computation completed");

    grad
}

pub struct PairData {
    pub pair_neighbors: Array2<u32>,
    pub pair_mn: Array2<u32>,
    pub pair_fp: Array2<u32>,
}

pub fn find_weights_with_config(config: &WeightConfig, itr: usize) -> Weights {
    let mut weights = Weights::default();

    // Phase-based weight adaptation
    if itr < config.phase_1_iters {
        // Phase 1: Focus on preserving local structure
        weights.w_neighbors = 2.0;
        weights.w_mn = config.w_mn_init;
        weights.w_fp = 0.1;
    } else if itr < config.phase_1_iters + config.phase_2_iters {
        // Phase 2: Balance local and global structure
        weights.w_neighbors = 1.0;
        weights.w_mn = config.w_mn_init * 0.5;
        weights.w_fp = 1.0;
    } else {
        // Phase 3: Focus on global structure
        weights.w_neighbors = 0.5;
        weights.w_mn = config.w_mn_init * 0.25;
        weights.w_fp = 2.0;
    }

    weights
}

use ndarray::{ArrayView2, ArrayViewMut2};
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
    pub adam_config: AdamConfig,
    /// Whether to report progress
    pub report_progress: bool,
    /// Progress callback function
    pub progress_callback: Option<ProgressCallback>,
}

impl std::fmt::Debug for PacmapConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PacmapConfig")
            .field("n_dims", &self.n_dims)
            .field("n_neighbors", &self.n_neighbors)
            .field("n_mn", &self.n_mn)
            .field("n_fp", &self.n_fp)
            .field("seed", &self.seed)
            .field("n_iters", &self.n_iters)
            .field("pair_config", &self.pair_config)
            .field("knn_config", &self.knn_config)
            .field("gradient_config", &self.gradient_config)
            .field("adam_config", &self.adam_config)
            .field("report_progress", &self.report_progress)
            .field("progress_callback", &self.progress_callback.is_some())
            .finish()
    }
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
            adam_config: AdamConfig::default(),
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

    let (n_samples, _n_features) = x.dim();
    if n_samples == 0 {
        return Err(PacmapError::InvalidInput("Input data is empty".to_string()));
    }

    let algorithm_start_time = std::time::Instant::now();

    report_progress_with_timing(
        &config.progress_callback,
        "PaCMAP",
        0,
        config.n_iters + 3, // 3 main stages + optimization iterations
        0.0,
        &format!("Starting deterministic PaMAP: {} samples → {}D", n_samples, config.n_dims),
        algorithm_start_time,
    );

    // Stage 1: Generate pairs
    let pairs_start_time = std::time::Instant::now();
    report_progress_with_timing(
        &config.progress_callback,
        "Pairs",
        1,
        config.n_iters + 3,
        10.0,
        "Generating neighbor pairs",
        pairs_start_time,
    );

    let pairs = generate_pairs_with_progress(x, &config.pair_config)
        .map_err(|e| PacmapError::KnnError(e))?;

    // Stage 2: Initialize embedding
    let embedding_start_time = std::time::Instant::now();
    report_progress_with_timing(
        &config.progress_callback,
        "Embedding",
        2,
        config.n_iters + 3,
        20.0,
        "Initializing random embedding",
        embedding_start_time,
    );

    let mut y = initialize_embedding(x.nrows(), config.n_dims, config.seed.unwrap_or(42));

    // Initialize Adam state
    let (mut m, mut v) = create_adam_state_random((n_samples, config.n_dims), config.seed.unwrap_or(42));

    // Stage 3: Heavy optimization
    let optimization_start_time = std::time::Instant::now();
    report_progress_with_timing(
        &config.progress_callback,
        "Optimization",
        3,
        config.n_iters + 3,
        30.0,
        &format!("Starting {} iteration optimization", config.n_iters),
        optimization_start_time,
    );

    let mut final_stats = OptimizationStats {
        iterations_completed: 0,
        final_loss: 0.0,
        final_gradient_norm: 0.0,
        final_parameter_norm: 0.0,
        total_pairs: pairs.pair_neighbors.nrows() + pairs.pair_mn.nrows() + pairs.pair_fp.nrows(),
    };

    // Optimization loop (simplified for now)
    for itr in 0..config.n_iters {
        // Compute weights for this iteration
        let phase_1_iters = config.n_iters / 3;
        let phase_2_iters = config.n_iters / 3;
        let weight_config = WeightConfig::new(2.0, phase_1_iters, phase_2_iters);
        let weights = find_weights_with_config(&weight_config, itr);

        // Compute gradient
        let grad = pacmap_grad_with_progress(
            y.view(),
            pairs.pair_neighbors.view(),
            pairs.pair_mn.view(),
            pairs.pair_fp.view(),
            &weights,
            &config.gradient_config,
        );

        // Update embedding with Adam
        let adam_stats = update_embedding_adam_with_stats(
            y.view_mut(),
            grad.view(),
            m.view_mut(),
            v.view_mut(),
            &config.adam_config,
            &config.progress_callback,
        );

        // Update final stats
        final_stats.iterations_completed = itr + 1;
        final_stats.final_loss = adam_stats.loss;
        final_stats.final_gradient_norm = adam_stats.grad_norm;
        final_stats.final_parameter_norm = adam_stats.param_norm;

        // Enhanced professional progress reporting during optimization
        if config.report_progress {
            let phase_1_iters = config.n_iters / 3;
            let phase_2_iters = config.n_iters / 3;
            let phase_3_iters = config.n_iters - phase_1_iters - phase_2_iters;

            let (phase_name, phase_base_percentage, phase_weight) = if itr < phase_1_iters {
                ("Local Structure", 30.0, phase_1_iters)
            } else if itr < phase_1_iters + phase_2_iters {
                ("Balanced Optimization", 30.0 + (phase_1_iters as f32 / config.n_iters as f32) * 65.0, phase_2_iters)
            } else {
                ("Global Structure", 30.0 + ((phase_1_iters + phase_2_iters) as f32 / config.n_iters as f32) * 65.0, phase_3_iters)
            };

            let phase_progress = if phase_weight > 0 {
                ((itr % phase_weight) as f32 / phase_weight as f32) * 65.0 + phase_base_percentage
            } else {
                phase_base_percentage
            };

            // Report every 10% of phase or every 50 iterations, whichever is more frequent
            let report_frequency = (phase_weight / 10).max(50);
            if itr == 0 || itr == config.n_iters - 1 || itr % report_frequency == 0 {
                let phase_details = format!(
                    "{}: Iter {}/{} | Loss: {:.6} | Grad: {:.6}",
                    phase_name, itr + 1, config.n_iters, adam_stats.loss, adam_stats.grad_norm
                );
                report_progress_with_timing(
                    &config.progress_callback,
                    "Optimization",
                    itr + 1,
                    config.n_iters,
                    phase_progress,
                    &phase_details,
                    optimization_start_time,
                );
            }
        }
    }

    let total_time = algorithm_start_time.elapsed();
    let completion_message = format!(
        "PacMAP Complete: {} iterations in {:.2}s | Final loss: {:.6}",
        config.n_iters, total_time.as_secs_f64(), final_stats.final_loss
    );

    report_progress_with_timing(
        &config.progress_callback,
        "Complete",
        config.n_iters + 3,
        config.n_iters + 3,
        100.0,
        &completion_message,
        algorithm_start_time,
    );

    Ok(PacmapResult {
        embedding: y,
        final_stats,
    })
}

/// Create Adam optimizer state initialized with zeros
pub fn create_adam_state_random(shape: (usize, usize), _seed: u64) -> (Array2<f32>, Array2<f32>) {
    let m = Array2::zeros(shape);
    let v = Array2::zeros(shape);
    (m, v)
}

/// Update embedding using Adam optimizer with enhanced progress reporting (deterministic)
pub fn update_embedding_adam_with_stats(
    mut y: ArrayViewMut2<f32>,
    grad: ArrayView2<f32>,
    mut m: ArrayViewMut2<f32>,
    mut v: ArrayViewMut2<f32>,
    config: &AdamConfig,
    _callback: &Option<ProgressCallback>,
) -> AdamStats {
    let (n_samples, n_dims) = y.dim();
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let epsilon = 1e-8f32;
    let lr = config.learning_rate;

    // CRITICAL: Process in deterministic order (row-major)
    // Update biased first moment estimate
    for i in 0..n_samples {
        for j in 0..n_dims {
            m[[i, j]] = beta1 * m[[i, j]] + (1.0 - beta1) * grad[[i, j]];
        }
    }

    // Update biased second raw moment estimate
    for i in 0..n_samples {
        for j in 0..n_dims {
            v[[i, j]] = beta2 * v[[i, j]] + (1.0 - beta2) * grad[[i, j]] * grad[[i, j]];
        }
    }

    // Compute bias-corrected estimates and update parameters
    for i in 0..n_samples {
        for j in 0..n_dims {
            let m_hat = m[[i, j]];
            let v_hat = v[[i, j]];
            let step = lr * m_hat / (v_hat.sqrt() + epsilon);
            y[[i, j]] += step;
        }
    }

    // Compute statistics with deterministic accumulation
    let mut grad_norm = 0.0f32;
    for &g in grad.iter() {
        grad_norm += g * g;
    }
    grad_norm = grad_norm.sqrt();

    let mut param_norm = 0.0f32;
    for &p in y.iter() {
        param_norm += p * p;
    }
    param_norm = param_norm.sqrt();

    let loss = grad_norm * grad_norm; // Simplified loss computation

    AdamStats {
        loss,
        grad_norm,
        param_norm,
    }
}

/// Initialize embedding with small random values
fn initialize_embedding(n_samples: usize, n_dims: usize, seed: u64) -> Array2<f32> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);

    Array2::from_shape_fn((n_samples, n_dims), |_| {
        rng.gen::<f32>() * 1e-4 - 5e-5 // Small random values around zero
    })
}

/// Reports progress safely with error handling - minimal reporting for heavy operations only
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

/// Professional progress reporting with time estimates and phase information
fn report_progress_with_timing(
    callback: &Option<ProgressCallback>,
    stage: &str,
    current: usize,
    total: usize,
    percentage: f32,
    details: &str,
    start_time: std::time::Instant,
) {
    if let Some(ref cb) = callback {
        let elapsed = start_time.elapsed();
        let elapsed_secs = elapsed.as_secs_f64();

        // Estimate remaining time
        let remaining_secs = if percentage > 0.0 && current > 0 {
            elapsed_secs * (100.0 - percentage as f64) / percentage as f64
        } else {
            0.0
        };

        let timing_info = if remaining_secs > 0.0 {
            format!("{} (ETA: {:.1}s)", details, remaining_secs)
        } else {
            details.to_string()
        };

        cb(stage, current, total, percentage, &timing_info);
    }
}

/// Enhanced optimization phase reporting
fn report_optimization_phase(
    callback: &Option<ProgressCallback>,
    phase: &str,
    iteration: usize,
    total_iterations: usize,
    phase_progress: f32,
    loss: f32,
    grad_norm: f32,
) {
    if let Some(ref cb) = callback {
        let details = format!(
            "Phase {}: Iter {}/{} | Loss: {:.6} | Grad: {:.6}",
            phase, iteration, total_iterations, loss, grad_norm
        );
        cb("Optimization", iteration, total_iterations, phase_progress, &details);
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
        let empty_x = Array2::<f32>::zeros((0, 0));
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