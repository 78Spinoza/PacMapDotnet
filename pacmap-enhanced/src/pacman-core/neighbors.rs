//! Manages sampling of point pairs for `PaCMAP` dimensionality reduction - Enhanced Version.
//!
//! This module generates three key types of point pairs that control how
//! structure is preserved during dimension reduction:
//!
//! - Nearest neighbor pairs preserve local neighborhoods and distances by
//!   connecting each point to its closest neighbors
//!
//! - Mid-near pairs maintain relationships between moderately distant points to
//!   capture intermediate structure and prevent overfitting to local
//!   neighborhoods
//!
//! - Far pairs prevent the embedding from collapsing by introducing repulsive
//!   forces between distant points
//!
//! This enhanced version provides deterministic pair generation with detailed
//! progress reporting and improved error handling.

use crate::distance::scale_dist_with_progress;
use crate::knn::{find_k_nearest_neighbors_with_progress, KnnConfig, KnnError};
use crate::sampling::{
    sample_fp_pair_with_progress, sample_mn_pair_with_progress, sample_neighbors_pair_with_progress,
    SamplingConfig,
};
use crate::Pairs;
use ndarray::{s, Array1, Array2, ArrayView2, Axis};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Progress callback type for pair generation
pub type ProgressCallback = Box<dyn Fn(&str, usize, usize, f32, &str) + Send + Sync>;

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

/// Configuration for pair generation with enhanced options
#[derive(Debug, Clone)]
pub struct PairConfig {
    /// Number of nearest neighbors to find per point
    pub n_neighbors: usize,
    /// Number of mid-near pairs to generate per point
    pub n_mn: usize,
    /// Number of far pairs to generate per point
    pub n_fp: usize,
    /// Random seed for deterministic pair sampling
    pub random_state: Option<u64>,
    /// Whether to use deterministic behavior
    pub deterministic: bool,
    /// Whether to report progress during generation
    pub report_progress: bool,
    /// Progress callback function
    pub progress_callback: Option<ProgressCallback>,
    /// KNN computation configuration
    pub knn_config: KnnConfig,
    /// Sampling configuration
    pub sampling_config: SamplingConfig,
}

impl Default for PairConfig {
    fn default() -> Self {
        Self {
            n_neighbors: 15,
            n_mn: 5,
            n_fp: 10,
            random_state: Some(42),
            deterministic: true,
            report_progress: false,
            progress_callback: None,
            knn_config: KnnConfig::default(),
            sampling_config: SamplingConfig::default(),
        }
    }
}

impl PairConfig {
    /// Create new pair configuration with basic parameters
    pub fn new(n_neighbors: usize, n_mn: usize, n_fp: usize) -> Self {
        Self {
            n_neighbors,
            n_mn,
            n_fp,
            random_state: Some(42),
            deterministic: true,
            ..Default::default()
        }
    }

    /// Set random seed for deterministic behavior
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self.deterministic = true;
        self.sampling_config.random_state = seed;
        self
    }

    /// Enable/disable deterministic behavior
    pub fn deterministic(mut self, deterministic: bool) -> Self {
        self.deterministic = deterministic;
        if !deterministic {
            self.random_state = None;
        }
        self
    }

    /// Enable progress reporting with callback
    pub fn with_progress_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&str, usize, usize, f32, &str) + Send + Sync + 'static,
    {
        self.report_progress = true;
        self.progress_callback = Some(Box::new(callback));
        self.knn_config.report_progress = true;
        self.knn_config.progress_callback = self.progress_callback.clone();
        self.sampling_config.report_progress = true;
        self.sampling_config.progress_callback = self.progress_callback.clone();
        self
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.n_neighbors == 0 {
            return Err("Number of neighbors must be positive".to_string());
        }
        if self.n_mn == 0 {
            return Err("Number of mid-near pairs must be positive".to_string());
        }
        if self.n_fp == 0 {
            return Err("Number of far pairs must be positive".to_string());
        }
        if let Some(seed) = self.random_state {
            self.sampling_config.random_state = seed;
        }
        self.knn_config.validate()?;
        Ok(())
    }
}

/// Generates the complete set of point pairs needed for `PaCMAP` optimization with enhanced progress reporting.
///
/// Creates all three pair types (nearest neighbor, mid-near, far) through a
/// multi-step process with detailed progress reporting:
///
/// 1. Finds k-nearest neighbors with padding for robust pair selection
/// 2. Computes scaling factors from moderately distant neighbors (indices 3-5)
/// 3. Applies distance scaling to normalize neighborhood sizes
/// 4. Samples neighbor pairs based on scaled distances
/// 5. Generates mid-near and far pairs deterministically
///
/// # Arguments
/// * `x` - Input data as an n × d matrix where each row is a point
/// * `config` - Pair generation configuration
///
/// # Returns
/// A `Pairs` struct containing the neighbor, mid-near, and far pair indices
///
/// # Errors
/// Returns `KnnError` if k-nearest neighbor search fails due to index
/// construction or memory allocation issues, or validation error if configuration is invalid
pub fn generate_pairs_with_progress(
    x: ArrayView2<f32>,
    config: &PairConfig,
) -> Result<Pairs, String> {
    // Validate configuration
    config.validate()?;

    let n = x.nrows();
    if n == 0 {
        return Err("Input data is empty".to_string());
    }

    report_progress(
        &config.progress_callback,
        "Pair Generation",
        0,
        5,
        0.0,
        &format!("Starting pair generation for {} points", n),
    );

    // Add padding neighbors for robust pair selection
    let n_neighbors_extra = (config.n_neighbors + 50).min(n - 1);
    let n_neighbors = config.n_neighbors.min(n - 1);

    report_progress(
        &config.progress_callback,
        "Pair Generation",
        1,
        5,
        20.0,
        &format!("Finding {} nearest neighbors with {} padding", n_neighbors, n_neighbors_extra - config.n_neighbors),
    );

    // Use deterministic exact neighbors (no approximate search for reproducibility)
    let mut knn_config = config.knn_config.clone();
    knn_config.k = n_neighbors_extra;
    knn_config.progress_callback = config.progress_callback.clone();

    let (neighbors, knn_distances) = find_k_nearest_neighbors_with_progress(x, &knn_config);

    report_progress(
        &config.progress_callback,
        "Pair Generation",
        2,
        5,
        40.0,
        "Computing distance scaling factors",
    );

    // Scale distances using moderately distant neighbors (indices 3-5)
    // for robust neighborhood size estimation
    let start = min(3, knn_distances.ncols().saturating_sub(1));
    let end = min(6, knn_distances.ncols());
    let sig = knn_distances
        .slice(s![.., start..end])
        .mean_axis(Axis(1))
        .map_or_else(|| Array1::from_elem(n, 1e-10), |d| d.mapv(|x| x.max(1e-10)));

    let neighbors_view = neighbors.view();
    let scaled_dist = scale_dist_with_progress(
        knn_distances.view(),
        sig.view(),
        neighbors_view,
        config.progress_callback.clone(),
    );

    report_progress(
        &config.progress_callback,
        "Pair Generation",
        3,
        5,
        60.0,
        "Sampling neighbor pairs based on scaled distances",
    );

    let pair_neighbors = sample_neighbors_pair_with_progress(
        x.view(),
        scaled_dist.view(),
        neighbors_view,
        n_neighbors,
        &config.progress_callback,
    );

    report_progress(
        &config.progress_callback,
        "Pair Generation",
        4,
        5,
        80.0,
        &format!("Generating {} mid-near and {} far pairs", config.n_mn, config.n_fp),
    );

    // Generate mid-near and far pairs deterministically
    let mut sampling_config = config.sampling_config.clone();
    sampling_config.progress_callback = config.progress_callback.clone();

    let pair_mn = sample_mn_pair_with_progress(x.view(), config.n_mn, &sampling_config);
    let pair_fp = sample_fp_pair_with_progress(
        x.view(),
        pair_neighbors.view(),
        n_neighbors,
        config.n_fp,
        &sampling_config,
    );

    report_progress(
        &config.progress_callback,
        "Pair Generation",
        5,
        5,
        100.0,
        &format!("Completed pair generation: {} neighbor, {} mid-near, {} far pairs",
                pair_neighbors.nrows(), pair_mn.nrows(), pair_fp.nrows()),
    );

    Ok(Pairs {
        pair_neighbors,
        pair_mn,
        pair_fp,
    })
}

/// Generates the complete set of point pairs needed for `PaCMAP` optimization (legacy API).
///
/// # Arguments
/// * `x` - Input data as an n × d matrix where each row is a point
/// * `n_neighbors` - Number of nearest neighbors to find per point
/// * `n_mn` - Number of mid-near pairs to generate per point
/// * `n_fp` - Number of far pairs to generate per point
/// * `random_state` - Optional seed for deterministic pair sampling
/// * `approx_threshold` - Row count threshold for switching to approximate
///   neighbor search (ignored in this deterministic implementation)
///
/// # Returns
/// A `Pairs` struct containing the neighbor, mid-near, and far pair indices
///
/// # Errors
/// Returns `KnnError` if k-nearest neighbor search fails due to index
/// construction or memory allocation issues
pub fn generate_pairs(
    x: ArrayView2<f32>,
    n_neighbors: usize,
    n_mn: usize,
    n_fp: usize,
    random_state: Option<u64>,
    _approx_threshold: usize, // Ignored - always use exact for determinism
) -> Result<Pairs, KnnError> {
    let config = if let Some(seed) = random_state {
        PairConfig::new(n_neighbors, n_mn, n_fp).with_seed(seed)
    } else {
        PairConfig::new(n_neighbors, n_mn, n_fp).deterministic(false)
    };

    generate_pairs_with_progress(x, &config).map_err(|e| KnnError::InvalidInput(e))
}

/// Generates mid-near and far pairs from pre-computed nearest neighbors with enhanced progress reporting.
///
/// Efficiently generates additional pairs when nearest neighbors have already
/// been computed, avoiding redundant distance calculations.
///
/// # Arguments
/// * `x` - Input data as an n × d matrix where each row is a point
/// * `n_neighbors` - Number of nearest neighbors used in input pairs
/// * `n_mn` - Number of mid-near pairs to generate per point
/// * `n_fp` - Number of far pairs to generate per point
/// * `pair_neighbors` - Pre-computed nearest neighbor pair indices
/// * `config` - Pair generation configuration (only uses sampling settings)
///
/// # Returns
/// A tuple containing:
/// - Mid-near pair indices as an (n × `n_mn`) × 2 array
/// - Far pair indices as an (n × `n_fp`) × 2 array
pub fn generate_pair_no_neighbors_with_progress(
    x: ArrayView2<f32>,
    n_neighbors: usize,
    n_mn: usize,
    n_fp: usize,
    pair_neighbors: ArrayView2<u32>,
    config: &PairConfig,
) -> (Array2<u32>, Array2<u32>) {
    let n = x.nrows();

    report_progress(
        &config.progress_callback,
        "Additional Pair Generation",
        0,
        2,
        0.0,
        "Generating additional pairs from pre-computed neighbors",
    );

    let mut sampling_config = config.sampling_config.clone();
    sampling_config.progress_callback = config.progress_callback.clone();

    let pair_mn = sample_mn_pair_with_progress(x.view(), n_mn, &sampling_config);

    report_progress(
        &config.progress_callback,
        "Additional Pair Generation",
        1,
        2,
        50.0,
        "Generated mid-near pairs, now generating far pairs",
    );

    let pair_fp = sample_fp_pair_with_progress(
        x.view(),
        pair_neighbors,
        n_neighbors,
        n_fp,
        &sampling_config,
    );

    report_progress(
        &config.progress_callback,
        "Additional Pair Generation",
        2,
        2,
        100.0,
        &format!("Completed additional pair generation: {} mid-near, {} far pairs",
                pair_mn.nrows(), pair_fp.nrows()),
    );

    (pair_mn, pair_fp)
}

/// Generates mid-near and far pairs from pre-computed nearest neighbors (legacy API).
///
/// # Arguments
/// * `x` - Input data as an n × d matrix where each row is a point
/// * `n_neighbors` - Number of nearest neighbors used in input pairs
/// * `n_mn` - Number of mid-near pairs to generate per point
/// * `n_fp` - Number of far pairs to generate per point
/// * `pair_neighbors` - Pre-computed nearest neighbor pair indices
/// * `random_seed` - Optional seed for deterministic sampling
///
/// # Returns
/// A tuple containing:
/// - Mid-near pair indices as an (n × `n_mn`) × 2 array
/// - Far pair indices as an (n × `n_fp`) × 2 array
pub fn generate_pair_no_neighbors(
    x: ArrayView2<f32>,
    n_neighbors: usize,
    n_mn: usize,
    n_fp: usize,
    pair_neighbors: ArrayView2<u32>,
    random_seed: Option<u64>,
) -> (Array2<u32>, Array2<u32>) {
    let config = if let Some(seed) = random_seed {
        PairConfig::new(n_neighbors, n_mn, n_fp).with_seed(seed)
    } else {
        PairConfig::new(n_neighbors, n_mn, n_fp).deterministic(false)
    };

    generate_pair_no_neighbors_with_progress(x, n_neighbors, n_mn, n_fp, pair_neighbors, &config)
}

/// Validates input data for pair generation
pub fn validate_pair_input(x: ArrayView2<f32>, config: &PairConfig) -> Result<(), String> {
    let (n, d) = x.dim();

    if n == 0 {
        return Err("Input data has zero points".to_string());
    }
    if d == 0 {
        return Err("Input data has zero dimensions".to_string());
    }

    // Validate pair counts against available points
    if config.n_neighbors >= n {
        return Err(format!(
            "Cannot request {} neighbors from {} points (need at least 1 fewer)",
            config.n_neighbors, n
        ));
    }

    // Check for NaN or infinite values
    for (i, row) in x.axis_iter(Axis(0)).enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if !val.is_finite() {
                return Err(format!("Non-finite value at position ({}, {}): {}", i, j, val));
            }
        }
    }

    // Check for excessive variance
    for j in 0..d {
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for i in 0..n {
            let val = x[[i, j]];
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }

        let range = max_val - min_val;
        if range > 1e12 {
            return Err(format!(
                "Excessive range in dimension {}: {} to {} (range: {})",
                j, min_val, max_val, range
            ));
        }
    }

    Ok(())
}

/// Computes statistics about generated pairs for monitoring
pub fn compute_pair_stats(pairs: &Pairs) -> PairStats {
    PairStats {
        neighbor_pairs: pairs.pair_neighbors.nrows(),
        mid_near_pairs: pairs.pair_mn.nrows(),
        far_pairs: pairs.pair_fp.nrows(),
        total_pairs: pairs.pair_neighbors.nrows() + pairs.pair_mn.nrows() + pairs.pair_fp.nrows(),
    }
}

/// Statistics about generated pairs
#[derive(Debug, Clone)]
pub struct PairStats {
    /// Number of nearest neighbor pairs
    pub neighbor_pairs: usize,
    /// Number of mid-near pairs
    pub mid_near_pairs: usize,
    /// Number of far pairs
    pub far_pairs: usize,
    /// Total number of pairs
    pub total_pairs: usize,
}

impl std::fmt::Display for PairStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PairStats {{ neighbors: {}, mid-near: {}, far: {}, total: {} }}",
            self.neighbor_pairs, self.mid_near_pairs, self.far_pairs, self.total_pairs
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_pair_config() {
        let config = PairConfig::default();
        assert_eq!(config.n_neighbors, 15);
        assert_eq!(config.n_mn, 5);
        assert_eq!(config.n_fp, 10);
        assert!(config.deterministic);
        assert_eq!(config.random_state, Some(42));

        let custom_config = PairConfig::new(10, 3, 7).with_seed(123);
        assert_eq!(custom_config.n_neighbors, 10);
        assert_eq!(custom_config.n_mn, 3);
        assert_eq!(custom_config.n_fp, 7);
        assert_eq!(custom_config.random_state, Some(123));

        // Test validation
        assert!(custom_config.validate().is_ok());

        let invalid_config = PairConfig::new(0, 3, 7);
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_generate_pairs() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let config = PairConfig::new(2, 1, 2).with_seed(42);

        let result = generate_pairs_with_progress(x.view(), &config);
        assert!(result.is_ok());

        let pairs = result.unwrap();
        assert_eq!(pairs.pair_neighbors.shape(), &[4 * 2, 2]); // n * n_neighbors
        assert_eq!(pairs.pair_mn.shape(), &[4 * 1, 2]);      // n * n_mn
        assert_eq!(pairs.pair_fp.shape(), &[4 * 2, 2]);      // n * n_fp
    }

    #[test]
    fn test_generate_pair_no_neighbors() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let pair_neighbors = array![[0, 1], [1, 2], [2, 0]];
        let config = PairConfig::new(2, 1, 1).with_seed(42);

        let (pair_mn, pair_fp) = generate_pair_no_neighbors_with_progress(
            x.view(),
            2,
            1,
            1,
            pair_neighbors.view(),
            &config,
        );

        assert_eq!(pair_mn.shape(), &[3 * 1, 2]); // n * n_mn
        assert_eq!(pair_fp.shape(), &[3 * 1, 2]); // n * n_fp
    }

    #[test]
    fn test_progress_callback() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let progress_calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let progress_calls_clone = progress_calls.clone();

        let config = PairConfig::new(1, 1, 1).with_progress_callback(move |stage, current, total, percentage, details| {
            let mut calls = progress_calls_clone.lock().unwrap();
            calls.push((stage.to_string(), current, total, percentage, details.to_string()));
        });

        let result = generate_pairs_with_progress(x.view(), &config);
        assert!(result.is_ok());

        let calls = progress_calls.lock().unwrap();
        assert!(!calls.is_empty());
        assert!(calls[0].0.contains("Pair Generation"));
    }

    #[test]
    fn test_input_validation() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let valid_config = PairConfig::new(1, 1, 1);

        assert!(validate_pair_input(x.view(), &valid_config).is_ok());

        // Empty data
        let empty_x = array![];
        assert!(validate_pair_input(empty_x.view(), &valid_config).is_err());

        // Too many neighbors
        let too_many_config = PairConfig::new(5, 1, 1);
        assert!(validate_pair_input(x.view(), &too_many_config).is_err());

        // NaN data
        let nan_x = array![[1.0, f32::NAN], [3.0, 4.0]];
        assert!(validate_pair_input(nan_x.view(), &valid_config).is_err());
    }

    #[test]
    fn test_pair_stats() {
        let pairs = Pairs {
            pair_neighbors: array![[0, 1], [1, 2]],
            pair_mn: array![[0, 2]],
            pair_fp: array![[1, 0], [2, 1], [0, 2]],
        };

        let stats = compute_pair_stats(&pairs);
        assert_eq!(stats.neighbor_pairs, 2);
        assert_eq!(stats.mid_near_pairs, 1);
        assert_eq!(stats.far_pairs, 3);
        assert_eq!(stats.total_pairs, 6);

        // Test display
        let display_str = format!("{}", stats);
        assert!(display_str.contains("PairStats"));
        assert!(display_str.contains("neighbors: 2"));
        assert!(display_str.contains("mid-near: 1"));
        assert!(display_str.contains("far: 3"));
        assert!(display_str.contains("total: 6"));
    }

    #[test]
    fn test_deterministic_behavior() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let config = PairConfig::new(2, 1, 1).with_seed(42);

        let pairs1 = generate_pairs_with_progress(x.view(), &config).unwrap();
        let pairs2 = generate_pairs_with_progress(x.view(), &config).unwrap();

        // Results should be identical
        assert_eq!(pairs1.pair_neighbors, pairs2.pair_neighbors);
        assert_eq!(pairs1.pair_mn, pairs2.pair_mn);
        assert_eq!(pairs1.pair_fp, pairs2.pair_fp);
    }

    #[test]
    fn test_legacy_api_compatibility() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let result_legacy = generate_pairs(x.view(), 2, 1, 1, Some(42), 1000);
        assert!(result_legacy.is_ok());

        let config = PairConfig::new(2, 1, 1).with_seed(42);
        let result_new = generate_pairs_with_progress(x.view(), &config);
        assert!(result_new.is_ok());

        let pairs_legacy = result_legacy.unwrap();
        let pairs_new = result_new.unwrap();

        // Results should be equivalent (may have slight differences due to enhanced algorithms)
        assert_eq!(pairs_legacy.pair_neighbors.shape(), pairs_new.pair_neighbors.shape());
        assert_eq!(pairs_legacy.pair_mn.shape(), pairs_new.pair_mn.shape());
        assert_eq!(pairs_legacy.pair_fp.shape(), pairs_new.pair_fp.shape());
    }
}