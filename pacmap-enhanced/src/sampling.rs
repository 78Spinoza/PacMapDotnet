//! `PaCMAP` pair sampling implementations - Deterministic Enhanced Version.
//!
//! This module provides deterministic sampling functions for three types of point pairs used
//! in `PaCMAP` dimensionality reduction with enhanced progress reporting:
//!
//! - Far pairs (FP): Random distant points sampled from outside each point's
//!   nearest neighbors
//! - Mid-near pairs (MN): Points sampled to preserve mid-range distances and
//!   global structure
//! - Nearest neighbors (NN): Close points based on distance metrics that
//!   preserve local structure
//!
//! All sampling strategies are deterministic with proper seeding and include
//! detailed progress reporting capabilities.

use crate::distance::array_euclidean_distance;
use ndarray::{s, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis, Zip};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::slice::ParallelSliceMut;
use std::cmp::min;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Progress callback type for sampling operations
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

/// Enhanced configuration for sampling operations with progress reporting
pub struct SamplingConfig {
    /// Random seed for deterministic sampling
    pub random_state: u64,
    /// Whether to report progress during sampling
    pub report_progress: bool,
    /// Progress callback function
    pub progress_callback: Option<ProgressCallback>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            random_state: 42,
            report_progress: false,
            progress_callback: None,
        }
    }
}

impl SamplingConfig {
    /// Create new sampling configuration
    pub fn new(random_state: u64) -> Self {
        Self {
            random_state,
            report_progress: false,
            progress_callback: None,
        }
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
}

impl Clone for SamplingConfig {
    fn clone(&self) -> Self {
        Self {
            random_state: self.random_state,
            report_progress: self.report_progress,
            progress_callback: None, // Can't clone closure, so reset to None
        }
    }
}

impl std::fmt::Debug for SamplingConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SamplingConfig")
            .field("random_state", &self.random_state)
            .field("report_progress", &self.report_progress)
            .field("progress_callback", &self.progress_callback.is_some())
            .finish()
    }
}

/// Samples random indices while avoiding excluded values (deterministic version).
///
/// # Arguments
/// * `n_samples` - Number of unique indices to sample
/// * `maximum` - Maximum index value (exclusive)
/// * `reject_ind` - Array of indices that cannot be sampled
/// * `self_ind` - Index to exclude from sampling (typically the source point index)
/// * `rng` - Random number generator to use
///
/// # Returns
/// A vector of `n_samples` unique indices, each < `maximum` and not in `reject_ind`
fn sample_fp_deterministic<R>(
    n_samples: usize,
    maximum: u32,
    reject_ind: ArrayView1<u32>,
    self_ind: u32,
    rng: &mut R,
) -> Vec<u32>
where
    R: Rng,
{
    let available_indices = (maximum as usize)
        .saturating_sub(reject_ind.len())
        .saturating_sub(usize::from(reject_ind.iter().all(|&i| i != self_ind)));

    let n_samples = min(n_samples, available_indices);
    let mut result = Vec::with_capacity(n_samples);

    while result.len() < n_samples {
        let j = rng.gen_range(0..maximum);
        if j != self_ind && !result.contains(&j) && reject_ind.iter().all(|&k| k != j) {
            result.push(j);
        }
    }
    result
}

/// Samples far pairs deterministically with enhanced progress reporting.
///
/// Generates pairs of points by selecting random indices far from each point's
/// nearest neighbors. The sampling is reproducible when using the same seed.
/// Includes detailed progress reporting for large datasets.
///
/// # Arguments
/// * `x` - Input data matrix where each row is a point
/// * `pair_neighbors` - Matrix of nearest neighbor indices for each point
/// * `n_neighbors` - Number of nearest neighbors per point
/// * `n_fp` - Number of far pairs to sample per point
/// * `config` - Sampling configuration with seed and progress reporting
///
/// # Returns
/// Matrix of shape `(n * n_fp, 2)` containing sampled far point pairs
pub fn sample_fp_pair_with_progress(
    x: ArrayView2<f32>,
    pair_neighbors: ArrayView2<u32>,
    n_neighbors: usize,
    n_fp: usize,
    config: &SamplingConfig,
) -> Array2<u32> {
    let n = x.nrows();
    let mut pair_fp = Array2::zeros((n * n_fp, 2));
    let n = n as u32;

    report_progress(
        &config.progress_callback,
        "Far Pair Sampling",
        0,
        n as usize,
        0.0,
        &format!("Starting far pair sampling for {} points", n),
    );

    // Use atomic counter for thread-safe progress tracking
    let progress_counter = AtomicUsize::new(0);

    // Sample n_fp far pairs for each point in parallel
    pair_fp
        .axis_chunks_iter_mut(Axis(0), n_fp)
        .enumerate()
        .for_each(|(i, mut pairs)| {
            let reject_ind =
                pair_neighbors.slice(s![i * n_neighbors..(i + 1) * n_neighbors, 1_usize]);

            let mut rng = StdRng::seed_from_u64(config.random_state + i as u64);
            let fp_index = sample_fp_deterministic(n_fp, n, reject_ind, i as u32, &mut rng);
            assign_pairs(i, &mut pairs, &fp_index);

            // Update progress (every 10% or for small datasets)
            let completed = progress_counter.fetch_add(1, Ordering::Relaxed) + 1;
            if config.report_progress && (completed % (n as usize / 10 + 1) == 0 || n <= 100) {
                let percentage = (completed as f32 / n as f32) * 100.0;
                report_progress(
                    &config.progress_callback,
                    "Far Pair Sampling",
                    completed,
                    n as usize,
                    percentage,
                    &format!("Processed {} points", completed),
                );
            }
        });

    report_progress(
        &config.progress_callback,
        "Far Pair Sampling",
        n as usize,
        n as usize,
        100.0,
        &format!("Completed far pair sampling: {} pairs generated", n * n_fp as u32),
    );

    pair_fp
}

/// Samples mid-near pairs deterministically with enhanced progress reporting.
///
/// Generates pairs of points with intermediate distances to help preserve
/// global structure. Each pair is selected by sampling 6 random points and
/// picking the second closest. Includes detailed progress reporting.
///
/// # Arguments
/// * `x` - Input data matrix where each row is a point
/// * `n_mn` - Number of mid-near pairs to sample per point
/// * `config` - Sampling configuration with seed and progress reporting
///
/// # Returns
/// Matrix of shape `(n * n_mn, 2)` containing sampled mid-near pairs
pub fn sample_mn_pair_with_progress(
    x: ArrayView2<f32>,
    n_mn: usize,
    config: &SamplingConfig,
) -> Array2<u32> {
    let n = x.nrows();
    let mut pair_mn = Array2::<u32>::zeros((n * n_mn, 2));
    let n = n as u32;

    report_progress(
        &config.progress_callback,
        "Mid-Near Pair Sampling",
        0,
        n as usize,
        0.0,
        &format!("Starting mid-near pair sampling for {} points", n),
    );

    // Use atomic counter for thread-safe progress tracking
    let progress_counter = AtomicUsize::new(0);

    // Sample n_mn mid-near pairs for each point in parallel
    pair_mn
        .axis_chunks_iter_mut(Axis(0), n_mn)
        .enumerate()
        .for_each(|(i, mut pairs)| {
            let mut rng = StdRng::seed_from_u64(config.random_state + i as u64 + 1000); // Offset to avoid correlation
            for j in 0..n_mn {
                let reject_ind = pairs.slice(s![0..j, 1_usize]);
                let sampled = sample_fp_deterministic(6, n, reject_ind, i as u32, &mut rng);
                sample_mn_pair_impl(x, pairs.row_mut(j), i, &sampled);
            }

            // Update progress (every 10% or for small datasets)
            let completed = progress_counter.fetch_add(1, Ordering::Relaxed) + 1;
            if config.report_progress && (completed % (n as usize / 10 + 1) == 0 || n <= 100) {
                let percentage = (completed as f32 / n as f32) * 100.0;
                report_progress(
                    &config.progress_callback,
                    "Mid-Near Pair Sampling",
                    completed,
                    n as usize,
                    percentage,
                    &format!("Processed {} points", completed),
                );
            }
        });

    report_progress(
        &config.progress_callback,
        "Mid-Near Pair Sampling",
        n as usize,
        n as usize,
        100.0,
        &format!("Completed mid-near pair sampling: {} pairs generated", n * n_mn as u32),
    );

    pair_mn
}

/// Samples nearest neighbor pairs based on scaled distances with progress reporting.
///
/// Generates pairs by connecting each point with its nearest neighbors
/// according to a distance matrix. Neighbors are sorted by their scaled
/// distances. This is a deterministic operation but includes progress reporting.
///
/// # Arguments
/// * `x` - Input data matrix where each row is a point
/// * `scaled_dist` - Matrix of scaled distances between points
/// * `neighbors` - Matrix of nearest neighbor indices for each point
/// * `n_neighbors` - Number of neighbors to sample per point
/// * `progress_callback` - Optional progress callback
///
/// # Returns
/// Matrix of shape `(n * n_neighbors, 2)` containing sampled neighbor pairs
pub fn sample_neighbors_pair_with_progress(
    x: ArrayView2<f32>,
    scaled_dist: ArrayView2<f32>,
    neighbors: ArrayView2<u32>,
    n_neighbors: usize,
    progress_callback: &Option<ProgressCallback>,
) -> Array2<u32> {
    let n = x.nrows();
    let mut sorted_dist_indices = Array2::<u32>::zeros(scaled_dist.dim());

    report_progress(
        progress_callback,
        "Neighbor Pair Sorting",
        0,
        2,
        0.0,
        "Starting distance sorting for neighbor pairs",
    );

    // Sort scaled distances for each point and store sorted indices
    let sort_counter = AtomicUsize::new(0);
    Zip::from(scaled_dist.axis_iter(Axis(0)))
        .and(sorted_dist_indices.axis_iter_mut(Axis(0)))
        .par_for_each(|distances, mut indices| {
            let mut distance_indices = distances.into_iter().enumerate().collect::<Vec<_>>();
            distance_indices.par_sort_unstable_by(|a, b| f32::total_cmp(a.1, b.1));
            for (i, (index, _)) in distance_indices.iter().enumerate() {
                indices[i] = *index as u32;
            }

            // Update progress
            let completed = sort_counter.fetch_add(1, Ordering::Relaxed) + 1;
            if progress_callback.is_some() && (completed % (n / 10 + 1) == 0 || n <= 100) {
                let percentage = (completed as f32 / n as f32) * 50.0; // First 50%
                report_progress(
                    progress_callback,
                    "Neighbor Pair Sorting",
                    completed,
                    n,
                    percentage,
                    &format!("Sorted distances for {} points", completed),
                );
            }
        });

    let mut pair_neighbors = Array2::zeros((n * n_neighbors, 2));
    let pair_counter = AtomicUsize::new(0);

    pair_neighbors
        .axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(pair_index, mut pair)| {
            let i = pair_index / n_neighbors;
            let j = pair_index % n_neighbors;
            let scaled_sort = sorted_dist_indices.row(i);

            pair[0] = i as u32;
            pair[1] = neighbors[[i, scaled_sort[j] as usize]];

            // Update progress
            let completed = pair_counter.fetch_add(1, Ordering::Relaxed) + 1;
            if progress_callback.is_some() && (completed % (n * n_neighbors / 10 + 1) == 0 || n * n_neighbors <= 100) {
                let percentage = 50.0 + (completed as f32 / (n * n_neighbors) as f32) * 50.0;
                report_progress(
                    progress_callback,
                    "Neighbor Pair Generation",
                    completed,
                    n * n_neighbors,
                    percentage,
                    &format!("Generated {} neighbor pairs", completed),
                );
            }
        });

    report_progress(
        progress_callback,
        "Neighbor Pair Generation",
        n * n_neighbors,
        n * n_neighbors,
        100.0,
        &format!("Completed neighbor pair sampling: {} pairs generated", n * n_neighbors),
    );

    pair_neighbors
}

/// Assigns pairs of indices to a point.
///
/// Sets the first column to the source point index `i` and the second column
/// to each target point index from `fp_index`. This creates pairs connecting
/// point `i` to each point in `fp_index`.
///
/// # Arguments
/// * `i` - Source point index to use in first column
/// * `pairs` - 2D array view to store the pairs in, with shape (n, 2)
/// * `fp_index` - Slice of target point indices to pair with source point
fn assign_pairs(i: usize, pairs: &mut ArrayViewMut2<u32>, fp_index: &[u32]) {
    pairs
        .rows_mut()
        .into_iter()
        .zip(fp_index)
        .for_each(|(mut pair, &index)| {
            pair[0] = i as u32;
            pair[1] = index;
        });
}

/// Creates a mid-near pair by finding the second closest point from sampled candidates.
///
/// # Arguments
/// * `x` - Input data matrix where each row is a point
/// * `pair` - Output array to store the sampled pair indices
/// * `i` - Index of source point
/// * `sampled` - Array of randomly sampled candidate indices
fn sample_mn_pair_impl(
    x: ArrayView2<f32>,
    mut pair: ArrayViewMut1<u32>,
    i: usize,
    sampled: &[u32],
) {
    let mut distance_indices = [(0.0, 0); 6];
    for (&s, entry) in sampled.iter().zip(distance_indices.iter_mut()) {
        let distance = array_euclidean_distance(x.row(i), x.row(s as usize));
        *entry = (distance, s);
    }

    distance_indices.sort_unstable_by(|a, b| f32::total_cmp(&a.0, &b.0));
    let picked = distance_indices[1].1;

    pair[0] = i as u32;
    pair[1] = picked;
}

// Legacy compatibility functions - maintain original API but use deterministic implementations

/// Samples far pairs deterministically using a fixed random seed (legacy API).
///
/// # Arguments
/// * `x` - Input data matrix where each row is a point
/// * `pair_neighbors` - Matrix of nearest neighbor indices for each point
/// * `n_neighbors` - Number of nearest neighbors per point
/// * `n_fp` - Number of far pairs to sample per point
/// * `random_state` - Seed for random number generation
///
/// # Returns
/// Matrix of shape `(n * n_fp, 2)` containing sampled far point pairs
pub fn sample_fp_pair_deterministic(
    x: ArrayView2<f32>,
    pair_neighbors: ArrayView2<u32>,
    n_neighbors: usize,
    n_fp: usize,
    random_state: u64,
) -> Array2<u32> {
    let config = SamplingConfig::new(random_state);
    sample_fp_pair_with_progress(x, pair_neighbors, n_neighbors, n_fp, &config)
}

/// Samples mid-near pairs deterministically using a fixed random seed (legacy API).
///
/// # Arguments
/// * `x` - Input data matrix where each row is a point
/// * `n_mn` - Number of mid-near pairs to sample per point
/// * `random_state` - Seed for random number generation
///
/// # Returns
/// Matrix of shape `(n * n_mn, 2)` containing sampled mid-near pairs
pub fn sample_mn_pair_deterministic(
    x: ArrayView2<f32>,
    n_mn: usize,
    random_state: u64,
) -> Array2<u32> {
    let config = SamplingConfig::new(random_state);
    sample_mn_pair_with_progress(x, n_mn, &config)
}

/// Samples nearest neighbor pairs based on scaled distances (legacy API).
pub fn sample_neighbors_pair(
    x: ArrayView2<f32>,
    scaled_dist: ArrayView2<f32>,
    neighbors: ArrayView2<u32>,
    n_neighbors: usize,
) -> Array2<u32> {
    sample_neighbors_pair_with_progress(x, scaled_dist, neighbors, n_neighbors, &None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use ndarray_rand::RandomExt;
    use rand::distributions::Uniform;

    #[test]
    fn test_sample_fp_deterministic() {
        let mut rng = StdRng::from_seed([0; 32]);
        let n_samples = 5;
        let maximum = 10;
        let reject_ind = array![2, 4, 6];

        let result = sample_fp_deterministic(n_samples, maximum, reject_ind.view(), 0, &mut rng);

        assert_eq!(result.len(), n_samples);
        for &x in result.iter() {
            assert!(x < maximum);
            assert!(!reject_ind.iter().any(|&k| k == x));
        }

        // Check for uniqueness
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                assert_ne!(result[i], result[j]);
            }
        }
    }

    #[test]
    fn test_sample_fp_pair_with_progress() {
        let x = Array2::ones((100, 10));
        let pair_neighbors = Array2::from_shape_fn((1000, 2), |(i, j)| {
            if j == 0 {
                (i / 10) as u32
            } else {
                (i % 10) as u32
            }
        });
        let n_neighbors = 10;
        let n_fp = 5;
        let config = SamplingConfig::new(42);

        let result = sample_fp_pair_with_progress(
            x.view(),
            pair_neighbors.view(),
            n_neighbors,
            n_fp,
            &config,
        );

        // Check shape
        assert_eq!(result.shape(), &[500, 2]);

        // Check that each point has exactly n_fp samples
        for i in 0..100 {
            assert_eq!(
                result
                    .slice(s![i * n_fp..(i + 1) * n_fp, 0])
                    .iter()
                    .all(|&x| x == i as u32),
                true
            );
        }

        // Check that samples are not in the nearest neighbors
        for i in 0..100 {
            let neighbors = pair_neighbors.slice(s![i * n_neighbors..(i + 1) * n_neighbors, 1]);
            for j in 0..n_fp {
                assert!(!neighbors.iter().any(|x| x == &result[[i * n_fp + j, 1]]));
            }
        }

        // Check determinism
        let result2 = sample_fp_pair_with_progress(
            x.view(),
            pair_neighbors.view(),
            n_neighbors,
            n_fp,
            &config,
        );

        assert_eq!(result, result2);
    }

    #[test]
    fn test_sample_mn_pair_with_progress() {
        let x = Array2::random((1000, 20), Uniform::new(-1.0, 1.0));
        let n_mn = 5;
        let config = SamplingConfig::new(42);

        let result = sample_mn_pair_with_progress(x.view(), n_mn, &config);

        // Check shape
        assert_eq!(result.shape(), &[1000 * n_mn, 2]);

        // Check if all pairs are valid
        for pair in result.rows() {
            assert!(pair[0] < 1000);
            assert!(pair[1] < 1000);
            assert_ne!(pair[0], pair[1]);
        }

        // Check determinism
        let result2 = sample_mn_pair_with_progress(x.view(), n_mn, &config);
        assert_eq!(result, result2);
    }

    #[test]
    fn test_progress_callback() {
        let x = Array2::ones((50, 10));
        let pair_neighbors = Array2::from_shape_fn((500, 2), |(i, j)| {
            if j == 0 {
                (i / 10) as u32
            } else {
                (i % 10) as u32
            }
        });

        let progress_calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let progress_calls_clone = progress_calls.clone();

        let config = SamplingConfig::new(42).with_progress_callback(move |stage, current, total, percentage, details| {
            let mut calls = progress_calls_clone.lock().unwrap();
            calls.push((stage.to_string(), current, total, percentage, details.to_string()));
        });

        let result = sample_fp_pair_with_progress(
            x.view(),
            pair_neighbors.view(),
            10,
            5,
            &config,
        );

        // Check that progress was reported
        let calls = progress_calls.lock().unwrap();
        assert!(!calls.is_empty());

        // Check that final call shows 100% completion
        let final_call = calls.last().unwrap();
        assert_eq!(final_call.3, 100.0); // percentage
        assert!(final_call.0.contains("Far Pair Sampling"));

        // Check result is still valid
        assert_eq!(result.shape(), &[250, 2]);
    }

    #[test]
    fn test_sampling_config() {
        let config = SamplingConfig::default();
        assert_eq!(config.random_state, 42);
        assert!(!config.report_progress);
        assert!(config.progress_callback.is_none());

        let config_with_seed = SamplingConfig::new(123);
        assert_eq!(config_with_seed.random_state, 123);

        let config_with_progress = config_with_seed.with_progress_callback(|_, _, _, _, _| {});
        assert!(config_with_progress.report_progress);
        assert!(config_with_progress.progress_callback.is_some());
    }

    // Legacy API tests
    #[test]
    fn test_legacy_api_compatibility() {
        let x = Array2::ones((100, 10));
        let pair_neighbors = Array2::from_shape_fn((1000, 2), |(i, j)| {
            if j == 0 {
                (i / 10) as u32
            } else {
                (i % 10) as u32
            }
        });

        let result_legacy = sample_fp_pair_deterministic(
            x.view(),
            pair_neighbors.view(),
            10,
            5,
            42,
        );

        let config = SamplingConfig::new(42);
        let result_new = sample_fp_pair_with_progress(
            x.view(),
            pair_neighbors.view(),
            10,
            5,
            &config,
        );

        assert_eq!(result_legacy, result_new);
    }
}