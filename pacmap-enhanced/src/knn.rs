//! K-nearest neighbor computation for `PaCMAP` dimensionality reduction - Deterministic Enhanced Version.
//!
//! Efficiently computes k-nearest neighbors for high-dimensional data points
//! using SIMD-accelerated Euclidean distance calculations and parallel
//! processing. Provides exact neighbor finding algorithms with deterministic
//! behavior and enhanced progress reporting.
//!
//! The neighbors and distances are used by `PaCMAP` to preserve local structure
//! during the dimensionality reduction process.
//!
//! This implementation prioritizes deterministic exact computation over approximate
//! methods to ensure reproducible results.

use crate::distance::simd_euclidean_distance;
use ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;
use std::cmp::min;
use std::sync::atomic::{AtomicUsize, Ordering};
use thiserror::Error;

/// Progress callback type for KNN operations
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

/// Configuration for KNN computation with progress reporting
pub struct KnnConfig {
    /// Number of nearest neighbors to find
    pub k: usize,
    /// Whether to report progress during computation
    pub report_progress: bool,
    /// Progress callback function
    pub progress_callback: Option<ProgressCallback>,
}

impl Default for KnnConfig {
    fn default() -> Self {
        Self {
            k: 15,
            report_progress: false,
            progress_callback: None,
        }
    }
}

impl KnnConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.k == 0 {
            return Err("k must be greater than 0".to_string());
        }
        Ok(())
    }
}

impl KnnConfig {
    /// Create new KNN configuration
    pub fn new(k: usize) -> Self {
        Self {
            k,
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

impl Clone for KnnConfig {
    fn clone(&self) -> Self {
        Self {
            k: self.k,
            report_progress: self.report_progress,
            progress_callback: None, // Can't clone closure, so reset to None
        }
    }
}

impl std::fmt::Debug for KnnConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KnnConfig")
            .field("k", &self.k)
            .field("report_progress", &self.report_progress)
            .field("progress_callback", &self.progress_callback.is_some())
            .finish()
    }
}

/// Finds k-nearest neighbors exactly by computing all pairwise distances with enhanced progress reporting.
///
/// Uses parallel processing and SIMD to accelerate the computation of pairwise
/// distances. This is a deterministic algorithm that produces reproducible results.
/// Appropriate for small to medium datasets where exact neighbors are desired.
///
/// # Arguments
/// * `data` - Input data matrix where each row is a point
/// * `config` - KNN configuration with parameters and progress reporting
///
/// # Returns
/// A tuple containing:
/// * `neighbor_array` - Matrix of shape `(n, min(k, n-1))` containing indices
///   of nearest neighbors
/// * `distance_array` - Matrix of shape `(n, min(k, n-1))` containing distances
///   to nearest neighbors
///
/// Returns empty arrays if input is empty. For a single input point, arrays
/// will have 0 columns.
pub fn find_k_nearest_neighbors_with_progress(
    data: ArrayView2<f32>,
    config: &KnnConfig,
) -> (Array2<u32>, Array2<f32>) {
    let n = data.nrows();

    report_progress(
        &config.progress_callback,
        "KNN Computation",
        0,
        3, // We'll report 3 major stages
        0.0,
        &format!("Starting KNN computation for {} points, k={}", n, config.k),
    );

    // Handle empty input case
    if n == 0 {
        report_progress(
            &config.progress_callback,
            "KNN Computation",
            3,
            3,
            100.0,
            "Completed KNN computation (empty input)",
        );
        return (Array2::<u32>::zeros((0, 0)), Array2::<f32>::zeros((0, 0)));
    }

    // Limit k to available neighbors
    let k = min(config.k, n - 1);
    let total_pairs = n * (n - 1) / 2; // Number of unique pairs

    report_progress(
        &config.progress_callback,
        "KNN Computation",
        1,
        3,
        33.3,
        &format!("Computing pairwise distances for {} pairs", total_pairs),
    );

    // Check if deterministic execution is forced via environment variable
    let force_deterministic = std::env::var("PACMAP_DETERMINISTIC")
        .unwrap_or_default()
        .parse::<bool>()
        .unwrap_or(false);

    // Compute upper triangular pairwise distances using SIMD
    let distances: Vec<_> = if force_deterministic {
        // Use sequential iteration for deterministic results
        report_progress(
            &config.progress_callback,
            "Distance Computation",
            0,
            n,
            33.3,
            &format!("Computing distances sequentially (deterministic mode) for {} pairs", total_pairs),
        );

        let mut completed = 0;
        (0..n)
            .flat_map(|i| {
                let row_i = data.row(i);
                let a_slice = row_i.as_slice().unwrap_or(&[]);
                (i + 1..n)
                    .map(move |j| {
                        let row_j = data.row(j);
                        let b_slice = row_j.as_slice().unwrap_or(&[]);
                        let dist = simd_euclidean_distance(a_slice, b_slice);

                        // Update progress periodically (simplified for sequential)
                        if config.report_progress && j % 100 == 0 {
                            completed += 1;
                            let percentage = 33.3 + (completed as f32 / n as f32) * 33.3;
                            report_progress(
                                &config.progress_callback,
                                "Distance Computation",
                                completed,
                                n,
                                percentage,
                                &format!("Computed distance {} of {}", completed, n),
                            );
                        }

                        (i as u32, j as u32, dist)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    } else {
        // Use parallel iteration (default - faster but non-deterministic)
        report_progress(
            &config.progress_callback,
            "Distance Computation",
            0,
            n,
            33.3,
            &format!("Computing distances in parallel (fast mode) for {} pairs", total_pairs),
        );

        use std::sync::Arc;
        let progress_counter = Arc::new(AtomicUsize::new(0));
        let progress_interval = if n > 1000 { n / 20 } else { 1 };

        (0..n)
            .into_par_iter()
            .flat_map(|i| {
            let row_i = data.row(i);
            let a_slice = row_i.as_slice().unwrap_or(&[]);
            let progress_counter_clone = progress_counter.clone();
            (i + 1..n)
                .map(move |j| {
                    let row_j = data.row(j);
                    let b_slice = row_j.as_slice().unwrap_or(&[]);
                    let dist = simd_euclidean_distance(a_slice, b_slice);

                    // Update progress periodically
                    if config.report_progress && i % progress_interval == 0 {
                        let completed = progress_counter_clone.fetch_add(1, Ordering::Relaxed) + 1;
                        let percentage = 33.3 + (completed as f32 / n as f32) * 33.3;
                        report_progress(
                            &config.progress_callback,
                            "Distance Computation",
                            completed,
                            n,
                            percentage,
                            &format!("Computed distances for {} of {} point pairs", completed, n),
                        );
                    }

                    (i as u32, j as u32, dist)
                })
                .collect::<Vec<_>>()
        })
        .collect()
    };

    report_progress(
        &config.progress_callback,
        "KNN Computation",
        2,
        3,
        66.6,
        &format!("Sorting {} computed distances", distances.len()),
    );

    // Sort distances to find k nearest neighbors (deterministic sort)
    let mut distances_sorted = distances;
    distances_sorted.par_sort_unstable_by(|a, b| f32::total_cmp(&a.2, &b.2));

    report_progress(
        &config.progress_callback,
        "KNN Computation",
        3,
        3,
        90.0,
        "Building neighbor and distance arrays",
    );

    // Initialize output arrays
    let mut neighbor_array = Array2::<u32>::zeros((n, k));
    let mut distance_array = Array2::<f32>::from_elem((n, k), f32::MAX);
    let mut counts = vec![0; n];

    // Fill arrays with k nearest neighbors for each point
    // Each distance pair (i,j) counts as a neighbor for both i and j
    for &(i, j, distance) in &distances_sorted {
        let ix = i as usize;
        let jx = j as usize;

        if counts[ix] < k {
            neighbor_array[(ix, counts[ix])] = j;
            distance_array[(ix, counts[ix])] = distance;
            counts[ix] += 1;
        }

        if counts[jx] < k {
            neighbor_array[(jx, counts[jx])] = i;
            distance_array[(jx, counts[jx])] = distance;
            counts[jx] += 1;
        }

        // Early exit if all points have k neighbors
        if counts.iter().all(|&count| count >= k) {
            break;
        }
    }

    report_progress(
        &config.progress_callback,
        "KNN Computation",
        3,
        3,
        100.0,
        &format!("Completed KNN computation: found {} neighbors for {} points", k, n),
    );

    (neighbor_array, distance_array)
}

/// Finds k-nearest neighbors exactly by computing all pairwise distances (legacy API).
///
/// Uses parallel processing and SIMD to accelerate the computation of pairwise
/// distances. Appropriate for small to medium datasets where exact neighbors
/// are desired.
///
/// # Arguments
/// * `data` - Input data matrix where each row is a point
/// * `k` - Number of nearest neighbors to find per point
///
/// # Returns
/// A tuple containing:
/// * `neighbor_array` - Matrix of shape `(n, min(k, n-1))` containing indices
///   of nearest neighbors
/// * `distance_array` - Matrix of shape `(n, min(k, n-1))` containing distances
///   to nearest neighbors
///
/// Returns empty arrays if input is empty. For a single input point, arrays
/// will have 0 columns.
#[must_use]
pub fn find_k_nearest_neighbors(data: ArrayView2<f32>, k: usize) -> (Array2<u32>, Array2<f32>) {
    let config = KnnConfig::new(k);
    find_k_nearest_neighbors_with_progress(data, &config)
}

/// Validates input data for KNN computation
pub fn validate_knn_input(data: ArrayView2<f32>) -> Result<(), String> {
    let n = data.nrows();
    let d = data.ncols();

    if d == 0 {
        return Err("Data has zero dimensions".to_string());
    }

    // Check for NaN or infinite values
    for (i, row) in data.axis_iter(Axis(0)).enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if !val.is_finite() {
                return Err(format!("Non-finite value at position ({}, {}): {}", i, j, val));
            }
        }
    }

    // Check for excessive variance that might cause numerical issues
    for j in 0..d {
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..n {
            let val = data[[i, j]];
            min_val = min_val.min(val);
            max_val = max_val.max(val);
            sum += val;
            count += 1;
        }

        let range = max_val - min_val;
        let mean = sum / count as f32;

        if range > 1e12 {
            return Err(format!(
                "Excessive range in dimension {}: {} to {} (range: {})",
                j, min_val, max_val, range
            ));
        }

        if mean.abs() > 1e10 {
            return Err(format!(
                "Excessive mean in dimension {}: {}",
                j, mean
            ));
        }
    }

    Ok(())
}

/// Computes KNN with input validation
pub fn find_k_nearest_neighbors_validated(
    data: ArrayView2<f32>,
    config: &KnnConfig,
) -> Result<(Array2<u32>, Array2<f32>), String> {
    validate_knn_input(data)?;
    Ok(find_k_nearest_neighbors_with_progress(data, config))
}

/// Errors that can occur during k-nearest neighbor computation.
#[derive(Debug, Error)]
pub enum KnnError {
    /// Invalid input data
    #[error("invalid input data: {0}")]
    InvalidInput(String),

    /// Failed to construct the spatial index (for future approximate implementations)
    #[error("failed to construct index: {0}")]
    Construction(String),

    /// Failed to allocate memory for the index (for future approximate implementations)
    #[error("failed to reserve space for vectors: {0}")]
    Reservation(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};
    use std::f32::consts::FRAC_1_SQRT_2;

    #[test]
    fn test_empty_embeddings() {
        let embeddings = Array2::<f32>::zeros((0, 128));
        let k = 5;
        let (neighbor_indices, distances) = find_k_nearest_neighbors(embeddings.view(), k);
        assert_eq!(neighbor_indices.shape(), &[0, 0]);
        assert_eq!(distances.shape(), &[0, 0]);
    }

    #[test]
    fn test_single_embedding() {
        let embeddings = array![[1.0, 0.0, 0.0]];
        let k = 1;
        let (neighbor_indices, distances) = find_k_nearest_neighbors(embeddings.view(), k);
        assert_eq!(neighbor_indices.shape(), &[1, 0]);
        assert_eq!(distances.shape(), &[1, 0]);
    }

    #[test]
    fn test_k_zero() {
        let embeddings = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
        let k = 0;
        let (neighbor_indices, distances) = find_k_nearest_neighbors(embeddings.view(), k);
        assert_eq!(neighbor_indices.shape(), &[3, 0]);
        assert_eq!(distances.shape(), &[3, 0]);
    }

    #[test]
    fn test_k_exceeds() {
        let embeddings: Array2<f32> = array![[1.0], [2.0], [3.0]];
        let k = 5;
        let (neighbor_indices, distances) = find_k_nearest_neighbors(embeddings.view(), k);
        assert_eq!(neighbor_indices.shape(), &[3, 2]);
        assert_eq!(distances.shape(), &[3, 2]);
        assert!(neighbor_indices.iter().all(|&idx| idx < 3));
        assert!(distances.iter().all(|&d| d >= 0.0));
    }

    #[test]
    fn test_normal_case() {
        let embeddings: Array2<f32> = array![
            [1.0, 0.0], // E0
            [0.0, 1.0], // E1
            [0.5, 0.5]  // E2
        ];

        let k = 2;
        let (neighbor_indices, distances) = find_k_nearest_neighbors(embeddings.view(), k);
        assert_eq!(neighbor_indices.shape(), &[3, 2]);
        assert_eq!(distances.shape(), &[3, 2]);

        let expected = vec![
            (0, vec![(2, FRAC_1_SQRT_2), (1, 1.4142)]),
            (1, vec![(2, FRAC_1_SQRT_2), (0, 1.4142)]),
            (2, vec![(0, FRAC_1_SQRT_2), (1, FRAC_1_SQRT_2)]),
        ];

        check_neighbors_and_distances(&neighbor_indices, &distances, &expected);
    }

    #[test]
    fn test_duplicate_embeddings() {
        let embeddings: Array2<f32> = array![
            [1.0, 0.0], // E0
            [1.0, 0.0], // E1 (duplicate of E0)
            [0.0, 1.0]  // E2
        ];

        let k = 2;
        let (neighbor_indices, distances) = find_k_nearest_neighbors(embeddings.view(), k);
        assert_eq!(neighbor_indices.shape(), &[3, 2]);
        assert_eq!(distances.shape(), &[3, 2]);

        let expected = vec![
            (0, vec![(1, 0.0), (2, 1.4142)]),
            (1, vec![(0, 0.0), (2, 1.4142)]),
            (2, vec![(0, 1.4142), (1, 1.4142)]),
        ];

        check_neighbors_and_distances(&neighbor_indices, &distances, &expected);
    }

    #[test]
    fn test_negative_components() {
        let embeddings: Array2<f32> = array![
            [1.0, 0.0],  // E0
            [-1.0, 0.0], // E1
            [0.0, 1.0]   // E2
        ];

        let k = 2;
        let (neighbor_indices, distances) = find_k_nearest_neighbors(embeddings.view(), k);
        assert_eq!(neighbor_indices.shape(), &[3, 2]);
        assert_eq!(distances.shape(), &[3, 2]);

        let expected = vec![
            (0, vec![(2, 1.4142), (1, 2.0)]),
            (1, vec![(2, 1.4142), (0, 2.0)]),
            (2, vec![(0, 1.4142), (1, 1.4142)]),
        ];

        check_neighbors_and_distances(&neighbor_indices, &distances, &expected);
    }

    #[test]
    fn test_progress_callback() {
        let embeddings = array![[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
        let config = KnnConfig::new(2).with_progress_callback(|stage, current, total, percentage, details| {
            // Just capture that the callback was called
            println!("Progress: {} - {}/{} ({:.1}%) - {}", stage, current, total, percentage, details);
        });

        let (neighbor_indices, distances) = find_k_nearest_neighbors_with_progress(embeddings.view(), &config);

        assert_eq!(neighbor_indices.shape(), &[3, 2]);
        assert_eq!(distances.shape(), &[3, 2]);
    }

    #[test]
    fn test_knn_config() {
        let config = KnnConfig::default();
        assert_eq!(config.k, 15);
        assert!(!config.report_progress);
        assert!(config.progress_callback.is_none());

        let config_with_k = KnnConfig::new(10);
        assert_eq!(config_with_k.k, 10);

        let config_with_progress = config_with_k.with_progress_callback(|_, _, _, _, _| {});
        assert!(config_with_progress.report_progress);
        assert!(config_with_progress.progress_callback.is_some());
    }

    #[test]
    fn test_input_validation() {
        // Valid input
        let valid_data = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(validate_knn_input(valid_data.view()).is_ok());

        // Empty dimensions
        let empty_dim_data = Array2::<f32>::zeros((2, 0));
        assert!(validate_knn_input(empty_dim_data.view()).is_err());

        // NaN values
        let nan_data = array![[1.0, f32::NAN], [3.0, 4.0]];
        assert!(validate_knn_input(nan_data.view()).is_err());

        // Infinite values
        let inf_data = array![[1.0, f32::INFINITY], [3.0, 4.0]];
        assert!(validate_knn_input(inf_data.view()).is_err());
    }

    #[test]
    fn test_deterministic_behavior() {
        let embeddings = array![
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 1.0],
            [0.5, 0.5, 1.5],
            [1.5, 1.0, 0.5]
        ];

        let config = KnnConfig::new(3);

        // Run computation multiple times to ensure deterministic results
        let (neighbors1, distances1) = find_k_nearest_neighbors_with_progress(embeddings.view(), &config);
        let (neighbors2, distances2) = find_k_nearest_neighbors_with_progress(embeddings.view(), &config);

        // Results should be identical
        assert_eq!(neighbors1, neighbors2);
        assert_eq!(distances1, distances2);
    }

    #[test]
    fn test_validated_knn() {
        let embeddings = array![[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
        let config = KnnConfig::new(2);

        let result = find_k_nearest_neighbors_validated(embeddings.view(), &config);
        assert!(result.is_ok());

        let (neighbors, distances) = result.unwrap();
        assert_eq!(neighbors.shape(), &[3, 2]);
        assert_eq!(distances.shape(), &[3, 2]);

        // Test with invalid data
        let invalid_data = array![[1.0, f32::NAN], [0.0, 1.0]];
        let result = find_k_nearest_neighbors_validated(invalid_data.view(), &config);
        assert!(result.is_err());
    }

    fn check_neighbors_and_distances(
        neighbor_indices: &Array2<u32>,
        distances: &Array2<f32>,
        expected: &Vec<(usize, Vec<(usize, f32)>)>,
    ) {
        for &(point_idx, ref expected_neighbors) in expected {
            let neighbors = neighbor_indices.row(point_idx);
            let neighbor_distances = distances.row(point_idx);

            assert_eq!(
                neighbors.len(),
                expected_neighbors.len(),
                "Mismatch in number of neighbors for point {}",
                point_idx
            );

            let mut neighbor_info: Vec<(usize, f32)> = neighbors
                .iter()
                .zip(neighbor_distances.iter())
                .map(|(&idx, &dist)| (idx as usize, dist))
                .collect();

            let mut expected_sorted = expected_neighbors.clone();
            neighbor_info.sort_by_key(|&(idx, _)| idx);
            expected_sorted.sort_by_key(|&(idx, _)| idx);

            for (&(neighbor_idx, distance), &(exp_neighbor_idx, exp_distance)) in
                neighbor_info.iter().zip(expected_sorted.iter())
            {
                assert_eq!(
                    neighbor_idx, exp_neighbor_idx,
                    "Mismatch in neighbor index for point {}",
                    point_idx
                );
                assert!((distance - exp_distance).abs() < 1e-4);
            }
        }
    }
}