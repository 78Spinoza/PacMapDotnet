//! Optimized distance calculations for `PaCMAP` dimensionality reduction - Enhanced Version.
//!
//! This module provides efficient implementations of distance metrics using
//! SIMD instructions where possible. It includes functions for:
//!
//! - Computing Euclidean distances between vectors using SIMD
//! - Scaling distances based on per-point sigma values
//! - Handling both contiguous and non-contiguous array views
//! - Enhanced error handling and progress reporting support

use ndarray::{Array2, ArrayView1, ArrayView2, Zip};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Progress callback type for distance operations
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

/// Scales distances between points using per-point sigma values for adaptive
/// scaling with optional progress reporting.
///
/// The scaling formula is: `scaled_dist` = dist^2 / (`sigma_i` * `sigma_j`)
/// where `sigma_i` and `sigma_j` are the scaling factors for points i and j.
///
/// # Arguments
/// * `knn_distances` - Matrix of distances to k-nearest neighbors
/// * `sig` - Scaling factor for each point
/// * `neighbors` - Indices of k-nearest neighbors for each point
/// * `progress_callback` - Optional progress callback for large matrices
///
/// # Returns
/// A matrix of scaled distances with same dimensions as `knn_distances`
pub fn scale_dist_with_progress(
    knn_distances: ArrayView2<f32>,
    sig: ArrayView1<f32>,
    neighbors: ArrayView2<u32>,
    progress_callback: Option<ProgressCallback>,
) -> Array2<f32> {
    let total_elements = knn_distances.len();

    if progress_callback.is_some() {
        report_progress(
            &progress_callback,
            "Distance Scaling",
            0,
            total_elements,
            0.0,
            "Starting distance scaling operation",
        );
    }

    let progress_counter = AtomicUsize::new(0);
    let progress_interval = total_elements / 20 + 1; // Report every 5%

    let result = Zip::indexed(knn_distances)
        .and(neighbors)
        .par_map_collect(|(i, _), knn_dist, neighbor| {
            let result = knn_dist * knn_dist / (sig[i] * sig[*neighbor as usize]);

            // Update progress periodically
            if progress_callback.is_some() {
                let completed = progress_counter.fetch_add(1, Ordering::Relaxed) + 1;
                if completed % progress_interval == 0 {
                    let percentage = (completed as f32 / total_elements as f32) * 100.0;
                    report_progress(
                        &progress_callback,
                        "Distance Scaling",
                        completed,
                        total_elements,
                        percentage,
                        &format!("Scaled {} distances", completed),
                    );
                }
            }

            result
        });

    if progress_callback.is_some() {
        report_progress(
            &progress_callback,
            "Distance Scaling",
            total_elements,
            total_elements,
            100.0,
            "Completed distance scaling operation",
        );
    }

    result
}

/// Scales distances between points using per-point sigma values (legacy API).
///
/// # Arguments
/// * `knn_distances` - Matrix of distances to k-nearest neighbors
/// * `sig` - Scaling factor for each point
/// * `neighbors` - Indices of k-nearest neighbors for each point
///
/// # Returns
/// A matrix of scaled distances with same dimensions as `knn_distances`
pub fn scale_dist(
    knn_distances: ArrayView2<f32>,
    sig: ArrayView1<f32>,
    neighbors: ArrayView2<u32>,
) -> Array2<f32> {
    scale_dist_with_progress(knn_distances, sig, neighbors, None)
}

/// Computes Euclidean distance between vectors using SIMD operations.
///
/// Processes vectors in chunks of 8 elements using SIMD instructions for
/// improved performance. Handles remaining elements sequentially.
/// This function is deterministic and has no side effects.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Panics
/// * If vectors have different lengths
pub fn simd_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    let a_chunks = a.chunks_exact(8);
    let a_remainder = a_chunks.remainder();

    let b_chunks = b.chunks_exact(8);
    let b_remainder = b_chunks.remainder();

    // Process 8 elements at a time using SIMD
    #[cfg(target_arch = "x86_64")]
    {
        use wide::f32x8;
        let mut sum_sq = f32x8::splat(0.0);
        for (a_chunk, b_chunk) in a_chunks.zip(b_chunks) {
            let diff = f32x8::from(a_chunk) - f32x8::from(b_chunk);
            sum_sq += diff * diff;
        }
        let mut total_sum_sq: f32 = sum_sq.as_array_ref().iter().sum();

        // Handle remaining elements sequentially
        for (a, b) in a_remainder.iter().zip(b_remainder) {
            let diff = a - b;
            total_sum_sq += diff * diff;
        }
        total_sum_sq.sqrt()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        // Fallback for non-x86_64 architectures
        let mut total_sum_sq: f32 = 0.0;

        // Process chunks using standard operations
        for (a_chunk, b_chunk) in a_chunks.zip(b_chunks) {
            for (a, b) in a_chunk.iter().zip(b_chunk) {
                let diff = a - b;
                total_sum_sq += diff * diff;
            }
        }

        // Handle remaining elements
        for (a, b) in a_remainder.iter().zip(b_remainder) {
            let diff = a - b;
            total_sum_sq += diff * diff;
        }

        total_sum_sq.sqrt()
    }
}

/// Computes Euclidean distance between array views with optimized path for
/// contiguous data.
///
/// Attempts to use SIMD operations on contiguous memory first, falling back to
/// slower methods for non-contiguous data with appropriate warnings.
/// This function is deterministic.
///
/// # Arguments
/// * `a` - First vector as array view
/// * `b` - Second vector as array view
///
/// # Returns
/// Euclidean distance between the vectors
pub fn array_euclidean_distance(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    let a_slice = a.as_slice();
    let b_slice = b.as_slice();

    match (a_slice, b_slice) {
        (Some(a), Some(b)) => simd_euclidean_distance(a, b),
        (Some(a), None) => {
            eprintln!("Warning: b is non-contiguous, requiring allocation to compute distance");
            simd_euclidean_distance(a, &b.to_vec())
        }
        (None, Some(b)) => {
            eprintln!("Warning: a is non-contiguous, requiring allocation to compute distance");
            simd_euclidean_distance(&a.to_vec(), b)
        }
        (None, None) => {
            eprintln!("Warning: both a and b are non-contiguous, requiring allocation to compute distance");
            simd_euclidean_distance(&a.to_vec(), &b.to_vec())
        }
    }
}

/// Computes Euclidean distance with enhanced error handling and validation.
///
/// # Arguments
/// * `a` - First vector as array view
/// * `b` - Second vector as array view
/// * `validate` - Whether to validate input vectors
///
/// # Returns
/// Result containing Euclidean distance or error message
pub fn array_euclidean_distance_validated(
    a: ArrayView1<f32>,
    b: ArrayView1<f32>,
    validate: bool,
) -> Result<f32, String> {
    if validate && a.len() != b.len() {
        return Err(format!("Vector length mismatch: {} vs {}", a.len(), b.len()));
    }

    if validate && a.len() == 0 {
        return Err("Empty vectors provided".to_string());
    }

    // Check for NaN or infinite values
    if validate {
        for (i, (&val_a, &val_b)) in a.iter().zip(b.iter()).enumerate() {
            if !val_a.is_finite() {
                return Err(format!("Non-finite value in vector a at index {}: {}", i, val_a));
            }
            if !val_b.is_finite() {
                return Err(format!("Non-finite value in vector b at index {}: {}", i, val_b));
            }
        }
    }

    Ok(array_euclidean_distance(a, b))
}

/// Batch computes Euclidean distances between pairs of vectors.
///
/// # Arguments
/// * `vector_pairs` - Iterator of vector pairs
/// * `progress_callback` - Optional progress callback
///
/// # Returns
/// Vector of computed distances
pub fn batch_euclidean_distances<I>(
    vector_pairs: I,
    progress_callback: Option<ProgressCallback>,
) -> Vec<f32>
where
    I: Iterator<Item = (ArrayView1<f32>, ArrayView1<f32>)>,
{
    let pairs: Vec<_> = vector_pairs.collect();
    let total_pairs = pairs.len();

    if progress_callback.is_some() {
        report_progress(
            &progress_callback,
            "Batch Distance Computation",
            0,
            total_pairs,
            0.0,
            "Starting batch distance computation",
        );
    }

    let progress_counter = AtomicUsize::new(0);
    let progress_interval = total_pairs / 20 + 1; // Report every 5%

    let distances: Vec<f32> = pairs
        .into_par_iter()
        .enumerate()
        .map(|(index, (a, b))| {
            let distance = array_euclidean_distance(a, b);

            // Update progress periodically
            if progress_callback.is_some() {
                let completed = progress_counter.fetch_add(1, Ordering::Relaxed) + 1;
                if completed % progress_interval == 0 {
                    let percentage = (completed as f32 / total_pairs as f32) * 100.0;
                    report_progress(
                        &progress_callback,
                        "Batch Distance Computation",
                        completed,
                        total_pairs,
                        percentage,
                        &format!("Computed {} distances", completed),
                    );
                }
            }

            distance
        })
        .collect();

    if progress_callback.is_some() {
        report_progress(
            &progress_callback,
            "Batch Distance Computation",
            total_pairs,
            total_pairs,
            100.0,
            "Completed batch distance computation",
        );
    }

    distances
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use ndarray::Array2;

    #[test]
    fn test_scale_dist() {
        let knn_distances = array![[1.0, 2.0], [3.0, 4.0]];
        let sig = array![1.0, 2.0];
        let neighbors = array![[0, 1], [1, 0]];

        let expected: Array2<f32> = array![[1.0, 2.0], [2.25, 8.0]];
        let result = scale_dist(knn_distances.view(), sig.view(), neighbors.view());

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scale_dist_with_progress() {
        let knn_distances = array![[1.0, 2.0], [3.0, 4.0]];
        let sig = array![1.0, 2.0];
        let neighbors = array![[0, 1], [1, 0]];

        let progress_calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let progress_calls_clone = progress_calls.clone();

        let callback = move |stage: &str, current: usize, total: usize, percentage: f32, details: &str| {
            let mut calls = progress_calls_clone.lock().unwrap();
            calls.push((stage.to_string(), current, total, percentage, details.to_string()));
        };

        let result = scale_dist_with_progress(
            knn_distances.view(),
            sig.view(),
            neighbors.view(),
            Some(Box::new(callback)),
        );

        // Check result is correct
        let expected: Array2<f32> = array![[1.0, 2.0], [2.25, 8.0]];
        assert_eq!(result, expected);

        // Check that progress was reported
        let calls = progress_calls.lock().unwrap();
        assert!(!calls.is_empty());
    }

    #[test]
    fn test_simd_euclidean_distance() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let distance = simd_euclidean_distance(&a, &b);
        let expected = 8.0f32.sqrt(); // sqrt((1^2)*8) = sqrt(8)

        assert!((distance - expected).abs() < 1e-6);
    }

    #[test]
    fn test_array_euclidean_distance() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![2.0, 3.0, 4.0];

        let distance = array_euclidean_distance(a.view(), b.view());
        let expected = 3.0f32.sqrt(); // sqrt(1^2 + 1^2 + 1^2)

        assert!((distance - expected).abs() < 1e-6);
    }

    #[test]
    fn test_array_euclidean_distance_validated() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![2.0, 3.0, 4.0];

        let result = array_euclidean_distance_validated(a.view(), b.view(), true);
        assert!(result.is_ok());
        assert!((result.unwrap() - 3.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_array_euclidean_distance_validated_errors() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![2.0, 3.0]; // Different length

        let result = array_euclidean_distance_validated(a.view(), b.view(), true);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("length mismatch"));

        // Test with NaN value
        let c = array![1.0, f32::NAN, 3.0];
        let d = array![2.0, 3.0, 4.0];

        let result = array_euclidean_distance_validated(c.view(), d.view(), true);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Non-finite"));
    }

    #[test]
    fn test_batch_euclidean_distances() {
        let a1 = array![1.0, 2.0];
        let b1 = array![2.0, 3.0];
        let a2 = array![1.0, 1.0];
        let b2 = array![2.0, 2.0];

        let pairs = vec![(a1.view(), b1.view()), (a2.view(), b2.view())];
        let distances = batch_euclidean_distances(pairs.into_iter(), None);

        assert_eq!(distances.len(), 2);
        assert!((distances[0] - 2.0f32.sqrt()).abs() < 1e-6);
        assert!((distances[1] - 2.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_batch_euclidean_distances_with_progress() {
        let a1 = array![1.0, 2.0];
        let b1 = array![2.0, 3.0];
        let a2 = array![1.0, 1.0];
        let b2 = array![2.0, 2.0];

        let pairs = vec![(a1.view(), b1.view()), (a2.view(), b2.view())];

        let progress_calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let progress_calls_clone = progress_calls.clone();

        let callback = move |stage: &str, current: usize, total: usize, percentage: f32, details: &str| {
            let mut calls = progress_calls_clone.lock().unwrap();
            calls.push((stage.to_string(), current, total, percentage, details.to_string()));
        };

        let distances = batch_euclidean_distances(pairs.into_iter(), Some(Box::new(callback)));

        assert_eq!(distances.len(), 2);

        // Check that progress was reported
        let calls = progress_calls.lock().unwrap();
        assert!(!calls.is_empty());
        assert!(calls[0].0.contains("Batch Distance Computation"));
    }

    #[test]
    fn test_distance_symmetry() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![2.0, 3.0, 4.0];

        let distance_ab = array_euclidean_distance(a.view(), b.view());
        let distance_ba = array_euclidean_distance(b.view(), a.view());

        assert!((distance_ab - distance_ba).abs() < f32::EPSILON);
    }

    #[test]
    fn test_zero_distance_when_identical() {
        let a = array![1.0, 2.0, 3.0];
        let distance = array_euclidean_distance(a.view(), a.view());
        assert_eq!(distance, 0.0);
    }

    #[test]
    fn test_large_vector_distance() {
        let a: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..1000).map(|i| (i + 1) as f32).collect();

        let distance = simd_euclidean_distance(&a, &b);
        let expected = (1000.0_f32).sqrt(); // sqrt(1^2 * 1000)

        assert!((distance - expected).abs() < 1e-4);
    }
}