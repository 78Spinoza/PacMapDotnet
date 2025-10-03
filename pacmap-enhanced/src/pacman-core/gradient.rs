//! `PaCMAP` gradient calculation implementation - Enhanced Version.
//!
//! This module provides the core gradient computation functionality for
//! `PaCMAP`'s loss function, which balances attractive forces between nearby
//! points and repulsive forces between distant points. The gradient is used to
//! iteratively optimize the low-dimensional embedding coordinates.
//!
//! This enhanced version includes deterministic computation with detailed
//! progress reporting and improved error handling.

use crate::weights::Weights;
use ndarray::{Array2, ArrayView2, Axis};
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Progress callback type for gradient computation
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

/// Configuration for gradient computation with progress reporting
#[derive(Debug, Clone)]
pub struct GradientConfig {
    /// Whether to report progress during computation
    pub report_progress: bool,
    /// Progress callback function
    pub progress_callback: Option<ProgressCallback>,
    /// Chunk size for parallel processing
    pub chunk_size: usize,
}

impl Default for GradientConfig {
    fn default() -> Self {
        Self {
            report_progress: false,
            progress_callback: None,
            chunk_size: 128 * 1024, // Default chunk size
        }
    }
}

impl GradientConfig {
    /// Create new gradient configuration with custom chunk size
    pub fn new(chunk_size: usize) -> Self {
        Self {
            report_progress: false,
            progress_callback: None,
            chunk_size,
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

/// Calculates the gradient of the `PaCMAP` loss function with enhanced progress reporting.
///
/// Computes contributions to the gradient from three types of point pairs:
/// - Nearest neighbor pairs that preserve local structure through attraction
/// - Mid-near pairs that preserve intermediate structure through attraction
/// - Far pairs that prevent collapse through repulsion
///
/// The gradient contributions are calculated in parallel using chunked
/// processing for memory efficiency. All computations are deterministic.
///
/// # Arguments
/// * `y` - Current embedding coordinates as an n × d matrix
/// * `pair_neighbors` - Matrix of nearest neighbor pair indices
/// * `pair_mn` - Matrix of mid-near pair indices
/// * `pair_fp` - Matrix of far pair indices
/// * `weights` - Weight parameters controlling the relative strength of each pair type
/// * `config` - Gradient computation configuration
///
/// # Returns
/// An (n+1) × d matrix containing:
/// - Gradient values for each point in the first n rows
/// - Total loss value in the first column of the last row
pub fn pacmap_grad_with_progress<'a>(
    y: ArrayView2<f32>,
    pair_neighbors: ArrayView2<'a, u32>,
    pair_mn: ArrayView2<'a, u32>,
    pair_fp: ArrayView2<'a, u32>,
    weights: &Weights,
    config: &GradientConfig,
) -> Array2<f32> {
    let (n, dim) = y.dim();

    report_progress(
        &config.progress_callback,
        "Gradient Computation",
        0,
        4,
        0.0,
        &format!("Starting gradient computation for {} points, {} dimensions", n, dim),
    );

    // Define parameters for each pair type:
    // (pairs, weight, denominator constant, weight constant, is_far_pair, name)
    let pair_params = [
        (pair_neighbors, weights.w_neighbors, 10.0, 20.0, false, "Neighbors"),
        (pair_mn, weights.w_mn, 10000.0, 20000.0, false, "Mid-Near"),
        (pair_fp, weights.w_fp, 1.0, 2.0, true, "Far Pairs"),
    ];

    // Calculate total pairs for progress reporting
    let total_pairs: usize = pair_params
        .iter()
        .map(|(pairs, _, _, _, _, _)| pairs.nrows())
        .sum();

    report_progress(
        &config.progress_callback,
        "Gradient Computation",
        1,
        4,
        25.0,
        &format!("Processing {} total pairs across 3 pair types", total_pairs),
    );

    // Process chunks of pairs in parallel and sum their gradient contributions
    let progress_counter = AtomicUsize::new(0);
    let progress_interval = if total_pairs > 10000 { total_pairs / 20 } else { 1 };

    let (mut grad, total_loss) = pair_params
        .iter()
        .enumerate()
        .flat_map(|(pair_type_idx, (pairs, w, denom_const, w_const, is_fp, name))| {
            pairs
                .axis_chunks_iter(Axis(0), config.chunk_size)
                .map(move |chunk| {
                    (chunk, *w, *denom_const, *w_const, *is_fp, *name, pair_type_idx)
                })
        })
        .par_bridge()
        .map(|(pairs, w, denom_const, w_const, is_fp, name, pair_type_idx)| {
            let (grad, loss) = process_pairs(y, pairs, w, denom_const, w_const, is_fp, n, dim);

            // Update progress periodically
            if config.report_progress {
                let completed = progress_counter.fetch_add(pairs.nrows(), Ordering::Relaxed) + pairs.nrows();
                if completed % progress_interval == 0 {
                    let percentage = 25.0 + (completed as f32 / total_pairs as f32) * 50.0;
                    report_progress(
                        &config.progress_callback,
                        "Pair Processing",
                        completed,
                        total_pairs,
                        percentage,
                        &format!("Processed {} pairs", completed),
                    );
                }
            }

            (grad, loss, name)
        })
        .reduce(
            || (Array2::zeros((n + 1, dim)), 0.0),
            |(mut grad1, loss1), (grad2, loss2, _name)| {
                grad1 += &grad2;
                (grad1, loss1 + loss2)
            },
        );

    report_progress(
        &config.progress_callback,
        "Gradient Computation",
        3,
        4,
        75.0,
        "Finalizing gradient computation",
    );

    // Store total loss in the last row, first column
    grad[[n, 0]] = total_loss;

    report_progress(
        &config.progress_callback,
        "Gradient Computation",
        4,
        4,
        100.0,
        &format!("Completed gradient computation, total loss: {:.6}", total_loss),
    );

    grad
}

/// Calculates the gradient of the `PaCMAP` loss function (legacy API).
///
/// # Arguments
/// * `y` - Current embedding coordinates as an n × d matrix
/// * `pair_neighbors` - Matrix of nearest neighbor pair indices
/// * `pair_mn` - Matrix of mid-near pair indices
/// * `pair_fp` - Matrix of far pair indices
/// * `weights` - Weight parameters controlling the relative strength of each pair type
///
/// # Returns
/// An (n+1) × d matrix containing:
/// - Gradient values for each point in the first n rows
/// - Total loss value in the first column of the last row
pub fn pacmap_grad<'a>(
    y: ArrayView2<f32>,
    pair_neighbors: ArrayView2<'a, u32>,
    pair_mn: ArrayView2<'a, u32>,
    pair_fp: ArrayView2<'a, u32>,
    weights: &Weights,
) -> Array2<f32> {
    let config = GradientConfig::default();
    pacmap_grad_with_progress(y, pair_neighbors, pair_mn, pair_fp, weights, &config)
}

/// Processes a chunk of point pairs to compute their gradient contributions.
///
/// For each pair, calculates:
/// - The squared distance between points in the current embedding
/// - The contribution to the loss based on pair type and distance
/// - Gradient updates that either attract or repel the points
///
/// This function is deterministic and has no side effects.
///
/// # Arguments
/// * `y` - Current embedding coordinates
/// * `pairs` - Chunk of point pair indices to process
/// * `w` - Weight factor for this pair type
/// * `denom_const` - Denominator constant in weight formula
/// * `w_const` - Weight constant in gradient formula
/// * `is_fp` - True if these are far pairs with repulsive forces
/// * `n` - Number of points
/// * `dim` - Embedding dimension
///
/// # Returns
/// A tuple containing:
/// - Gradient matrix for this chunk of pairs
/// - Loss value for this chunk
#[allow(clippy::too_many_arguments)]
fn process_pairs(
    y: ArrayView2<f32>,
    pairs: ArrayView2<u32>,
    w: f32,
    denom_const: f32,
    w_const: f32,
    is_fp: bool,
    n: usize,
    dim: usize,
) -> (Array2<f32>, f32) {
    let mut grad = Array2::zeros((n + 1, dim));
    let mut loss = 0.0;
    let mut y_ij = vec![0.0; dim];

    for pair_row in pairs.rows() {
        let i = pair_row[0] as usize;
        let j = pair_row[1] as usize;

        if i == j {
            continue;
        }

        // Calculate squared distance between points
        let mut d_ij = 1.0f32;
        for d in 0..dim {
            y_ij[d] = y[[i, d]] - y[[j, d]];
            d_ij += y_ij[d].powi(2);
        }

        if is_fp {
            // Repulsive updates for far pairs
            loss += w * (1.0 / (1.0 + d_ij));
            let w1 = w * (2.0 / (1.0 + d_ij).powi(2));

            for d in 0..dim {
                let grad_update = w1 * y_ij[d];
                grad[[i, d]] -= grad_update;
                grad[[j, d]] += grad_update;
            }
        } else {
            // Attractive updates for neighbor/mid-near pairs
            loss += w * (d_ij / (denom_const + d_ij));
            let w1 = w * (w_const / (denom_const + d_ij).powi(2));

            for d in 0..dim {
                let grad_update = w1 * y_ij[d];
                grad[[i, d]] += grad_update;
                grad[[j, d]] -= grad_update;
            }
        }
    }

    (grad, loss)
}

/// Validates gradient computation inputs
pub fn validate_gradient_inputs(
    y: ArrayView2<f32>,
    pair_neighbors: ArrayView2<u32>,
    pair_mn: ArrayView2<u32>,
    pair_fp: ArrayView2<u32>,
    weights: &Weights,
) -> Result<(), String> {
    let (n, dim) = y.dim();

    if n == 0 || dim == 0 {
        return Err("Embedding has zero dimensions".to_string());
    }

    // Check pair matrices have correct dimensions
    for (name, pairs) in [
        ("neighbors", pair_neighbors),
        ("mid-near", pair_mn),
        ("far pairs", pair_fp),
    ] {
        if pairs.ncols() != 2 {
            return Err(format!("{} pairs must have 2 columns, found {}", name, pairs.ncols()));
        }

        // Validate pair indices are within bounds
        for pair in pairs.rows() {
            let i = pair[0] as usize;
            let j = pair[1] as usize;
            if i >= n || j >= n {
                return Err(format!(
                    "{} pair contains invalid indices: ({}, {}) for {} points",
                    name, i, j, n
                ));
            }
        }
    }

    // Validate weights are finite and non-negative
    if !weights.w_neighbors.is_finite() || weights.w_neighbors < 0.0 {
        return Err(format!("Invalid neighbor weight: {}", weights.w_neighbors));
    }
    if !weights.w_mn.is_finite() || weights.w_mn < 0.0 {
        return Err(format!("Invalid mid-near weight: {}", weights.w_mn));
    }
    if !weights.w_fp.is_finite() || weights.w_fp < 0.0 {
        return Err(format!("Invalid far pair weight: {}", weights.w_fp));
    }

    // Check for NaN or infinite values in embedding
    for (i, row) in y.axis_iter(Axis(0)).enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if !val.is_finite() {
                return Err(format!("Non-finite embedding value at ({}, {}): {}", i, j, val));
            }
        }
    }

    Ok(())
}

/// Computes gradient with input validation
pub fn pacmap_grad_validated<'a>(
    y: ArrayView2<f32>,
    pair_neighbors: ArrayView2<'a, u32>,
    pair_mn: ArrayView2<'a, u32>,
    pair_fp: ArrayView2<'a, u32>,
    weights: &Weights,
    config: &GradientConfig,
) -> Result<Array2<f32>, String> {
    validate_gradient_inputs(y, pair_neighbors, pair_mn, pair_fp, weights)?;
    Ok(pacmap_grad_with_progress(y, pair_neighbors, pair_mn, pair_fp, weights, config))
}

/// Computes gradient statistics for monitoring optimization progress
pub fn compute_gradient_stats(grad: ArrayView2<f32>) -> GradientStats {
    let (n, dim) = grad.dim();
    let n_points = n - 1; // Last row contains loss

    let mut grad_norm = 0.0f32;
    let mut max_grad = 0.0f32;
    let mut min_grad = f32::INFINITY;

    // Compute gradient statistics excluding the loss row
    for i in 0..n_points {
        for j in 0..dim {
            let g = grad[[i, j]];
            grad_norm += g * g;
            max_grad = max_grad.max(g.abs());
            min_grad = min_grad.min(g.abs());
        }
    }

    grad_norm = grad_norm.sqrt();
    let mean_grad = grad_norm / (n_points * dim) as f32;

    GradientStats {
        grad_norm,
        mean_grad,
        max_grad,
        min_grad,
        total_loss: grad[[n_points, 0]],
    }
}

/// Gradient computation statistics
#[derive(Debug, Clone)]
pub struct GradientStats {
    /// L2 norm of the gradient vector
    pub grad_norm: f32,
    /// Mean absolute gradient value
    pub mean_grad: f32,
    /// Maximum absolute gradient value
    pub max_grad: f32,
    /// Minimum absolute gradient value
    pub min_grad: f32,
    /// Total loss value
    pub total_loss: f32,
}

impl std::fmt::Display for GradientStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GradientStats {{ norm: {:.6}, mean: {:.6}, max: {:.6}, min: {:.6}, loss: {:.6} }}",
            self.grad_norm, self.mean_grad, self.max_grad, self.min_grad, self.total_loss
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Zip};

    fn create_test_weights() -> Weights {
        Weights {
            w_neighbors: 0.5,
            w_mn: 0.3,
            w_fp: 0.2,
        }
    }

    #[test]
    fn test_pacmap_grad() {
        let y_test = array![
            [-0.70575494, 0.4136191],
            [-0.5127779, 1.060248],
            [-1.0165913, -1.1657093],
            [-0.8206925, 0.9737984],
            [-1.0650787, -1.5299057],
            [-0.02214996, -1.4788837],
            [0.37072298, 1.6783544],
            [-1.0666362, 1.1047112],
            [-0.2004564, -0.08376265],
            [-1.1240833, 0.10645787],
        ];

        let pair_neighbors = array![[0, 1], [2, 3], [4, 5]];
        let pair_mn = array![[6, 7], [8, 9]];
        let pair_fp = array![[0, 2], [3, 5]];

        let grad_python = array![
            [-0.020605005, -0.07924966],
            [0.014705758, 0.04927617],
            [-0.0021341527, -0.057763252],
            [0.012299123, 0.0746348],
            [-0.071347743, -0.0034904587],
            [0.067082018, 0.016592404],
            [0.00008618303, 0.000034395234],
            [-0.00008618303, -0.000034395234],
            [0.000055396682, -0.000011408921],
            [-0.000055396682, 0.000011408921],
            [0.39661729, 0.0],
        ];

        let weights = create_test_weights();

        let grad_rust = pacmap_grad(
            y_test.view(),
            pair_neighbors.view(),
            pair_mn.view(),
            pair_fp.view(),
            &weights,
        );

        Zip::from(grad_python.view())
            .and(grad_rust.view())
            .for_each(|&a, &b| {
                assert_abs_diff_eq!(a, b, epsilon = 1e-6);
            });
    }

    #[test]
    fn test_pacmap_grad_with_progress() {
        let y = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let pair_neighbors = array![[0, 1], [1, 2]];
        let pair_mn = array![[0, 2]];
        let pair_fp = array![[0, 1]];

        let weights = create_test_weights();
        let config = GradientConfig::new(100).with_progress_callback(|stage, current, total, percentage, details| {
            println!("Progress: {} - {}/{} ({:.1}%) - {}", stage, current, total, percentage, details);
        });

        let grad = pacmap_grad_with_progress(
            y.view(),
            pair_neighbors.view(),
            pair_mn.view(),
            pair_fp.view(),
            &weights,
            &config,
        );

        assert_eq!(grad.shape(), &[4, 2]); // 3 points + 1 loss row
    }

    #[test]
    fn test_gradient_config() {
        let config = GradientConfig::default();
        assert!(!config.report_progress);
        assert!(config.progress_callback.is_none());
        assert_eq!(config.chunk_size, 128 * 1024);

        let config_with_size = GradientConfig::new(512);
        assert_eq!(config_with_size.chunk_size, 512);

        let config_with_progress = config_with_size.with_progress_callback(|_, _, _, _, _| {});
        assert!(config_with_progress.report_progress);
        assert!(config_with_progress.progress_callback.is_some());
    }

    #[test]
    fn test_input_validation() {
        let y = array![[1.0, 2.0], [3.0, 4.0]];
        let valid_pairs = array![[0, 1], [1, 0]];
        let weights = create_test_weights();

        // Valid inputs
        let result = validate_gradient_inputs(
            y.view(),
            valid_pairs.view(),
            valid_pairs.view(),
            valid_pairs.view(),
            &weights,
        );
        assert!(result.is_ok());

        // Invalid embedding (empty)
        let empty_y = array![];
        let result = validate_gradient_inputs(
            empty_y.view(),
            valid_pairs.view(),
            valid_pairs.view(),
            valid_pairs.view(),
            &weights,
        );
        assert!(result.is_err());

        // Invalid pair indices
        let invalid_pairs = array![[0, 5]]; // Index 5 doesn't exist
        let result = validate_gradient_inputs(
            y.view(),
            invalid_pairs.view(),
            valid_pairs.view(),
            valid_pairs.view(),
            &weights,
        );
        assert!(result.is_err());

        // Invalid weights
        let invalid_weights = Weights {
            w_neighbors: -1.0,
            w_mn: 0.3,
            w_fp: 0.2,
        };
        let result = validate_gradient_inputs(
            y.view(),
            valid_pairs.view(),
            valid_pairs.view(),
            valid_pairs.view(),
            &invalid_weights,
        );
        assert!(result.is_err());

        // NaN in embedding
        let nan_y = array![[1.0, f32::NAN], [3.0, 4.0]];
        let result = validate_gradient_inputs(
            nan_y.view(),
            valid_pairs.view(),
            valid_pairs.view(),
            valid_pairs.view(),
            &weights,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_validated_gradient() {
        let y = array![[1.0, 2.0], [3.0, 4.0]];
        let pair_neighbors = array![[0, 1]];
        let pair_mn = array![[1, 0]];
        let pair_fp = array![[0, 1]];
        let weights = create_test_weights();
        let config = GradientConfig::default();

        let result = pacmap_grad_validated(
            y.view(),
            pair_neighbors.view(),
            pair_mn.view(),
            pair_fp.view(),
            &weights,
            &config,
        );

        assert!(result.is_ok());
        let grad = result.unwrap();
        assert_eq!(grad.shape(), &[3, 2]); // 2 points + 1 loss row
    }

    #[test]
    fn test_gradient_stats() {
        let grad = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [0.1, 0.0] // Loss row (ignored in stats except for loss)
        ];

        let stats = compute_gradient_stats(grad.view());

        assert!(stats.grad_norm > 0.0);
        assert!(stats.mean_grad > 0.0);
        assert!(stats.max_grad > 0.0);
        assert!(stats.min_grad >= 0.0);
        assert_eq!(stats.total_loss, 0.1);

        // Test display formatting
        let display_str = format!("{}", stats);
        assert!(display_str.contains("GradientStats"));
        assert!(display_str.contains("loss:"));
    }

    #[test]
    fn test_deterministic_behavior() {
        let y = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let pair_neighbors = array![[0, 1], [1, 2]];
        let pair_mn = array![[0, 2]];
        let pair_fp = array![[0, 1]];
        let weights = create_test_weights();

        // Run gradient computation multiple times
        let grad1 = pacmap_grad(
            y.view(),
            pair_neighbors.view(),
            pair_mn.view(),
            pair_fp.view(),
            &weights,
        );

        let grad2 = pacmap_grad(
            y.view(),
            pair_neighbors.view(),
            pair_mn.view(),
            pair_fp.view(),
            &weights,
        );

        // Results should be identical
        assert_eq!(grad1, grad2);
    }
}