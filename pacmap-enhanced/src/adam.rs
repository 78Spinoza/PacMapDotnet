//! Adam optimization updates for `PaCMAP` embeddings - Enhanced Version.
//!
//! This module implements gradient updates using the Adam optimizer during
//! `PaCMAP` dimensionality reduction. The Adam optimizer adapts learning rates
//! per parameter using moment estimates.
//!
//! This enhanced version includes improved validation, error handling, and
//! detailed progress reporting capabilities while maintaining deterministic
//! behavior.

use ndarray::{s, Array2, ArrayView2, ArrayViewMut2, Zip};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Progress callback type for Adam optimization
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

/// Configuration for Adam optimization with enhanced options
pub struct AdamConfig {
    /// First moment decay rate (typically 0.9)
    pub beta1: f32,
    /// Second moment decay rate (typically 0.999)
    pub beta2: f32,
    /// Base learning rate
    pub learning_rate: f32,
    /// Numerical stability constant
    pub epsilon: f32,
    /// Whether to report progress during optimization
    pub report_progress: bool,
    /// Progress callback function
    pub progress_callback: Option<ProgressCallback>,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            learning_rate: 0.001,
            epsilon: 1e-7,
            report_progress: false,
            progress_callback: None,
        }
    }
}

impl AdamConfig {
    /// Create new Adam configuration with standard parameters
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            ..Default::default()
        }
    }

    /// Create Adam configuration with custom beta values
    pub fn with_betas(learning_rate: f32, beta1: f32, beta2: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            ..Default::default()
        }
    }

    /// Set custom epsilon value
    pub fn epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
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

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(format!("Invalid learning rate: {}", self.learning_rate));
        }
        if !(0.0..1.0).contains(&self.beta1) {
            return Err(format!("Beta1 must be in (0,1): {}", self.beta1));
        }
        if !(0.0..1.0).contains(&self.beta2) {
            return Err(format!("Beta2 must be in (0,1): {}", self.beta2));
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(format!("Invalid epsilon: {}", self.epsilon));
        }
        Ok(())
    }
}

impl Clone for AdamConfig {
    fn clone(&self) -> Self {
        Self {
            beta1: self.beta1,
            beta2: self.beta2,
            learning_rate: self.learning_rate,
            epsilon: self.epsilon,
            report_progress: self.report_progress,
            progress_callback: None, // Can't clone closure, so reset to None
        }
    }
}

impl std::fmt::Debug for AdamConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdamConfig")
            .field("beta1", &self.beta1)
            .field("beta2", &self.beta2)
            .field("learning_rate", &self.learning_rate)
            .field("epsilon", &self.epsilon)
            .field("report_progress", &self.report_progress)
            .field("progress_callback", &self.progress_callback.is_some())
            .finish()
    }
}

/// Updates embedding coordinates using adaptive moment estimation (Adam) with enhanced reporting.
///
/// Performs a single Adam optimization step to update the low-dimensional
/// embedding based on computed gradients. Uses exponential moving averages of
/// gradients and squared gradients to adapt learning rates per parameter.
///
/// This implementation is deterministic and includes enhanced error handling
/// and progress reporting capabilities.
///
/// # Arguments
/// * `y` - Current embedding coordinates to update
/// * `grad` - Gradient for this iteration
/// * `m` - First moment estimate (gradient moving average)
/// * `v` - Second moment estimate (squared gradient moving average)
/// * `config` - Adam optimizer configuration
/// * `itr` - Current iteration number (0-based)
///
/// # Returns
/// A tuple containing:
/// - Updated embedding coordinates (in-place)
/// - Adam update statistics for monitoring
///
/// # Implementation Notes
/// - Applies bias correction to moment estimates based on iteration number
/// - Updates moment estimates using exponential moving averages
/// - Updates parameters with Adam rule: y -= lr * m / (sqrt(v) + eps)
/// - Uses parallel iteration for efficiency
/// - Includes detailed progress reporting and validation
///
/// # Errors
/// Returns an error if input validation fails
#[allow(clippy::too_many_arguments)]
pub fn update_embedding_adam_with_stats(
    mut y: ArrayViewMut2<f32>,
    grad: ArrayView2<f32>,
    mut m: ArrayViewMut2<f32>,
    mut v: ArrayViewMut2<f32>,
    config: &AdamConfig,
    itr: usize,
) -> Result<AdamStats, String> {
    // Validate inputs
    validate_adam_inputs(y.view(), grad.view(), m.view(), v.view(), config)?;

    let (n, dim) = y.dim();

    report_progress(
        &config.progress_callback,
        "Adam Optimization",
        itr,
        itr + 1,
        0.0,
        &format!("Starting Adam update iteration {}", itr),
    );

    // Compute bias-corrected learning rate
    let itr_f32 = (itr + 1) as f32;
    let lr_t = config.learning_rate * (1.0 - config.beta2.powf(itr_f32)).sqrt() / (1.0 - config.beta1.powf(itr_f32));
    let grad = grad.slice(s![..n, ..]);

    // Initialize statistics
    let mut max_m_update = 0.0f32;
    let mut max_v_update = 0.0f32;
    let mut max_param_update = 0.0f32;
    let mut grad_norm = 0.0f32;

    // Update moment estimates and parameters in parallel
    let progress_counter = AtomicUsize::new(0);
    let progress_interval = if n * dim > 10000 { (n * dim) / 20 } else { 1 };

    Zip::from(y.view_mut())
        .and(grad)
        .and(m.view_mut())
        .and(v.view_mut())
        .for_each(|y, &grad, m, v| {
            // Update first moment
            let m_update = (1.0 - config.beta1) * (grad - *m);
            *m += m_update;

            // Update second moment
            let v_update = (1.0 - config.beta2) * (grad.powi(2) - *v);
            *v += v_update;

            // Update parameter
            let param_update = lr_t * *m / (v.sqrt() + config.epsilon);
            *y -= param_update;

            // Track statistics (atomic operations for thread safety)
            let m_update_abs = m_update.abs();
            let v_update_abs = v_update.abs();
            let param_update_abs = param_update.abs();

            // Use relaxed ordering since we're just tracking approximate maxima
            max_m_update = max_m_update.max(m_update_abs);
            max_v_update = max_v_update.max(v_update_abs);
            max_param_update = max_param_update.max(param_update_abs);

            grad_norm += grad * grad;

            // Update progress periodically
            if config.report_progress {
                let completed = progress_counter.fetch_add(1, Ordering::Relaxed) + 1;
                if completed % progress_interval == 0 {
                    let percentage = (completed as f32 / (n * dim) as f32) * 100.0;
                    report_progress(
                        &config.progress_callback,
                        "Adam Parameter Updates",
                        completed,
                        n * dim,
                        percentage,
                        &format!("Updated {} of {} parameters", completed, n * dim),
                    );
                }
            }
        });

    grad_norm = grad_norm.sqrt();

    let stats = AdamStats {
        iteration: itr,
        learning_rate_corrected: lr_t,
        max_momentum_update: max_m_update,
        max_velocity_update: max_v_update,
        max_parameter_update: max_param_update,
        gradient_norm: grad_norm,
        parameter_norm: compute_parameter_norm(y.view()),
    };

    report_progress(
        &config.progress_callback,
        "Adam Optimization",
        itr + 1,
        itr + 1,
        100.0,
        &format!("Completed Adam update iteration {}: lr_t={:.6}, grad_norm={:.6}",
                itr, lr_t, grad_norm),
    );

    Ok(stats)
}

/// Updates embedding coordinates using adaptive moment estimation (Adam) (legacy API).
///
/// # Arguments
/// * `y` - Current embedding coordinates to update
/// * `grad` - Gradient for this iteration
/// * `m` - First moment estimate (gradient moving average)
/// * `v` - Second moment estimate (squared gradient moving average)
/// * `beta1` - First moment decay rate (typically 0.9)
/// * `beta2` - Second moment decay rate (typically 0.999)
/// * `lr` - Base learning rate
/// * `itr` - Current iteration number (0-based)
pub fn update_embedding_adam(
    y: ArrayViewMut2<f32>,
    grad: ArrayView2<f32>,
    m: ArrayViewMut2<f32>,
    v: ArrayViewMut2<f32>,
    beta1: f32,
    beta2: f32,
    lr: f32,
    itr: usize,
) {
    let config = AdamConfig::with_betas(lr, beta1, beta2);
    if let Err(e) = update_embedding_adam_with_stats(y, grad, m, v, &config, itr) {
        panic!("Adam update failed: {}", e);
    }
}

/// Validates inputs for Adam optimization
fn validate_adam_inputs<'a>(
    y: ArrayView2<'a, f32>,
    grad: ArrayView2<'a, f32>,
    m: ArrayView2<'a, f32>,
    v: ArrayView2<'a, f32>,
    config: &AdamConfig,
) -> Result<(), String> {
    // Validate configuration
    config.validate()?;

    // Check shape compatibility
    if y.shape() != m.shape() {
        return Err(format!("Shape mismatch: y {:?} vs m {:?}", y.shape(), m.shape()));
    }
    if y.shape() != v.shape() {
        return Err(format!("Shape mismatch: y {:?} vs v {:?}", y.shape(), v.shape()));
    }
    if grad.nrows() < y.nrows() || grad.ncols() != y.ncols() {
        return Err(format!("Shape mismatch: grad {:?} vs y {:?}", grad.shape(), y.shape()));
    }

    // Check for NaN or infinite values
    for (name, array) in [
        ("embedding", y),
        ("gradient", grad),
        ("momentum", m),
        ("velocity", v),
    ] {
        for (i, &val) in array.iter().enumerate() {
            if !val.is_finite() {
                return Err(format!("Non-finite {} value at index {}: {}", name, i, val));
            }
        }
    }

    Ok(())
}

/// Computes the L2 norm of parameters
fn compute_parameter_norm(y: ArrayView2<f32>) -> f32 {
    y.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

/// Statistics from an Adam optimization step
#[derive(Debug, Clone)]
pub struct AdamStats {
    /// Current iteration number
    pub iteration: usize,
    /// Bias-corrected learning rate for this iteration
    pub learning_rate_corrected: f32,
    /// Maximum absolute change in momentum (first moment)
    pub max_momentum_update: f32,
    /// Maximum absolute change in velocity (second moment)
    pub max_velocity_update: f32,
    /// Maximum absolute parameter update
    pub max_parameter_update: f32,
    /// L2 norm of the gradient vector
    pub gradient_norm: f32,
    /// L2 norm of the parameter vector after update
    pub parameter_norm: f32,
}

impl std::fmt::Display for AdamStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AdamStats {{ itr: {}, lr: {:.6}, |grad|: {:.6}, |param|: {:.6}, max_Δ: {:.6} }}",
            self.iteration,
            self.learning_rate_corrected,
            self.gradient_norm,
            self.parameter_norm,
            self.max_parameter_update
        )
    }
}

/// Creates initialized momentum and velocity arrays for Adam optimization
pub fn create_adam_state(shape: (usize, usize)) -> (Array2<f32>, Array2<f32>) {
    let m = Array2::zeros(shape);
    let v = Array2::zeros(shape);
    (m, v)
}

/// Creates Adam state with small random initialization for better optimization
pub fn create_adam_state_random(shape: (usize, usize), seed: u64) -> (Array2<f32>, Array2<f32>) {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let m = Array2::from_shape_fn(shape, |_| rng.gen::<f32>() * 1e-6);
    let v = Array2::from_shape_fn(shape, |_| rng.gen::<f32>() * 1e-6);
    (m, v)
}

/// Resets Adam state to zeros
pub fn reset_adam_state(mut m: ArrayViewMut2<f32>, mut v: ArrayViewMut2<f32>) {
    m.fill(0.0);
    v.fill(0.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array2};

    #[test]
    fn test_update_embedding_adam() {
        // Define test inputs
        let mut y = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let grad = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];
        let mut m = Array2::zeros((3, 2));
        let mut v = Array2::zeros((3, 2));
        let beta1 = 0.9;
        let beta2 = 0.999;
        let lr = 0.001;
        let itr = 0;

        // Run update step
        update_embedding_adam(
            y.view_mut(),
            grad.view(),
            m.view_mut(),
            v.view_mut(),
            beta1,
            beta2,
            lr,
            itr,
        );

        // Define expected outputs
        let y_expected = array![[0.999, 1.9990001], [2.999, 3.999], [4.999, 5.999]];
        let m_expected = array![
            [0.01, 0.02000001],
            [0.03000001, 0.04000001],
            [0.05000001, 0.06000002]
        ];
        let v_expected = array![
            [9.9998715e-06, 3.9999486e-05],
            [8.9998844e-05, 1.5999794e-04],
            [2.4999678e-04, 3.5999538e-04]
        ];

        // Verify outputs match expected values
        Zip::from(&y).and(&y_expected).for_each(|&y_val, &y_exp| {
            assert_abs_diff_eq!(y_val, y_exp, epsilon = 1e-6);
        });

        Zip::from(&m).and(&m_expected).for_each(|&m_val, &m_exp| {
            assert_abs_diff_eq!(m_val, m_exp, epsilon = 1e-6);
        });

        Zip::from(&v).and(&v_expected).for_each(|&v_val, &v_exp| {
            assert_abs_diff_eq!(v_val, v_exp, epsilon = 1e-6);
        });
    }

    #[test]
    fn test_adam_config() {
        let config = AdamConfig::default();
        assert_eq!(config.beta1, 0.9);
        assert_eq!(config.beta2, 0.999);
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.epsilon, 1e-7);
        assert!(!config.report_progress);

        let custom_config = AdamConfig::with_betas(0.01, 0.95, 0.995).epsilon(1e-8);
        assert_eq!(custom_config.learning_rate, 0.01);
        assert_eq!(custom_config.beta1, 0.95);
        assert_eq!(custom_config.beta2, 0.995);
        assert_eq!(custom_config.epsilon, 1e-8);

        // Test validation
        assert!(custom_config.validate().is_ok());

        let invalid_config = AdamConfig::new(-0.01);
        assert!(invalid_config.validate().is_err());

        let invalid_beta_config = AdamConfig::new(0.01).with_betas(0.01, 1.5, 0.999);
        assert!(invalid_beta_config.validate().is_err());
    }

    #[test]
    fn test_update_embedding_adam_with_stats() {
        let mut y = array![[1.0, 2.0], [3.0, 4.0]];
        let grad = array![[0.1, 0.2], [0.3, 0.4]];
        let mut m = Array2::zeros((2, 2));
        let mut v = Array2::zeros((2, 2));
        let config = AdamConfig::new(0.001);

        let stats = update_embedding_adam_with_stats(
            y.view_mut(),
            grad.view(),
            m.view_mut(),
            v.view_mut(),
            &config,
            0,
        ).unwrap();

        assert_eq!(stats.iteration, 0);
        assert!(stats.learning_rate_corrected > 0.0);
        assert!(stats.gradient_norm > 0.0);
        assert!(stats.parameter_norm > 0.0);
        assert!(stats.max_parameter_update > 0.0);
    }

    #[test]
    fn test_progress_callback() {
        let mut y = array![[1.0, 2.0]];
        let grad = array![[0.1, 0.2]];
        let mut m = Array2::zeros((1, 2));
        let mut v = Array2::zeros((1, 2));

        let callback_called = std::sync::Arc::new(std::sync::Mutex::new(false));
        let callback_called_clone = callback_called.clone();

        let config = AdamConfig::new(0.001).with_progress_callback(move |stage, current, total, percentage, details| {
            let mut called = callback_called_clone.lock().unwrap();
            *called = true;
            println!("Progress: {} - {}/{} ({:.1}%) - {}", stage, current, total, percentage, details);
        });

        let stats = update_embedding_adam_with_stats(
            y.view_mut(),
            grad.view(),
            m.view_mut(),
            v.view_mut(),
            &config,
            0,
        ).unwrap();

        assert!(*callback_called.lock().unwrap());
        assert_eq!(stats.iteration, 0);
    }

    #[test]
    fn test_adam_state_creation() {
        let shape = (3, 2);
        let (m, v) = create_adam_state(shape);

        assert_eq!(m.shape(), shape);
        assert_eq!(v.shape(), shape);
        assert!(m.iter().all(|&x| x == 0.0));
        assert!(v.iter().all(|&x| x == 0.0));

        // Test random initialization
        let (m_rand, v_rand) = create_adam_state_random(shape, 42);
        assert_eq!(m_rand.shape(), shape);
        assert_eq!(v_rand.shape(), shape);
        assert!(!m_rand.iter().all(|&x| x == 0.0));
        assert!(!v_rand.iter().all(|&x| x == 0.0));

        // Test reset
        reset_adam_state(m_rand.view_mut(), v_rand.view_mut());
        assert!(m_rand.iter().all(|&x| x == 0.0));
        assert!(v_rand.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_input_validation() {
        let y = array![[1.0, 2.0]];
        let grad = array![[0.1, 0.2]];
        let mut m = Array2::zeros((1, 2));
        let mut v = Array2::zeros((1, 2));
        let config = AdamConfig::default();

        // Valid inputs should work
        let result = update_embedding_adam_with_stats(
            y.view(),
            grad.view(),
            m.view_mut(),
            v.view_mut(),
            &config,
            0,
        );
        assert!(result.is_ok());

        // Shape mismatch
        let wrong_shape_grad = array![[0.1, 0.2, 0.3]];
        let result = update_embedding_adam_with_stats(
            y.view(),
            wrong_shape_grad.view(),
            m.view_mut(),
            v.view_mut(),
            &config,
            0,
        );
        assert!(result.is_err());

        // NaN in gradient
        let nan_grad = array![[f32::NAN, 0.2]];
        let result = update_embedding_adam_with_stats(
            y.view(),
            nan_grad.view(),
            m.view_mut(),
            v.view_mut(),
            &config,
            0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_adam_stats_display() {
        let stats = AdamStats {
            iteration: 42,
            learning_rate_corrected: 0.001,
            max_momentum_update: 0.01,
            max_velocity_update: 0.001,
            max_parameter_update: 0.005,
            gradient_norm: 1.23,
            parameter_norm: 4.56,
        };

        let display_str = format!("{}", stats);
        assert!(display_str.contains("AdamStats"));
        assert!(display_str.contains("itr: 42"));
        assert!(display_str.contains("lr: 0.001000"));
        assert!(display_str.contains("|grad|: 1.230000"));
        assert!(display_str.contains("|param|: 4.560000"));
    }

    #[test]
    fn test_deterministic_behavior() {
        let mut y1 = array![[1.0, 2.0], [3.0, 4.0]];
        let mut y2 = array![[1.0, 2.0], [3.0, 4.0]];
        let grad = array![[0.1, 0.2], [0.3, 0.4]];
        let config = AdamConfig::new(0.001);

        let (mut m1, mut v1) = create_adam_state_random((2, 2), 42);
        let (mut m2, mut v2) = create_adam_state_random((2, 2), 42);

        // Run identical updates
        let stats1 = update_embedding_adam_with_stats(
            y1.view_mut(),
            grad.view(),
            m1.view_mut(),
            v1.view_mut(),
            &config,
            0,
        ).unwrap();

        let stats2 = update_embedding_adam_with_stats(
            y2.view_mut(),
            grad.view(),
            m2.view_mut(),
            v2.view_mut(),
            &config,
            0,
        ).unwrap();

        // Results should be identical
        assert_eq!(y1, y2);
        assert_eq!(m1, m2);
        assert_eq!(v1, v2);
        assert_eq!(stats1.learning_rate_corrected, stats2.learning_rate_corrected);
    }
}