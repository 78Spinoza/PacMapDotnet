//! Manages weights used during the `PaCMAP` optimization process - Enhanced Version.
//!
//! The weights control how different types of point pairs influence the
//! embedding:
//! - Nearest neighbor pairs preserve local structure
//! - Mid-near pairs preserve medium-range structure
//! - Far pairs prevent collapse by maintaining separation
//!
//! The weights evolve through three phases during optimization:
//! 1. Gradually reduce mid-near weight to allow initial structure formation
//! 2. Balance local and global structure with equal mid-near and neighbor
//!    weights
//! 3. Focus on local structure by zeroing mid-near weight
//!
//! This enhanced version includes improved validation and detailed progress
//! reporting capabilities.

/// Weight parameters applied to each type of point pair during optimization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weights {
    /// Weight for mid-near pairs, controlling medium-range structure
    /// preservation. Gradually decreases from initial value to 0.0 across
    /// optimization phases.
    pub w_mn: f32,

    /// Weight for nearest neighbor pairs, controlling local structure
    /// preservation. Varies between 1.0-3.0 across optimization phases.
    pub w_neighbors: f32,

    /// Weight for far pairs, controlling global structure by preventing
    /// collapse. Remains constant at 1.0 throughout optimization.
    pub w_fp: f32,
}

impl Default for Weights {
    fn default() -> Self {
        Self {
            w_mn: 2.0,
            w_neighbors: 2.0,
            w_fp: 1.0,
        }
    }
}

impl Weights {
    /// Create new weights with specified values
    pub fn new(w_mn: f32, w_neighbors: f32, w_fp: f32) -> Self {
        Self {
            w_mn,
            w_neighbors,
            w_fp,
        }
    }

    /// Validate weight values
    pub fn validate(&self) -> Result<(), String> {
        if !self.w_mn.is_finite() || self.w_mn < 0.0 {
            return Err(format!("Invalid mid-near weight: {}", self.w_mn));
        }
        if !self.w_neighbors.is_finite() || self.w_neighbors < 0.0 {
            return Err(format!("Invalid neighbor weight: {}", self.w_neighbors));
        }
        if !self.w_fp.is_finite() || self.w_fp < 0.0 {
            return Err(format!("Invalid far pair weight: {}", self.w_fp));
        }
        Ok(())
    }

    /// Get the total weight (sum of all weights)
    pub fn total(&self) -> f32 {
        self.w_mn + self.w_neighbors + self.w_fp
    }

    /// Get normalized weights (sums to 1.0)
    pub fn normalized(&self) -> Weights {
        let total = self.total();
        if total == 0.0 {
            Weights::default()
        } else {
            Weights {
                w_mn: self.w_mn / total,
                w_neighbors: self.w_neighbors / total,
                w_fp: self.w_fp / total,
            }
        }
    }

    /// Get weight ratios relative to far pairs
    pub fn ratios(&self) -> (f32, f32, f32) {
        if self.w_fp == 0.0 {
            (0.0, 0.0, 0.0)
        } else {
            (self.w_mn / self.w_fp, self.w_neighbors / self.w_fp, 1.0)
        }
    }
}

impl std::fmt::Display for Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Weights {{ mn: {:.3}, neighbors: {:.3}, fp: {:.3}, total: {:.3} }}",
            self.w_mn,
            self.w_neighbors,
            self.w_fp,
            self.total()
        )
    }
}

/// Configuration for weight calculation with enhanced options
#[derive(Debug, Clone)]
pub struct WeightConfig {
    /// Initial weight for mid-near pairs
    pub w_mn_init: f32,
    /// Number of iterations in phase 1 (mid-near reduction)
    pub phase_1_iters: usize,
    /// Number of iterations in phase 2 (balanced weights)
    pub phase_2_iters: usize,
    /// Whether to use custom phase weights
    pub custom_phase_weights: Option<PhaseWeights>,
}

/// Custom weight values for each optimization phase
#[derive(Debug, Clone)]
pub struct PhaseWeights {
    /// Weights for phase 1 (mid-near reduction)
    pub phase_1: Weights,
    /// Weights for phase 2 (balanced)
    pub phase_2: Weights,
    /// Weights for phase 3 (local focus)
    pub phase_3: Weights,
}

impl Default for WeightConfig {
    fn default() -> Self {
        Self {
            w_mn_init: 2.0,
            phase_1_iters: 100,
            phase_2_iters: 100,
            custom_phase_weights: None,
        }
    }
}

impl WeightConfig {
    /// Create new weight configuration
    pub fn new(w_mn_init: f32, phase_1_iters: usize, phase_2_iters: usize) -> Self {
        Self {
            w_mn_init,
            phase_1_iters,
            phase_2_iters,
            custom_phase_weights: None,
        }
    }

    /// Set custom phase weights
    pub fn with_custom_weights(mut self, phase_weights: PhaseWeights) -> Self {
        self.custom_phase_weights = Some(phase_weights);
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if !self.w_mn_init.is_finite() || self.w_mn_init < 0.0 {
            return Err(format!("Invalid initial mid-near weight: {}", self.w_mn_init));
        }
        if self.phase_1_iters == 0 {
            return Err("Phase 1 iterations must be positive".to_string());
        }
        if self.phase_2_iters == 0 {
            return Err("Phase 2 iterations must be positive".to_string());
        }

        if let Some(ref weights) = self.custom_phase_weights {
            weights.phase_1.validate()?;
            weights.phase_2.validate()?;
            weights.phase_3.validate()?;
        }

        Ok(())
    }
}

/// Calculate optimization weights based on the current iteration and phase
/// durations with enhanced configuration options.
///
/// The weights smoothly transition through three phases:
/// 1. Reduce mid-near weight linearly from initial value to 3.0, with fixed
///    neighbor and far weights
/// 2. Fix all weights to balance local and global structure (mid-near=3.0,
///    neighbor=3.0, far=1.0)
/// 3. Zero mid-near weight and reduce neighbor weight to focus on local
///    structure
///
/// # Arguments
/// * `config` - Weight configuration including initial weight and phase durations
/// * `itr` - Current iteration number
///
/// # Returns
/// A `Weights` struct containing the calculated weights for this iteration
#[allow(clippy::cast_precision_loss)]
pub fn find_weights_with_config(config: &WeightConfig, itr: usize) -> Weights {
    if let Some(ref custom_weights) = config.custom_phase_weights {
        return find_weights_custom(custom_weights, itr, config.phase_1_iters, config.phase_2_iters);
    }

    find_weights(config.w_mn_init, itr, config.phase_1_iters, config.phase_2_iters)
}

/// Calculate optimization weights using custom phase weights
fn find_weights_custom(
    custom_weights: &PhaseWeights,
    itr: usize,
    phase_1_iters: usize,
    phase_2_iters: usize,
) -> Weights {
    if itr < phase_1_iters {
        // Phase 1: Linear interpolation from initial to target
        let progress = itr as f32 / phase_1_iters as f32;
        let w_mn = (1.0 - progress) * custom_weights.phase_1.w_mn
                 + progress * custom_weights.phase_2.w_mn;
        let w_neighbors = (1.0 - progress) * custom_weights.phase_1.w_neighbors
                        + progress * custom_weights.phase_2.w_neighbors;
        let w_fp = (1.0 - progress) * custom_weights.phase_1.w_fp
                 + progress * custom_weights.phase_2.w_fp;

        Weights { w_mn, w_neighbors, w_fp }
    } else if itr < phase_1_iters + phase_2_iters {
        // Phase 2: Use phase 2 weights
        custom_weights.phase_2
    } else {
        // Phase 3: Use phase 3 weights
        custom_weights.phase_3
    }
}

/// Calculate optimization weights based on the current iteration and phase
/// durations (legacy API).
///
/// The weights smoothly transition through three phases:
/// 1. Reduce mid-near weight linearly from initial value to 3.0, with fixed
///    neighbor and far weights
/// 2. Fix all weights to balance local and global structure (mid-near=3.0,
///    neighbor=3.0, far=1.0)
/// 3. Zero mid-near weight and reduce neighbor weight to focus on local
///    structure
///
/// # Arguments
/// * `w_mn_init` - Initial weight for mid-near pairs
/// * `itr` - Current iteration number
/// * `phase_1_iters` - Number of iterations in phase 1 (mid-near reduction)
/// * `phase_2_iters` - Number of iterations in phase 2 (balanced weights)
///
/// # Returns
/// A `Weights` struct containing the calculated weights for this iteration
#[allow(clippy::cast_precision_loss)]
pub fn find_weights(
    w_mn_init: f32,
    itr: usize,
    phase_1_iters: usize,
    phase_2_iters: usize,
) -> Weights {
    if itr < phase_1_iters {
        // Phase 1: Linear interpolation of mid-near weight
        let progress = itr as f32 / phase_1_iters as f32;
        Weights {
            w_mn: (1.0 - progress) * w_mn_init + progress * 3.0,
            w_neighbors: 2.0,
            w_fp: 1.0,
        }
    } else if itr < phase_1_iters + phase_2_iters {
        // Phase 2: Fixed balanced weights
        Weights {
            w_mn: 3.0,
            w_neighbors: 3.0,
            w_fp: 1.0,
        }
    } else {
        // Phase 3: Local structure focus
        Weights {
            w_mn: 0.0,
            w_neighbors: 1.0,
            w_fp: 1.0,
        }
    }
}

/// Get the current phase information for an iteration
pub fn get_phase_info(itr: usize, phase_1_iters: usize, phase_2_iters: usize) -> PhaseInfo {
    if itr < phase_1_iters {
        PhaseInfo {
            phase: 1,
            progress: itr as f32 / phase_1_iters as f32,
            iterations_in_phase: itr,
            total_phase_iterations: phase_1_iters,
            name: "Mid-near reduction",
        }
    } else if itr < phase_1_iters + phase_2_iters {
        let phase_2_itr = itr - phase_1_iters;
        PhaseInfo {
            phase: 2,
            progress: phase_2_itr as f32 / phase_2_iters as f32,
            iterations_in_phase: phase_2_itr,
            total_phase_iterations: phase_2_iters,
            name: "Balanced weights",
        }
    } else {
        PhaseInfo {
            phase: 3,
            progress: 1.0, // Final phase doesn't progress
            iterations_in_phase: itr - phase_1_iters - phase_2_iters,
            total_phase_iterations: usize::MAX, // Ongoing
            name: "Local structure focus",
        }
    }
}

/// Information about the current optimization phase
#[derive(Debug, Clone)]
pub struct PhaseInfo {
    /// Current phase number (1, 2, or 3)
    pub phase: u8,
    /// Progress through current phase (0.0 to 1.0)
    pub progress: f32,
    /// Number of iterations completed in current phase
    pub iterations_in_phase: usize,
    /// Total number of iterations in current phase
    pub total_phase_iterations: usize,
    /// Descriptive name of the phase
    pub name: &'static str,
}

impl std::fmt::Display for PhaseInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.total_phase_iterations == usize::MAX {
            write!(
                f,
                "Phase {}: {} (iteration {} in phase)",
                self.phase, self.name, self.iterations_in_phase
            )
        } else {
            write!(
                f,
                "Phase {}: {} ({:.1}% complete, {}/{})",
                self.phase,
                self.name,
                self.progress * 100.0,
                self.iterations_in_phase,
                self.total_phase_iterations
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_find_weight() {
        let w_mn_init = 1000.0;

        // Test Phase 1
        let w0 = find_weights(w_mn_init, 0, 100, 100);
        assert_abs_diff_eq!(w0.w_mn, 1000.0);
        assert_abs_diff_eq!(w0.w_neighbors, 2.0);
        assert_abs_diff_eq!(w0.w_fp, 1.0);

        let w50 = find_weights(w_mn_init, 50, 100, 100);
        assert_abs_diff_eq!(w50.w_mn, 501.5);
        assert_abs_diff_eq!(w50.w_neighbors, 2.0);
        assert_abs_diff_eq!(w50.w_fp, 1.0);

        // Test Phase 2
        let w150 = find_weights(w_mn_init, 150, 100, 100);
        assert_abs_diff_eq!(w150.w_mn, 3.0);
        assert_abs_diff_eq!(w150.w_neighbors, 3.0);
        assert_abs_diff_eq!(w150.w_fp, 1.0);

        // Test Phase 3
        let w300 = find_weights(w_mn_init, 300, 100, 100);
        assert_abs_diff_eq!(w300.w_mn, 0.0);
        assert_abs_diff_eq!(w300.w_neighbors, 1.0);
        assert_abs_diff_eq!(w300.w_fp, 1.0);
    }

    #[test]
    fn test_weights_struct() {
        let weights = Weights::new(1.0, 2.0, 3.0);

        assert_eq!(weights.w_mn, 1.0);
        assert_eq!(weights.w_neighbors, 2.0);
        assert_eq!(weights.w_fp, 3.0);
        assert_eq!(weights.total(), 6.0);

        // Test normalized weights
        let normalized = weights.normalized();
        assert_abs_diff_eq!(normalized.w_mn, 1.0/6.0);
        assert_abs_diff_eq!(normalized.w_neighbors, 2.0/6.0);
        assert_abs_diff_eq!(normalized.w_fp, 3.0/6.0);

        // Test ratios
        let ratios = weights.ratios();
        assert_abs_diff_eq!(ratios.0, 1.0/3.0); // w_mn / w_fp
        assert_abs_diff_eq!(ratios.1, 2.0/3.0); // w_neighbors / w_fp
        assert_abs_diff_eq!(ratios.2, 1.0);     // w_fp / w_fp

        // Test validation
        assert!(weights.validate().is_ok());

        let invalid_weights = Weights::new(-1.0, 2.0, f32::NAN);
        assert!(invalid_weights.validate().is_err());

        // Test display
        let display_str = format!("{}", weights);
        assert!(display_str.contains("Weights"));
        assert!(display_str.contains("mn: 1.000"));
        assert!(display_str.contains("neighbors: 2.000"));
        assert!(display_str.contains("fp: 3.000"));
        assert!(display_str.contains("total: 6.000"));
    }

    #[test]
    fn test_weight_config() {
        let config = WeightConfig::new(5.0, 50, 100);

        assert_eq!(config.w_mn_init, 5.0);
        assert_eq!(config.phase_1_iters, 50);
        assert_eq!(config.phase_2_iters, 100);
        assert!(config.custom_phase_weights.is_none());

        // Test validation
        assert!(config.validate().is_ok());

        let invalid_config = WeightConfig::new(-1.0, 0, 100);
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_custom_weights() {
        let phase_weights = PhaseWeights {
            phase_1: Weights::new(10.0, 1.0, 1.0),
            phase_2: Weights::new(5.0, 3.0, 1.0),
            phase_3: Weights::new(0.0, 2.0, 1.0),
        };

        let config = WeightConfig::new(10.0, 50, 50).with_custom_weights(phase_weights);

        // Test phase 1 (interpolation)
        let w_start = find_weights_with_config(&config, 0);
        assert_abs_diff_eq!(w_start.w_mn, 10.0);
        assert_abs_diff_eq!(w_start.w_neighbors, 1.0);
        assert_abs_diff_eq!(w_start.w_fp, 1.0);

        let w_mid = find_weights_with_config(&config, 25);
        assert!(w_mid.w_mn > 5.0 && w_mid.w_mn < 10.0); // Interpolated
        assert!(w_mid.w_neighbors > 1.0 && w_mid.w_neighbors < 3.0); // Interpolated

        // Test phase 2 (fixed)
        let w_phase2 = find_weights_with_config(&config, 75);
        assert_abs_diff_eq!(w_phase2.w_mn, 5.0);
        assert_abs_diff_eq!(w_phase2.w_neighbors, 3.0);
        assert_abs_diff_eq!(w_phase2.w_fp, 1.0);

        // Test phase 3 (fixed)
        let w_phase3 = find_weights_with_config(&config, 200);
        assert_abs_diff_eq!(w_phase3.w_mn, 0.0);
        assert_abs_diff_eq!(w_phase3.w_neighbors, 2.0);
        assert_abs_diff_eq!(w_phase3.w_fp, 1.0);
    }

    #[test]
    fn test_phase_info() {
        let phase1 = get_phase_info(25, 100, 100);
        assert_eq!(phase1.phase, 1);
        assert_eq!(phase1.name, "Mid-near reduction");
        assert_abs_diff_eq!(phase1.progress, 0.25);

        let phase2 = get_phase_info(150, 100, 100);
        assert_eq!(phase2.phase, 2);
        assert_eq!(phase2.name, "Balanced weights");
        assert_eq!(phase2.iterations_in_phase, 50);
        assert_eq!(phase2.total_phase_iterations, 100);

        let phase3 = get_phase_info(250, 100, 100);
        assert_eq!(phase3.phase, 3);
        assert_eq!(phase3.name, "Local structure focus");
        assert_eq!(phase3.progress, 1.0);
        assert_eq!(phase3.iterations_in_phase, 50);

        // Test display
        let display_str = format!("{}", phase1);
        assert!(display_str.contains("Phase 1"));
        assert!(display_str.contains("Mid-near reduction"));
        assert!(display_str.contains("25.0% complete"));
    }

    #[test]
    fn test_weights_equality() {
        let w1 = Weights::new(1.0, 2.0, 3.0);
        let w2 = Weights::new(1.0, 2.0, 3.0);
        let w3 = Weights::new(1.0, 2.0, 3.1);

        assert_eq!(w1, w2);
        assert_ne!(w1, w3);
    }

    #[test]
    fn test_deterministic_behavior() {
        // Multiple calls with same parameters should produce identical results
        let w1 = find_weights(10.0, 42, 100, 100);
        let w2 = find_weights(10.0, 42, 100, 100);
        let w3 = find_weights(10.0, 42, 100, 100);

        assert_eq!(w1, w2);
        assert_eq!(w2, w3);
    }

    #[test]
    fn test_default_weights() {
        let default_weights = Weights::default();
        assert_eq!(default_weights.w_mn, 2.0);
        assert_eq!(default_weights.w_neighbors, 2.0);
        assert_eq!(default_weights.w_fp, 1.0);
        assert_eq!(default_weights.total(), 5.0);
    }

    #[test]
    fn test_weight_validation() {
        // Valid weights
        let valid_weights = Weights::new(1.0, 2.0, 3.0);
        assert!(valid_weights.validate().is_ok());

        // Zero weights (valid)
        let zero_weights = Weights::new(0.0, 0.0, 0.0);
        assert!(zero_weights.validate().is_ok());

        // Negative weight (invalid)
        let negative_weights = Weights::new(-1.0, 2.0, 3.0);
        assert!(negative_weights.validate().is_err());

        // NaN weight (invalid)
        let nan_weights = Weights::new(1.0, f32::NAN, 3.0);
        assert!(nan_weights.validate().is_err());

        // Infinity weight (invalid)
        let inf_weights = Weights::new(1.0, 2.0, f32::INFINITY);
        assert!(inf_weights.validate().is_err());
    }
}