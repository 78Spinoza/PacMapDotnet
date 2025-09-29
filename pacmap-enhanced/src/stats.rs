use ndarray::{Array2, Axis};
use ordered_float::OrderedFloat;
use serde::{Serialize, Deserialize};
use std::error::Error;
use std::fmt;
use rand::Rng;

/// Normalization modes supported by the system
/// Following UMAP's approach to feature scaling with saved parameters
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub enum NormalizationMode {
    /// Z-score normalization: (x - μ) / σ
    /// Most common, assumes normal distribution
    ZScore,
    /// Min-max scaling: (x - min) / (max - min)
    /// Maps to [0, 1] range, preserves relative distances
    MinMax,
    /// Robust scaling: (x - median) / IQR
    /// Less sensitive to outliers, uses median and interquartile range
    Robust,
    /// No normalization applied
    None,
}

impl Default for NormalizationMode {
    fn default() -> Self {
        NormalizationMode::ZScore
    }
}

/// Normalization parameters stored with the model
/// Critical for consistent transform behavior - mirrors UMAP's approach
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NormalizationParams {
    /// Feature means (for Z-score normalization)
    pub means: Vec<f64>,
    /// Feature standard deviations (for Z-score normalization)
    pub stds: Vec<f64>,
    /// Feature minimum values (for MinMax normalization)
    pub mins: Vec<f64>,
    /// Feature maximum values (for MinMax normalization)
    pub maxs: Vec<f64>,
    /// Feature medians (for Robust normalization)
    pub medians: Vec<f64>,
    /// Feature interquartile ranges (for Robust normalization)
    pub iqrs: Vec<f64>,
    /// Normalization mode used
    pub mode: NormalizationMode,
    /// Number of features
    pub n_features: usize,
    /// Whether normalization was applied during training
    pub is_fitted: bool,
}

impl Default for NormalizationParams {
    fn default() -> Self {
        Self {
            means: Vec::new(),
            stds: Vec::new(),
            mins: Vec::new(),
            maxs: Vec::new(),
            medians: Vec::new(),
            iqrs: Vec::new(),
            mode: NormalizationMode::default(),
            n_features: 0,
            is_fitted: false,
        }
    }
}

#[derive(Debug)]
pub struct NormalizationError {
    message: String,
}

impl fmt::Display for NormalizationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Normalization error: {}", self.message)
    }
}

impl Error for NormalizationError {}

impl NormalizationError {
    pub fn new(message: &str) -> Self {
        Self {
            message: message.to_string(),
        }
    }
}

impl NormalizationParams {
    /// Create new normalization parameters for the given number of features
    pub fn new(n_features: usize, mode: NormalizationMode) -> Self {
        Self {
            means: vec![0.0; n_features],
            stds: vec![1.0; n_features],
            mins: vec![0.0; n_features],
            maxs: vec![1.0; n_features],
            medians: vec![0.0; n_features],
            iqrs: vec![1.0; n_features],
            mode,
            n_features,
            is_fitted: false,
        }
    }

    /// Fit normalization parameters from training data
    /// This computes and stores the parameters needed for consistent normalization
    pub fn fit(&mut self, data: &Array2<f64>) -> Result<(), NormalizationError> {
        let (n_samples, n_features) = data.dim();

        if n_samples == 0 {
            return Err(NormalizationError::new("Cannot fit on empty data"));
        }

        if self.n_features == 0 {
            self.n_features = n_features;
        } else if self.n_features != n_features {
            return Err(NormalizationError::new(&format!(
                "Feature dimension mismatch: expected {}, got {}",
                self.n_features, n_features
            )));
        }

        // Resize vectors to match feature count
        self.means.resize(n_features, 0.0);
        self.stds.resize(n_features, 1.0);
        self.mins.resize(n_features, 0.0);
        self.maxs.resize(n_features, 1.0);
        self.medians.resize(n_features, 0.0);
        self.iqrs.resize(n_features, 1.0);

        match self.mode {
            NormalizationMode::ZScore => {
                self.compute_zscore_params(data)?;
            }
            NormalizationMode::MinMax => {
                self.compute_minmax_params(data)?;
            }
            NormalizationMode::Robust => {
                self.compute_robust_params(data)?;
            }
            NormalizationMode::None => {
                // No parameters to compute
            }
        }

        self.is_fitted = true;
        Ok(())
    }

    /// Apply normalization to data using fitted parameters
    /// Critical: This must use the saved parameters for consistency
    pub fn transform(&self, data: &mut Array2<f64>) -> Result<(), NormalizationError> {
        if !self.is_fitted {
            return Err(NormalizationError::new("Normalization parameters not fitted"));
        }

        let (_, n_features) = data.dim();
        if n_features != self.n_features {
            return Err(NormalizationError::new(&format!(
                "Feature dimension mismatch: expected {}, got {}",
                self.n_features, n_features
            )));
        }

        match self.mode {
            NormalizationMode::ZScore => {
                self.apply_zscore_normalization(data)?;
            }
            NormalizationMode::MinMax => {
                self.apply_minmax_normalization(data)?;
            }
            NormalizationMode::Robust => {
                self.apply_robust_normalization(data)?;
            }
            NormalizationMode::None => {
                // No normalization to apply
            }
        }

        Ok(())
    }

    /// Fit and transform in one step (for training data)
    pub fn fit_transform(&mut self, data: &mut Array2<f64>) -> Result<(), NormalizationError> {
        self.fit(data)?;
        self.transform(data)?;
        Ok(())
    }

    fn compute_zscore_params(&mut self, data: &Array2<f64>) -> Result<(), NormalizationError> {
        let (n_samples, n_features) = data.dim();

        // Use ndarray built-in functions for better performance
        let means = data.mean_axis(Axis(0)).unwrap();
        let variances = data.var_axis(Axis(0), 1.0); // ddof=1 for sample variance

        for j in 0..n_features {
            self.means[j] = means[j];

            self.stds[j] = if n_samples > 1 {
                variances[j].sqrt()
            } else {
                1.0
            };

            // Avoid division by zero
            if self.stds[j] < 1e-8 {
                self.stds[j] = 1.0;
            }
        }

        Ok(())
    }

    fn compute_minmax_params(&mut self, data: &Array2<f64>) -> Result<(), NormalizationError> {
        let (_n_samples, n_features) = data.dim();

        // Use ndarray built-in column operations
        for j in 0..n_features {
            let column = data.column(j);

            if let (Some(min_val), Some(max_val)) = (
                column.iter().min_by(|a, b| a.partial_cmp(b).unwrap()),
                column.iter().max_by(|a, b| a.partial_cmp(b).unwrap())
            ) {
                self.mins[j] = *min_val;
                self.maxs[j] = *max_val;
            } else {
                self.mins[j] = 0.0;
                self.maxs[j] = 1.0;
            }

            // Avoid division by zero
            if (self.maxs[j] - self.mins[j]).abs() < 1e-8 {
                self.maxs[j] = self.mins[j] + 1.0;
            }
        }

        Ok(())
    }

    fn compute_robust_params(&mut self, data: &Array2<f64>) -> Result<(), NormalizationError> {
        let (n_samples, n_features) = data.dim();

        for j in 0..n_features {
            // Extract column values
            let mut values: Vec<f64> = (0..n_samples).map(|i| data[[i, j]]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Compute median
            self.medians[j] = if values.len() % 2 == 0 {
                let mid = values.len() / 2;
                (values[mid - 1] + values[mid]) / 2.0
            } else {
                values[values.len() / 2]
            };

            // Compute IQR (Q3 - Q1)
            let q1_idx = values.len() / 4;
            let q3_idx = 3 * values.len() / 4;
            let q1 = values[q1_idx];
            let q3 = values[q3_idx];

            self.iqrs[j] = q3 - q1;

            // Avoid division by zero
            if self.iqrs[j] < 1e-8 {
                self.iqrs[j] = 1.0;
            }
        }

        Ok(())
    }

    fn apply_zscore_normalization(&self, data: &mut Array2<f64>) -> Result<(), NormalizationError> {
        let (n_samples, n_features) = data.dim();

        for j in 0..n_features {
            for i in 0..n_samples {
                data[[i, j]] = (data[[i, j]] - self.means[j]) / self.stds[j];
            }
        }

        Ok(())
    }

    fn apply_minmax_normalization(&self, data: &mut Array2<f64>) -> Result<(), NormalizationError> {
        let (n_samples, n_features) = data.dim();

        for j in 0..n_features {
            let range = self.maxs[j] - self.mins[j];
            for i in 0..n_samples {
                data[[i, j]] = (data[[i, j]] - self.mins[j]) / range;
            }
        }

        Ok(())
    }

    fn apply_robust_normalization(&self, data: &mut Array2<f64>) -> Result<(), NormalizationError> {
        let (n_samples, n_features) = data.dim();

        for j in 0..n_features {
            for i in 0..n_samples {
                data[[i, j]] = (data[[i, j]] - self.medians[j]) / self.iqrs[j];
            }
        }

        Ok(())
    }

    /// Validate that the parameters are consistent and reasonable
    pub fn validate(&self) -> Result<(), NormalizationError> {
        if !self.is_fitted {
            return Err(NormalizationError::new("Parameters not fitted"));
        }

        if self.n_features == 0 {
            return Err(NormalizationError::new("Invalid feature count"));
        }

        let expected_len = self.n_features;
        if self.means.len() != expected_len || self.stds.len() != expected_len ||
           self.mins.len() != expected_len || self.maxs.len() != expected_len ||
           self.medians.len() != expected_len || self.iqrs.len() != expected_len {
            return Err(NormalizationError::new("Parameter vector length mismatch"));
        }

        // Check for invalid values
        match self.mode {
            NormalizationMode::ZScore => {
                for (i, &std) in self.stds.iter().enumerate() {
                    if std <= 0.0 || !std.is_finite() {
                        return Err(NormalizationError::new(&format!(
                            "Invalid standard deviation at feature {}: {}", i, std
                        )));
                    }
                }
            }
            NormalizationMode::MinMax => {
                for (i, (&min, &max)) in self.mins.iter().zip(&self.maxs).enumerate() {
                    if !min.is_finite() || !max.is_finite() || max <= min {
                        return Err(NormalizationError::new(&format!(
                            "Invalid min/max at feature {}: min={}, max={}", i, min, max
                        )));
                    }
                }
            }
            NormalizationMode::Robust => {
                for (i, &iqr) in self.iqrs.iter().enumerate() {
                    if iqr <= 0.0 || !iqr.is_finite() {
                        return Err(NormalizationError::new(&format!(
                            "Invalid IQR at feature {}: {}", i, iqr
                        )));
                    }
                }
            }
            NormalizationMode::None => {
                // No validation needed
            }
        }

        Ok(())
    }
}

/// Compute distance statistics for outlier detection with smart approximation
/// Uses sampling for large datasets to avoid O(n²) performance issues
pub fn compute_distance_stats(embedding: &Array2<f64>) -> (f64, f64, f64) {
    let n = embedding.shape()[0];
    if n < 2 {
        return (0.0, 0.0, 0.0);
    }

    // Use approximation for large datasets to avoid O(n²) bottleneck
    const EXACT_THRESHOLD: usize = 5000;
    const MAX_SAMPLE_PAIRS: usize = 50000;

    let mut distances = Vec::new();

    if n <= EXACT_THRESHOLD {
        // Exact computation for small datasets
        for i in 0..n {
            for j in (i + 1)..n {
                let mut diff = 0.0;
                for (&a, &b) in embedding.row(i).iter().zip(embedding.row(j).iter()) {
                    let d = a - b;
                    diff += d * d;
                }
                let dist = diff.sqrt();
                distances.push(OrderedFloat(dist));
            }
        }
    } else {
        // Approximation using random sampling for large datasets
        if std::env::var("PACMAP_VERBOSE").is_ok() {
            eprintln!("DEBUG: Using distance stats approximation for {} samples (n>{}) to avoid O(n²) bottleneck", n, EXACT_THRESHOLD);
        }

        use std::collections::HashSet;
        let mut rng = rand::thread_rng();
        let mut sampled_pairs = HashSet::new();

        // Calculate how many pairs we want to sample
        let total_possible_pairs = (n * (n - 1)) / 2;
        let sample_size = MAX_SAMPLE_PAIRS.min(total_possible_pairs);

        while sampled_pairs.len() < sample_size {
            let i = (rng.gen::<f64>() * n as f64) as usize;
            let j = (rng.gen::<f64>() * n as f64) as usize;

            if i != j {
                let (min_idx, max_idx) = if i < j { (i, j) } else { (j, i) };
                sampled_pairs.insert((min_idx, max_idx));
            }
        }

        for &(i, j) in &sampled_pairs {
            let mut diff = 0.0;
            for (&a, &b) in embedding.row(i).iter().zip(embedding.row(j).iter()) {
                let d = a - b;
                diff += d * d;
            }
            let dist = diff.sqrt();
            distances.push(OrderedFloat(dist));
        }

        if std::env::var("PACMAP_VERBOSE").is_ok() {
            eprintln!("SUCCESS: Sampled {} distance pairs ({}% of total)", distances.len(),
                     (distances.len() * 100) / total_possible_pairs);
        }
    }

    if distances.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    distances.sort();
    let mean = distances.iter().map(|x| x.0).sum::<f64>() / distances.len() as f64;
    let p95_idx = ((distances.len() as f64 * 0.95) as usize).min(distances.len() - 1);
    let p95 = distances[p95_idx].0;
    let max = distances.last().map(|x| x.0).unwrap_or(0.0);
    (mean, p95, max)
}

/// Determine optimal normalization mode based on data characteristics
pub fn recommend_normalization_mode(data: &Array2<f64>) -> NormalizationMode {
    let (n_samples, n_features) = data.dim();

    if n_samples < 10 || n_features == 0 {
        return NormalizationMode::None;
    }

    // Simple heuristic: check if data has outliers
    let mut has_outliers = false;

    for j in 0..n_features {
        let mut values: Vec<f64> = (0..n_samples).map(|i| data[[i, j]]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Check for extreme outliers using IQR method
        let q1_idx = values.len() / 4;
        let q3_idx = 3 * values.len() / 4;
        if q1_idx < values.len() && q3_idx < values.len() {
            let q1 = values[q1_idx];
            let q3 = values[q3_idx];
            let iqr = q3 - q1;

            if iqr > 1e-8 {
                let lower_bound = q1 - 1.5 * iqr;
                let upper_bound = q3 + 1.5 * iqr;

                for &val in &values {
                    if val < lower_bound || val > upper_bound {
                        has_outliers = true;
                        break;
                    }
                }
            }
        }

        if has_outliers {
            break;
        }
    }

    if has_outliers {
        NormalizationMode::Robust
    } else {
        NormalizationMode::ZScore
    }
}
