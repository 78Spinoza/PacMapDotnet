// HNSW Parameter Auto-scaling Module
// Based on UMAP Enhanced patterns for optimal performance across dataset sizes

use serde::{Deserialize, Serialize};

/// HNSW parameters with auto-scaling support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswParams {
    /// HNSW graph connectivity parameter (M)
    /// Higher values = better recall, more memory
    pub m: usize,

    /// HNSW construction quality parameter (ef_construction)
    /// Higher values = better index quality, slower construction
    pub ef_construction: usize,

    /// HNSW search quality parameter (ef_search)
    /// Higher values = better recall, slower search
    pub ef_search: usize,

    /// Maximum connections per layer 0 node (max_m0)
    /// Usually 2 * M
    pub max_m0: usize,

    /// Memory usage estimate in bytes
    pub estimated_memory_bytes: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        // Default to medium dataset parameters
        Self {
            m: 16,
            ef_construction: 128,
            ef_search: 64,
            max_m0: 32,
            estimated_memory_bytes: 0,
        }
    }
}

impl HnswParams {
    /// Auto-scale HNSW parameters based on dataset characteristics
    /// Following UMAP Enhanced proven patterns
    pub fn auto_scale(n_samples: usize, n_features: usize, target_neighbors: usize) -> Self {
        let (m, ef_construction, ef_search) = match n_samples {
            // Small datasets: Prioritize accuracy over speed/memory
            0..=50_000 => (16, 64, 32),

            // Medium datasets: Balanced performance
            50_001..=1_000_000 => (32, 128, 64),

            // Large datasets: Prioritize memory efficiency and speed
            _ => (64, 128, 128),
        };

        // Apply dimension-based scaling (from UMAP research)
        let dim_scale = (n_features as f32).sqrt();
        let ef_search_scaled = std::cmp::max(
            ef_search,
            (dim_scale * 2.0) as usize
        );

        // Apply neighbor-count scaling (UMAP pattern)
        let log_scale = (n_samples as f32).log2();
        let ef_search_final = std::cmp::max(
            ef_search_scaled,
            (target_neighbors as f32 * log_scale) as usize
        );

        let max_m0 = m * 2;
        let estimated_memory = Self::estimate_memory(n_samples, m, max_m0);

        Self {
            m,
            ef_construction,
            ef_search: ef_search_final,
            max_m0,
            estimated_memory_bytes: estimated_memory,
        }
    }

    /// Create parameters optimized for specific use cases
    pub fn for_use_case(use_case: HnswUseCase, n_samples: usize, n_features: usize) -> Self {
        match use_case {
            HnswUseCase::FastConstruction => {
                // Minimize construction time
                Self {
                    m: 8,
                    ef_construction: 32,
                    ef_search: 16,
                    max_m0: 16,
                    estimated_memory_bytes: Self::estimate_memory(n_samples, 8, 16),
                }
            },
            HnswUseCase::HighAccuracy => {
                // Maximize recall at expense of speed/memory
                Self {
                    m: 64,
                    ef_construction: 256,
                    ef_search: 256,
                    max_m0: 128,
                    estimated_memory_bytes: Self::estimate_memory(n_samples, 64, 128),
                }
            },
            HnswUseCase::MemoryOptimized => {
                // Minimize memory usage
                Self {
                    m: 8,
                    ef_construction: 64,
                    ef_search: 32,
                    max_m0: 16,
                    estimated_memory_bytes: Self::estimate_memory(n_samples, 8, 16),
                }
            },
            HnswUseCase::Balanced => {
                Self::auto_scale(n_samples, n_features, 15) // Default neighbor count
            },
        }
    }

    /// Estimate memory usage for HNSW index
    /// Based on research and empirical measurements
    pub fn estimate_memory(n_samples: usize, m: usize, max_m0: usize) -> usize {
        // Base memory per node
        let base_memory_per_node = 32; // Node overhead, level info, etc.

        // Memory for connections (approximate)
        let avg_connections_per_node = (m + max_m0) as f32 / 2.0;
        let connection_memory_per_node = (avg_connections_per_node * 4.0) as usize; // 4 bytes per connection ID

        // Memory for vectors (assuming f32)
        let vector_memory_per_node = 0; // Vectors stored separately in our case

        let total_per_node = base_memory_per_node + connection_memory_per_node + vector_memory_per_node;
        n_samples * total_per_node
    }

    /// Validate parameters and suggest corrections
    pub fn validate(&mut self) -> Result<(), String> {
        // Validate M parameter
        if self.m < 2 {
            return Err("M must be at least 2 for HNSW connectivity".to_string());
        }
        if self.m > 128 {
            return Err("M > 128 may cause excessive memory usage".to_string());
        }

        // Validate ef_construction
        if self.ef_construction < self.m {
            self.ef_construction = self.m * 2;
        }

        // Validate ef_search
        if self.ef_search < self.m {
            self.ef_search = self.m;
        }

        // Validate max_m0
        if self.max_m0 < self.m {
            self.max_m0 = self.m * 2;
        }

        Ok(())
    }

    /// Get human-readable performance characteristics
    pub fn get_characteristics(&self) -> HnswCharacteristics {
        let construction_speed = match self.ef_construction {
            0..=64 => PerformanceLevel::Fast,
            65..=128 => PerformanceLevel::Medium,
            _ => PerformanceLevel::Slow,
        };

        let search_speed = match self.ef_search {
            0..=32 => PerformanceLevel::Fast,
            33..=64 => PerformanceLevel::Medium,
            _ => PerformanceLevel::Slow,
        };

        let memory_usage = match self.m {
            0..=16 => PerformanceLevel::Low,
            17..=32 => PerformanceLevel::Medium,
            _ => PerformanceLevel::High,
        };

        let accuracy = match (self.m, self.ef_search) {
            (m, ef) if m >= 32 && ef >= 64 => PerformanceLevel::High,
            (m, ef) if m >= 16 && ef >= 32 => PerformanceLevel::Medium,
            _ => PerformanceLevel::Low,
        };

        HnswCharacteristics {
            construction_speed,
            search_speed,
            memory_usage,
            accuracy,
            estimated_memory_mb: self.estimated_memory_bytes / (1024 * 1024),
        }
    }
}

/// HNSW use case optimization targets
#[derive(Debug, Clone, Copy)]
pub enum HnswUseCase {
    /// Minimize index construction time
    FastConstruction,
    /// Maximize search accuracy/recall
    HighAccuracy,
    /// Minimize memory footprint
    MemoryOptimized,
    /// Balanced performance across all metrics
    Balanced,
}

/// Performance level indicators
#[derive(Debug, Clone, Copy)]
pub enum PerformanceLevel {
    Low,
    Medium,
    High,
    Fast, // For speed metrics
    Slow,
}

impl std::fmt::Display for PerformanceLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PerformanceLevel::Low => write!(f, "Low"),
            PerformanceLevel::Medium => write!(f, "Medium"),
            PerformanceLevel::High => write!(f, "High"),
            PerformanceLevel::Fast => write!(f, "Fast"),
            PerformanceLevel::Slow => write!(f, "Slow"),
        }
    }
}

/// Human-readable HNSW performance characteristics
#[derive(Debug)]
pub struct HnswCharacteristics {
    pub construction_speed: PerformanceLevel,
    pub search_speed: PerformanceLevel,
    pub memory_usage: PerformanceLevel,
    pub accuracy: PerformanceLevel,
    pub estimated_memory_mb: usize,
}

impl std::fmt::Display for HnswCharacteristics {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f,
            "HNSW Characteristics:\n\
             • Construction Speed: {}\n\
             • Search Speed: {}\n\
             • Memory Usage: {} (~{}MB)\n\
             • Expected Accuracy: {}",
            self.construction_speed,
            self.search_speed,
            self.memory_usage,
            self.estimated_memory_mb,
            self.accuracy
        )
    }
}

/// Recommend optimal HNSW parameters based on dataset and requirements
#[allow(dead_code)]
pub fn recommend_hnsw_params(
    n_samples: usize,
    n_features: usize,
    _target_neighbors: usize,
    memory_limit_mb: Option<usize>,
    priority: HnswUseCase,
) -> Result<HnswParams, String> {
    // Start with use-case optimized parameters
    let mut params = HnswParams::for_use_case(priority, n_samples, n_features);

    // Apply memory constraints if specified
    if let Some(limit_mb) = memory_limit_mb {
        let limit_bytes = limit_mb * 1024 * 1024;

        if params.estimated_memory_bytes > limit_bytes {
            // Reduce M to fit memory constraint
            while params.estimated_memory_bytes > limit_bytes && params.m > 4 {
                params.m = std::cmp::max(4, params.m - 4);
                params.max_m0 = params.m * 2;
                params.estimated_memory_bytes = HnswParams::estimate_memory(n_samples, params.m, params.max_m0);
            }

            if params.estimated_memory_bytes > limit_bytes {
                return Err(format!(
                    "Cannot fit HNSW index within {}MB memory limit. \
                     Minimum estimated: {}MB",
                    limit_mb,
                    params.estimated_memory_bytes / (1024 * 1024)
                ));
            }
        }
    }

    // Validate final parameters
    params.validate()?;

    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_scale_small_dataset() {
        let params = HnswParams::auto_scale(1000, 50, 15);
        assert_eq!(params.m, 16);
        assert_eq!(params.ef_construction, 64);
        assert!(params.ef_search >= 32); // May be scaled up
    }

    #[test]
    fn test_auto_scale_medium_dataset() {
        let params = HnswParams::auto_scale(100_000, 100, 15);
        assert_eq!(params.m, 32);
        assert_eq!(params.ef_construction, 128);
        assert!(params.ef_search >= 64);
    }

    #[test]
    fn test_auto_scale_large_dataset() {
        let params = HnswParams::auto_scale(2_000_000, 200, 15);
        assert_eq!(params.m, 64);
        assert_eq!(params.ef_construction, 128);
        assert!(params.ef_search >= 128);
    }

    #[test]
    fn test_memory_estimation() {
        let memory = HnswParams::estimate_memory(10_000, 16, 32);
        assert!(memory > 0);
        assert!(memory < 10_000_000); // Reasonable upper bound
    }

    #[test]
    fn test_parameter_validation() {
        let mut params = HnswParams {
            m: 1, // Too small
            ef_construction: 8, // Too small relative to M
            ef_search: 4, // Too small relative to M
            max_m0: 8, // Too small relative to M
            estimated_memory_bytes: 0,
        };

        assert!(params.validate().is_err());
    }

    #[test]
    fn test_use_case_optimization() {
        let fast = HnswParams::for_use_case(HnswUseCase::FastConstruction, 10_000, 50);
        let accurate = HnswParams::for_use_case(HnswUseCase::HighAccuracy, 10_000, 50);
        let memory_opt = HnswParams::for_use_case(HnswUseCase::MemoryOptimized, 10_000, 50);

        // Fast construction should have lower parameters
        assert!(fast.ef_construction < accurate.ef_construction);

        // High accuracy should have higher parameters
        assert!(accurate.m > memory_opt.m);
        assert!(accurate.ef_search > memory_opt.ef_search);

        // Memory optimized should use less memory
        assert!(memory_opt.estimated_memory_bytes <= fast.estimated_memory_bytes);
    }

    #[test]
    fn test_memory_limit_recommendation() {
        let result = recommend_hnsw_params(
            100_000, // samples
            100,     // features
            15,      // neighbors
            Some(50), // 50MB limit
            HnswUseCase::Balanced
        );

        assert!(result.is_ok());
        let params = result.unwrap();
        assert!(params.estimated_memory_bytes <= 50 * 1024 * 1024);
    }
}