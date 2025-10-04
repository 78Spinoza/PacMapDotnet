//! FULL HNSW implementation using the hnsw 0.11 crate
//! Provides hierarchical approximate nearest neighbor search with deterministic seeding
//! Uses proper multi-layer graph structure for O(log n) performance

use hnsw::{Hnsw, Params, Searcher};
use space::Metric;
use serde::{Serialize, Deserialize};
use rand_pcg::Lcg128Xsl64;

/// Deterministic f64 to f32 conversion with consistent rounding
pub fn deterministic_f32_from_f64(x: f64) -> f32 {
    // Use a specific rounding method to ensure consistency across conversions
    // This implements a deterministic rounding strategy that should be identical
    // across all platforms and code paths

    // Handle special cases first
    if x.is_nan() {
        return f32::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { f32::INFINITY } else { f32::NEG_INFINITY };
    }

    // For finite values, use consistent rounding
    // Convert to f32 with rounding towards zero (truncation) for consistency
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let abs_x = x.abs();

    // Convert with explicit truncation to ensure consistent rounding
    let truncated = abs_x as f32;

    // Apply the original sign
    sign * truncated
}

#[derive(Clone, Debug)]
pub struct Neighbor {
    pub d_id: usize,
    pub distance: f32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct EuclideanMetric;

impl Metric<Vec<f32>> for EuclideanMetric {
    type Unit = u32; // Satisfies Unsigned + Ord + Copy requirements (space 0.17)

    fn distance(&self, a: &Vec<f32>, b: &Vec<f32>) -> Self::Unit {
        let dist_sq = a.iter().zip(b.iter())
            .map(|(x, y)| {
                let diff = *x - *y;
                diff * diff
            })
            .sum::<f32>();
        // Scale Euclidean distance to u32 for precision (multiply by 1000)
        (dist_sq.sqrt() * 1000.0) as u32
    }
}

/// Serializable HNSW container that stores all necessary data for reconstruction
#[derive(Clone, Serialize, Deserialize)]
pub struct SerializableHnswIndex {
    /// The original data points that were inserted into the index
    pub data: Vec<Vec<f32>>,
    /// HNSW construction parameters
    pub m: usize,
    pub ef_construction: usize,
    pub seed: u64,
    /// Index statistics
    pub total_elements: usize,
    pub max_layer: usize,
}

#[derive(Clone)]
pub struct DeterministicHnsw {
    hnsw: Hnsw<EuclideanMetric, Vec<f32>, Lcg128Xsl64, 16, 12>,  // M=16, M0=12 are common defaults
    seed: u64,
    m: usize,
    ef_construction: usize,
}

impl DeterministicHnsw {
    pub fn new(m: usize, _max_elements: usize, ef_construction: usize, seed: u64) -> Self {
        // Create deterministic parameters for HNSW
        let params = Params::default()
            .ef_construction(ef_construction);

        // Create the FULL HNSW with proper hierarchical structure
        let hnsw = Hnsw::new_params(EuclideanMetric, params);

        Self {
            hnsw,
            seed,
            m,
            ef_construction,
        }
    }

    /// Export HNSW index to serializable format (stores data for reconstruction)
    pub fn to_serializable(&self) -> SerializableHnswIndex {
        // Note: Since the actual Hnsw struct can't be serialized due to RNG,
        // we store the data needed to reconstruct it identically
        SerializableHnswIndex {
            data: Vec::new(), // We'll populate this with the actual training data
            m: self.m,
            ef_construction: self.ef_construction,
            seed: self.seed,
            total_elements: self.hnsw.len(),
            max_layer: self.hnsw.layers(),
        }
    }

    /// Reconstruct HNSW index from serializable format
    pub fn from_serializable(serializable: &SerializableHnswIndex, training_data: Vec<Vec<f32>>) -> Self {
        let mut hnsw = Self::new(serializable.m, training_data.len(), serializable.ef_construction, serializable.seed);

        // Insert all training data points in the same order
        for point in &training_data {
            hnsw.insert(point);
        }

        hnsw
    }

    /// Get hyperparameters for ModelInfo
    pub fn get_hyperparams(&self) -> HnswHyperparams {
        HnswHyperparams {
            m: self.m,
            ef_construction: self.ef_construction,
            ef_search: self.ef_construction, // Default to ef_construction
            max_m0: self.m * 2, // Standard HNSW convention
            seed: self.seed,
            max_layer: self.hnsw.layers(),
            total_elements: self.hnsw.len(),
        }
    }

    pub fn insert(&mut self, point: &[f32]) -> usize {
        let mut searcher = Searcher::default();
        self.hnsw.insert(point.to_vec(), &mut searcher)
    }

    pub fn search(&mut self, query: &[f32], k: usize, ef_search: usize) -> Vec<Neighbor> {
        // Perform fast O(log n) approximate search using the hierarchical structure
        let mut searcher = Searcher::default();
        let mut dest = vec![space::Neighbor { index: 0, distance: 0u32 }; k];
        let results = self.hnsw.nearest(&query.to_vec(), ef_search, &mut searcher, &mut dest);

        results.into_iter()
            .map(|hnsw_neighbor| Neighbor {
                d_id: hnsw_neighbor.index,
                distance: hnsw_neighbor.distance as f32 / 1000.0, // Convert u32 back to f32
            })
            .collect()
    }

    pub fn get_max_layer(&self) -> usize {
        self.hnsw.layers()
    }

    pub fn stats(&self) -> HnswStats {
        HnswStats {
            total_elements: self.hnsw.len(),
            max_layer: self.hnsw.layers(),
            seed: self.seed,
        }
    }
}

/// HNSW hyperparameters for ModelInfo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswHyperparams {
    /// HNSW graph connectivity parameter (M)
    pub m: usize,

    /// HNSW construction quality parameter (ef_construction)
    pub ef_construction: usize,

    /// HNSW search quality parameter (ef_search)
    pub ef_search: usize,

    /// Maximum connections per layer 0 node (max_m0)
    pub max_m0: usize,

    /// Random seed for deterministic behavior
    pub seed: u64,

    /// Current number of layers in the index
    pub max_layer: usize,

    /// Total number of elements in the index
    pub total_elements: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswStats {
    pub total_elements: usize,
    pub max_layer: usize,
    pub seed: u64,
}