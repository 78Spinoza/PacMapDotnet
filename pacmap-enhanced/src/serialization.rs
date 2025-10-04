use serde::{Serialize, Deserialize};
use zstd::{stream::encode_all as compress, stream::decode_all as decompress};
use std::fs::File;
use std::io::{Read, Write};
use ndarray::{Array2, Array1};
use crate::quantize::{quantize_embedding_linear, QuantizedEmbedding, dequantize_embedding};
use crate::stats::NormalizationParams;
use crate::hnsw_params::HnswParams;

// HNSW serialization disabled - indices will be rebuilt on demand
// The new deterministic hnsw crate doesn't support the same serialization
//#[cfg(feature = "use_hnsw")]
//use crate::hnsw_wrapper::DeterministicHnsw;

// Import is_verbose function from lib.rs
use crate::is_verbose;

// Define vprint macro locally since macros don't auto-import
macro_rules! vprint {
    ($($arg:tt)*) => {
        if is_verbose() {
            eprintln!($($arg)*);
        }
    };
}

/// A serializable container for all the data needed to rebuild an HNSW index.
/// This is a robust alternative to trying to serialize the Hnsw struct itself.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SerializableHnswIndex {
    /// The original data points that were inserted into the index.
    pub data: Vec<Vec<f32>>,
    /// The HNSW parameters used to build the index.
    pub params: HnswParams,
    /// The number of layers in the original index.
    pub max_layer: usize,
}

/// Statistics calculated for a transformed data point to assess its relationship
/// to the original training data's embedding. Used for "No Man's Land" detection.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TransformStats {
    /// The low-dimensional coordinates of the transformed point.
    pub coordinates: Array1<f64>,

    /// The distance to the single closest neighbor in the training embedding.
    /// A primary indicator of outlierness.
    pub distance_to_closest_neighbor: f64,

    /// The mean distance to the k nearest neighbors in the training embedding.
    /// Provides a more stable measure of local density.
    pub mean_distance_to_k_neighbors: f64,

    /// The distance from the point to the centroid of the entire training embedding.
    /// A global measure of how far the point is from the "center" of the data.
    pub distance_to_training_centroid: f64,
}

impl Default for TransformStats {
    fn default() -> Self {
        Self {
            coordinates: Array1::zeros(2), // Default 2D embedding
            distance_to_closest_neighbor: 0.0,
            mean_distance_to_k_neighbors: 0.0,
            distance_to_training_centroid: 0.0,
        }
    }
}

/// Core PacMAP model parameters (serializable subset of Configuration)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PacMAPConfig {
    pub n_neighbors: usize,
    pub embedding_dim: usize,
    pub n_epochs: usize,
    pub learning_rate: f64,
    pub mid_near_ratio: f64,
    pub far_pair_ratio: f64,
    pub seed: Option<u64>,
    /// HNSW parameters for neighbor search optimization
    pub hnsw_params: HnswParams,
    /// Whether HNSW was actually used during fitting (vs exact KNN)
    #[serde(default)]
    pub used_hnsw: bool,
    /// Force direct KNN instead of HNSW regardless of dataset size
    #[serde(default)]
    pub force_knn: bool,
}

impl Default for PacMAPConfig {
    fn default() -> Self {
        Self {
            n_neighbors: 10,
            embedding_dim: 2,
            n_epochs: 450,
            learning_rate: 1.0,
            mid_near_ratio: 0.5,
            far_pair_ratio: 2.0,
            seed: None,
            hnsw_params: HnswParams::default(),
            used_hnsw: false,
            force_knn: false,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct PaCMAP {
    pub embedding: Array2<f64>,
    pub config: PacMAPConfig,
    pub stats: DistanceStats,
    /// Normalization parameters - CRITICAL for consistent transforms (like UMAP)
    /// These parameters are computed during training and must be applied to new data
    pub normalization: NormalizationParams,
    /// If true, the model will be saved with the embedding quantized to `f16`
    /// to reduce the on‑disk size. The quantization is performed lazily when
    /// `save_compressed` is called.
    #[serde(default)]
    pub quantize_on_save: bool,
    /// Quantized embedding with parameters (replaces simple embedding_q)
    #[serde(default)]
    pub quantized_embedding: Option<QuantizedEmbedding>,

    /// TRANSFORM SUPPORT: Original high-dimensional training data (full precision for exact KNN)
    /// Used for exact neighbor search when KNN algorithm is used
    #[serde(default)]
    pub original_data_full: Option<Array2<f64>>,

    /// TRANSFORM SUPPORT: Original high-dimensional training data (quantized for HNSW)
    /// Used for initial neighbor search when transforming new points with HNSW
    #[serde(default)]
    pub original_data: Option<QuantizedEmbedding>,

    /// TRANSFORM SUPPORT: Fitted low-dimensional projections (ALWAYS SAVED)
    /// Critical for accurate transforms - contains exact fitted coordinates
    /// Used for final neighbor refinement in embedding space
    pub fitted_projections: Array2<f64>,

    /// The centroid of the fitted embedding space.
    /// Used for global outlierness detection during transform ("No Man's Land" detection).
    #[serde(default)]
    pub embedding_centroid: Option<Array1<f64>>,

    /// HNSW indices - re-enabled with new deterministic hnsw crate
    #[serde(skip)]
    #[cfg(feature = "use_hnsw")]
    pub hnsw_index: Option<crate::hnsw_wrapper::DeterministicHnsw>,
    #[serde(skip)]
    #[cfg(feature = "use_hnsw")]
    pub embedding_hnsw_index: Option<crate::hnsw_wrapper::DeterministicHnsw>,

    /// The raw bytes of a serialized HNSW index for original data.
    /// Used for massive datasets where rebuilding is too slow.
    #[serde(default)]
    pub serialized_hnsw_index: Option<Vec<u8>>,

    // REMOVED: Never save transformed space HNSW index
    // Always rebuild from fitted_projections for accuracy
    // pub serialized_embedding_hnsw_index: Option<Vec<u8>>, // REMOVED

    /// CRC32 checksum of the serialized HNSW index for original data.
    /// Used for integrity validation when loading.
    #[serde(default)]
    pub hnsw_index_crc32: Option<u32>,

    /// CRC32 checksum of the fitted projections data (replaces embedding HNSW CRC32)
    /// Used for integrity validation of critical fitted projection data
    #[serde(default)]
    pub fitted_projections_crc32: Option<u32>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DistanceStats {
    pub mean_distance: f64,
    pub p95_distance: f64,
    pub max_distance: f64,
}

impl Default for DistanceStats {
    fn default() -> Self {
        Self {
            mean_distance: 0.0,
            p95_distance: 0.0,
            max_distance: 0.0,
        }
    }
}

impl PaCMAP {
    /// Prepare a quantized copy of the embedding for saving.
    /// This is called automatically by `save_compressed` when
    /// `quantize_on_save` is true.
    ///
    /// The quantized embedding with parameters is stored in `quantized_embedding`.
    /// The original high‑precision `embedding` is left untouched.
    pub fn quantize_for_save(&mut self) {
        if self.quantized_embedding.is_none() {
            self.quantized_embedding = Some(quantize_embedding_linear(&self.embedding));
        }
    }

    /// Store original training data and fitted projections for transform support
    pub fn store_transform_data(&mut self, original_data: &Array2<f64>, fitted_projections: &Array2<f64>) {
        vprint!("STORING: Storing transform data: original {}x{}, fitted {}x{}",
               original_data.shape()[0], original_data.shape()[1],
               fitted_projections.shape()[0], fitted_projections.shape()[1]);

        // Store original data based on algorithm type
        // HNSW models: NO original data storage (only index), Exact KNN: original data
        if self.config.used_hnsw {
            // HNSW mode: NO original data storage - HNSW index handles neighbor searches
            vprint!("HNSW MODE: No original data storage (using HNSW index for neighbor search)");
        } else if self.quantize_on_save {
            // Exact KNN with quantization enabled - save quantized original data for space efficiency
            self.original_data = Some(quantize_embedding_linear(original_data));
            vprint!("QUANTIZATION ENABLED: Stored quantized original training data for Exact KNN");
        } else {
            // Full precision storage for Exact KNN models
            self.original_data_full = Some(original_data.clone());
            vprint!("FULL PRECISION: Stored full precision original training data for Exact KNN");
        }

        // Store fitted projections (low-dim embeddings) - always full precision
        self.fitted_projections = fitted_projections.clone();

        vprint!("SUCCESS: Transform data stored successfully");
    }

    /// Get original training data for transforms (full precision preferred)
    pub fn get_original_data(&self) -> Option<Array2<f64>> {
        // Always prefer full precision data when available
        if let Some(ref full_data) = self.original_data_full {
            vprint!("TRANSFORM: Using full precision original training data");
            return Some(full_data.clone());
        }

        // Use quantized data only when explicitly enabled during save
        if let Some(ref quantized) = self.original_data {
            vprint!("TRANSFORM: Using quantized original training data (quantization was enabled)");
            return Some(dequantize_embedding(quantized));
        }

        vprint!("WARNING: No original training data available for transform");
        None
    }

    /// Get the embedding in f64 precision, dequantizing if necessary
    pub fn get_embedding(&self) -> Array2<f64> {
        if let Some(ref quantized) = self.quantized_embedding {
            // If we have quantized data, dequantize it
            dequantize_embedding(quantized)
        } else {
            // Return original embedding
            self.embedding.clone()
        }
    }

    /// Temporarily disabled during migration to deterministic hnsw crate
    /// TODO: Re-implement with DeterministicHnsw wrapper
    #[cfg(feature = "use_hnsw")]
    pub fn ensure_hnsw_index(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Re-enabled with deterministic hnsw 0.11 implementation
        if self.hnsw_index.is_some() {
            return Ok(());
        }

        if let Some(ref serialized_bytes) = self.serialized_hnsw_index {
            // FAST PATH: Load from serialized index
            vprint!("LOADING: Loading HNSW index from serialized data (fast path)");
            let serialized_index = deserialize_hnsw_from_bytes(serialized_bytes)?;

            // Build HNSW directly from serialized data using deterministic hnsw 0.11
            let n_samples = serialized_index.data.len();
            let seed = self.config.seed.unwrap_or(42);
            let mut hnsw = crate::hnsw_wrapper::DeterministicHnsw::new(
                serialized_index.params.m,
                n_samples,
                serialized_index.params.ef_construction,
                seed,
            );

            // Insert all data points
            for point in &serialized_index.data {
                hnsw.insert(point.as_slice());
            }

            self.hnsw_index = Some(hnsw);
            vprint!("SUCCESS: HNSW index loaded from serialized data");
        } else {
            // SLOW PATH: Rebuild from stored data for exact KNN, HNSW should never reach here
            if self.config.used_hnsw {
                // HNSW models should always have serialized index - this is an error case
                return Err("HNSW index not found in serialized data and original training data not stored. Cannot rebuild HNSW index.".into());
            }

            vprint!("REBUILDING: HNSW index not found, rebuilding from stored training data...");
            let original_data = self.get_original_data().ok_or("Cannot rebuild HNSW index: original training data not stored")?;
            let (n_samples, _) = original_data.dim();

            let hnsw_params = &self.config.hnsw_params;
            let points: Vec<Vec<f32>> = (0..n_samples)
                .map(|i| original_data.row(i).iter().map(|&x| crate::hnsw_wrapper::deterministic_f32_from_f64(x)).collect())
                .collect();

            // Create deterministic HNSW using our wrapper
            let seed = self.config.seed.unwrap_or(42);
            let mut hnsw = crate::hnsw_wrapper::DeterministicHnsw::new(
                hnsw_params.m,
                n_samples,
                hnsw_params.ef_construction,
                seed,
            );

            // Insert all points with deterministic HNSW 0.11 API
            for point in &points {
                hnsw.insert(point.as_slice());
            }

            self.hnsw_index = Some(hnsw);
            vprint!("SUCCESS: HNSW index rebuilt successfully.");
        }
        Ok(())
    }

    /* OLD IMPLEMENTATION - DISABLED
    #[cfg(feature = "use_hnsw")]
    pub fn ensure_hnsw_index_OLD(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.hnsw_index.is_some() {
            return Ok(());
        }

        if let Some(ref serialized_bytes) = self.serialized_hnsw_index {
            // FAST PATH: Load from serialized index
            vprint!("LOADING: Loading HNSW index from serialized data (fast path)");
            let serialized_index = deserialize_hnsw_from_bytes(serialized_bytes)?;

            // Build HNSW directly from serialized data
            let n_samples = serialized_index.data.len();
            // Rebuild HNSW from serialized data using deterministic hnsw 0.11
            use hnsw::Hnsw;
            use space::Metric;
            use rand_pcg::Pcg64;

            let seed = self.config.seed.unwrap_or(42);
            let rng = Pcg64::seed_from_u64(seed);

            let hnsw = Hnsw::<f32, crate::hnsw_wrapper::EuclideanMetric>::new(
                serialized_index.params.m,
                n_samples,
                serialized_index.params.ef_construction,
                crate::hnsw_wrapper::EuclideanMetric,
                rng,
            );

            // Insert all data points
            for point in &serialized_index.data {
                hnsw.insert(point.as_slice());
            }

            self.hnsw_index = Some(hnsw);
        } else {
            // SLOW PATH: Rebuild from stored data for exact KNN, HNSW should never reach here
            if self.config.used_hnsw {
                // HNSW models should always have serialized index - this is an error case
                return Err("HNSW index not found in serialized data and original training data not stored. Cannot rebuild HNSW index.".into());
            }

            vprint!("REBUILDING: HNSW index not found, rebuilding from stored training data...");
            let original_data = self.get_original_data().ok_or("Cannot rebuild HNSW index: original training data not stored")?;
            let (n_samples, _) = original_data.dim();

            let hnsw_params = &self.config.hnsw_params;
            let points: Vec<Vec<f32>> = (0..n_samples)
                .map(|i| original_data.row(i).iter().map(|&x| crate::hnsw_wrapper::deterministic_f32_from_f64(x)).collect())
                .collect();

            let max_layer = ((n_samples as f32).ln() / (hnsw_params.m as f32).ln()).ceil() as usize + 1;
            let max_layer = max_layer.min(32).max(4);

            // Create deterministic HNSW using our wrapper
            let seed = self.config.seed.unwrap_or(42);
            let mut hnsw = crate::hnsw_wrapper::DeterministicHnsw::new(
                hnsw_params.m,
                n_samples,
                hnsw_params.ef_construction,
                seed,
            );

            // Insert all points with deterministic HNSW 0.11 API
            for point in &points {
                hnsw.insert(point.as_slice());
            }

            self.hnsw_index = Some(hnsw);
            vprint!("SUCCESS: HNSW index rebuilt successfully.");
        }
        Ok(())
    }
    */ // End of OLD ensure_hnsw_index

    /// Temporarily disabled during migration to deterministic hnsw crate
    #[cfg(feature = "use_hnsw")]
    pub fn ensure_embedding_hnsw_index(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Temporarily disabled - HNSW indices must be built externally
        eprintln!("WARNING: HNSW serialization temporarily disabled during migration");
        Ok(())
    }

    /* OLD IMPLEMENTATION - DISABLED
    #[cfg(feature = "use_hnsw")]
    pub fn ensure_embedding_hnsw_index_OLD(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.embedding_hnsw_index.is_some() {
            return Ok(());
        }

        // ALWAYS REBUILD: Never use serialized embedding HNSW index for accuracy
        vprint!("REBUILDING: Building embedding HNSW index from exact fitted projections...");
        let fitted_projections = &self.fitted_projections; // No longer optional - always saved
        let (n_samples, _) = fitted_projections.dim();

        let hnsw_params = &self.config.hnsw_params;
        let points: Vec<Vec<f32>> = (0..n_samples)
            .map(|i| fitted_projections.row(i).iter().map(|&x| crate::hnsw_wrapper::deterministic_f32_from_f64(x)).collect())
            .collect();

        let max_layer = ((n_samples as f32).ln() / (hnsw_params.m as f32).ln()).ceil() as usize + 1;
        let max_layer = max_layer.min(32).max(4);

        // Create deterministic HNSW with seed from config
        use hnsw::Hnsw;
        use space::Metric;
        use rand_pcg::Pcg64;

        let seed = self.config.seed.unwrap_or(42);
        let rng = Pcg64::seed_from_u64(seed);

        let hnsw = Hnsw::<f32, crate::hnsw_wrapper::EuclideanMetric>::new(
            hnsw_params.m,
            n_samples,
            hnsw_params.ef_construction,
            crate::hnsw_wrapper::EuclideanMetric,
            rng,
        );

        // Insert all points with deterministic HNSW 0.11 API
        for point in &points {
            hnsw.insert(point.as_slice());
        }

        self.embedding_hnsw_index = Some(hnsw);
        vprint!("SUCCESS: Embedding HNSW index rebuilt successfully.");
        Ok(())
    }
    */ // End of OLD ensure_embedding_hnsw_index

    /// Save without quantization (preserves full precision)
    pub fn save_uncompressed(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.save_uncompressed_with_progress(path, None)
    }

    /// Save without quantization with progress reporting
    pub fn save_uncompressed_with_progress(
        &self,
        path: &str,
        progress_callback: Option<&crate::ProgressCallback>
    ) -> Result<(), Box<dyn std::error::Error>> {
        let progress = |phase: &str, current: usize, total: usize, percent: f32, message: &str| {
            if let Some(callback) = progress_callback {
                callback(phase, current, total, percent, message);
            }
        };

        progress("Preparing", 0, 100, 0.0, "Preparing model for uncompressed save");

        // Display model settings
        self.print_model_settings("Saving Uncompressed Model");

        progress("Serializing", 20, 100, 20.0, "Serializing model to MessagePack format");
        let serialized = rmp_serde::to_vec(self)?;

        progress("Compressing", 60, 100, 60.0, "Compressing with ZSTD");
        let compressed = compress(&serialized[..], 3)?;

        progress("Writing", 80, 100, 80.0, &format!("Writing to file: {}", path));
        File::create(path)?.write_all(&compressed)?;

        let file_size = std::fs::metadata(path)?.len();
        progress("Complete", 100, 100, 100.0, &format!("Save completed - {} bytes", file_size));

        println!("SUCCESS: Model saved successfully to: {}", path);
        println!("    File size: {} bytes ({:.2} KB)", file_size, file_size as f64 / 1024.0);

        Ok(())
    }

    pub fn save_compressed(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.save_compressed_with_progress(path, None)
    }

    /// Save with optional quantization and progress reporting
    pub fn save_compressed_with_progress(
        &mut self,
        path: &str,
        progress_callback: Option<&crate::ProgressCallback>
    ) -> Result<(), Box<dyn std::error::Error>> {
        let progress = |phase: &str, current: usize, total: usize, percent: f32, message: &str| {
            if let Some(callback) = progress_callback {
                callback(phase, current, total, percent, message);
            }
        };

        progress("Preparing", 0, 100, 0.0, "Preparing model for compressed save");

        // Display model settings
        let save_type = if self.quantize_on_save { "Saving Quantized Model" } else { "Saving Compressed Model" };
        self.print_model_settings(save_type);

        // If quantization is enabled, quantize and remove original to save space
        if self.quantize_on_save {
            progress("Quantizing", 20, 100, 20.0, "Quantizing embedding to reduce file size");
            self.quantize_for_save();
            // Replace original embedding with minimal placeholder to save space
            self.embedding = Array2::zeros((1, 1));
        }

        progress("Serializing", 40, 100, 40.0, "Serializing model to MessagePack format");
        let serialized = rmp_serde::to_vec(self)?;

        progress("Compressing", 70, 100, 70.0, "Compressing with ZSTD");
        let compressed = compress(&serialized[..], 3)?;

        progress("Writing", 90, 100, 90.0, &format!("Writing to file: {}", path));
        File::create(path)?.write_all(&compressed)?;

        let file_size = std::fs::metadata(path)?.len();
        progress("Complete", 100, 100, 100.0, &format!("Save completed - {} bytes", file_size));

        println!("SUCCESS: Model saved successfully to: {}", path);
        println!("    File size: {} bytes ({:.2} KB)", file_size, file_size as f64 / 1024.0);
        if self.quantize_on_save {
            println!("     Quantization enabled - model size optimized");
        }

        Ok(())
    }

    pub fn load_compressed(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_compressed_with_progress(path, None)
    }

    /// Load model with progress reporting
    pub fn load_compressed_with_progress(
        path: &str,
        progress_callback: Option<&crate::ProgressCallback>
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let progress = |phase: &str, current: usize, total: usize, percent: f32, message: &str| {
            if let Some(callback) = progress_callback {
                callback(phase, current, total, percent, message);
            }
        };

        progress("Reading", 0, 100, 0.0, &format!("Reading model file: {}", path));

        let mut file = File::open(path)?;
        let mut compressed = Vec::new();
        file.read_to_end(&mut compressed)?;

        let file_size = compressed.len();
        progress("Decompressing", 30, 100, 30.0, &format!("Decompressing {} bytes", file_size));

        let decompressed = decompress(std::io::Cursor::new(&compressed))?;

        progress("Deserializing", 70, 100, 70.0, "Deserializing model from MessagePack");
        let model: PaCMAP = rmp_serde::from_slice(&decompressed)?;

        progress("Validating", 90, 100, 90.0, "Validating loaded model");

        // Display loaded model settings
        model.print_model_settings("Loaded Model");

        progress("Complete", 100, 100, 100.0, "Model loading completed successfully");

        println!("SUCCESS: Model loaded successfully from: {}", path);
        println!("    File size: {} bytes ({:.2} KB)", file_size, file_size as f64 / 1024.0);

        Ok(model)
    }

    /// Print comprehensive model settings
    pub fn print_model_settings(&self, title: &str) {
        println!();
        println!("===========================================");
        println!("   {}", title);
        println!("===========================================");
        println!();

        // Core PacMAP Configuration
        println!("  Core PacMAP Configuration:");
        println!("   - Embedding dimensions: {}D", self.config.embedding_dim);
        println!("   - Number of neighbors: {}", self.config.n_neighbors);
        println!("   - Number of epochs: {}", self.config.n_epochs);
        println!("   - Learning rate: {:.3}", self.config.learning_rate);
        println!("   - Mid-near ratio: {:.2}", self.config.mid_near_ratio);
        println!("   - Far-pair ratio: {:.2}", self.config.far_pair_ratio);
        if let Some(seed) = self.config.seed {
            println!("   - Random seed: {} (deterministic)", seed);
        } else {
            println!("   - Random seed: None (non-deterministic)");
        }

        println!();

        // HNSW Configuration
        println!(" HNSW Neighbor Search Configuration:");
        println!("   - M parameter: {}", self.config.hnsw_params.m);
        println!("   - ef_construction: {}", self.config.hnsw_params.ef_construction);
        println!("   - ef_search: {}", self.config.hnsw_params.ef_search);
        println!("   - Estimated memory: {} MB", self.config.hnsw_params.estimated_memory_bytes / (1024 * 1024));
        println!("   - Characteristics: {}", self.config.hnsw_params.get_characteristics());

        println!();

        // Normalization Configuration
        println!(" Data Normalization Configuration:");
        println!("   - Mode: {:?}", self.normalization.mode);
        println!("   - Features: {}", self.normalization.n_features);
        match self.normalization.mode {
            crate::stats::NormalizationMode::ZScore => {
                if !self.normalization.means.is_empty() {
                    println!("   - Means: [{:.3}, ..., {:.3}] (length: {})",
                             self.normalization.means[0],
                             self.normalization.means.last().unwrap_or(&0.0),
                             self.normalization.means.len());
                }
                if !self.normalization.stds.is_empty() {
                    println!("   - Std devs: [{:.3}, ..., {:.3}] (length: {})",
                             self.normalization.stds[0],
                             self.normalization.stds.last().unwrap_or(&1.0),
                             self.normalization.stds.len());
                }
            },
            crate::stats::NormalizationMode::MinMax => {
                if !self.normalization.mins.is_empty() {
                    println!("   - Minimums: [{:.3}, ..., {:.3}] (length: {})",
                             self.normalization.mins[0],
                             self.normalization.mins.last().unwrap_or(&0.0),
                             self.normalization.mins.len());
                }
                if !self.normalization.maxs.is_empty() {
                    println!("   - Maximums: [{:.3}, ..., {:.3}] (length: {})",
                             self.normalization.maxs[0],
                             self.normalization.maxs.last().unwrap_or(&1.0),
                             self.normalization.maxs.len());
                }
            },
            crate::stats::NormalizationMode::Robust => {
                if !self.normalization.medians.is_empty() {
                    println!("   - Medians: [{:.3}, ..., {:.3}] (length: {})",
                             self.normalization.medians[0],
                             self.normalization.medians.last().unwrap_or(&0.0),
                             self.normalization.medians.len());
                }
                if !self.normalization.iqrs.is_empty() {
                    println!("   - IQRs: [{:.3}, ..., {:.3}] (length: {})",
                             self.normalization.iqrs[0],
                             self.normalization.iqrs.last().unwrap_or(&1.0),
                             self.normalization.iqrs.len());
                }
            },
            _ => {}
        }

        println!();

        // Embedding Information
        println!(" Embedding Information:");
        println!("   - Shape: {:?}", self.embedding.shape());
        let non_zero_count = self.embedding.iter().filter(|&&x| x.abs() > 1e-10).count();
        println!("   - Non-zero values: {}/{} ({:.1}%)",
                 non_zero_count, self.embedding.len(),
                 (non_zero_count as f64 / self.embedding.len() as f64) * 100.0);

        println!();

        // Distance Statistics
        println!("Distance Statistics:");
        println!("   - Mean distance: {:.6}", self.stats.mean_distance);
        println!("   - 95th percentile: {:.6}", self.stats.p95_distance);
        println!("   - Maximum distance: {:.6}", self.stats.max_distance);

        println!();

        // Quantization Information
        if self.quantize_on_save {
            println!("  Quantization Configuration:");
            println!("   - Quantize on save: Enabled");
            if let Some(ref quantized) = self.quantized_embedding {
                println!("   - Quantized shape: {:?}", quantized.data.shape());
                println!("   - Min value: {:.6}", quantized.params.min_value);
                println!("   - Max value: {:.6}", quantized.params.max_value);
                println!("   - Scale factor: {:.6}", quantized.params.scale);
                println!("   - Zero point: {}", quantized.params.zero_point);
                if let Some(ref centroids) = quantized.params.centroids {
                    if !centroids.is_empty() {
                        println!("   - Centroids: {} values", centroids.len());
                    }
                } else {
                    println!("   - Centroids: None");
                }
            } else {
                println!("   - Quantized embedding: Not yet computed");
            }
        } else {
            println!("  Quantization: Disabled (full precision preserved)");
        }

        println!();
        println!("===========================================");
        println!();
    }
}

impl Clone for PaCMAP {
    fn clone(&self) -> Self {
        Self {
            embedding: self.embedding.clone(),
            config: self.config.clone(),
            stats: self.stats.clone(),
            normalization: self.normalization.clone(),
            quantize_on_save: self.quantize_on_save,
            quantized_embedding: self.quantized_embedding.clone(),
            original_data_full: self.original_data_full.clone(),
            original_data: self.original_data.clone(),
            fitted_projections: self.fitted_projections.clone(),
            embedding_centroid: self.embedding_centroid.clone(),
            #[cfg(feature = "use_hnsw")]
            hnsw_index: None, // Don't clone HNSW indices - they'll be rebuilt on demand
            #[cfg(feature = "use_hnsw")]
            embedding_hnsw_index: None,
            serialized_hnsw_index: self.serialized_hnsw_index.clone(),
            // REMOVED: Never save transformed space HNSW index
            // serialized_embedding_hnsw_index: self.serialized_embedding_hnsw_index.clone(), // REMOVED
            hnsw_index_crc32: self.hnsw_index_crc32,
            fitted_projections_crc32: self.fitted_projections_crc32,
        }
    }
}

/// Create serialized HNSW index using binary MessagePack format (much faster than JSON)
#[cfg(feature = "use_hnsw")]
pub fn custom_serialize_hnsw(data: &Array2<f64>, hnsw_params: &HnswParams, max_layer: usize) -> Result<SerializableHnswIndex, Box<dyn std::error::Error>> {
    let data_f32: Vec<Vec<f32>> = data.rows()
        .into_iter()
        .map(|row| row.iter().map(|&x| x as f32).collect())
        .collect();

    let serializable = SerializableHnswIndex {
        data: data_f32,
        params: hnsw_params.clone(),
        max_layer,
    };

    Ok(serializable)
}

/// Serialize SerializableHnswIndex to binary bytes using MessagePack
pub fn serialize_hnsw_to_bytes(serializable_index: &SerializableHnswIndex) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let serialized_bytes = rmp_serde::to_vec(serializable_index)?;
    Ok(serialized_bytes)
}

/// Deserialize SerializableHnswIndex from binary bytes using MessagePack
pub fn deserialize_hnsw_from_bytes(serialized_bytes: &[u8]) -> Result<SerializableHnswIndex, Box<dyn std::error::Error>> {
    let serializable_index: SerializableHnswIndex = rmp_serde::from_slice(serialized_bytes)?;
    Ok(serializable_index)
}

// Load HNSW from serialized index - Updated for new HNSW 0.11
#[cfg(feature = "use_hnsw")]
pub fn load_hnsw_from_serialized(serialized_index: &SerializableHnswIndex) -> Result<crate::hnsw_wrapper::DeterministicHnsw, Box<dyn std::error::Error>> {
    use crate::hnsw_wrapper::DeterministicHnsw;

    let n_samples = serialized_index.data.len();

    let mut hnsw = DeterministicHnsw::new(
        serialized_index.params.m,
        n_samples,
        serialized_index.params.ef_construction,
        42 // Fixed seed for reproducibility
    );

    // Insert all data points
    for (_i, point) in serialized_index.data.iter().enumerate() {
        hnsw.insert(point.as_slice());
    }

    Ok(hnsw)
}
