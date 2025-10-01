use ndarray::Array2;
use pacmap::{Configuration, fit_transform, PairConfiguration};
use crate::pairs::{compute_pairs_hnsw, get_knn_indices};
#[cfg(feature = "use_hnsw")]
use hnsw_rs::{hnsw::Hnsw, dist::DistL2};
use ndarray::Array1;
use std::sync::atomic::{AtomicBool, Ordering};
// Removed unused HashSet import

/// Global verbose toggle to control print statements
static VERBOSE_MODE: AtomicBool = AtomicBool::new(false);

/// Enable or disable verbose logging
pub fn set_verbose(enabled: bool) {
    VERBOSE_MODE.store(enabled, Ordering::Relaxed);
}

/// Check if verbose mode is enabled
pub fn is_verbose() -> bool {
    VERBOSE_MODE.load(Ordering::Relaxed) || std::env::var("PACMAP_VERBOSE").is_ok()
}

/// Conditional print macro - only prints if verbose mode is enabled
macro_rules! vprint {
    ($($arg:tt)*) => {
        if is_verbose() {
            eprintln!($($arg)*);
        }
    };
}
use crate::quantize::{quantize_embedding};
use half::f16;
use crate::serialization::{PaCMAP, DistanceStats, PacMAPConfig};
use crate::stats::{compute_distance_stats, NormalizationParams, NormalizationMode, recommend_normalization_mode};
use crate::hnsw_params::HnswParams;
use crate::recall_validation::validate_hnsw_quality_with_retry;
mod pairs;
mod quantize;
pub mod serialization;
mod stats;
mod hnsw_params;
pub mod ffi;
#[cfg(test)]
mod test_normalization;
#[cfg(test)]
mod test_hnsw_params;
#[cfg(test)]
mod test_ffi;
#[cfg(test)]
mod test_working_hnsw;
mod recall_validation;

/// Fit the data using HNSW-accelerated PaCMAP transformation
pub fn fit_transform_hnsw(data: Array2<f64>, config: Configuration, force_exact_knn: bool, progress_callback: Option<&(dyn Fn(&str, usize, usize, f32, &str) + Send + Sync)>) -> Result<(Array2<f64>, Option<HnswParams>), Box<dyn std::error::Error>> {
    use std::time::Instant;
    let start_time = Instant::now();
    let n_neighbors = config.override_neighbors.unwrap_or(10);
    let seed = config.seed.unwrap_or(42);
    let (n_samples, _) = data.dim();

    // Determine whether to use HNSW or fall back to standard PaCMAP
    // Force exact KNN overrides the size threshold
    let use_hnsw = !force_exact_knn && n_samples > 1000;

    // DEBUG: Report actual parameters via callback
    let debug_msg = format!(" DLL DEBUG: force_exact_knn={}, n_samples={}, use_hnsw={}", force_exact_knn, n_samples, use_hnsw);
    if let Some(callback) = progress_callback {
        callback("DLL Debug", 5, 100, 5.0, &debug_msg);
    }

    // Helper function to call enhanced progress callback with timing
    let report_progress = |phase: &str, current: usize, total: usize, percent: f32, message: &str| {
        if let Some(callback) = progress_callback {
            let elapsed = start_time.elapsed();
            let enhanced_message = format!("{} (⏱️ {})", message, format_duration(elapsed));
            callback(phase, current, total, percent, &enhanced_message);
        }
    };

    // Track optimized HNSW parameters to return them for model storage
    let mut optimized_hnsw_params: Option<HnswParams> = None;

    let updated_config = if use_hnsw {
        vprint!(" DEBUG: use_hnsw=true, force_exact_knn={}, n_samples={}", force_exact_knn, n_samples);
        vprint!(" Using HNSW-accelerated neighbor search for {} samples", n_samples);
        report_progress("KNN Method", 25, 100, 25.0, "Using HNSW for fast approximate neighbor search");

        // Compute HNSW parameters for validation and optimization
        let (_, n_features) = data.dim();
        let mut hnsw_params = HnswParams::auto_scale(n_samples, n_features, n_neighbors);

        // Validate HNSW recall quality with auto-retry and ef_search optimization
        match validate_hnsw_quality_with_retry(data.view(), n_neighbors, seed, hnsw_params.ef_search, Some(&report_progress)) {
            Ok((quality_msg, optimized_ef_search)) => {
                // Update ef_search with optimized value from retry mechanism
                if optimized_ef_search != hnsw_params.ef_search {
                    vprint!("HNSW ef_search optimized: {} -> {} (auto-retry tuning)", hnsw_params.ef_search, optimized_ef_search);
                    hnsw_params.ef_search = optimized_ef_search;
                }
                vprint!("HNSW validation result: {}", quality_msg);
            },
            Err(error) => {
                return Err(format!("HNSW validation failed: {}", error).into());
            }
        }

        // Store optimized parameters for model storage
        optimized_hnsw_params = Some(hnsw_params.clone());

        // Compute neighbor pairs using HNSW with optimized parameters
        vprint!("DEBUG: Starting HNSW neighbor computation with ef_search={}...", hnsw_params.ef_search);
        let original_hnsw_pairs = compute_pairs_hnsw(data.view(), n_neighbors, seed);
        vprint!("DEBUG: HNSW neighbor computation completed");

        // FIXED: Use per-point symmetrization with proper distance handling
        vprint!("DEBUG: Converting to per-point neighbors with distances...");
        let hnsw_pairs = match crate::pairs::compute_pairs_hnsw_to_per_point(data.view(), n_neighbors, seed) {
            Ok(mut nn_per_point) => {
                vprint!("DEBUG: Applying symmetric per-point merging...");
                symmetrize_per_point(&mut nn_per_point, n_neighbors);

                // Convert back to pairs format with guaranteed size
                let mut pairs = Vec::with_capacity(n_samples * n_neighbors);
                for (i, neighbors) in nn_per_point.iter().enumerate() {
                    for &(j, _dist) in neighbors {
                        pairs.push((i, j));
                    }
                }
                vprint!("DEBUG: Per-point symmetrization completed: {} pairs", pairs.len());
                pairs
            },
            Err(e) => {
                vprint!("ERROR: Failed to compute per-point neighbors: {}, using original pairs without symmetrization", e);
                // Fallback to original pairs without symmetrization
                original_hnsw_pairs
            }
        };


        // Handle HNSW pair count - truncate or warn if mismatch
        let expected_pairs = n_samples * n_neighbors;
        let actual_pairs = hnsw_pairs.len();

        if actual_pairs != expected_pairs {
            vprint!("WARNING:  HNSW pair count mismatch: expected {} ({}×{}), got {} - adjusting",
                   expected_pairs, n_samples, n_neighbors, actual_pairs);
        }

        // Check if HNSW returned sufficient pairs (at least 90% of expected)
        let min_acceptable_pairs = (expected_pairs as f32 * 0.9) as usize;

        if actual_pairs < min_acceptable_pairs {
            vprint!("WARNING: HNSW returned insufficient pairs ({} < {}), falling back to exact KNN", actual_pairs, min_acceptable_pairs);
            // Force exact KNN by using default PairConfiguration (not HNSW-based)
            Configuration {
                pair_configuration: PairConfiguration::default(),  // Force exact KNN - no HNSW
                ..config
            }
        } else {

        // Convert pairs to required format: Array2<u32> with shape (expected_pairs, 2)
        let mut pair_neighbors = Array2::<u32>::zeros((expected_pairs, 2));

        // Fill with available pairs, truncated to expected count
        for (idx, &(i, j)) in hnsw_pairs.iter().take(expected_pairs).enumerate() {
            pair_neighbors[[idx, 0]] = i as u32;
            pair_neighbors[[idx, 1]] = j as u32;
        }

        vprint!("SUCCESS: HNSW pairs validated: using {} pairs for PaCMAP (truncated from {})", expected_pairs, actual_pairs);

        // Create configuration with precomputed neighbors
        Configuration {
            pair_configuration: PairConfiguration::NeighborsProvided { pair_neighbors },
            ..config
        }
        }
    } else {
        vprint!(" DEBUG: use_hnsw=false, force_exact_knn={}, n_samples={}", force_exact_knn, n_samples);
        if force_exact_knn {
            vprint!(" Using exact KNN search for {} samples (forced by user)", n_samples);
            report_progress("Exact KNN", 25, 100, 25.0, "SUCCESS: EXACT KNN ENABLED - Using O(n^2) brute-force neighbor search (precise)");
        } else {
            vprint!(" Using exact KNN search for {} samples (small dataset: auto-fallback)", n_samples);
            report_progress("Exact KNN", 25, 100, 25.0, "SUCCESS: EXACT KNN AUTO - Using O(n^2) exact search (dataset <1000 samples)");
        }
        // FORCE exact KNN by using default PairConfiguration (not HNSW-based)
        Configuration {
            pair_configuration: PairConfiguration::default(),  // Force exact KNN - no HNSW
            ..config
        }
    };

    // Convert data to f32 for PaCMAP
    let data_f32 = data.mapv(|v| v as f32);

    // Perform PaCMAP dimensionality reduction with HNSW neighbors (if computed)
    vprint!("DEBUG: Calling external PacMAP fit_transform...");
    let (embedding_f32, _) = fit_transform(data_f32.view(), updated_config)?;
    vprint!("DEBUG: External PacMAP fit_transform completed");

    // Convert embedding back to f64 for the public API
    let embedding_f64 = embedding_f32.mapv(|v| v as f64);
    Ok((embedding_f64, optimized_hnsw_params))
}

/// Progress callback function type for Rust usage
pub type ProgressCallback = Box<dyn Fn(&str, usize, usize, f32, &str) + Send + Sync>;

/// Enhanced fit function with normalization, HNSW auto-scaling, and progress reporting with force_exact_knn control
/// This version properly normalizes data and auto-scales HNSW parameters for optimal performance
pub fn fit_transform_normalized_with_progress_and_force_knn(
    mut data: Array2<f64>,
    config: Configuration,
    normalization_mode: Option<NormalizationMode>,
    progress_callback: Option<ProgressCallback>,
    force_exact_knn: bool,
    use_quantization: bool
) -> Result<(Array2<f64>, PaCMAP), Box<dyn std::error::Error>> {
    let progress = |phase: &str, current: usize, total: usize, percent: f32, message: &str| {
        if let Some(ref callback) = progress_callback {
            callback(phase, current, total, percent, message);
        }
    };

    progress("Initializing", 0, 100, 0.0, "Preparing dataset for PacMAP fitting");
    let (n_samples, n_features) = data.dim();

    // Determine normalization mode (auto-detect if not specified)
    progress("Analyzing", 5, 100, 5.0, "Analyzing data characteristics for normalization");
    let norm_mode = normalization_mode.unwrap_or_else(|| recommend_normalization_mode(&data));

    // Initialize and fit normalization parameters
    progress("Normalizing", 10, 100, 10.0, &format!("Applying {:?} normalization", norm_mode));
    let mut normalization = NormalizationParams::new(n_features, norm_mode);
    if norm_mode != NormalizationMode::None {
        normalization.fit_transform(&mut data)?;
    }

    // Auto-scale HNSW parameters based on dataset characteristics (only if not forcing exact KNN)
    let n_neighbors = config.override_neighbors.unwrap_or(10);
    let mut hnsw_params = if !force_exact_knn {
        progress("HNSW Config", 20, 100, 20.0, "Auto-scaling HNSW parameters for dataset");
        HnswParams::auto_scale(n_samples, n_features, n_neighbors)
    } else {
        progress("Exact KNN", 20, 100, 20.0, "Skipping HNSW configuration - using exact KNN");
        HnswParams::default() // Not used, but needed for struct
    };

    // Log HNSW parameter selection for user information (only if using HNSW)
    if !force_exact_knn {
        let characteristics = hnsw_params.get_characteristics();
        let hnsw_message = format!("HNSW: M={}, ef_construction={}, ef_search={}",
                                   hnsw_params.m, hnsw_params.ef_construction, hnsw_params.ef_search);
        progress("HNSW Ready", 25, 100, 25.0, &hnsw_message);

        vprint!("DEBUG: Auto-scaled HNSW parameters for {}k samples, {} features:", n_samples / 1000, n_features);
        vprint!("   M={}, ef_construction={}, ef_search={}",
                  hnsw_params.m, hnsw_params.ef_construction, hnsw_params.ef_search);
        vprint!("   {}", characteristics);
    } else {
        progress("Exact KNN Ready", 25, 100, 25.0, "Exact KNN configuration complete - high precision mode");
        vprint!("DEBUG: Using exact KNN for {} samples, {} features (high precision mode)", n_samples, n_features);
    }

    // Perform fit using HNSW‑enhanced PaCMAP on normalized data
    progress("Embedding", 30, 100, 30.0, "Computing PacMAP embedding (this may take time for large datasets)");
    let callback_ref = progress_callback.as_ref().map(|cb| cb.as_ref());
    let (embedding, optimized_hnsw_params) = fit_transform_hnsw(data.clone(), config.clone(), force_exact_knn, callback_ref)?;
    progress("Embedding Done", 80, 100, 80.0, "PacMAP embedding computation completed");

    // Compute statistics over the embedding for outlier detection
    progress("Finalizing", 90, 100, 90.0, "Computing embedding statistics and building model");
    let (mean, p95, max) = compute_distance_stats(&embedding);

    // Create serializable config with optimized HNSW parameters (if available)
    let final_hnsw_params = optimized_hnsw_params.unwrap_or(hnsw_params);
    let pacmap_config = PacMAPConfig {
        n_neighbors,
        embedding_dim: config.embedding_dimensions,
        n_epochs: 450, // Default for now - could be extracted from config
        learning_rate: 1.0,
        min_dist: 0.1,
        mid_near_ratio: 0.5,
        far_pair_ratio: 0.5,
        seed: config.seed,
        hnsw_params: final_hnsw_params,
        used_hnsw: !force_exact_knn, // Track whether HNSW was used
    };

    // Build complete model struct with normalization parameters
    let mut model = PaCMAP {
        embedding: embedding.clone(),
        config: pacmap_config,
        stats: DistanceStats {
            mean_distance: mean,
            p95_distance: p95,
            max_distance: max
        },
        normalization, // Store fitted normalization parameters
        quantize_on_save: use_quantization,
        quantized_embedding: None,
        original_data: None,
        fitted_projections: None,
    };

    // Store transform data: original input and fitted projections
    // This enables proper 2-stage neighbor search for new points
    model.store_transform_data(&data, &embedding);

    progress("Complete", 100, 100, 100.0, "PacMAP fitting completed successfully");

    // Display final model settings
    model.print_model_settings("Fitted Model");

    Ok((embedding, model))
}

/// Enhanced fit function with normalization, HNSW auto-scaling, and progress reporting
/// This version properly normalizes data and auto-scales HNSW parameters for optimal performance
pub fn fit_transform_normalized_with_progress(
    data: Array2<f64>,
    config: Configuration,
    normalization_mode: Option<NormalizationMode>,
    progress_callback: Option<ProgressCallback>
) -> Result<(Array2<f64>, PaCMAP), Box<dyn std::error::Error>> {
    // Call the extended version with force_exact_knn = false (use HNSW by default)
    fit_transform_normalized_with_progress_and_force_knn(data, config, normalization_mode, progress_callback, false, false)
}

/// Enhanced fit function with normalization and HNSW auto-scaling
/// This version properly normalizes data and auto-scales HNSW parameters for optimal performance
pub fn fit_transform_normalized(
    data: Array2<f64>,
    config: Configuration,
    normalization_mode: Option<NormalizationMode>
) -> Result<(Array2<f64>, PaCMAP), Box<dyn std::error::Error>> {
    // Call the progress version without a callback (silent mode)
    fit_transform_normalized_with_progress(data, config, normalization_mode, None)
}

/// Legacy function - now calls the enhanced version with auto-normalization
pub fn fit_transform_quantized(data: Array2<f64>, config: Configuration) -> Result<(Array2<f16>, PaCMAP), Box<dyn std::error::Error>> {
    let (embedding, model) = fit_transform_normalized(data, config, None)?;

    // Quantize embedding to f16 for compactness
    let quantized = quantize_embedding(&embedding);
    Ok((quantized, model))
}

/// Transform new data using a fitted model with consistent normalization
/// This is critical - must use the same normalization as training data
pub fn transform_with_model(model: &PaCMAP, mut new_data: Array2<f64>) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    let (n_samples, n_features) = new_data.dim();

    // Validate feature dimensions match training data
    if n_features != model.normalization.n_features {
        return Err(format!(
            "Feature dimension mismatch: expected {}, got {}",
            model.normalization.n_features, n_features
        ).into());
    }

    // Apply the same normalization used during training
    if model.normalization.mode != NormalizationMode::None {
        model.normalization.transform(&mut new_data)?;
    }

    // Check if we have transform data available
    let original_data = model.get_original_data().ok_or("No original training data stored for transforms")?;
    let fitted_projections = model.fitted_projections.as_ref().ok_or("No fitted projections stored for transforms")?;

    // STAGE 1: Find initial neighbors in original high-dimensional space
    // Use the same method (HNSW/exact KNN) as during fitting
    let k_initial = (model.config.n_neighbors * 3).max(50); // More candidates for better coverage
    let initial_neighbors = find_neighbors_in_original_space(&new_data, &original_data, k_initial, model)?;

    // STAGE 2: Project new points to 2D space using initial 3D neighbors
    let embedding_dim = fitted_projections.shape()[1];
    let mut transformed = Array2::zeros((n_samples, embedding_dim));

    for (i, neighbor_indices) in initial_neighbors.iter().enumerate() {
        // Use weighted interpolation based on distances in original space
        let mut weighted_position = Array1::<f64>::zeros(embedding_dim);
        let mut total_weight = 0.0;

        for &neighbor_idx in neighbor_indices.iter().take(model.config.n_neighbors) {
            if neighbor_idx < fitted_projections.shape()[0] {
                // Calculate inverse distance weight in original space
                let orig_dist = euclidean_distance(new_data.row(i), original_data.row(neighbor_idx));
                let weight = 1.0 / (orig_dist + 1e-8); // Avoid division by zero

                // Add weighted contribution from neighbor's projection
                for j in 0..embedding_dim {
                    weighted_position[j] += weight * fitted_projections[[neighbor_idx, j]];
                }
                total_weight += weight;
            }
        }

        // Normalize by total weight
        if total_weight > 0.0 {
            for j in 0..embedding_dim {
                transformed[[i, j]] = weighted_position[j] / total_weight;
            }
        }
    }

    // STAGE 3: Find final neighbors in the 2D embedding space
    // This is what the user actually wants - neighbors in the compressed space!
    vprint!("🔍 Transform STAGE 3: Finding neighbors in 2D embedding space");

    let _final_neighbors = find_neighbors_in_embedding_space(&transformed, fitted_projections, model.config.n_neighbors, model)?;

    vprint!("✅ Transform completed: 2D projection with neighbors in embedding space");

    Ok(transformed)
}

/// Enhanced transform function that returns both coordinates AND neighbors in 2D space
/// This is what users typically want: where the point projects to + what it's near
pub fn transform_with_neighbors(model: &PaCMAP, new_data: Array2<f64>) -> Result<(Array2<f64>, Vec<Vec<usize>>), Box<dyn std::error::Error>> {
    let (n_samples, n_features) = new_data.dim();

    // Validate feature dimensions match training data
    if n_features != model.normalization.n_features {
        return Err(format!(
            "Feature dimension mismatch: expected {}, got {}",
            model.normalization.n_features, n_features
        ).into());
    }

    // Apply the same normalization used during training
    let mut normalized_data = new_data;
    if model.normalization.mode != NormalizationMode::None {
        model.normalization.transform(&mut normalized_data)?;
    }

    // Check if we have transform data available
    let original_data = model.get_original_data().ok_or("No original training data stored for transforms")?;
    let fitted_projections = model.fitted_projections.as_ref().ok_or("No fitted projections stored for transforms")?;

    // STAGE 1: Find initial neighbors in original high-dimensional space
    let k_initial = (model.config.n_neighbors * 3).max(50);
    let _initial_neighbors = find_neighbors_in_original_space(&normalized_data, &original_data, k_initial, model)?;

    // STAGE 2: Project to 2D space (same as transform_with_model)
    let embedding_dim = fitted_projections.shape()[1];
    let mut transformed = Array2::zeros((n_samples, embedding_dim));

    for (i, neighbor_indices) in _initial_neighbors.iter().enumerate() {
        let mut weighted_position = Array1::<f64>::zeros(embedding_dim);
        let mut total_weight = 0.0;

        for &neighbor_idx in neighbor_indices.iter().take(model.config.n_neighbors) {
            if neighbor_idx < fitted_projections.shape()[0] {
                let orig_dist = euclidean_distance(normalized_data.row(i), original_data.row(neighbor_idx));
                let weight = 1.0 / (orig_dist + 1e-8);

                for j in 0..embedding_dim {
                    weighted_position[j] += weight * fitted_projections[[neighbor_idx, j]];
                }
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            for j in 0..embedding_dim {
                transformed[[i, j]] = weighted_position[j] / total_weight;
            }
        }
    }

    // STAGE 3: Find final neighbors in the 2D embedding space
    let final_neighbors = find_neighbors_in_embedding_space(&transformed, fitted_projections, model.config.n_neighbors, model)?;

    vprint!("✅ Transform with neighbors completed: {} points projected with 2D neighbors", n_samples);

    Ok((transformed, final_neighbors))
}

#[no_mangle]
pub extern "C" fn pacmap_fit_transform_quantized(
    data: *const f64,
    rows: usize,
    cols: usize,
    n_dims: usize,
    n_neighbors: usize,
    seed: u64,
) -> *mut f16 {
    // SAFETY: Caller must ensure data points to a valid slice of length rows*cols
    let data_vec = unsafe { std::slice::from_raw_parts(data, rows * cols) }.to_vec();
    let data_arr = Array2::from_shape_vec((rows, cols), data_vec).expect("Invalid shape");
    let config = Configuration {
        embedding_dimensions: n_dims,
        override_neighbors: Some(n_neighbors),
        seed: Some(seed),
        ..Default::default()
    };
    let (embedding_f16, _model) = fit_transform_quantized(data_arr, config).expect("Fit failed");
    // Convert Vec<f16> to raw pointer for C caller
    let (vec, _offset) = embedding_f16.into_raw_vec_and_offset();
    let boxed = vec.into_boxed_slice();
    Box::into_raw(boxed) as *mut f16
}

#[no_mangle]
pub extern "C" fn pacmap_save_model(model: *mut PaCMAP, path: *const u8, path_len: usize) -> i32 {
    let model_ref = unsafe { &mut *model };
    let path_slice = unsafe { std::slice::from_raw_parts(path, path_len) };
    let path_str = std::str::from_utf8(path_slice).expect("Invalid UTF-8 path");
    // Save without quantization by default
    model_ref.quantize_on_save = false;
    match model_ref.save_uncompressed(path_str) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

#[no_mangle]
pub extern "C" fn pacmap_save_model_quantized(
    model: *mut PaCMAP,
    path: *const u8,
    path_len: usize,
    quantize: bool,
) -> i32 {
    let model_ref = unsafe { &mut *model };
    model_ref.quantize_on_save = quantize;
    let path_slice = unsafe { std::slice::from_raw_parts(path, path_len) };
    let path_str = std::str::from_utf8(path_slice).expect("Invalid UTF-8 path");
    match model_ref.save_compressed(path_str) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

#[no_mangle]
pub extern "C" fn pacmap_load_model(path: *const u8, path_len: usize) -> *mut PaCMAP {
    let path_slice = unsafe { std::slice::from_raw_parts(path, path_len) };
    let path_str = std::str::from_utf8(path_slice).expect("Invalid UTF-8 path");
    match PaCMAP::load_compressed(path_str) {
        Ok(model) => Box::into_raw(Box::new(model)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn pacmap_free_model(model: *mut PaCMAP) {
    if !model.is_null() {
        unsafe {
            let _ = Box::from_raw(model);
        }
    }
}

#[no_mangle]
pub extern "C" fn pacmap_free_f16(ptr: *mut f16, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(ptr, len));
        }
    }
}

#[no_mangle]
pub extern "C" fn pacmap_get_knn_indices(
    data: *const f64,
    rows: usize,
    cols: usize,
    n_neighbors: usize,
    seed: u64,
) -> *mut usize {
    let data_vec = unsafe { std::slice::from_raw_parts(data, rows * cols) }.to_vec();
    let data_arr = Array2::from_shape_vec((rows, cols), data_vec).expect("Invalid shape");
    let indices = get_knn_indices(data_arr.view(), n_neighbors, seed);
    let flat: Vec<usize> = indices.into_iter().flatten().collect();
    let boxed = flat.into_boxed_slice();
    Box::into_raw(boxed) as *mut usize
}

#[no_mangle]
pub extern "C" fn pacmap_free_usize(ptr: *mut usize, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(ptr, len));
        }
    }
}

#[no_mangle]
pub extern "C" fn pacmap_distance_stats(
    embedding: *const f64,
    rows: usize,
    cols: usize,
    _k: usize,
    mean: *mut f64,
    p95: *mut f64,
    max: *mut f64,
) {
    let emb_vec = unsafe { std::slice::from_raw_parts(embedding, rows * cols) }.to_vec();
    let emb_arr = Array2::from_shape_vec((rows, cols), emb_vec).expect("Invalid shape");
    let (m, p, mx) = compute_distance_stats(&emb_arr);
    unsafe {
        *mean = m;
        *p95 = p;
        *max = mx;
    }
}

#[no_mangle]
pub extern "C" fn pacmap_get_model_stats(
    model: *const PaCMAP,
    mean: *mut f64,
    p95: *mut f64,
    max: *mut f64,
) {
    let model_ref = unsafe { &*model };
    unsafe {
        *mean = model_ref.stats.mean_distance;
        *p95 = model_ref.stats.p95_distance;
        *max = model_ref.stats.max_distance;
    }
}

/// Symmetrize k-NN graph to improve connectivity and reduce artifacts
/// Makes the graph undirected: if i is neighbor of j, ensure j is neighbor of i
/// Symmetrize per-point neighbor lists with proper distance-based selection
/// Ensures fixed size n_neighbors per point using distance-weighted merging
fn symmetrize_per_point(nn_per_point: &mut Vec<Vec<(usize, f64)>>, n_neighbors: usize) {
    let n_samples = nn_per_point.len();

    // Collect all bidirectional connections with minimum distance
    let mut bidirectional: std::collections::HashMap<(usize, usize), f64> = std::collections::HashMap::new();

    for i in 0..n_samples {
        for &(j, dist) in &nn_per_point[i] {
            let key = if i < j { (i, j) } else { (j, i) };
            bidirectional.entry(key).and_modify(|existing_dist| {
                *existing_dist = existing_dist.min(dist); // Use minimum distance
            }).or_insert(dist);
        }
    }

    // Rebuild neighbor lists ensuring symmetry and fixed size
    for i in 0..n_samples {
        let mut all_candidates: Vec<(usize, f64)> = Vec::new();

        // Collect all valid symmetric neighbors for point i
        for (edge, &dist) in &bidirectional {
            let (a, b) = *edge;
            if a == i {
                all_candidates.push((b, dist));
            } else if b == i {
                all_candidates.push((a, dist));
            }
        }

        // Sort by distance and take exactly n_neighbors
        all_candidates.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
        nn_per_point[i] = all_candidates.into_iter().take(n_neighbors).collect();
    }

    if std::env::var("PACMAP_VERBOSE").is_ok() {
        let total_pairs = bidirectional.len() * 2; // Each bidirectional edge = 2 directed pairs
        eprintln!("🔄 SYMMETRIC PER-POINT: {} bidirectional edges → {} total directed pairs",
                 bidirectional.len(), total_pairs);
    }
}

/// Find neighbors for new points in the original high-dimensional space
/// Uses the same method and parameters as during fitting
/// Returns Vec<Vec<usize>> where each inner Vec contains neighbor indices
fn find_neighbors_in_original_space(
    new_data: &Array2<f64>,
    original_data: &Array2<f64>,
    k: usize,
    model: &PaCMAP
) -> Result<Vec<Vec<usize>>, Box<dyn std::error::Error>> {
    let (n_new, _n_features) = new_data.dim();
    let (n_orig, _) = original_data.dim();

    let mut all_neighbors = Vec::with_capacity(n_new);

    // Use the same neighbor search method that was used during fitting
    if model.config.used_hnsw {
        #[cfg(feature = "use_hnsw")]
        {
            vprint!("🔍 Transform: Using HNSW for neighbor search in original space ({} points)", n_orig);

            // Use the exact same HNSW parameters as during fitting
            let hnsw_params = &model.config.hnsw_params;
            let max_layer = ((n_orig as f32).log2() * 0.8) as usize;
            let max_layer = max_layer.min(32).max(4);

            let mut hnsw = Hnsw::<f32, DistL2>::new(
                hnsw_params.m,
                n_orig,
                max_layer,
                hnsw_params.ef_construction,
                DistL2{}
            );

            // Insert original data points (convert to f32)
            for (i, row) in original_data.axis_iter(ndarray::Axis(0)).enumerate() {
                let point: Vec<f32> = row.iter().map(|&x| x as f32).collect();
                hnsw.insert((&point, i));
            }

            // Search for neighbors of each new point
            for i in 0..n_new {
                let query: Vec<f32> = new_data.row(i).iter().map(|&x| x as f32).collect();
                let neighbors = hnsw.search(&query, k, hnsw_params.ef_search);
                let neighbor_indices: Vec<usize> = neighbors.into_iter().map(|n| n.d_id).collect();
                all_neighbors.push(neighbor_indices);
            }

            vprint!("✅ Transform: HNSW neighbor search completed");
            return Ok(all_neighbors);
        }
        #[cfg(not(feature = "use_hnsw"))]
        {
            return Err("Transform failed: Model was trained with HNSW but HNSW feature is not enabled".into());
        }
    }

    // Fallback to exact KNN search
    vprint!("🔍 Transform: Using exact KNN for neighbor search ({} points)", n_orig);

    for i in 0..n_new {
        let mut distances: Vec<(usize, f64)> = Vec::with_capacity(n_orig);

        // Calculate distances to all original points
        for j in 0..n_orig {
            let dist = euclidean_distance(new_data.row(i), original_data.row(j));
            distances.push((j, dist));
        }

        // Sort by distance and take top k
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let neighbors: Vec<usize> = distances.into_iter()
            .take(k.min(n_orig))
            .map(|(idx, _)| idx)
            .collect();

        all_neighbors.push(neighbors);
    }

    Ok(all_neighbors)
}

/// Find neighbors for new points in the 2D embedding space using HNSW
/// Always uses HNSW for consistency with the training process
fn find_neighbors_in_embedding_space(
    new_projections: &Array2<f64>,
    fitted_projections: &Array2<f64>,
    k: usize,
    model: &PaCMAP
) -> Result<Vec<Vec<usize>>, Box<dyn std::error::Error>> {
    let (n_new, _) = new_projections.dim();
    let (n_fitted, _) = fitted_projections.dim();

    let mut all_neighbors = Vec::with_capacity(n_new);

    vprint!("🔍 Finding neighbors in 2D embedding space for {} new points among {} fitted points", n_new, n_fitted);
    vprint!("📊 Using HNSW with model parameters: M={}, ef_search={}",
            model.config.hnsw_params.m, model.config.hnsw_params.ef_search);

    // ALWAYS use HNSW in 2D embedding space for consistency
    #[cfg(feature = "use_hnsw")]
    {
        // Use the same HNSW parameters as the original model
        let hnsw_params = &model.config.hnsw_params;
        let max_layer = ((n_fitted as f32).log2() * 0.8) as usize;
        let max_layer = max_layer.min(32).max(4);

        let mut hnsw = Hnsw::<f32, DistL2>::new(
            hnsw_params.m,
            n_fitted,
            max_layer,
            hnsw_params.ef_construction,
            DistL2{}
        );

        // Insert fitted projections (convert to f32)
        for (i, row) in fitted_projections.axis_iter(ndarray::Axis(0)).enumerate() {
            let point: Vec<f32> = row.iter().map(|&x| x as f32).collect();
            hnsw.insert((&point, i));
        }

        // Search for neighbors of each new projection
        for i in 0..n_new {
            let query: Vec<f32> = new_projections.row(i).iter().map(|&x| x as f32).collect();
            let neighbors = hnsw.search(&query, k, hnsw_params.ef_search);
            let neighbor_indices: Vec<usize> = neighbors.into_iter().map(|n| n.d_id).collect();
            all_neighbors.push(neighbor_indices);
        }

        vprint!("✅ Found {} neighbors using HNSW in 2D space - indices point to original fitted data", k);
        return Ok(all_neighbors);
    }

    #[cfg(not(feature = "use_hnsw"))]
    {
        return Err("HNSW feature not enabled but required for 2D neighbor search".into());
    }
}

/// Calculate Euclidean distance between two array views
fn euclidean_distance(a: ndarray::ArrayView1<f64>, b: ndarray::ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Format duration in human-readable format
fn format_duration(duration: std::time::Duration) -> String {
    let total_secs = duration.as_secs();
    let millis = duration.subsec_millis();

    if total_secs >= 60 {
        let mins = total_secs / 60;
        let secs = total_secs % 60;
        format!("{}m{}s", mins, secs)
    } else if total_secs >= 1 {
        if millis > 0 {
            format!("{}.{}s", total_secs, millis / 100)
        } else {
            format!("{}s", total_secs)
        }
    } else {
        format!("{}ms", millis)
    }
}
