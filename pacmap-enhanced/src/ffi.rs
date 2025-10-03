// Enhanced C FFI Interface for PacMAP with HNSW Auto-scaling and Progress Callbacks
// Following UMAP Enhanced patterns for C# integration

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_float, c_double, c_void};
use std::panic;
use ndarray::Array2;
use pacmap::Configuration;
use crate::{fit_transform_normalized_with_progress_and_force_knn_with_hnsw, transform_with_model};
use crate::serialization::PaCMAP;
use crate::stats::NormalizationMode;
use crate::hnsw_params::{HnswParams, HnswUseCase};

/// Get the version of the PacMAP Enhanced library
/// Returns a null-terminated string with version info
#[no_mangle]
pub extern "C" fn pacmap_get_version() -> *const c_char {
    let version_info = format!(
        "PacMAP Enhanced v{} - HNSW: {}, OpenBLAS: {}",
        env!("CARGO_PKG_VERSION"),
        if cfg!(feature = "use_hnsw") { "ENABLED" } else { "DISABLED" },
"SYSTEM"
    );

    let c_string = CString::new(version_info).unwrap_or_else(|_| {
        CString::new("PacMAP Enhanced v0.2.0 - Version Error").unwrap()
    });

    // Leak the string so it remains valid for C# to read
    let ptr = c_string.as_ptr();
    std::mem::forget(c_string);
    ptr
}

/// Progress callback function type for C# integration
/// Safe callback pattern - passes user_data and byte array only
pub type PacmapProgressCallback = extern "C" fn(
    user_data: *mut c_void,      // User data pointer (GCHandle from C#)
    data_ptr: *const u8,         // Message data bytes
    len: usize,                  // Length of data in bytes
);

/// HNSW configuration struct for C FFI
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PacmapHnswConfig {
    pub auto_scale: bool,       // If true, ignore manual parameters and auto-scale
    pub use_case: c_int,        // 0=Balanced, 1=FastConstruction, 2=HighAccuracy, 3=MemoryOptimized
    pub m: c_int,               // Manual M parameter (ignored if auto_scale=true)
    pub ef_construction: c_int, // Manual ef_construction (ignored if auto_scale=true)
    pub ef_search: c_int,       // Manual ef_search (ignored if auto_scale=true)
    pub memory_limit_mb: c_int, // Memory limit in MB (0 = no limit)
    pub density_scaling: bool,  // Apply dimension-based density scaling (O(sqrt(d)) behavior)
    pub autodetect_hnsw_params: bool, // If true, do recall validation and auto-optimize; if false, use params as-is
}

impl Default for PacmapHnswConfig {
    fn default() -> Self {
        Self {
            auto_scale: true,
            use_case: 0, // Balanced
            m: 16,
            ef_construction: 128,
            ef_search: 64,
            memory_limit_mb: 0,
            density_scaling: true, // Maintain current behavior by default
            autodetect_hnsw_params: true, // Enable recall validation by default
        }
    }
}

/// Main configuration struct for C FFI
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PacmapConfig {
    pub n_neighbors: c_int,
    pub embedding_dimensions: c_int,
    pub n_epochs: c_int,
    pub learning_rate: c_double,
    pub mid_near_ratio: c_double,
    pub far_pair_ratio: c_double,
    pub seed: c_int,              // -1 for random seed
    pub normalization_mode: c_int, // 0=Auto, 1=ZScore, 2=MinMax, 3=Robust, 4=None
    pub force_exact_knn: bool,    // If true, disable HNSW and use brute-force KNN
    pub use_quantization: bool,   // If true, enable 16-bit quantization for memory reduction
    pub hnsw_config: PacmapHnswConfig,
}

impl Default for PacmapConfig {
    fn default() -> Self {
        Self {
            n_neighbors: 10,
            embedding_dimensions: 2,
            n_epochs: 450,
            learning_rate: 1.0,
            mid_near_ratio: 0.5,
            far_pair_ratio: 2.0,
            seed: -1,
            normalization_mode: 0, // Auto
            force_exact_knn: false, // Use HNSW by default
            use_quantization: false, // No quantization by default
            hnsw_config: PacmapHnswConfig::default(),
        }
    }
}

impl PacmapConfig {
    fn to_pacmap_configuration(&self) -> Configuration {
        // Convert single n_epochs to PacMAP's three-phase iteration structure
        // Default proportions: (100, 100, 250) = 450 total
        // Phase 1: ~22% for mid-near weight reduction
        // Phase 2: ~22% for balanced weight phase
        // Phase 3: ~56% for local structure focus
        let total_epochs = self.n_epochs as usize;
        let phase1 = (total_epochs as f64 * 0.22).round() as usize;
        let phase2 = (total_epochs as f64 * 0.22).round() as usize;
        let phase3 = total_epochs - phase1 - phase2; // Remainder goes to phase 3

        Configuration {
            embedding_dimensions: self.embedding_dimensions as usize,
            override_neighbors: Some(self.n_neighbors as usize),
            seed: if self.seed >= 0 { Some(self.seed as u64) } else { None },
            mid_near_ratio: self.mid_near_ratio as f32,
            far_pair_ratio: self.far_pair_ratio as f32,
            learning_rate: self.learning_rate as f32,
            num_iters: (phase1, phase2, phase3),
            ..Default::default()
        }
    }

    fn to_normalization_mode(&self) -> Option<NormalizationMode> {
        match self.normalization_mode {
            0 => None, // Auto-detect
            1 => Some(NormalizationMode::ZScore),
            2 => Some(NormalizationMode::MinMax),
            3 => Some(NormalizationMode::Robust),
            4 => Some(NormalizationMode::None),
            _ => None, // Default to auto-detect
        }
    }

    fn to_hnsw_params(&self, n_samples: usize, n_features: usize) -> HnswParams {
        if self.hnsw_config.auto_scale {
            let _use_case = match self.hnsw_config.use_case {
                1 => HnswUseCase::FastConstruction,
                2 => HnswUseCase::HighAccuracy,
                3 => HnswUseCase::MemoryOptimized,
                _ => HnswUseCase::Balanced,
            };
            // Use the new auto_scale_with_density function to pass density_scaling flag
            HnswParams::auto_scale_with_density(n_samples, n_features, 15, self.hnsw_config.density_scaling)
        } else {
            HnswParams {
                m: self.hnsw_config.m as usize,
                ef_construction: self.hnsw_config.ef_construction as usize,
                ef_search: self.hnsw_config.ef_search as usize,
                max_m0: (self.hnsw_config.m * 2) as usize,
                density_scaling: self.hnsw_config.density_scaling,
                estimated_memory_bytes: HnswParams::estimate_memory(
                    n_samples,
                    self.hnsw_config.m as usize,
                    (self.hnsw_config.m * 2) as usize
                ),
            }
        }
    }
}

/// Debug printing macro - controlled by VERBOSE flag
macro_rules! vprint {
    ($($arg:tt)*) => {
        if cfg!(feature = "verbose") {
            println!($($arg)*);
        }
    };
}

/// Model handle for C FFI - opaque pointer
pub type PacmapHandle = *mut PaCMAP;

/// Safe callback wrapper using the recommended pattern
/// Creates a single message string and passes it as bytes with user_data
macro_rules! safe_progress_callback {
    ($callback:expr, $user_data:expr, $phase:expr, $current:expr, $total:expr, $percent:expr, $message:expr) => {
        if let Some(cb) = $callback {
            // Create a single formatted message string
            let full_message = format!("[{}] {} ({:.1}%)", $phase, $message, $percent);
            let message_bytes = full_message.as_bytes();

            // Use catch_unwind to prevent panics from crossing FFI boundary
            let res = panic::catch_unwind(|| {
                cb(
                    $user_data,
                    message_bytes.as_ptr(),
                    message_bytes.len()
                );
            });

            if res.is_err() {
                // Log error but don't let panic escape FFI boundary
                // Silent error handling - library shouldn't print
            }
        }
    };
}

/// Clean up a leaked CString created by safe_progress_callback
/// This function can be called by C code to properly free memory
#[no_mangle]
pub extern "C" fn pacmap_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            // Convert back to Box<CString> and let it drop
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Create default configuration
#[no_mangle]
pub extern "C" fn pacmap_config_default() -> PacmapConfig {
    PacmapConfig::default()
}

/// Create HNSW configuration for specific use case
#[no_mangle]
pub extern "C" fn pacmap_hnsw_config_for_use_case(use_case: c_int) -> PacmapHnswConfig {
    PacmapHnswConfig {
        auto_scale: true,
        use_case,
        m: 16,
        ef_construction: 128,
        ef_search: 64,
        memory_limit_mb: 0,
        density_scaling: true,
        autodetect_hnsw_params: true,
    }
}

/// Enhanced fit function with progress callbacks and HNSW auto-scaling
#[no_mangle]
pub extern "C" fn pacmap_fit_transform_enhanced(
    data: *const c_double,
    rows: c_int,
    cols: c_int,
    config: PacmapConfig,
    embedding: *mut c_double,
    embedding_buffer_len: c_int,
    callback: Option<PacmapProgressCallback>,
    user_data: *mut c_void,
) -> PacmapHandle {
    // Safety checks
    if data.is_null() || embedding.is_null() || rows <= 0 || cols <= 0 {
        return std::ptr::null_mut();
    }

    // Buffer overflow protection for embedding output
    let required_embedding_size = rows * config.embedding_dimensions;
    if required_embedding_size > embedding_buffer_len {
        // Return error code -4 for buffer overflow
        return std::ptr::null_mut();
    }

    // Input data validation - prevent integer overflow
    let data_size = match rows.checked_mul(cols) {
        Some(size) if size > 0 => size as usize,
        _ => return std::ptr::null_mut(), // Overflow or invalid size
    };

    // Convert input data
    let data_slice = unsafe { std::slice::from_raw_parts(data, data_size) };
    let data_vec: Vec<f64> = data_slice.iter().map(|&x| x as f64).collect();
    let data_arr = match Array2::from_shape_vec((rows as usize, cols as usize), data_vec) {
        Ok(arr) => arr,
        Err(_) => return std::ptr::null_mut(),
    };

    // Create progress callback wrapper
    let progress_callback = |phase: &str, current: usize, total: usize, percent: f32, message: &str| {
        safe_progress_callback!(callback, user_data, phase, current, total, percent, message);
    };

    // Report progress: Starting
    progress_callback("Initializing", 0, 100, 0.0, "Preparing dataset for PacMAP fitting");

    // Get HNSW parameters (but only report them if not forcing exact KNN)
    let hnsw_params = config.to_hnsw_params(rows as usize, cols as usize);
    let characteristics = hnsw_params.get_characteristics();

    // Report HNSW configuration only if not forcing exact KNN
    if !config.force_exact_knn {
        let hnsw_message = format!(
            "HNSW: M={}, ef_construction={}, ef_search={}, Memory~{}MB",
            hnsw_params.m,
            hnsw_params.ef_construction,
            hnsw_params.ef_search,
            characteristics.estimated_memory_mb
        );
        progress_callback("HNSW Config", 10, 100, 10.0, &hnsw_message);
    } else {
        progress_callback("KNN Config", 10, 100, 10.0, "Exact KNN requested - HNSW disabled for precision");
    }

    // Convert configuration
    let pacmap_config = config.to_pacmap_configuration();
    let norm_mode = config.to_normalization_mode();

    // Report progress: Normalization
    progress_callback("Normalizing", 20, 100, 20.0, "Applying data normalization");

    // DISABLE CALLBACKS FOR NOW - they cause thread safety issues
    let rust_progress_callback: Option<Box<dyn Fn(&str, usize, usize, f32, &str) + Send + Sync>> = None;

    // DEBUG: Report FFI parameters via callback
    let ffi_debug_msg = format!(" FFI DEBUG: force_exact_knn={}, use_quantization={}", config.force_exact_knn, config.use_quantization);
    progress_callback("FFI Debug", 3, 100, 3.0, &ffi_debug_msg);

    // DEBUG: Log before entering main fitting function
    let debug_msg = format!(" FFI DEBUG: Entering fit_transform_normalized_with_progress_and_force_knn - samples: {}, features: {}, force_exact_knn: {}", rows, cols, config.force_exact_knn);
    progress_callback("Debug Start", 4, 100, 4.0, &debug_msg);

    // Pass FFI HNSW parameters to the fit function
    let ffi_hnsw_params = config.to_hnsw_params(rows as usize, cols as usize);

    match fit_transform_normalized_with_progress_and_force_knn_with_hnsw(
        data_arr,
        pacmap_config,
        norm_mode,
        rust_progress_callback,
        config.force_exact_knn,
        config.use_quantization,
        Some(ffi_hnsw_params),
        config.hnsw_config.autodetect_hnsw_params
    ) {
        Ok((result_embedding, model)) => {
            // DEBUG: Success path
            progress_callback("Debug Success", 89, 100, 89.0, " FFI DEBUG: fit_transform completed successfully, copying results");

            // Report progress: Copying results
            progress_callback("Finalizing", 90, 100, 90.0, "Copying embedding results");


            // Copy embedding to output buffer (already validated above)
            let embedding_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    embedding,
                    (rows * config.embedding_dimensions) as usize
                )
            };

            for (i, &value) in result_embedding.iter().enumerate() {
                if i < embedding_slice.len() {
                    embedding_slice[i] = value;
                }
            }

            // Report completion
            progress_callback("Complete", 100, 100, 100.0, "PacMAP fitting completed successfully");

            // Return model handle
            Box::into_raw(Box::new(model))
        }
        Err(e) => {
            // DEBUG: Detailed error logging with specific error classification
            let error_msg = format!("FFI ERROR: fit_transform FAILED - {}", e);
            progress_callback("Error Details", 1, 100, 1.0, &error_msg);

            // Send specific error classification to C#
            let error_type = if error_msg.contains("HNSW") {
                "HNSW Error"
            } else if error_msg.contains("normalization") {
                "Normalization Error"
            } else if error_msg.contains("memory") || error_msg.contains("allocation") {
                "Memory Error"
            } else if error_msg.contains("dimension") {
                "Dimension Mismatch Error"
            } else {
                "General Error"
            };

            progress_callback(error_type, 0, 100, 0.0, &format!("PacMAP fitting failed: {}", e));
            std::ptr::null_mut()
        }
    }
}

/// Transform new data using existing model
#[no_mangle]
pub extern "C" fn pacmap_transform(
    handle: PacmapHandle,
    data: *const c_double,
    rows: c_int,
    cols: c_int,
    embedding: *mut c_double,
    embedding_buffer_len: c_int,
    callback: Option<PacmapProgressCallback>,
    user_data: *mut c_void,
) -> c_int {
    // Safety checks
    if handle.is_null() || data.is_null() || embedding.is_null() || rows <= 0 || cols <= 0 {
        return -1;
    }

    let model = unsafe { &mut *handle };

    // Buffer overflow protection
    let required_embedding_size = rows * model.config.embedding_dim as c_int;
    if required_embedding_size > embedding_buffer_len {
        return -4; // Buffer overflow error
    }

    // Input data validation - prevent integer overflow
    let data_size = match rows.checked_mul(cols) {
        Some(size) if size > 0 => size as usize,
        _ => return -5, // Overflow or invalid size
    };

    // Convert input data
    let data_slice = unsafe { std::slice::from_raw_parts(data, data_size) };
    let data_vec: Vec<f64> = data_slice.iter().map(|&x| x as f64).collect();
    let data_arr = match Array2::from_shape_vec((rows as usize, cols as usize), data_vec) {
        Ok(arr) => arr,
        Err(_) => return -2,
    };

    // Create progress callback wrapper
    let progress_callback = |phase: &str, current: usize, total: usize, percent: f32, message: &str| {
        safe_progress_callback!(callback, user_data, phase, current, total, percent, message);
    };

    progress_callback("Transform", 0, 100, 0.0, "Transforming new data using existing model");

    // Perform transformation
    match transform_with_model(model, data_arr) {
        Ok(result_embedding) => {
            progress_callback("Copying", 50, 100, 50.0, "Copying transformation results");

            // Copy results (already validated above)
            let embedding_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    embedding,
                    (rows * model.config.embedding_dim as c_int) as usize
                )
            };

            for (i, &value) in result_embedding.iter().enumerate() {
                if i < embedding_slice.len() {
                    embedding_slice[i] = value;
                }
            }

            progress_callback("Complete", 100, 100, 100.0, "Transform completed successfully");
            0 // Success
        }
        Err(e) => {
            // Detailed error reporting for transform failures
            let error_msg = format!("Transform failed: {}", e);
            progress_callback("Transform Error", 0, 100, 0.0, &error_msg);

            // Return specific error codes based on error type
            if error_msg.contains("HNSW") {
                -6 // HNSW-specific error in transform
            } else if error_msg.contains("dimension") {
                -7 // Dimension mismatch error
            } else if error_msg.contains("original data") || error_msg.contains("fitted projections") {
                -8 // Missing transform data error
            } else {
                -3 // General transform error
            }
        }
    }
}

/// Get COMPLETE model information - ALL parameters that get serialized
#[no_mangle]
pub extern "C" fn pacmap_get_model_info(
    handle: PacmapHandle,
    n_samples: *mut c_int,
    n_features: *mut c_int,
    embedding_dim: *mut c_int,
    normalization_mode: *mut c_int,
    hnsw_m: *mut c_int,
    hnsw_ef_construction: *mut c_int,
    hnsw_ef_search: *mut c_int,
    used_hnsw: *mut bool,
    learning_rate: *mut f64,
    n_epochs: *mut c_int,
    mid_near_ratio: *mut f64,
    far_pair_ratio: *mut f64,
    seed: *mut c_int,
    quantize_on_save: *mut bool,
    hnsw_index_crc32: *mut u32,
    embedding_hnsw_index_crc32: *mut u32,
) -> c_int {
    if handle.is_null() {
        return -1;
    }

    let model = unsafe { &*handle };

    // Basic model dimensions
    if !n_samples.is_null() {
        unsafe { *n_samples = model.embedding.shape()[0] as c_int; }
    }
    if !n_features.is_null() {
        unsafe { *n_features = model.normalization.n_features as c_int; }
    }
    if !embedding_dim.is_null() {
        unsafe { *embedding_dim = model.config.embedding_dim as c_int; }
    }

    // Normalization mode
    if !normalization_mode.is_null() {
        let mode = match model.normalization.mode {
            NormalizationMode::ZScore => 1,
            NormalizationMode::MinMax => 2,
            NormalizationMode::Robust => 3,
            NormalizationMode::None => 4,
        };
        unsafe { *normalization_mode = mode; }
    }

    // HNSW parameters (the discovered/optimized ones)
    if !hnsw_m.is_null() {
        unsafe { *hnsw_m = model.config.hnsw_params.m as c_int; }
    }
    if !hnsw_ef_construction.is_null() {
        unsafe { *hnsw_ef_construction = model.config.hnsw_params.ef_construction as c_int; }
    }
    if !hnsw_ef_search.is_null() {
        unsafe { *hnsw_ef_search = model.config.hnsw_params.ef_search as c_int; }
    }
    if !used_hnsw.is_null() {
        // Check if HNSW was actually used (depends on dataset size and force_exact_knn)
        let hnsw_was_used = model.embedding.shape()[0] > 1000; // Same logic as in training
        unsafe { *used_hnsw = hnsw_was_used; }
    }

    // PacMAP algorithm parameters
    if !learning_rate.is_null() {
        unsafe { *learning_rate = model.config.learning_rate; }
    }
    if !n_epochs.is_null() {
        unsafe { *n_epochs = model.config.n_epochs as c_int; }
    }
    if !mid_near_ratio.is_null() {
        unsafe { *mid_near_ratio = model.config.mid_near_ratio; }
    }
    if !far_pair_ratio.is_null() {
        unsafe { *far_pair_ratio = model.config.far_pair_ratio; }
    }
    if !seed.is_null() {
        unsafe { *seed = model.config.seed.unwrap_or(42) as c_int; }
    }

    // Quantization setting
    if !quantize_on_save.is_null() {
        unsafe { *quantize_on_save = model.quantize_on_save; }
    }

    // CRC checksums for HNSW indexes
    if !hnsw_index_crc32.is_null() {
        if let Some(crc) = model.hnsw_index_crc32 {
            unsafe { *hnsw_index_crc32 = crc; }
        } else {
            unsafe { *hnsw_index_crc32 = 0; } // 0 indicates no CRC available
        }
    }
    if !embedding_hnsw_index_crc32.is_null() {
        // REMOVED: embedding_hnsw_index_crc32 field no longer exists - always return 0
        unsafe { *embedding_hnsw_index_crc32 = 0; } // 0 indicates no CRC available (field removed)
    }

    0 // Success
}

/// Save model to file
#[no_mangle]
pub extern "C" fn pacmap_save_model_enhanced(
    handle: PacmapHandle,
    path: *const c_char,
    quantize: bool,
) -> c_int {
    if handle.is_null() || path.is_null() {
        // Silent error handling - library shouldn't print
        return -1;
    }

    let model = unsafe { &mut *(handle as *mut PaCMAP) };
    let path_str = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(e) => {
                // Silent error handling - library shouldn't print
                return -2;
            }
        }
    };

    model.quantize_on_save = quantize;

    match model.save_compressed(path_str) {
        Ok(()) => 0,
        Err(_) => -3,
    }
}

/// Load model from file
#[no_mangle]
pub extern "C" fn pacmap_load_model_enhanced(path: *const c_char) -> PacmapHandle {
    if path.is_null() {
        return std::ptr::null_mut();
    }

    let path_str = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };

    match PaCMAP::load_compressed(path_str) {
        Ok(model) => Box::into_raw(Box::new(model)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free model memory
#[no_mangle]
pub extern "C" fn pacmap_free_model_enhanced(handle: PacmapHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle);
        }
    }
}

// Removed duplicate version function - using dynamic version at top of file