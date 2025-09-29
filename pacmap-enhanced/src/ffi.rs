// Enhanced C FFI Interface for PacMAP with HNSW Auto-scaling and Progress Callbacks
// Following UMAP Enhanced patterns for C# integration

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_float, c_double};
use ndarray::Array2;
use pacmap::Configuration;
use crate::{fit_transform_normalized_with_progress_and_force_knn, transform_with_model};
use crate::serialization::PaCMAP;
use crate::stats::NormalizationMode;
use crate::hnsw_params::{HnswParams, HnswUseCase};

/// Progress callback function type for C# integration
/// Follows UMAP Enhanced v2 callback pattern
pub type PacmapProgressCallback = extern "C" fn(
    phase: *const c_char,        // Current phase: "Normalizing", "Building HNSW", "PacMAP", etc.
    current: c_int,              // Current progress counter
    total: c_int,                // Total items to process
    percent: c_float,            // Progress percentage (0-100)
    message: *const c_char,      // Time estimates, warnings, or NULL
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
    pub min_dist: c_double,
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
            min_dist: 0.1,
            mid_near_ratio: 0.5,
            far_pair_ratio: 0.5,
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
            let use_case = match self.hnsw_config.use_case {
                1 => HnswUseCase::FastConstruction,
                2 => HnswUseCase::HighAccuracy,
                3 => HnswUseCase::MemoryOptimized,
                _ => HnswUseCase::Balanced,
            };
            HnswParams::for_use_case(use_case, n_samples, n_features)
        } else {
            HnswParams {
                m: self.hnsw_config.m as usize,
                ef_construction: self.hnsw_config.ef_construction as usize,
                ef_search: self.hnsw_config.ef_search as usize,
                max_m0: (self.hnsw_config.m * 2) as usize,
                estimated_memory_bytes: HnswParams::estimate_memory(
                    n_samples,
                    self.hnsw_config.m as usize,
                    (self.hnsw_config.m * 2) as usize
                ),
            }
        }
    }
}

/// Model handle for C FFI - opaque pointer
pub type PacmapHandle = *mut PaCMAP;

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
    callback: Option<PacmapProgressCallback>,
) -> PacmapHandle {
    // Safety check
    if data.is_null() || embedding.is_null() || rows <= 0 || cols <= 0 {
        return std::ptr::null_mut();
    }

    // Convert input data
    let data_slice = unsafe { std::slice::from_raw_parts(data, (rows * cols) as usize) };
    let data_vec: Vec<f64> = data_slice.iter().map(|&x| x as f64).collect();
    let data_arr = match Array2::from_shape_vec((rows as usize, cols as usize), data_vec) {
        Ok(arr) => arr,
        Err(_) => return std::ptr::null_mut(),
    };

    // Create progress callback wrapper
    let progress_callback = |phase: &str, current: usize, total: usize, percent: f32, message: &str| {
        if let Some(cb) = callback {
            let phase_cstr = CString::new(phase).unwrap_or_else(|_| CString::new("Unknown").unwrap());
            let message_cstr = if message.is_empty() {
                None
            } else {
                Some(CString::new(message).unwrap_or_else(|_| CString::new("").unwrap()))
            };

            let message_ptr = message_cstr.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());

            cb(
                phase_cstr.as_ptr(),
                current as c_int,
                total as c_int,
                percent,
                message_ptr,
            );
        }
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

    // Perform fitting with progress callback and force_exact_knn control
    let rust_progress_callback = if callback.is_some() {
        Some(Box::new(move |phase: &str, current: usize, total: usize, percent: f32, message: &str| {
            if let Some(cb) = callback {
                // Convert Rust strings to C strings for FFI
                let phase_cstr = CString::new(phase).unwrap_or_else(|_| CString::new("Unknown").unwrap());
                let message_cstr = CString::new(message).unwrap_or_else(|_| CString::new("").unwrap());
                cb(
                    phase_cstr.as_ptr(),
                    current as c_int,
                    total as c_int,
                    percent,
                    message_cstr.as_ptr(),
                );
            }
        }) as Box<dyn Fn(&str, usize, usize, f32, &str) + Send + Sync>)
    } else {
        None
    };

    // DEBUG: Report FFI parameters via callback
    let ffi_debug_msg = format!(" FFI DEBUG: force_exact_knn={}, use_quantization={}", config.force_exact_knn, config.use_quantization);
    progress_callback("FFI Debug", 3, 100, 3.0, &ffi_debug_msg);

    match fit_transform_normalized_with_progress_and_force_knn(
        data_arr,
        pacmap_config,
        norm_mode,
        rust_progress_callback,
        config.force_exact_knn,
        config.use_quantization
    ) {
        Ok((result_embedding, model)) => {
            // Report progress: Copying results
            progress_callback("Finalizing", 90, 100, 90.0, "Copying embedding results");


            // Copy embedding to output buffer
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
        Err(_) => {
            progress_callback("Error", 0, 100, 0.0, "PacMAP fitting failed");
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
    callback: Option<PacmapProgressCallback>,
) -> c_int {
    // Safety checks
    if handle.is_null() || data.is_null() || embedding.is_null() || rows <= 0 || cols <= 0 {
        return -1;
    }

    let model = unsafe { &*handle };

    // Convert input data
    let data_slice = unsafe { std::slice::from_raw_parts(data, (rows * cols) as usize) };
    let data_vec: Vec<f64> = data_slice.iter().map(|&x| x as f64).collect();
    let data_arr = match Array2::from_shape_vec((rows as usize, cols as usize), data_vec) {
        Ok(arr) => arr,
        Err(_) => return -2,
    };

    // Create progress callback wrapper
    let progress_callback = |phase: &str, current: usize, total: usize, percent: f32, message: &str| {
        if let Some(cb) = callback {
            let phase_cstr = CString::new(phase).unwrap_or_else(|_| CString::new("Unknown").unwrap());
            let message_cstr = if message.is_empty() {
                None
            } else {
                Some(CString::new(message).unwrap_or_else(|_| CString::new("").unwrap()))
            };

            let message_ptr = message_cstr.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());

            cb(
                phase_cstr.as_ptr(),
                current as c_int,
                total as c_int,
                percent,
                message_ptr,
            );
        }
    };

    progress_callback("Transform", 0, 100, 0.0, "Transforming new data using existing model");

    // Perform transformation
    match transform_with_model(model, data_arr) {
        Ok(result_embedding) => {
            progress_callback("Copying", 50, 100, 50.0, "Copying transformation results");

            // Copy results
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
        Err(_) => {
            progress_callback("Error", 0, 100, 0.0, "Transform failed");
            -3
        }
    }
}

/// Get model information
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
    memory_usage_mb: *mut c_int,
) -> c_int {
    if handle.is_null() {
        return -1;
    }

    let model = unsafe { &*handle };

    // Fill in model information
    if !n_samples.is_null() {
        unsafe { *n_samples = model.embedding.shape()[0] as c_int; }
    }
    if !n_features.is_null() {
        unsafe { *n_features = model.normalization.n_features as c_int; }
    }
    if !embedding_dim.is_null() {
        unsafe { *embedding_dim = model.config.embedding_dim as c_int; }
    }
    if !normalization_mode.is_null() {
        let mode = match model.normalization.mode {
            NormalizationMode::ZScore => 1,
            NormalizationMode::MinMax => 2,
            NormalizationMode::Robust => 3,
            NormalizationMode::None => 4,
        };
        unsafe { *normalization_mode = mode; }
    }
    if !hnsw_m.is_null() {
        unsafe { *hnsw_m = model.config.hnsw_params.m as c_int; }
    }
    if !hnsw_ef_construction.is_null() {
        unsafe { *hnsw_ef_construction = model.config.hnsw_params.ef_construction as c_int; }
    }
    if !hnsw_ef_search.is_null() {
        unsafe { *hnsw_ef_search = model.config.hnsw_params.ef_search as c_int; }
    }
    if !memory_usage_mb.is_null() {
        unsafe { *memory_usage_mb = (model.config.hnsw_params.estimated_memory_bytes / (1024 * 1024)) as c_int; }
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
        eprintln!("ERROR: Save model: null handle or path");
        return -1;
    }

    let model = unsafe { &mut *(handle as *mut PaCMAP) };
    let path_str = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("ERROR: Save model: invalid UTF-8 in path: {:?}", e);
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

/// Get version information
#[no_mangle]
pub extern "C" fn pacmap_get_version() -> *const c_char {
    static VERSION: &str = "PacMAP Enhanced v0.1.0 with HNSW Auto-scaling\0";
    VERSION.as_ptr() as *const c_char
}