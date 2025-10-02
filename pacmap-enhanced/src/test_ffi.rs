// Tests for enhanced C FFI interface
use crate::ffi::*;
use std::ffi::CString;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        println!("\n--- Test: Configuration Creation ---");

        let config = pacmap_config_default();
        println!("Default config: neighbors={}, embedding_dim={}, auto_scale={}",
                 config.n_neighbors, config.embedding_dimensions, config.hnsw_config.auto_scale);

        assert_eq!(config.n_neighbors, 10);
        assert_eq!(config.embedding_dimensions, 2);
        assert!(config.hnsw_config.auto_scale);

        println!("SUCCESS: Configuration creation test passed");
    }

    #[test]
    fn test_hnsw_config_creation() {
        println!("\n--- Test: HNSW Configuration Creation ---");

        let balanced = pacmap_hnsw_config_for_use_case(0); // Balanced
        let fast = pacmap_hnsw_config_for_use_case(1);     // FastConstruction
        let accurate = pacmap_hnsw_config_for_use_case(2); // HighAccuracy
        let memory_opt = pacmap_hnsw_config_for_use_case(3); // MemoryOptimized

        println!("Balanced: auto_scale={}, use_case={}", balanced.auto_scale, balanced.use_case);
        println!("Fast: auto_scale={}, use_case={}", fast.auto_scale, fast.use_case);
        println!("Accurate: auto_scale={}, use_case={}", accurate.auto_scale, accurate.use_case);
        println!("Memory-opt: auto_scale={}, use_case={}", memory_opt.auto_scale, memory_opt.use_case);

        assert!(balanced.auto_scale);
        assert_eq!(balanced.use_case, 0);
        assert_eq!(fast.use_case, 1);
        assert_eq!(accurate.use_case, 2);
        assert_eq!(memory_opt.use_case, 3);

        println!("SUCCESS: HNSW configuration creation test passed");
    }

    #[test]
    fn test_fit_transform_basic() {
        println!("\n--- Test: Basic Fit Transform ---");

        // Create test data (15 samples, 3 features) - more samples than neighbors
        let data: Vec<f64> = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            2.0, 3.0, 4.0,
            5.0, 6.0, 7.0,
            8.0, 9.0, 1.0,
            3.0, 4.0, 5.0,
            6.0, 7.0, 8.0,
            9.0, 1.0, 2.0,
            4.0, 5.0, 6.0,
            1.5, 2.5, 3.5,
            4.5, 5.5, 6.5,
            7.5, 8.5, 9.5,
            2.5, 3.5, 4.5,
            5.5, 6.5, 7.5,
        ];

        let rows = 15;
        let cols = 3;
        let config = pacmap_config_default();
        let mut embedding = vec![0.0; rows * config.embedding_dimensions as usize];

        println!("Input data: {} x {}", rows, cols);
        println!("Expected embedding: {} x {}", rows, config.embedding_dimensions);

        // Test callback
        extern "C" fn test_callback(
            phase: *const std::os::raw::c_char,
            current: std::os::raw::c_int,
            total: std::os::raw::c_int,
            percent: std::os::raw::c_float,
            message: *const std::os::raw::c_char,
        ) {
            let phase_str = if phase.is_null() {
                "Unknown".to_string()
            } else {
                unsafe { std::ffi::CStr::from_ptr(phase) }
                    .to_string_lossy()
                    .to_string()
            };

            let message_str = if message.is_null() {
                String::new()
            } else {
                unsafe { std::ffi::CStr::from_ptr(message) }
                    .to_string_lossy()
                    .to_string()
            };

            println!("  FFI Callback: [{:.1}%] {} ({}/{}) - {}",
                     percent, phase_str, current, total, message_str);
        }

        // Call FFI function
        let handle = pacmap_fit_transform_enhanced(
            data.as_ptr(),
            rows as std::os::raw::c_int,
            cols as std::os::raw::c_int,
            config,
            embedding.as_mut_ptr(),
            (rows * config.embedding_dimensions as usize) as std::os::raw::c_int, // buffer length
            Some(test_callback),
        );

        if handle.is_null() {
            panic!("FFI fit_transform returned null handle");
        }

        println!("Model handle created successfully: {:?}", handle);

        // Test model info retrieval
        let mut n_samples = 0;
        let mut n_features = 0;
        let mut embedding_dim = 0;
        let mut normalization_mode = 0;
        let mut hnsw_m = 0;
        let mut hnsw_ef_construction = 0;
        let mut hnsw_ef_search = 0;
        let memory_usage_mb = 0;

        let mut used_hnsw = false;
        let mut learning_rate = 0.0;
        let mut n_epochs = 0;
        let mut mid_near_ratio = 0.0;
        let mut far_pair_ratio = 0.0;
        let mut seed = 0;
        let mut quantize_on_save = false;
        let mut hnsw_index_crc32 = 0;
        let mut embedding_hnsw_index_crc32 = 0;

        let info_result = pacmap_get_model_info(
            handle,
            &mut n_samples,
            &mut n_features,
            &mut embedding_dim,
            &mut normalization_mode,
            &mut hnsw_m,
            &mut hnsw_ef_construction,
            &mut hnsw_ef_search,
            &mut used_hnsw,
            &mut learning_rate,
            &mut n_epochs,
            &mut mid_near_ratio,
            &mut far_pair_ratio,
            &mut seed,
            &mut quantize_on_save,
            &mut hnsw_index_crc32,
            &mut embedding_hnsw_index_crc32,
        );

        assert_eq!(info_result, 0, "Model info retrieval should succeed");
        assert_eq!(n_samples, rows as std::os::raw::c_int);
        assert_eq!(n_features, cols as std::os::raw::c_int);
        assert_eq!(embedding_dim, config.embedding_dimensions);

        println!("Model info:");
        println!("  Samples: {}, Features: {}, Embedding dim: {}", n_samples, n_features, embedding_dim);
        println!("  Normalization mode: {}", normalization_mode);
        println!("  HNSW: M={}, ef_construction={}, ef_search={}", hnsw_m, hnsw_ef_construction, hnsw_ef_search);
        println!("  Memory usage: {} MB", memory_usage_mb);

        // Verify embedding has reasonable values (not all zeros)
        let non_zero_count = embedding.iter().filter(|&&x| x.abs() > 1e-10).count();
        println!("Non-zero embedding values: {}/{}", non_zero_count, embedding.len());

        // At least some values should be non-zero
        assert!(non_zero_count > 0, "Embedding should have some non-zero values");

        // Clean up
        pacmap_free_model_enhanced(handle);
        println!("SUCCESS: Basic fit transform test passed");
    }

    #[test]
    fn test_ffi_quantization() {
        println!("\n--- Test: FFI Quantization ---");

        // Test data
        let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.05).collect(); // 20x5 data
        let rows = 20;
        let cols = 5;
        let config = pacmap_config_default();
        let mut embedding = vec![0.0; rows * config.embedding_dimensions as usize];

        // Fit model
        let handle = pacmap_fit_transform_enhanced(
            data.as_ptr(),
            rows as std::os::raw::c_int,
            cols as std::os::raw::c_int,
            config,
            embedding.as_mut_ptr(),
            (rows * config.embedding_dimensions as usize) as std::os::raw::c_int, // buffer length
            None,
        );
        assert!(!handle.is_null(), "Model fitting should succeed");

        // Test save without quantization
        let save_path_no_quant = std::ffi::CString::new("test_ffi_no_quant.bin").unwrap();
        let save_result = pacmap_save_model_enhanced(handle, save_path_no_quant.as_ptr(), false);
        assert_eq!(save_result, 0, "Save without quantization should succeed");

        // Test save with quantization
        let save_path_with_quant = std::ffi::CString::new("test_ffi_with_quant.bin").unwrap();
        let save_result = pacmap_save_model_enhanced(handle, save_path_with_quant.as_ptr(), true);
        assert_eq!(save_result, 0, "Save with quantization should succeed");

        // Compare file sizes
        let size_no_quant = std::fs::metadata("test_ffi_no_quant.bin").unwrap().len();
        let size_with_quant = std::fs::metadata("test_ffi_with_quant.bin").unwrap().len();
        println!(" FFI file sizes:");
        println!("   No quantization: {} bytes", size_no_quant);
        println!("   With quantization: {} bytes", size_with_quant);
        if size_with_quant < size_no_quant {
            println!("   SUCCESS: Quantization reduced file size by {} bytes", size_no_quant - size_with_quant);
        }

        // Test loading both
        let load_handle_no_quant = pacmap_load_model_enhanced(save_path_no_quant.as_ptr());
        let load_handle_with_quant = pacmap_load_model_enhanced(save_path_with_quant.as_ptr());

        assert!(!load_handle_no_quant.is_null(), "Load without quantization should succeed");
        assert!(!load_handle_with_quant.is_null(), "Load with quantization should succeed");

        // Cleanup
        pacmap_free_model_enhanced(handle);
        pacmap_free_model_enhanced(load_handle_no_quant);
        pacmap_free_model_enhanced(load_handle_with_quant);
        std::fs::remove_file("test_ffi_no_quant.bin").ok();
        std::fs::remove_file("test_ffi_with_quant.bin").ok();

        println!("SUCCESS: FFI quantization tests passed");
    }

    #[test]
    fn test_quantization_save_load() {
        println!("\n--- Test: Quantization Save/Load ---");

        // Create test data
        let data = ndarray::Array2::from_shape_vec((20, 4), (0..80).map(|i| i as f64 * 0.1).collect()).unwrap();
        let config = pacmap::Configuration::default();

        // Create model
        let (embedding, mut model) = crate::fit_transform_normalized(data, config, None).unwrap();
        println!("Original embedding shape: {:?}", embedding.shape());

        // Test 1: Save without quantization
        model.quantize_on_save = false;
        model.save_compressed("test_no_quant.bin").unwrap();
        let loaded_no_quant = crate::serialization::PaCMAP::load_compressed("test_no_quant.bin").unwrap();
        println!("SUCCESS: No quantization: saved and loaded successfully");
        println!("   Embedding shape: {:?}", loaded_no_quant.embedding.shape());
        println!("   Has quantized embedding: {}", loaded_no_quant.quantized_embedding.is_some());

        // Test 2: Save with quantization
        model.quantize_on_save = true;
        model.quantize_for_save(); // Manually quantize
        model.save_compressed("test_with_quant.bin").unwrap();
        let loaded_with_quant = crate::serialization::PaCMAP::load_compressed("test_with_quant.bin").unwrap();
        println!("SUCCESS: With quantization: saved and loaded successfully");
        println!("   Embedding shape: {:?}", loaded_with_quant.embedding.shape());
        println!("   Has quantized embedding: {}", loaded_with_quant.quantized_embedding.is_some());

        // Test 3: Compare file sizes
        let size_no_quant = std::fs::metadata("test_no_quant.bin").unwrap().len();
        let size_with_quant = std::fs::metadata("test_with_quant.bin").unwrap().len();
        println!(" File sizes:");
        println!("   No quantization: {} bytes", size_no_quant);
        println!("   With quantization: {} bytes", size_with_quant);
        println!("   Compression ratio: {:.1}%", (size_with_quant as f64 / size_no_quant as f64) * 100.0);

        // Cleanup
        std::fs::remove_file("test_no_quant.bin").ok();
        std::fs::remove_file("test_with_quant.bin").ok();

        println!("SUCCESS: Quantization tests passed");
    }

    #[test]
    fn test_simple_save_load() {
        println!("\n--- Test: Simple Save/Load (no FFI) ---");

        // Create a simple model directly without FFI
        let data = ndarray::Array2::from_shape_vec((15, 3), (0..45).map(|i| i as f64).collect()).unwrap();
        let mut config = pacmap::Configuration::default();
        config.override_neighbors = Some(5); // Ensure neighbors < samples
        let norm_mode = None;

        // Create model using internal function
        match crate::fit_transform_normalized(data, config, norm_mode) {
            Ok((_embedding, mut model)) => {
                println!("Model created with {} samples", model.embedding.shape()[0]);

                // Test save
                match model.save_compressed("test_simple.bin") {
                    Ok(()) => println!("SUCCESS: Save succeeded"),
                    Err(e) => panic!("Save failed: {:?}", e),
                }

                // Test load
                match crate::serialization::PaCMAP::load_compressed("test_simple.bin") {
                    Ok(loaded) => {
                        println!("SUCCESS: Load succeeded with {} samples", loaded.embedding.shape()[0]);
                        assert_eq!(model.embedding.shape(), loaded.embedding.shape());
                        println!("SUCCESS: Simple save/load test passed");
                    },
                    Err(e) => panic!("Load failed: {:?}", e),
                }

                std::fs::remove_file("test_simple.bin").ok();
            },
            Err(e) => panic!("Model creation failed: {:?}", e),
        }
    }

    #[test]
    fn test_model_save_load() {
        println!("\n--- Test: Model Save/Load ---");

        // Use a larger dataset that might work better with PacMAP
        let mut data: Vec<f64> = Vec::new();
        let rows = 50;  // Increase sample size
        let cols = 5;   // Increase feature size

        // Generate more realistic data with some structure
        for i in 0..rows {
            for j in 0..cols {
                // Create some structure in the data
                let value = (i as f64 * 0.1) + (j as f64 * 0.3) + ((i * j) as f64 * 0.01);
                data.push(value);
            }
        }
        let config = pacmap_config_default();
        let mut embedding = vec![0.0; rows * config.embedding_dimensions as usize];

        // Fit model
        let handle = pacmap_fit_transform_enhanced(
            data.as_ptr(),
            rows as std::os::raw::c_int,
            cols as std::os::raw::c_int,
            config,
            embedding.as_mut_ptr(),
            (rows * config.embedding_dimensions as usize) as std::os::raw::c_int, // buffer length
            None, // No callback for this test
        );

        assert!(!handle.is_null(), "Model fitting should succeed");

        // Save model
        let save_path = CString::new("test_ffi_model.bin").unwrap();
        let save_result = pacmap_save_model_enhanced(handle, save_path.as_ptr(), false);
        assert_eq!(save_result, 0, "Model save should succeed");

        println!("Model saved successfully");

        // Get original model info
        let mut orig_samples = 0;
        let mut orig_features = 0;
        let mut orig_hnsw_m = 0;
        pacmap_get_model_info(
            handle,
            &mut orig_samples,
            &mut orig_features,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut orig_hnsw_m,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );


        // Free original model
        pacmap_free_model_enhanced(handle);


        // Load model
        let load_handle = pacmap_load_model_enhanced(save_path.as_ptr());
        assert!(!load_handle.is_null(), "Model load should succeed");

        println!("Model loaded successfully");

        // Get loaded model info
        let mut loaded_samples = 0;
        let mut loaded_features = 0;
        let mut loaded_hnsw_m = 0;
        let info_result = pacmap_get_model_info(
            load_handle,
            &mut loaded_samples,
            &mut loaded_features,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut loaded_hnsw_m,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );

        assert_eq!(info_result, 0, "Loaded model info should be accessible");

        // Verify consistency
        assert_eq!(orig_samples, loaded_samples, "Sample count should match");
        assert_eq!(orig_features, loaded_features, "Feature count should match");
        assert_eq!(orig_hnsw_m, loaded_hnsw_m, "HNSW M parameter should match");

        println!("Model consistency verified:");
        println!("  Samples: {} -> {}", orig_samples, loaded_samples);
        println!("  Features: {} -> {}", orig_features, loaded_features);
        println!("  HNSW M: {} -> {}", orig_hnsw_m, loaded_hnsw_m);

        // Clean up
        pacmap_free_model_enhanced(load_handle);
        std::fs::remove_file("test_ffi_model.bin").ok();

        println!("SUCCESS: Model save/load test passed");
    }

    #[test]
    fn test_version_info() {
        println!("\n--- Test: Version Information ---");

        let version_ptr = pacmap_get_version();
        assert!(!version_ptr.is_null(), "Version pointer should not be null");

        let version_str = unsafe {
            std::ffi::CStr::from_ptr(version_ptr).to_string_lossy().to_string()
        };

        println!("Library version: {}", version_str);
        assert!(version_str.contains("PacMAP Enhanced"), "Version should contain library name");
        assert!(version_str.contains("HNSW"), "Version should mention HNSW");

        println!("SUCCESS: Version information test passed");
    }
}