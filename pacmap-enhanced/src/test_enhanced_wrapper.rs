// Enhanced wrapper test for PacMAP Enhanced FFI interface
// Tests the complete C FFI wrapper functionality

use crate::ffi::*;
use std::ffi::CString;
use std::ptr;

#[cfg(test)]
mod tests {
    use super::*;

    // Test configuration constants
    const TEST_SAMPLES: usize = 200;
    const TEST_FEATURES: usize = 10;
    const TEST_NEIGHBORS: usize = 10;
    const TEST_EMBEDDING_DIM: usize = 2;

    #[test]
    fn test_enhanced_wrapper_comprehensive() {
        println!("\n===========================================");
        println!("   PacMAP Enhanced Wrapper Test");
        println!("===========================================");
        println!();

        let mut all_tests_passed = true;

        // Test 1: Version information
        println!("[TEST 1] Version Information");
        println!("----------------------------");
        let version_result = test_version_info();
        if version_result {
            println!("[PASS] Version information test");
        } else {
            println!("[FAIL] Version information test");
            all_tests_passed = false;
        }

        // Test 2: Configuration creation and validation
        println!("\n[TEST 2] Configuration Creation and Validation");
        println!("----------------------------------------------");
        let config_result = test_configuration_creation();
        if config_result {
            println!("[PASS] Configuration creation test");
        } else {
            println!("[FAIL] Configuration creation test");
            all_tests_passed = false;
        }

        // Test 3: HNSW configuration auto-scaling
        println!("\n[TEST 3] HNSW Configuration Auto-scaling");
        println!("----------------------------------------");
        let hnsw_result = test_hnsw_configuration();
        if hnsw_result {
            println!("[PASS] HNSW configuration test");
        } else {
            println!("[FAIL] HNSW configuration test");
            all_tests_passed = false;
        }

        // Test 4: Enhanced fit/transform with callbacks
        println!("\n[TEST 4] Enhanced Fit/Transform with Callbacks");
        println!("---------------------------------------------");
        let fit_result = test_enhanced_fit_transform();
        if fit_result {
            println!("[PASS] Enhanced fit/transform test");
        } else {
            println!("[FAIL] Enhanced fit/transform test");
            all_tests_passed = false;
        }

        // Test 5: Model information retrieval
        println!("\n[TEST 5] Model Information Retrieval");
        println!("-----------------------------------");
        let model_info_result = test_model_information();
        if model_info_result {
            println!("[PASS] Model information test");
        } else {
            println!("[FAIL] Model information test");
            all_tests_passed = false;
        }

        // Test 6: Model persistence (save/load)
        println!("\n[TEST 6] Model Persistence (Save/Load)");
        println!("-------------------------------------");
        let persistence_result = test_model_persistence();
        if persistence_result {
            println!("[PASS] Model persistence test");
        } else {
            println!("[FAIL] Model persistence test");
            all_tests_passed = false;
        }

        // Test 7: Quantization functionality
        println!("\n[TEST 7] Quantization Functionality");
        println!("-----------------------------------");
        let quantization_result = test_quantization_functionality();
        if quantization_result {
            println!("[PASS] Quantization functionality test");
        } else {
            println!("[FAIL] Quantization functionality test");
            all_tests_passed = false;
        }

        // Test 8: Error handling and edge cases
        println!("\n[TEST 8] Error Handling and Edge Cases");
        println!("-------------------------------------");
        let error_handling_result = test_error_handling();
        if error_handling_result {
            println!("[PASS] Error handling test");
        } else {
            println!("[FAIL] Error handling test");
            all_tests_passed = false;
        }

        // Final summary
        println!("\n===========================================");
        println!("   Enhanced Wrapper Test Summary");
        println!("===========================================");
        println!();

        if all_tests_passed {
            println!("🎉 ALL ENHANCED WRAPPER TESTS PASSED!");
            println!("✅ Version information working");
            println!("✅ Configuration system working");
            println!("✅ HNSW auto-scaling working");
            println!("✅ Enhanced fit/transform working");
            println!("✅ Model information retrieval working");
            println!("✅ Model persistence working");
            println!("✅ Quantization functionality working");
            println!("✅ Error handling working");
            println!();
            println!("PacMAP Enhanced wrapper is fully functional!");
        } else {
            println!("❌ SOME ENHANCED WRAPPER TESTS FAILED!");
            println!("Check individual test results above for details.");
        }

        assert!(all_tests_passed, "Enhanced wrapper comprehensive test failed");
    }

    fn test_version_info() -> bool {
        let version_ptr = pacmap_get_version();
        if version_ptr.is_null() {
            println!("[ERROR] Version pointer is null");
            return false;
        }

        let version_str = unsafe {
            std::ffi::CStr::from_ptr(version_ptr).to_string_lossy().to_string()
        };

        println!("[INFO] Library version: {}", version_str);

        if !version_str.contains("PacMAP Enhanced") {
            println!("[ERROR] Version string doesn't contain 'PacMAP Enhanced'");
            return false;
        }

        if !version_str.contains("HNSW") {
            println!("[ERROR] Version string doesn't mention HNSW");
            return false;
        }

        true
    }

    fn test_configuration_creation() -> bool {
        let config = pacmap_config_default();

        // Check default values
        if config.n_neighbors != 10 {
            println!("[ERROR] Default n_neighbors is {}, expected 10", config.n_neighbors);
            return false;
        }

        if config.embedding_dimensions != 2 {
            println!("[ERROR] Default embedding_dimensions is {}, expected 2", config.embedding_dimensions);
            return false;
        }

        if !config.hnsw_config.auto_scale {
            println!("[ERROR] Default HNSW auto_scale should be true");
            return false;
        }

        println!("[INFO] Default config validated: neighbors={}, embedding_dim={}, auto_scale={}",
                 config.n_neighbors, config.embedding_dimensions, config.hnsw_config.auto_scale);

        true
    }

    fn test_hnsw_configuration() -> bool {
        // Test all HNSW use cases
        let use_cases = [
            (0, "Balanced"),
            (1, "FastConstruction"),
            (2, "HighAccuracy"),
            (3, "MemoryOptimized"),
        ];

        for (use_case_id, use_case_name) in use_cases {
            let hnsw_config = pacmap_hnsw_config_for_use_case(use_case_id);

            if !hnsw_config.auto_scale {
                println!("[ERROR] HNSW config for {} should have auto_scale=true", use_case_name);
                return false;
            }

            if hnsw_config.use_case != use_case_id {
                println!("[ERROR] HNSW config use_case mismatch: got {}, expected {}",
                         hnsw_config.use_case, use_case_id);
                return false;
            }

            println!("[INFO] HNSW config for {}: auto_scale={}, use_case={}",
                     use_case_name, hnsw_config.auto_scale, hnsw_config.use_case);
        }

        true
    }

    fn test_enhanced_fit_transform() -> bool {
        // Generate test data
        let data = generate_test_data(TEST_SAMPLES, TEST_FEATURES);
        let config = pacmap_config_default();
        let mut embedding = vec![0.0; TEST_SAMPLES * TEST_EMBEDDING_DIM];

        println!("[INFO] Testing fit/transform with {} samples, {} features",
                 TEST_SAMPLES, TEST_FEATURES);

        // Test with progress callback
        let mut callback_count = 0;
        extern "C" fn progress_callback(
            phase: *const std::os::raw::c_char,
            current: std::os::raw::c_int,
            total: std::os::raw::c_int,
            percent: std::os::raw::c_float,
            message: *const std::os::raw::c_char,
        ) {
            static mut CALLBACK_COUNT: i32 = 0;
            unsafe {
                CALLBACK_COUNT += 1;
                if CALLBACK_COUNT <= 5 || CALLBACK_COUNT % 10 == 0 { // Limit output
                    let phase_str = if phase.is_null() {
                        "Unknown".to_string()
                    } else {
                        std::ffi::CStr::from_ptr(phase).to_string_lossy().to_string()
                    };

                    let message_str = if message.is_null() {
                        String::new()
                    } else {
                        std::ffi::CStr::from_ptr(message).to_string_lossy().to_string()
                    };

                    println!("[CALLBACK] [{:.1}%] {} ({}/{}) - {}",
                             percent, phase_str, current, total, message_str);
                }
            }
        }

        let handle = pacmap_fit_transform_enhanced(
            data.as_ptr(),
            TEST_SAMPLES as std::os::raw::c_int,
            TEST_FEATURES as std::os::raw::c_int,
            config,
            embedding.as_mut_ptr(),
            Some(progress_callback),
        );

        if handle.is_null() {
            println!("[ERROR] Enhanced fit/transform returned null handle");
            return false;
        }

        println!("[INFO] Enhanced fit/transform completed successfully");

        // Verify embedding has non-zero values
        let non_zero_count = embedding.iter().filter(|&&x| x.abs() > 1e-10).count();
        if non_zero_count == 0 {
            println!("[ERROR] Embedding contains all zero values");
            pacmap_free_model_enhanced(handle);
            return false;
        }

        println!("[INFO] Embedding has {} non-zero values out of {}",
                 non_zero_count, embedding.len());

        // Clean up
        pacmap_free_model_enhanced(handle);
        true
    }

    fn test_model_information() -> bool {
        let data = generate_test_data(TEST_SAMPLES, TEST_FEATURES);
        let config = pacmap_config_default();
        let mut embedding = vec![0.0; TEST_SAMPLES * TEST_EMBEDDING_DIM];

        let handle = pacmap_fit_transform_enhanced(
            data.as_ptr(),
            TEST_SAMPLES as std::os::raw::c_int,
            TEST_FEATURES as std::os::raw::c_int,
            config,
            embedding.as_mut_ptr(),
            None,
        );

        if handle.is_null() {
            println!("[ERROR] Failed to create model for information test");
            return false;
        }

        // Test model info retrieval
        let mut n_samples = 0;
        let mut n_features = 0;
        let mut embedding_dim = 0;
        let mut normalization_mode = 0;
        let mut hnsw_m = 0;
        let mut hnsw_ef_construction = 0;
        let mut hnsw_ef_search = 0;
        let mut memory_usage_mb = 0;

        let info_result = pacmap_get_model_info(
            handle,
            &mut n_samples,
            &mut n_features,
            &mut embedding_dim,
            &mut normalization_mode,
            &mut hnsw_m,
            &mut hnsw_ef_construction,
            &mut hnsw_ef_search,
            &mut memory_usage_mb,
        );

        if info_result != 0 {
            println!("[ERROR] Model info retrieval failed with code {}", info_result);
            pacmap_free_model_enhanced(handle);
            return false;
        }

        // Validate retrieved information
        if n_samples != TEST_SAMPLES as std::os::raw::c_int {
            println!("[ERROR] Sample count mismatch: got {}, expected {}",
                     n_samples, TEST_SAMPLES);
            pacmap_free_model_enhanced(handle);
            return false;
        }

        if n_features != TEST_FEATURES as std::os::raw::c_int {
            println!("[ERROR] Feature count mismatch: got {}, expected {}",
                     n_features, TEST_FEATURES);
            pacmap_free_model_enhanced(handle);
            return false;
        }

        if embedding_dim != TEST_EMBEDDING_DIM as std::os::raw::c_int {
            println!("[ERROR] Embedding dimension mismatch: got {}, expected {}",
                     embedding_dim, TEST_EMBEDDING_DIM);
            pacmap_free_model_enhanced(handle);
            return false;
        }

        println!("[INFO] Model information validated:");
        println!("       Samples: {}, Features: {}, Embedding dim: {}",
                 n_samples, n_features, embedding_dim);
        println!("       Normalization: {}, HNSW M: {}, Memory: {} MB",
                 normalization_mode, hnsw_m, memory_usage_mb);

        pacmap_free_model_enhanced(handle);
        true
    }

    fn test_model_persistence() -> bool {
        let data = generate_test_data(TEST_SAMPLES, TEST_FEATURES);
        let config = pacmap_config_default();
        let mut embedding = vec![0.0; TEST_SAMPLES * TEST_EMBEDDING_DIM];

        // Create and fit model
        let handle = pacmap_fit_transform_enhanced(
            data.as_ptr(),
            TEST_SAMPLES as std::os::raw::c_int,
            TEST_FEATURES as std::os::raw::c_int,
            config,
            embedding.as_mut_ptr(),
            None,
        );

        if handle.is_null() {
            println!("[ERROR] Failed to create model for persistence test");
            return false;
        }

        // Save model
        let save_path = CString::new("test_enhanced_wrapper.bin").unwrap();
        let save_result = pacmap_save_model_enhanced(handle, save_path.as_ptr(), false);
        if save_result != 0 {
            println!("[ERROR] Model save failed with code {}", save_result);
            pacmap_free_model_enhanced(handle);
            return false;
        }

        println!("[INFO] Model saved successfully");

        // Get original model info for comparison
        let mut orig_samples = 0;
        let mut orig_features = 0;
        let mut orig_hnsw_m = 0;
        pacmap_get_model_info(
            handle,
            &mut orig_samples,
            &mut orig_features,
            ptr::null_mut(),
            ptr::null_mut(),
            &mut orig_hnsw_m,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        );

        // Free original model
        pacmap_free_model_enhanced(handle);

        // Load model
        let load_handle = pacmap_load_model_enhanced(save_path.as_ptr());
        if load_handle.is_null() {
            println!("[ERROR] Model load failed");
            std::fs::remove_file("test_enhanced_wrapper.bin").ok();
            return false;
        }

        println!("[INFO] Model loaded successfully");

        // Get loaded model info for comparison
        let mut loaded_samples = 0;
        let mut loaded_features = 0;
        let mut loaded_hnsw_m = 0;
        let info_result = pacmap_get_model_info(
            load_handle,
            &mut loaded_samples,
            &mut loaded_features,
            ptr::null_mut(),
            ptr::null_mut(),
            &mut loaded_hnsw_m,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        );

        if info_result != 0 {
            println!("[ERROR] Loaded model info retrieval failed");
            pacmap_free_model_enhanced(load_handle);
            std::fs::remove_file("test_enhanced_wrapper.bin").ok();
            return false;
        }

        // Verify consistency
        if orig_samples != loaded_samples || orig_features != loaded_features ||
           orig_hnsw_m != loaded_hnsw_m {
            println!("[ERROR] Model consistency check failed");
            println!("        Original: samples={}, features={}, hnsw_m={}",
                     orig_samples, orig_features, orig_hnsw_m);
            println!("        Loaded:   samples={}, features={}, hnsw_m={}",
                     loaded_samples, loaded_features, loaded_hnsw_m);
            pacmap_free_model_enhanced(load_handle);
            std::fs::remove_file("test_enhanced_wrapper.bin").ok();
            return false;
        }

        println!("[INFO] Model persistence verified successfully");

        // Clean up
        pacmap_free_model_enhanced(load_handle);
        std::fs::remove_file("test_enhanced_wrapper.bin").ok();
        true
    }

    fn test_quantization_functionality() -> bool {
        let data = generate_test_data(TEST_SAMPLES, TEST_FEATURES);
        let config = pacmap_config_default();
        let mut embedding = vec![0.0; TEST_SAMPLES * TEST_EMBEDDING_DIM];

        // Create model
        let handle = pacmap_fit_transform_enhanced(
            data.as_ptr(),
            TEST_SAMPLES as std::os::raw::c_int,
            TEST_FEATURES as std::os::raw::c_int,
            config,
            embedding.as_mut_ptr(),
            None,
        );

        if handle.is_null() {
            println!("[ERROR] Failed to create model for quantization test");
            return false;
        }

        // Test save without quantization
        let save_path_no_quant = CString::new("test_no_quant_wrapper.bin").unwrap();
        let save_result = pacmap_save_model_enhanced(handle, save_path_no_quant.as_ptr(), false);
        if save_result != 0 {
            println!("[ERROR] Save without quantization failed");
            pacmap_free_model_enhanced(handle);
            return false;
        }

        // Test save with quantization
        let save_path_with_quant = CString::new("test_with_quant_wrapper.bin").unwrap();
        let save_result = pacmap_save_model_enhanced(handle, save_path_with_quant.as_ptr(), true);
        if save_result != 0 {
            println!("[ERROR] Save with quantization failed");
            pacmap_free_model_enhanced(handle);
            std::fs::remove_file("test_no_quant_wrapper.bin").ok();
            return false;
        }

        // Compare file sizes
        let size_no_quant = std::fs::metadata("test_no_quant_wrapper.bin")
            .map(|m| m.len()).unwrap_or(0);
        let size_with_quant = std::fs::metadata("test_with_quant_wrapper.bin")
            .map(|m| m.len()).unwrap_or(0);

        println!("[INFO] Quantization file sizes:");
        println!("       No quantization: {} bytes", size_no_quant);
        println!("       With quantization: {} bytes", size_with_quant);

        // For small test data, quantization might not save space due to overhead
        // Just verify both files were created and can be loaded
        let load_handle_no_quant = pacmap_load_model_enhanced(save_path_no_quant.as_ptr());
        let load_handle_with_quant = pacmap_load_model_enhanced(save_path_with_quant.as_ptr());

        let success = !load_handle_no_quant.is_null() && !load_handle_with_quant.is_null();

        if success {
            println!("[INFO] Both quantized and non-quantized models loaded successfully");
        } else {
            println!("[ERROR] Failed to load quantized models");
        }

        // Clean up
        pacmap_free_model_enhanced(handle);
        if !load_handle_no_quant.is_null() {
            pacmap_free_model_enhanced(load_handle_no_quant);
        }
        if !load_handle_with_quant.is_null() {
            pacmap_free_model_enhanced(load_handle_with_quant);
        }
        std::fs::remove_file("test_no_quant_wrapper.bin").ok();
        std::fs::remove_file("test_with_quant_wrapper.bin").ok();

        success
    }

    fn test_error_handling() -> bool {
        println!("[INFO] Testing error handling and edge cases...");

        // Test 1: Null pointer handling
        let null_handle = ptr::null_mut();
        let save_path = CString::new("test_null.bin").unwrap();
        let save_result = pacmap_save_model_enhanced(null_handle, save_path.as_ptr(), false);
        if save_result == 0 {
            println!("[ERROR] Save should fail with null handle");
            return false;
        }

        // Test 2: Invalid file path for loading
        let invalid_path = CString::new("nonexistent_file.bin").unwrap();
        let load_handle = pacmap_load_model_enhanced(invalid_path.as_ptr());
        if !load_handle.is_null() {
            println!("[ERROR] Load should fail with invalid path");
            pacmap_free_model_enhanced(load_handle);
            return false;
        }

        // Test 3: Model info with null handle
        let mut dummy = 0;
        let info_result = pacmap_get_model_info(
            null_handle,
            &mut dummy, &mut dummy, &mut dummy, &mut dummy,
            &mut dummy, &mut dummy, &mut dummy, &mut dummy,
        );
        if info_result == 0 {
            println!("[ERROR] Model info should fail with null handle");
            return false;
        }

        // Test 4: Free null handle (should not crash)
        pacmap_free_model_enhanced(null_handle); // Should be safe

        println!("[INFO] Error handling tests completed successfully");
        true
    }

    fn generate_test_data(n_samples: usize, n_features: usize) -> Vec<f64> {
        let mut data = Vec::with_capacity(n_samples * n_features);

        for i in 0..n_samples {
            for j in 0..n_features {
                // Generate structured test data
                let value = (i as f64 * 0.01) + (j as f64 * 0.1) +
                           ((i * j) as f64 * 0.001).sin();
                data.push(value);
            }
        }

        data
    }
}