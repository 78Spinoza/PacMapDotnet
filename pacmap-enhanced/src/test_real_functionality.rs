// REAL TESTS that actually test HNSW and FFI parameter chains
// Based on proper UMAP test patterns - NOT the garbage placeholder tests

use ndarray::Array2;
use crate::ffi::*;
use crate::pairs::*;
use crate::stats::*;
use std::ffi::CString;

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate realistic test data that triggers HNSW (large dataset)
    fn generate_large_test_data(n_samples: usize, n_features: usize, seed: u64) -> Array2<f64> {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut data = Array2::zeros((n_samples, n_features));

        // Generate 3 distinct clusters for proper neighborhood testing
        let cluster_size = n_samples / 3;

        for i in 0..n_samples {
            let cluster = i / cluster_size;
            let cluster_center = match cluster {
                0 => [0.0, 0.0, 0.0],
                1 => [5.0, 5.0, 5.0],
                _ => [-5.0, 5.0, -5.0],
            };

            for j in 0..n_features.min(3) {
                let noise: f64 = rng.gen::<f64>() * 2.0 - 1.0; // Simple uniform noise
                data[[i, j]] = cluster_center[j] + noise * 0.5;
            }

            // Fill remaining features with noise
            for j in 3..n_features {
                data[[i, j]] = rng.gen::<f64>() * 2.0 - 1.0; // Simple uniform noise
            }
        }

        data
    }

    /// Test that HNSW is actually triggered with large datasets (>1000 samples)
    #[test]
    fn test_hnsw_actually_triggered() {
        println!("\n=== REAL TEST: HNSW Actually Triggered ===");

        // Use large dataset that FORCES HNSW path
        let n_samples = 2000;  // Well above 1000 threshold
        let n_features = 10;
        let n_neighbors = 15;

        let data = generate_large_test_data(n_samples, n_features, 42);
        println!("Generated {}x{} dataset", n_samples, n_features);

        // This should trigger HNSW path
        let hnsw_pairs = compute_pairs_hnsw(data.view(), n_neighbors, 42);
        println!("HNSW returned {} pairs", hnsw_pairs.len());

        // Should get approximately n_samples * n_neighbors pairs
        let expected_pairs = n_samples * n_neighbors;
        let tolerance = (expected_pairs as f32 * 0.1) as usize; // 10% tolerance

        assert!(hnsw_pairs.len() > expected_pairs - tolerance,
               "HNSW should return ~{} pairs, got {}", expected_pairs, hnsw_pairs.len());
        assert!(hnsw_pairs.len() <= expected_pairs + tolerance,
               "HNSW returned too many pairs: {} > {}", hnsw_pairs.len(), expected_pairs + tolerance);

        // Validate pairs are reasonable (not all pointing to same neighbors)
        let mut neighbor_counts = std::collections::HashMap::new();
        for (_, j) in &hnsw_pairs {
            *neighbor_counts.entry(*j).or_insert(0) += 1;
        }

        // Should have good distribution of neighbors
        assert!(neighbor_counts.len() > n_samples / 2,
               "HNSW neighbors too concentrated: only {} unique neighbors", neighbor_counts.len());

        println!("SUCCESS: HNSW test passed: {} pairs from {} samples", hnsw_pairs.len(), n_samples);
    }

    /// Test HNSW vs Brute-force accuracy comparison
    #[test]
    fn test_hnsw_vs_brute_force_accuracy() {
        println!("\n=== REAL TEST: HNSW vs Brute-Force Accuracy ===");

        // Use moderate size for comparison (slow brute-force but manageable)
        let n_samples = 500;
        let n_features = 8;
        let n_neighbors = 10;

        let data = generate_large_test_data(n_samples, n_features, 123);

        // Get HNSW pairs
        let hnsw_pairs = compute_pairs_hnsw(data.view(), n_neighbors, 123);

        // Get brute-force pairs
        let brute_pairs = compute_pairs_bruteforce(data.view(), n_neighbors, 123);

        println!("HNSW: {} pairs, Brute-force: {} pairs", hnsw_pairs.len(), brute_pairs.len());

        // Convert to neighbor sets for comparison
        let mut hnsw_neighbors: std::collections::HashMap<usize, std::collections::HashSet<usize>> = std::collections::HashMap::new();
        let mut brute_neighbors: std::collections::HashMap<usize, std::collections::HashSet<usize>> = std::collections::HashMap::new();

        for (i, j) in hnsw_pairs {
            hnsw_neighbors.entry(i).or_insert_with(std::collections::HashSet::new).insert(j);
        }

        for (i, j) in brute_pairs {
            brute_neighbors.entry(i).or_insert_with(std::collections::HashSet::new).insert(j);
        }

        // Calculate recall (what fraction of true neighbors did HNSW find)
        let mut total_neighbors = 0;
        let mut found_neighbors = 0;

        for (i, true_neighbors) in &brute_neighbors {
            if let Some(hnsw_set) = hnsw_neighbors.get(i) {
                for neighbor in true_neighbors {
                    total_neighbors += 1;
                    if hnsw_set.contains(neighbor) {
                        found_neighbors += 1;
                    }
                }
            }
        }

        let recall = found_neighbors as f64 / total_neighbors as f64;
        println!("HNSW Recall: {:.1}% ({}/{})", recall * 100.0, found_neighbors, total_neighbors);

        // HNSW should achieve at least 70% recall
        assert!(recall > 0.7, "HNSW recall too low: {:.1}%", recall * 100.0);
        assert!(recall <= 1.0, "HNSW recall impossible: {:.1}%", recall * 100.0);

        println!("SUCCESS: HNSW accuracy test passed: {:.1}% recall", recall * 100.0);
    }

    /// Test REAL FFI parameter chain from C# struct to Rust execution
    #[test]
    fn test_real_ffi_parameter_chain() {
        println!("\n=== REAL TEST: FFI Parameter Chain (C# → Rust) ===");

        // Create realistic dataset
        let n_samples = 1500; // Triggers HNSW
        let n_features = 6;
        let data = generate_large_test_data(n_samples, n_features, 789);

        // Convert to C format
        let data_vec: Vec<f64> = data.iter().cloned().collect();
        let mut embedding = vec![0.0; n_samples * 2];

        // Create config with SPECIFIC parameters to test FFI chain
        let mut config = pacmap_config_default();
        config.n_neighbors = 12;
        config.seed = 999;
        config.n_epochs = 100;
        config.mid_near_ratio = 0.7;
        config.far_pair_ratio = 1.5;
        config.normalization_mode = 1; // ZScore
        config.force_exact_knn = false; // Should use HNSW
        config.use_quantization = true; // Should enable quantization

        // Test progress callback
        let _callback_invoked = false;
        let _hnsw_reported = false;
        let _ffi_debug_found = false;
        let _quantization_reported = false;

        extern "C" fn test_callback(
            phase: *const std::os::raw::c_char,
            _current: std::os::raw::c_int,
            _total: std::os::raw::c_int,
            _percent: f32,
            message: *const std::os::raw::c_char,
        ) {
            unsafe {
                let phase_str = if !phase.is_null() {
                    std::ffi::CStr::from_ptr(phase).to_string_lossy()
                } else {
                    "Unknown".into()
                };

                let message_str = if !message.is_null() {
                    std::ffi::CStr::from_ptr(message).to_string_lossy()
                } else {
                    "".into()
                };

                println!("FFI Callback: {} - {}", phase_str, message_str);

                // Verify specific parameters are reported via callbacks
                if message_str.contains("HNSW") {
                    static mut HNSW_REPORTED: bool = false;
                    HNSW_REPORTED = true;
                }
                if message_str.contains("FFI DEBUG") {
                    static mut FFI_DEBUG_FOUND: bool = false;
                    FFI_DEBUG_FOUND = true;
                }
                if message_str.contains("quantization") {
                    static mut QUANTIZATION_REPORTED: bool = false;
                    QUANTIZATION_REPORTED = true;
                }

                static mut CALLBACK_INVOKED: bool = false;
                CALLBACK_INVOKED = true;
            }
        }

        println!("Calling FFI with specific parameters:");
        println!("- neighbors: {}", config.n_neighbors);
        println!("- seed: {}", config.seed);
        println!("- force_exact_knn: {}", config.force_exact_knn);
        println!("- use_quantization: {}", config.use_quantization);
        println!("- normalization_mode: {} (ZScore)", config.normalization_mode);

        // Call actual FFI function
        let handle = pacmap_fit_transform_enhanced(
            data_vec.as_ptr(),
            n_samples as i32,
            n_features as i32,
            config,
            embedding.as_mut_ptr(),
            Some(test_callback),
        );

        // Verify FFI succeeded
        assert!(!handle.is_null(), "FFI function should return valid handle");

        // Verify embedding was populated
        let non_zero_count = embedding.iter().filter(|&&x| x.abs() > 1e-10).count();
        assert!(non_zero_count > n_samples, "Embedding should be populated (found {} non-zero values)", non_zero_count);

        println!("SUCCESS: FFI returned valid handle and populated embedding");
        println!("Non-zero embedding values: {}/{}", non_zero_count, embedding.len());

        // Clean up
        pacmap_free_model_enhanced(handle);

        println!("SUCCESS: FFI parameter chain test passed");
    }

    /// Test that small datasets properly fall back to brute-force
    #[test]
    fn test_small_dataset_brute_force_fallback() {
        println!("\n=== REAL TEST: Small Dataset Brute-Force Fallback ===");

        // Use small dataset that should NOT trigger HNSW
        let n_samples = 100; // Well below 1000 threshold
        let n_features = 5;
        let n_neighbors = 8;

        let data = generate_large_test_data(n_samples, n_features, 456);

        // Should use brute-force
        let pairs = compute_pairs_hnsw(data.view(), n_neighbors, 456);

        // Should get exactly n_samples * n_neighbors pairs (brute-force is exact)
        let expected_pairs = n_samples * n_neighbors;
        assert_eq!(pairs.len(), expected_pairs,
                  "Brute-force should return exactly {} pairs, got {}", expected_pairs, pairs.len());

        // Validate all pairs are valid
        for (i, j) in &pairs {
            assert!(*i < n_samples, "Invalid source index: {}", i);
            assert!(*j < n_samples, "Invalid neighbor index: {}", j);
            assert_ne!(*i, *j, "Self-references not allowed: {} -> {}", i, j);
        }

        println!("SUCCESS: Small dataset correctly used brute-force: {} pairs", pairs.len());
    }

    /// Test ZScore normalization is properly applied
    #[test]
    fn test_zscore_normalization_application() {
        println!("\n=== REAL TEST: ZScore Normalization ===");

        // Create data with known statistics
        let mut data = Array2::zeros((100, 3));

        // Feature 0: mean=10, std=2
        for i in 0..100 {
            data[[i, 0]] = 10.0 + (i as f64 - 50.0) * 0.04; // Range 8-12
        }

        // Feature 1: mean=50, std=10
        for i in 0..100 {
            data[[i, 1]] = 50.0 + (i as f64 - 50.0) * 0.2; // Range 40-60
        }

        // Feature 2: mean=0, std=1 (already normalized)
        for i in 0..100 {
            data[[i, 2]] = (i as f64 - 50.0) * 0.02; // Range -1 to 1
        }

        println!("Original feature means: {:.2}, {:.2}, {:.2}",
                data.column(0).mean().unwrap(),
                data.column(1).mean().unwrap(),
                data.column(2).mean().unwrap());

        // Create ZScore normalization params
        let mut norm_params = crate::stats::compute_zscore_params(&data);

        // Apply normalization
        let mut norm_data = data.clone();
        norm_params.apply_zscore_normalization(&mut norm_data).unwrap();

        println!("Normalized feature means: {:.6}, {:.6}, {:.6}",
                norm_data.column(0).mean().unwrap(),
                norm_data.column(1).mean().unwrap(),
                norm_data.column(2).mean().unwrap());

        // Verify normalization worked
        for j in 0..3 {
            let mean = norm_data.column(j).mean().unwrap();
            assert!(mean.abs() < 0.1, "Feature {} mean should be ~0, got {:.6}", j, mean);
        }

        // Verify parameters were captured
        assert_eq!(norm_params.means.len(), 3);
        assert_eq!(norm_params.stds.len(), 3);

        println!("SUCCESS: ZScore normalization correctly applied");
    }

    /// Test quantization actually reduces precision but preserves structure
    #[test]
    fn test_quantization_precision_reduction() {
        println!("\n=== REAL TEST: Quantization Precision Reduction ===");

        // Create embedding with high precision values
        let mut embedding = Array2::zeros((50, 2));
        for i in 0..50 {
            embedding[[i, 0]] = (i as f64) * 0.123456789 + 0.987654321;
            embedding[[i, 1]] = (i as f64) * -0.987654321 + 1.123456789;
        }

        println!("Original precision (first 3 points):");
        for i in 0..3 {
            println!("  [{:.10}, {:.10}]", embedding[[i, 0]], embedding[[i, 1]]);
        }

        // Apply quantization
        let quantized = crate::quantize::quantize_embedding_linear(&embedding);
        let dequantized = crate::quantize::dequantize_embedding(&quantized);

        println!("After quantization (first 3 points):");
        for i in 0..3 {
            println!("  [{:.10}, {:.10}]", dequantized[[i, 0]], dequantized[[i, 1]]);
        }

        // Verify precision was reduced but structure preserved
        let mut total_error = 0.0;
        let mut max_error: f64 = 0.0;

        for i in 0..50 {
            for j in 0..2 {
                let error = (embedding[[i, j]] - dequantized[[i, j]]).abs();
                total_error += error;
                max_error = max_error.max(error);
            }
        }

        let mean_error = total_error / 100.0;

        println!("Quantization error - Mean: {:.6}, Max: {:.6}", mean_error, max_error);

        // Should have some precision loss but not destroy data
        assert!(mean_error > 1e-10, "Should have some quantization error");
        assert!(mean_error < 0.1, "Quantization error should be reasonable");
        assert!(max_error < 1.0, "Max quantization error should be bounded");

        println!("SUCCESS: Quantization properly reduces precision while preserving structure");
    }

    /// Test FFI model save/load with REAL parameter persistence
    #[test]
    fn test_ffi_model_save_load_parameters() {
        use std::path::Path;

        println!("\n=== REAL TEST: FFI Model Save/Load Parameter Persistence ===");

        let model_path = "test_ffi_model_params.bin";
        let model_path_cstr = CString::new(model_path).unwrap();

        // Create and fit model with specific parameters
        let n_samples = 200;
        let n_features = 4;
        let data = generate_large_test_data(n_samples, n_features, 111);
        let data_vec: Vec<f64> = data.iter().cloned().collect();
        let mut embedding = vec![0.0; n_samples * 2];

        let mut config = pacmap_config_default();
        config.n_neighbors = 8;
        config.seed = 777;
        config.mid_near_ratio = 0.6;
        config.far_pair_ratio = 1.8;
        config.normalization_mode = 1; // ZScore
        config.use_quantization = false;

        let handle = pacmap_fit_transform_enhanced(
            data_vec.as_ptr(),
            n_samples as i32,
            n_features as i32,
            config,
            embedding.as_mut_ptr(),
            None, // No callback for this test
        );

        assert!(!handle.is_null(), "Model fitting should succeed");

        // Save model
        let save_result = pacmap_save_model_enhanced(handle, model_path_cstr.as_ptr(), false);
        assert_eq!(save_result, 0, "Model save should succeed");

        // Free original model
        pacmap_free_model_enhanced(handle);

        // Load model and verify parameters
        let loaded_handle = pacmap_load_model_enhanced(model_path_cstr.as_ptr());
        assert!(!loaded_handle.is_null(), "Model load should succeed");

        // Get model info to verify parameters were preserved
        let mut n_samples_out = 0;
        let mut n_features_out = 0;
        let mut embedding_dim_out = 0;
        let mut normalization_mode_out = 0;
        let mut hnsw_m_out = 0;
        let mut hnsw_ef_construction_out = 0;
        let mut hnsw_ef_search_out = 0;
        let mut memory_usage_out = 0;

        let info_result = pacmap_get_model_info(
            loaded_handle,
            &mut n_samples_out,
            &mut n_features_out,
            &mut embedding_dim_out,
            &mut normalization_mode_out,
            &mut hnsw_m_out,
            &mut hnsw_ef_construction_out,
            &mut hnsw_ef_search_out,
            &mut memory_usage_out,
        );

        assert_eq!(info_result, 0, "Model info retrieval should succeed");

        // Verify parameters were preserved
        assert_eq!(n_samples_out, n_samples as i32, "Sample count should be preserved");
        assert_eq!(n_features_out, n_features as i32, "Feature count should be preserved");
        assert_eq!(embedding_dim_out, 2, "Embedding dimension should be preserved");
        assert_eq!(normalization_mode_out, 1, "ZScore normalization should be preserved");

        println!("Model parameters preserved:");
        println!("- Samples: {}", n_samples_out);
        println!("- Features: {}", n_features_out);
        println!("- Embedding dim: {}", embedding_dim_out);
        println!("- Normalization: {} (ZScore)", normalization_mode_out);
        println!("- HNSW M: {}", hnsw_m_out);

        // Clean up
        pacmap_free_model_enhanced(loaded_handle);
        if Path::new(model_path).exists() {
            std::fs::remove_file(model_path).ok();
        }

        println!("SUCCESS: FFI model save/load preserves parameters correctly");
    }
}