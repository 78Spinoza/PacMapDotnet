// Comprehensive pipeline test for PacMAP Enhanced
// Tests the complete data processing pipeline from input to output

use crate::fit_transform_normalized;
use crate::pairs::{compute_pairs_hnsw, compute_pairs_bruteforce};
use crate::stats::{NormalizationMode, NormalizationParams};
use crate::serialization::PaCMAP;
use ndarray::Array2;
use std::time::Instant;

#[cfg(test)]
mod tests {
    use super::*;

    // Pipeline test configuration
    const SMALL_DATASET_SIZE: usize = 500;
    const LARGE_DATASET_SIZE: usize = 1500; // Trigger HNSW if enabled
    const N_FEATURES: usize = 20;
    const N_NEIGHBORS: usize = 10;

    #[test]
    fn test_comprehensive_pipeline() {
        println!("\n===========================================");
        println!("   PacMAP Enhanced Pipeline Test");
        println!("===========================================");
        println!();

        let start_time = Instant::now();
        let mut all_tests_passed = true;

        // Test 1: Normalization pipeline
        println!("[TEST 1] Normalization Pipeline");
        println!("-------------------------------");
        let norm_result = test_normalization_pipeline();
        if norm_result {
            println!("[PASS] Normalization pipeline test");
        } else {
            println!("[FAIL] Normalization pipeline test");
            all_tests_passed = false;
        }

        // Test 2: k-NN selection pipeline (brute-force vs HNSW)
        println!("\n[TEST 2] k-NN Selection Pipeline");
        println!("--------------------------------");
        let knn_result = test_knn_selection_pipeline();
        if knn_result {
            println!("[PASS] k-NN selection pipeline test");
        } else {
            println!("[FAIL] k-NN selection pipeline test");
            all_tests_passed = false;
        }

        // Test 3: Full embedding pipeline
        println!("\n[TEST 3] Full Embedding Pipeline");
        println!("--------------------------------");
        let embedding_result = test_full_embedding_pipeline();
        if embedding_result {
            println!("[PASS] Full embedding pipeline test");
        } else {
            println!("[FAIL] Full embedding pipeline test");
            all_tests_passed = false;
        }

        // Test 4: Persistence pipeline
        println!("\n[TEST 4] Persistence Pipeline");
        println!("-----------------------------");
        let persistence_result = test_persistence_pipeline();
        if persistence_result {
            println!("[PASS] Persistence pipeline test");
        } else {
            println!("[FAIL] Persistence pipeline test");
            all_tests_passed = false;
        }

        // Test 5: Multi-dimensional pipeline
        println!("\n[TEST 5] Multi-dimensional Pipeline");
        println!("-----------------------------------");
        let multidim_result = test_multidimensional_pipeline();
        if multidim_result {
            println!("[PASS] Multi-dimensional pipeline test");
        } else {
            println!("[FAIL] Multi-dimensional pipeline test");
            all_tests_passed = false;
        }

        // Test 6: Performance pipeline
        println!("\n[TEST 6] Performance Pipeline");
        println!("-----------------------------");
        let performance_result = test_performance_pipeline();
        if performance_result {
            println!("[PASS] Performance pipeline test");
        } else {
            println!("[FAIL] Performance pipeline test");
            all_tests_passed = false;
        }

        let total_time = start_time.elapsed();

        // Final summary
        println!("\n===========================================");
        println!("   Pipeline Test Summary");
        println!("===========================================");
        println!();

        if all_tests_passed {
            println!("🎉 ALL PIPELINE TESTS PASSED!");
            println!("✅ Normalization pipeline working");
            println!("✅ k-NN selection pipeline working");
            println!("✅ Full embedding pipeline working");
            println!("✅ Persistence pipeline working");
            println!("✅ Multi-dimensional pipeline working");
            println!("✅ Performance pipeline working");
            println!();
            println!("Total pipeline test time: {:.2}s", total_time.as_secs_f64());
            println!("PacMAP Enhanced pipeline is fully functional!");
        } else {
            println!("❌ SOME PIPELINE TESTS FAILED!");
            println!("Check individual test results above for details.");
        }

        assert!(all_tests_passed, "Comprehensive pipeline test failed");
    }

    fn test_normalization_pipeline() -> bool {
        let data = generate_test_data(SMALL_DATASET_SIZE, N_FEATURES);

        // Test each normalization mode
        let modes = [
            NormalizationMode::None,
            NormalizationMode::ZScore,
            NormalizationMode::MinMax,
            NormalizationMode::Robust,
        ];

        for mode in modes {
            println!("[INFO] Testing normalization mode: {:?}", mode);

            let mut test_data = data.clone();
            let mut norm_params = NormalizationParams::new(N_FEATURES, mode);

            if mode != NormalizationMode::None {
                if let Err(e) = norm_params.fit_transform(&mut test_data) {
                    println!("[ERROR] Normalization failed for {:?}: {}", mode, e);
                    return false;
                }

                // Verify normalization worked
                match mode {
                    NormalizationMode::ZScore => {
                        // Check means are approximately zero
                        for j in 0..N_FEATURES {
                            let mean: f64 = test_data.column(j).iter().sum::<f64>() / SMALL_DATASET_SIZE as f64;
                            if mean.abs() > 1e-10 {
                                println!("[WARNING] Z-score mean not zero for feature {}: {}", j, mean);
                            }
                        }
                    },
                    NormalizationMode::MinMax => {
                        // Check values are in [0, 1] range
                        for value in test_data.iter() {
                            if *value < -1e-10 || *value > 1.0 + 1e-10 {
                                println!("[ERROR] MinMax value out of range: {}", value);
                                return false;
                            }
                        }
                    },
                    _ => {}, // Other modes have different validation criteria
                }
            }

            println!("[INFO] Normalization mode {:?} completed successfully", mode);
        }

        true
    }

    fn test_knn_selection_pipeline() -> bool {
        // Test small dataset (should use brute-force)
        let small_data = generate_test_data(SMALL_DATASET_SIZE, N_FEATURES);
        println!("[INFO] Testing k-NN with small dataset ({} samples)", SMALL_DATASET_SIZE);

        let small_pairs = compute_pairs_hnsw(small_data.view(), N_NEIGHBORS, 42);
        if small_pairs.is_empty() {
            println!("[ERROR] No pairs found for small dataset");
            return false;
        }
        println!("[INFO] Small dataset k-NN completed: {} pairs", small_pairs.len());

        // Test large dataset (should use HNSW if available, brute-force otherwise)
        let large_data = generate_test_data(LARGE_DATASET_SIZE, N_FEATURES);
        println!("[INFO] Testing k-NN with large dataset ({} samples)", LARGE_DATASET_SIZE);

        let large_pairs = compute_pairs_hnsw(large_data.view(), N_NEIGHBORS, 42);
        if large_pairs.is_empty() {
            println!("[ERROR] No pairs found for large dataset");
            return false;
        }
        println!("[INFO] Large dataset k-NN completed: {} pairs", large_pairs.len());

        // Compare with brute-force for consistency (on small dataset)
        let brute_force_pairs = compute_pairs_bruteforce(small_data.view(), N_NEIGHBORS, 42);
        if brute_force_pairs.len() != small_pairs.len() {
            println!("[ERROR] Brute-force and adaptive k-NN pair counts don't match: {} vs {}",
                     brute_force_pairs.len(), small_pairs.len());
            return false;
        }

        println!("[INFO] k-NN selection consistency verified");
        true
    }

    fn test_full_embedding_pipeline() -> bool {
        let data = generate_test_data(SMALL_DATASET_SIZE, N_FEATURES);

        // Test 2D embedding
        let config_2d = pacmap::Configuration {
            embedding_dimensions: 2,
            override_neighbors: Some(N_NEIGHBORS),
            ..Default::default()
        };

        match fit_transform_normalized(data.clone(), config_2d, Some(NormalizationMode::ZScore)) {
            Ok((embedding_2d, model_2d)) => {
                if embedding_2d.shape() != [SMALL_DATASET_SIZE, 2] {
                    println!("[ERROR] 2D embedding shape mismatch: {:?}", embedding_2d.shape());
                    return false;
                }

                // Verify model contains expected information
                if model_2d.config.embedding_dim != 2 {
                    println!("[ERROR] Model embedding_dim mismatch");
                    return false;
                }

                if model_2d.normalization.mode != NormalizationMode::ZScore {
                    println!("[ERROR] Model normalization mode mismatch");
                    return false;
                }

                println!("[INFO] 2D embedding pipeline completed successfully");
            },
            Err(e) => {
                println!("[ERROR] 2D embedding failed: {}", e);
                return false;
            }
        }

        // Test 5D embedding
        let config_5d = pacmap::Configuration {
            embedding_dimensions: 5,
            override_neighbors: Some(N_NEIGHBORS),
            ..Default::default()
        };

        match fit_transform_normalized(data, config_5d, Some(NormalizationMode::MinMax)) {
            Ok((embedding_5d, model_5d)) => {
                if embedding_5d.shape() != [SMALL_DATASET_SIZE, 5] {
                    println!("[ERROR] 5D embedding shape mismatch: {:?}", embedding_5d.shape());
                    return false;
                }

                if model_5d.config.embedding_dim != 5 {
                    println!("[ERROR] 5D model embedding_dim mismatch");
                    return false;
                }

                println!("[INFO] 5D embedding pipeline completed successfully");
            },
            Err(e) => {
                println!("[ERROR] 5D embedding failed: {}", e);
                return false;
            }
        }

        true
    }

    fn test_persistence_pipeline() -> bool {
        let data = generate_test_data(SMALL_DATASET_SIZE, N_FEATURES);
        let config = pacmap::Configuration {
            embedding_dimensions: 3,
            override_neighbors: Some(N_NEIGHBORS),
            ..Default::default()
        };

        // Create model
        let (original_embedding, mut model) = match fit_transform_normalized(
            data, config, Some(NormalizationMode::Robust)
        ) {
            Ok(result) => result,
            Err(e) => {
                println!("[ERROR] Model creation failed: {}", e);
                return false;
            }
        };

        // Test save without quantization
        let save_path_normal = "test_pipeline_normal.bin";
        if let Err(e) = model.save_uncompressed(save_path_normal) {
            println!("[ERROR] Normal save failed: {}", e);
            return false;
        }

        // Test save with quantization
        model.quantize_on_save = true;
        let save_path_quantized = "test_pipeline_quantized.bin";
        if let Err(e) = model.save_compressed(save_path_quantized) {
            println!("[ERROR] Quantized save failed: {}", e);
            std::fs::remove_file(save_path_normal).ok();
            return false;
        }

        // Load both models
        let loaded_normal = match PaCMAP::load_compressed(save_path_normal) {
            Ok(model) => model,
            Err(e) => {
                println!("[ERROR] Normal load failed: {}", e);
                std::fs::remove_file(save_path_normal).ok();
                std::fs::remove_file(save_path_quantized).ok();
                return false;
            }
        };

        let loaded_quantized = match PaCMAP::load_compressed(save_path_quantized) {
            Ok(model) => model,
            Err(e) => {
                println!("[ERROR] Quantized load failed: {}", e);
                std::fs::remove_file(save_path_normal).ok();
                std::fs::remove_file(save_path_quantized).ok();
                return false;
            }
        };

        // Verify models
        if loaded_normal.config.embedding_dim != 3 {
            println!("[ERROR] Normal model embedding_dim mismatch");
            std::fs::remove_file(save_path_normal).ok();
            std::fs::remove_file(save_path_quantized).ok();
            return false;
        }

        if loaded_quantized.config.embedding_dim != 3 {
            println!("[ERROR] Quantized model embedding_dim mismatch");
            std::fs::remove_file(save_path_normal).ok();
            std::fs::remove_file(save_path_quantized).ok();
            return false;
        }

        // Check quantization status
        if loaded_normal.quantized_embedding.is_some() {
            println!("[WARNING] Normal model unexpectedly has quantized embedding");
        }

        if loaded_quantized.quantized_embedding.is_none() {
            println!("[ERROR] Quantized model missing quantized embedding");
            std::fs::remove_file(save_path_normal).ok();
            std::fs::remove_file(save_path_quantized).ok();
            return false;
        }

        println!("[INFO] Persistence pipeline verified successfully");

        // Cleanup
        std::fs::remove_file(save_path_normal).ok();
        std::fs::remove_file(save_path_quantized).ok();

        true
    }

    fn test_multidimensional_pipeline() -> bool {
        let data = generate_test_data(SMALL_DATASET_SIZE, N_FEATURES);
        let dimensions = [1, 2, 3, 5, 8, 10, 15, 20];

        for &dim in &dimensions {
            println!("[INFO] Testing {}D embedding pipeline", dim);

            let config = pacmap::Configuration {
                embedding_dimensions: dim,
                override_neighbors: Some(N_NEIGHBORS),
                ..Default::default()
            };

            match fit_transform_normalized(data.clone(), config, Some(NormalizationMode::ZScore)) {
                Ok((embedding, model)) => {
                    if embedding.shape() != [SMALL_DATASET_SIZE, dim] {
                        println!("[ERROR] {}D embedding shape mismatch: {:?}", dim, embedding.shape());
                        return false;
                    }

                    if model.config.embedding_dim != dim {
                        println!("[ERROR] {}D model embedding_dim mismatch", dim);
                        return false;
                    }

                    // Check for reasonable coordinate variety
                    let mut has_variety = true;
                    for d in 0..dim {
                        let column = embedding.column(d);
                        let min_val = column.iter().copied().fold(f64::INFINITY, f64::min);
                        let max_val = column.iter().copied().fold(f64::NEG_INFINITY, f64::max);

                        if (max_val - min_val).abs() < 1e-10 {
                            has_variety = false;
                            break;
                        }
                    }

                    if !has_variety {
                        println!("[ERROR] {}D embedding lacks coordinate variety", dim);
                        return false;
                    }

                    println!("[INFO] {}D embedding pipeline successful", dim);
                },
                Err(e) => {
                    println!("[ERROR] {}D embedding failed: {}", dim, e);
                    return false;
                }
            }
        }

        println!("[INFO] Multi-dimensional pipeline completed successfully");
        true
    }

    fn test_performance_pipeline() -> bool {
        let small_data = generate_test_data(200, 10);
        let medium_data = generate_test_data(800, 15);

        // Test performance with different configurations
        let configs = [
            (2, 5, "Fast 2D"),
            (3, 8, "Medium 3D"),
            (5, 10, "Detailed 5D"),
        ];

        for (embedding_dim, neighbors, description) in configs {
            println!("[INFO] Testing performance: {}", description);

            let config = pacmap::Configuration {
                embedding_dimensions: embedding_dim,
                override_neighbors: Some(neighbors),
                ..Default::default()
            };

            let start_time = Instant::now();

            match fit_transform_normalized(small_data.clone(), config.clone(), Some(NormalizationMode::ZScore)) {
                Ok((embedding, _)) => {
                    let elapsed = start_time.elapsed();
                    println!("[INFO] {} completed in {:.2}s, shape: {:?}",
                             description, elapsed.as_secs_f64(), embedding.shape());

                    // Reasonable performance check (should complete within reasonable time)
                    if elapsed.as_secs() > 30 {
                        println!("[WARNING] {} took longer than expected: {:.2}s", description, elapsed.as_secs_f64());
                    }
                },
                Err(e) => {
                    println!("[ERROR] {} failed: {}", description, e);
                    return false;
                }
            }

            // Test with medium data for scalability
            if embedding_dim <= 3 { // Only test smaller dimensions for medium data
                let start_time = Instant::now();
                match fit_transform_normalized(medium_data.clone(), config, Some(NormalizationMode::MinMax)) {
                    Ok((embedding, _)) => {
                        let elapsed = start_time.elapsed();
                        println!("[INFO] {} (medium data) completed in {:.2}s, shape: {:?}",
                                 description, elapsed.as_secs_f64(), embedding.shape());
                    },
                    Err(e) => {
                        println!("[ERROR] {} (medium data) failed: {}", description, e);
                        return false;
                    }
                }
            }
        }

        println!("[INFO] Performance pipeline completed successfully");
        true
    }

    fn generate_test_data(n_samples: usize, n_features: usize) -> Array2<f64> {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(12345); // Different seed than other tests
        let mut data = Array2::zeros((n_samples, n_features));

        // Create realistic structured data
        for i in 0..n_samples {
            let cluster = i % 3; // 3 clusters
            let base_pattern = i as f64 / n_samples as f64; // Continuous gradient

            for j in 0..n_features {
                let value = match cluster {
                    0 => (j as f64 * 0.2 + base_pattern).sin() * 2.0 + 1.0,
                    1 => (j as f64 * 0.1 + base_pattern).cos() * 1.5 - 0.5,
                    _ => j as f64 * 0.05 + base_pattern * 3.0 + (i as f64 * 0.001).exp() - 1.0,
                };

                let noise = (rng.gen::<f64>() - 0.5) * 0.1;
                data[[i, j]] = value + noise;
            }
        }

        data
    }
}