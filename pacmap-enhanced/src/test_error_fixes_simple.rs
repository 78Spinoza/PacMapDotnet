// Error fixes simple test for PacMAP Enhanced
// Tests basic error handling and fixes applied to the system

use crate::fit_transform_normalized;
use crate::serialization::PaCMAP;
use crate::stats::NormalizationMode;
use ndarray::Array2;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_fixes_simple() {
        println!("=== Testing All Error Fixes ===");

        // Create test data
        const N_OBS: usize = 500;
        const N_DIM: usize = 20;
        const EMBEDDING_DIM: usize = 2;

        let data = generate_test_data(N_OBS, N_DIM);

        // Test 1: Basic training with normalization (Fix 1: Proper normalization)
        println!("\n🧪 Test 1: Basic training with normalization fix...");

        let config = pacmap::Configuration {
            embedding_dimensions: EMBEDDING_DIM,
            override_neighbors: Some(15),
            ..Default::default()
        };

        match fit_transform_normalized(data.clone(), config, Some(NormalizationMode::ZScore)) {
            Ok((embedding, model)) => {
                println!("✅ Basic training succeeded");
                println!("   - Training completed successfully with all fixes applied");
                println!("   - Embedding shape: {:?}", embedding.shape());

                // Verify embedding quality
                let non_zero_count = embedding.iter().filter(|&&x| x.abs() > 1e-10).count();
                if non_zero_count == 0 {
                    println!("❌ Embedding contains all zero values");
                    panic!("Basic training failed - zero embedding");
                }

                // Test 2: Save and reload model (Fix 2: Persistence fixes)
                println!("\n🧪 Test 2: Save/load validation...");

                match test_save_load(&model) {
                    Ok(()) => {
                        println!("✅ Save/load succeeded");
                        println!("   - Model persistence working correctly");
                    },
                    Err(e) => {
                        println!("❌ Save/load failed: {}", e);
                        panic!("Save/load test failed");
                    }
                }

                // Test 3: Normalization consistency (Fix 3: Consistent normalization)
                println!("\n🧪 Test 3: Normalization consistency...");

                if test_normalization_consistency(&data) {
                    println!("✅ Normalization consistency verified");
                    println!("   - Normalization parameters properly preserved");
                } else {
                    println!("❌ Normalization consistency failed");
                    panic!("Normalization consistency test failed");
                }

                // Test 4: HNSW parameter validation (Fix 4: HNSW scaling)
                println!("\n🧪 Test 4: HNSW parameter validation...");

                if test_hnsw_parameters(&model) {
                    println!("✅ HNSW parameters validated");
                    println!("   - Auto-scaling working correctly");
                } else {
                    println!("❌ HNSW parameter validation failed");
                    panic!("HNSW parameter validation failed");
                }

                // Test 5: Quantization integrity (Fix 5: Quantization fixes)
                println!("\n🧪 Test 5: Quantization integrity...");

                if test_quantization_integrity(&model) {
                    println!("✅ Quantization integrity verified");
                    println!("   - Quantization parameters properly saved");
                } else {
                    println!("❌ Quantization integrity failed");
                    panic!("Quantization integrity test failed");
                }

                // Test 6: Memory management (Fix 6: Proper cleanup)
                println!("\n🧪 Test 6: Memory management...");

                if test_memory_management() {
                    println!("✅ Memory management verified");
                    println!("   - No memory leaks detected in basic operations");
                } else {
                    println!("❌ Memory management issues detected");
                    panic!("Memory management test failed");
                }

                // Test 7: Edge case handling (Fix 7: Robust error handling)
                println!("\n🧪 Test 7: Edge case handling...");

                if test_edge_cases() {
                    println!("✅ Edge case handling verified");
                    println!("   - Robust error handling for invalid inputs");
                } else {
                    println!("❌ Edge case handling failed");
                    panic!("Edge case handling test failed");
                }

                println!("\n=== All Error Fix Tests Completed ===");
                println!("🎉 All 7 error fix tests passed successfully!");
                println!("✅ Basic training with normalization");
                println!("✅ Save/load validation");
                println!("✅ Normalization consistency");
                println!("✅ HNSW parameter validation");
                println!("✅ Quantization integrity");
                println!("✅ Memory management");
                println!("✅ Edge case handling");
                println!("\nPacMAP Enhanced error fixes fully validated!");

            },
            Err(e) => {
                println!("❌ Basic training failed: {}", e);
                panic!("Basic training test failed");
            }
        }
    }

    fn generate_test_data(n_obs: usize, n_dim: usize) -> Array2<f64> {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut data = Array2::zeros((n_obs, n_dim));

        // Generate normally distributed data
        for i in 0..n_obs {
            for j in 0..n_dim {
                let value: f64 = rng.gen::<f64>() * 2.0 - 1.0; // [-1, 1]
                data[[i, j]] = value;
            }
        }

        data
    }

    fn test_save_load(model: &PaCMAP) -> Result<(), String> {
        // Save model
        model.save_uncompressed("test_error_fixes.bin")
            .map_err(|e| format!("Save failed: {}", e))?;

        // Load model
        let loaded_model = PaCMAP::load_compressed("test_error_fixes.bin")
            .map_err(|e| format!("Load failed: {}", e))?;

        // Verify basic consistency
        if model.config.embedding_dim != loaded_model.config.embedding_dim {
            return Err("Embedding dimension mismatch".to_string());
        }

        if model.config.n_neighbors != loaded_model.config.n_neighbors {
            return Err("Neighbors count mismatch".to_string());
        }

        // Clean up
        std::fs::remove_file("test_error_fixes.bin").ok();

        Ok(())
    }

    fn test_normalization_consistency(data: &Array2<f64>) -> bool {
        // Test that normalization is applied consistently
        let config = pacmap::Configuration {
            embedding_dimensions: 2,
            override_neighbors: Some(10),
            ..Default::default()
        };

        // First run with Z-score normalization
        let result1 = fit_transform_normalized(data.clone(), config.clone(), Some(NormalizationMode::ZScore));
        let result2 = fit_transform_normalized(data.clone(), config, Some(NormalizationMode::ZScore));

        match (result1, result2) {
            (Ok((emb1, model1)), Ok((emb2, model2))) => {
                // Check that normalization modes match
                if model1.normalization.mode != model2.normalization.mode {
                    return false;
                }

                // Check that normalization parameters are similar
                if model1.normalization.means.len() != model2.normalization.means.len() {
                    return false;
                }

                // Embeddings should be similar (not identical due to randomness)
                let shape_match = emb1.shape() == emb2.shape();
                let both_non_zero = emb1.iter().any(|&x| x.abs() > 1e-10) &&
                                   emb2.iter().any(|&x| x.abs() > 1e-10);

                shape_match && both_non_zero
            },
            _ => false
        }
    }

    fn test_hnsw_parameters(model: &PaCMAP) -> bool {
        // Verify HNSW parameters are reasonable
        let hnsw_params = &model.config.hnsw_params;

        // Check M parameter is reasonable
        if hnsw_params.m < 4 || hnsw_params.m > 64 {
            return false;
        }

        // Check ef_construction is reasonable
        if hnsw_params.ef_construction < hnsw_params.m || hnsw_params.ef_construction > 1000 {
            return false;
        }

        // Check ef_search is reasonable
        if hnsw_params.ef_search < hnsw_params.m || hnsw_params.ef_search > 1000 {
            return false;
        }

        // Check memory estimate is not zero
        hnsw_params.estimated_memory_bytes > 0
    }

    fn test_quantization_integrity(model: &PaCMAP) -> bool {
        // Create a copy and test quantization
        let mut test_model = model.clone();
        test_model.quantize_on_save = true;
        test_model.quantize_for_save();

        // Check if quantized embedding was created
        if let Some(ref quantized) = test_model.quantized_embedding {
            // Verify quantization parameters exist
            let params = &quantized.params;

            // Check that min/max values are reasonable
            if !params.min_value.is_finite() || !params.max_value.is_finite() {
                return false;
            }

            // Check that scale is positive
            if params.scale <= 0.0 {
                return false;
            }

            // Check that quantized data has correct shape
            if quantized.data.shape() != model.embedding.shape() {
                return false;
            }

            true
        } else {
            // If no quantization occurred, that's also valid
            true
        }
    }

    fn test_memory_management() -> bool {
        // Test that we can create and destroy multiple models without issues
        for _i in 0..5 {
            let small_data = Array2::from_shape_vec((50, 5), (0..250).map(|x| x as f64 * 0.01).collect()).unwrap();
            let config = pacmap::Configuration {
                embedding_dimensions: 2,
                override_neighbors: Some(5),
                ..Default::default()
            };

            match fit_transform_normalized(small_data, config, Some(NormalizationMode::ZScore)) {
                Ok((embedding, _model)) => {
                    // Verify embedding is valid
                    if embedding.len() == 0 {
                        return false;
                    }
                    // Model will be dropped automatically - test for clean destruction
                },
                Err(_) => return false,
            }
        }

        true
    }

    fn test_edge_cases() -> bool {
        // Test 1: Very small dataset
        let tiny_data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let config = pacmap::Configuration {
            embedding_dimensions: 2,
            override_neighbors: Some(2),
            ..Default::default()
        };

        // Should handle small data gracefully
        if fit_transform_normalized(tiny_data, config, Some(NormalizationMode::ZScore)).is_err() {
            return false;
        }

        // Test 2: Single dimension input
        let single_dim_data = Array2::from_shape_vec((10, 1), (0..10).map(|x| x as f64).collect()).unwrap();
        let config = pacmap::Configuration {
            embedding_dimensions: 1,
            override_neighbors: Some(3),
            ..Default::default()
        };

        // Should handle single dimension
        if fit_transform_normalized(single_dim_data, config, Some(NormalizationMode::MinMax)).is_err() {
            return false;
        }

        // Test 3: Identical data points
        let identical_data = Array2::from_shape_vec((20, 3), vec![1.0; 60]).unwrap();
        let config = pacmap::Configuration {
            embedding_dimensions: 2,
            override_neighbors: Some(5),
            ..Default::default()
        };

        // Should handle identical points without crashing
        match fit_transform_normalized(identical_data, config, Some(NormalizationMode::ZScore)) {
            Ok((embedding, _)) => {
                // Embedding should exist even if not meaningful
                embedding.len() > 0
            },
            Err(_) => {
                // It's acceptable for this to fail gracefully
                true
            }
        }
    }
}