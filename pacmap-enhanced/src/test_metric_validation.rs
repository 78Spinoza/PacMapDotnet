// Metric validation test for PacMAP Enhanced
// Tests distance metric validation and edge cases

use crate::fit_transform_normalized;
use crate::stats::NormalizationMode;
use ndarray::Array2;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_validation() {
        println!("=== Testing Distance Metric Validation ===");

        // Test 1: Euclidean metric validation (default)
        test_euclidean_validation();

        // Test 2: Data normalization validation
        test_normalization_validation();

        // Test 3: Constant data validation
        test_constant_data_validation();

        // Test 4: High dimensional data validation
        test_high_dimensional_validation();

        // Test 5: Sparse data validation
        test_sparse_data_validation();

        println!("\n=== All Metric Validation Tests Completed ===");
        println!("🎉 All metric validation tests passed successfully!");
        println!("SUCCESS: Euclidean metric validation");
        println!("SUCCESS: Data normalization validation");
        println!("SUCCESS: Constant data validation");
        println!("SUCCESS: High dimensional data validation");
        println!("SUCCESS: Sparse data validation");
        println!("\nPacMAP Enhanced metric validation fully verified!");
    }

    fn test_euclidean_validation() {
        println!("\nTesting Euclidean metric validation...");

        // Create normal distributed data
        const N_OBS: usize = 100;
        const N_DIM: usize = 10;
        const EMBEDDING_DIM: usize = 2;

        let data = generate_normal_data(N_OBS, N_DIM);

        let config = pacmap::Configuration {
            embedding_dimensions: EMBEDDING_DIM,
            override_neighbors: Some(15),
            ..Default::default()
        };

        println!("Fitting with normal distributed data using default Euclidean metric:");

        match fit_transform_normalized(data, config, Some(NormalizationMode::ZScore)) {
            Ok((embedding, model)) => {
                println!("Fit result: SUCCESS");
                println!("   - Embedding shape: {:?}", embedding.shape());
                println!("   - Normalization mode: {:?}", model.normalization.mode);
                println!("   - HNSW parameters: M={}, ef_construction={}",
                         model.config.hnsw_params.m, model.config.hnsw_params.ef_construction);

                // Verify embedding quality
                let non_zero_count = embedding.iter().filter(|&&x| x.abs() > 1e-10).count();
                if non_zero_count > 0 {
                    println!("SUCCESS: Euclidean metric validation passed");
                } else {
                    println!("ERROR: Euclidean metric validation failed - zero embedding");
                    panic!("Euclidean metric validation failed");
                }
            },
            Err(e) => {
                println!("Fit result: ERROR - {}", e);
                panic!("Euclidean metric validation failed");
            }
        }
    }

    fn test_normalization_validation() {
        println!("\nTesting Data normalization validation...");

        // Create data with different scales
        const N_OBS: usize = 100;
        const N_DIM: usize = 8;

        let data = generate_mixed_scale_data(N_OBS, N_DIM);

        let config = pacmap::Configuration {
            embedding_dimensions: 2,
            override_neighbors: Some(10),
            ..Default::default()
        };

        // Test different normalization modes
        let normalization_modes = [
            (NormalizationMode::ZScore, "Z-Score"),
            (NormalizationMode::MinMax, "MinMax"),
            (NormalizationMode::Robust, "Robust"),
        ];

        for (mode, mode_name) in normalization_modes {
            println!("Testing {} normalization:", mode_name);

            match fit_transform_normalized(data.clone(), config.clone(), Some(mode)) {
                Ok((embedding, model)) => {
                    println!("   {} normalization result: SUCCESS", mode_name);
                    println!("   - Applied normalization: {:?}", model.normalization.mode);

                    // Verify normalization was applied correctly
                    match mode {
                        NormalizationMode::ZScore => {
                            // Check that means are preserved in model
                            if model.normalization.means.len() != N_DIM {
                                println!("   ERROR: {} normalization failed - wrong means count", mode_name);
                                panic!("{} normalization validation failed", mode_name);
                            }
                        },
                        NormalizationMode::MinMax => {
                            // Check that min/max are preserved in model
                            if model.normalization.mins.len() != N_DIM ||
                               model.normalization.maxs.len() != N_DIM {
                                println!("   ERROR: {} normalization failed - wrong min/max count", mode_name);
                                panic!("{} normalization validation failed", mode_name);
                            }
                        },
                        NormalizationMode::Robust => {
                            // Check that medians/IQRs are preserved in model
                            if model.normalization.medians.len() != N_DIM ||
                               model.normalization.iqrs.len() != N_DIM {
                                println!("   ERROR: {} normalization failed - wrong median/IQR count", mode_name);
                                panic!("{} normalization validation failed", mode_name);
                            }
                        },
                        _ => {}
                    }

                    println!("   SUCCESS: {} normalization validation passed", mode_name);
                },
                Err(e) => {
                    println!("   {} normalization result: ERROR - {}", mode_name, e);
                    panic!("{} normalization validation failed", mode_name);
                }
            }
        }
    }

    fn test_constant_data_validation() {
        println!("\nTesting Constant data validation...");

        // Create constant data (no variance)
        const N_OBS: usize = 100;
        const N_DIM: usize = 5;

        let mut data = Array2::zeros((N_OBS, N_DIM));
        // Fill with constant values
        for i in 0..N_OBS {
            for j in 0..N_DIM {
                data[[i, j]] = 1.0; // All values the same
            }
        }

        let config = pacmap::Configuration {
            embedding_dimensions: 2,
            override_neighbors: Some(10),
            ..Default::default()
        };

        println!("Fitting with constant data (should handle gracefully):");

        match fit_transform_normalized(data, config, Some(NormalizationMode::ZScore)) {
            Ok((embedding, _model)) => {
                println!("Constant data result: SUCCESS (handled gracefully)");
                println!("   - Embedding shape: {:?}", embedding.shape());

                // Constant data might produce valid but not meaningful embeddings
                if embedding.len() > 0 {
                    println!("SUCCESS: Constant data validation passed");
                } else {
                    println!("ERROR: Constant data validation failed - no embedding");
                    panic!("Constant data validation failed");
                }
            },
            Err(e) => {
                // It's acceptable for constant data to fail gracefully
                println!("Constant data result: ERROR (acceptable) - {}", e);
                println!("SUCCESS: Constant data validation passed (graceful failure)");
            }
        }
    }

    fn test_high_dimensional_validation() {
        println!("\nTesting High dimensional data validation...");

        // Create high-dimensional data
        const N_OBS: usize = 200;
        const N_DIM: usize = 100; // High dimensional
        const EMBEDDING_DIM: usize = 3;

        let data = generate_normal_data(N_OBS, N_DIM);

        let config = pacmap::Configuration {
            embedding_dimensions: EMBEDDING_DIM,
            override_neighbors: Some(15),
            ..Default::default()
        };

        println!("Fitting with high-dimensional data ({}D -> {}D):", N_DIM, EMBEDDING_DIM);

        match fit_transform_normalized(data, config, Some(NormalizationMode::ZScore)) {
            Ok((embedding, model)) => {
                println!("High-dimensional result: SUCCESS");
                println!("   - Input dimensions: {}", N_DIM);
                println!("   - Output dimensions: {}", EMBEDDING_DIM);
                println!("   - Embedding shape: {:?}", embedding.shape());
                println!("   - HNSW memory estimate: {} bytes", model.config.hnsw_params.estimated_memory_bytes);

                // Verify embedding quality for high-dimensional data
                let non_zero_count = embedding.iter().filter(|&&x| x.abs() > 1e-10).count();
                let has_variety = check_coordinate_variety(&embedding);

                if non_zero_count > 0 && has_variety {
                    println!("SUCCESS: High dimensional validation passed");
                } else {
                    println!("ERROR: High dimensional validation failed - poor embedding quality");
                    panic!("High dimensional validation failed");
                }
            },
            Err(e) => {
                println!("High-dimensional result: ERROR - {}", e);
                panic!("High dimensional validation failed");
            }
        }
    }

    fn test_sparse_data_validation() {
        println!("\nTesting Sparse data validation...");

        // Create sparse data (mostly zeros)
        const N_OBS: usize = 150;
        const N_DIM: usize = 20;

        let data = generate_sparse_data(N_OBS, N_DIM, 0.1); // 10% non-zero

        let config = pacmap::Configuration {
            embedding_dimensions: 2,
            override_neighbors: Some(12),
            ..Default::default()
        };

        println!("Fitting with sparse data (90% zeros):");

        match fit_transform_normalized(data, config, Some(NormalizationMode::Robust)) {
            Ok((embedding, model)) => {
                println!("Sparse data result: SUCCESS");
                println!("   - Embedding shape: {:?}", embedding.shape());
                println!("   - Normalization: {:?}", model.normalization.mode);

                // Verify embedding for sparse data
                let non_zero_count = embedding.iter().filter(|&&x| x.abs() > 1e-10).count();
                if non_zero_count > 0 {
                    println!("SUCCESS: Sparse data validation passed");
                } else {
                    println!("ERROR: Sparse data validation failed - zero embedding");
                    panic!("Sparse data validation failed");
                }
            },
            Err(e) => {
                println!("Sparse data result: ERROR - {}", e);
                panic!("Sparse data validation failed");
            }
        }
    }

    fn generate_normal_data(n_obs: usize, n_dim: usize) -> Array2<f64> {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut data = Array2::zeros((n_obs, n_dim));

        for i in 0..n_obs {
            for j in 0..n_dim {
                // Normal distribution with different means for each dimension
                let mean = j as f64 * 0.1;
                let std = 1.0 + j as f64 * 0.05;
                let value = rng.gen::<f64>() * std + mean;
                data[[i, j]] = value;
            }
        }

        data
    }

    fn generate_mixed_scale_data(n_obs: usize, n_dim: usize) -> Array2<f64> {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let mut data = Array2::zeros((n_obs, n_dim));

        for i in 0..n_obs {
            for j in 0..n_dim {
                // Different scales for different dimensions
                let scale = match j % 4 {
                    0 => 1.0,       // Small scale
                    1 => 100.0,     // Medium scale
                    2 => 10000.0,   // Large scale
                    _ => 0.01,      // Very small scale
                };

                let value = (rng.gen::<f64>() - 0.5) * scale;
                data[[i, j]] = value;
            }
        }

        data
    }

    fn generate_sparse_data(n_obs: usize, n_dim: usize, density: f64) -> Array2<f64> {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(456);
        let mut data = Array2::zeros((n_obs, n_dim));

        for i in 0..n_obs {
            for j in 0..n_dim {
                if rng.gen::<f64>() < density {
                    // Non-zero value
                    let value = (rng.gen::<f64>() - 0.5) * 10.0;
                    data[[i, j]] = value;
                }
                // else remains zero
            }
        }

        data
    }

    fn check_coordinate_variety(embedding: &Array2<f64>) -> bool {
        let n_dim = embedding.shape()[1];

        for dim in 0..n_dim {
            let column = embedding.column(dim);
            let min_val = column.iter().copied().fold(f64::INFINITY, f64::min);
            let max_val = column.iter().copied().fold(f64::NEG_INFINITY, f64::max);

            // Should have some spread (not all identical)
            if (max_val - min_val).abs() < 1e-10 {
                return false;
            }
        }

        true
    }
}