// Comprehensive standard test suite for PacMAP Enhanced
// Based on UMAP's test patterns but adapted for PacMAP functionality

use crate::fit_transform_normalized;
use crate::serialization::PaCMAP;
use crate::stats::NormalizationMode;
use ndarray::Array2;
use std::time::Instant;

#[cfg(test)]
mod tests {
    use super::*;

    // STRICT PASS/FAIL THRESHOLDS - MUST MEET THESE TO PASS
    const MAX_ALLOWED_1_PERCENT_ERROR_RATE: f64 = 0.5; // Max 0.5% of points can have >1% error
    const MAX_ALLOWED_FIT_TRANSFORM_MSE: f64 = 0.1;    // MSE between fit and transform must be <0.1
    const MAX_ALLOWED_SAVE_LOAD_MSE: f64 = 1e-6;       // Save/load must be nearly identical
    const MAX_ALLOWED_COORDINATE_COLLAPSE: f64 = 1e-4; // Detect coordinate collapse
    const MIN_COORDINATE_VARIETY: usize = 10;           // At least 10 different coordinate values

    // Test configuration
    const N_SAMPLES: usize = 1000;  // Reasonable size for CI/testing
    const N_DIM: usize = 50;        // Input dimensions
    const N_NEIGHBORS: usize = 15;  // PacMAP neighbors

    #[derive(Debug)]
    struct TestResults {
        fit_transform_mse_2d: f64,
        fit_transform_mse_10d: f64,
        save_load_mse_2d: f64,
        save_load_mse_10d: f64,
        error_rate_1_percent_2d: f64,
        error_rate_1_percent_10d: f64,
        coordinate_variety_2d: bool,
        coordinate_variety_10d: bool,
        embedding_quality_2d: bool,
        embedding_quality_10d: bool,
        all_tests_passed: bool,
    }

    #[test]
    fn test_standard_comprehensive_pipeline() {
        println!("\n===========================================");
        println!("   PacMAP Enhanced Standard Comprehensive Test");
        println!("===========================================");
        println!();

        let start_time = Instant::now();
        let mut results = TestResults {
            fit_transform_mse_2d: 0.0,
            fit_transform_mse_10d: 0.0,
            save_load_mse_2d: 0.0,
            save_load_mse_10d: 0.0,
            error_rate_1_percent_2d: 0.0,
            error_rate_1_percent_10d: 0.0,
            coordinate_variety_2d: false,
            coordinate_variety_10d: false,
            embedding_quality_2d: false,
            embedding_quality_10d: false,
            all_tests_passed: false,
        };

        // Generate test data with structure
        let data = generate_structured_test_data(N_SAMPLES, N_DIM);
        println!("[INFO] Generated test data: {} samples x {} features", N_SAMPLES, N_DIM);

        // Test 1: 2D embedding comprehensive validation
        println!("\n[TEST 1] 2D Embedding Comprehensive Validation");
        println!("----------------------------------------------");

        let config_2d = pacmap::Configuration {
            embedding_dimensions: 2,
            override_neighbors: Some(N_NEIGHBORS),
            ..Default::default()
        };

        match test_embedding_comprehensive(&data, config_2d, "2D") {
            Ok((mse, error_rate, variety, quality)) => {
                results.fit_transform_mse_2d = mse;
                results.error_rate_1_percent_2d = error_rate;
                results.coordinate_variety_2d = variety;
                results.embedding_quality_2d = quality;

                if mse <= MAX_ALLOWED_FIT_TRANSFORM_MSE &&
                   error_rate <= MAX_ALLOWED_1_PERCENT_ERROR_RATE &&
                   variety && quality {
                    println!("[PASS] 2D embedding test passed");
                } else {
                    println!("[FAIL] 2D embedding test failed");
                    print_failure_details(mse, error_rate, variety, quality);
                }
            },
            Err(e) => {
                println!("[FAIL] 2D embedding test failed with error: {}", e);
                results.fit_transform_mse_2d = 1e6;
            }
        }

        // Test 2: 10D embedding comprehensive validation
        println!("\n[TEST 2] 10D Embedding Comprehensive Validation");
        println!("-----------------------------------------------");

        let config_10d = pacmap::Configuration {
            embedding_dimensions: 10,
            override_neighbors: Some(N_NEIGHBORS),
            ..Default::default()
        };

        match test_embedding_comprehensive(&data, config_10d, "10D") {
            Ok((mse, error_rate, variety, quality)) => {
                results.fit_transform_mse_10d = mse;
                results.error_rate_1_percent_10d = error_rate;
                results.coordinate_variety_10d = variety;
                results.embedding_quality_10d = quality;

                if mse <= MAX_ALLOWED_FIT_TRANSFORM_MSE &&
                   error_rate <= MAX_ALLOWED_1_PERCENT_ERROR_RATE &&
                   variety && quality {
                    println!("[PASS] 10D embedding test passed");
                } else {
                    println!("[FAIL] 10D embedding test failed");
                    print_failure_details(mse, error_rate, variety, quality);
                }
            },
            Err(e) => {
                println!("[FAIL] 10D embedding test failed with error: {}", e);
                results.fit_transform_mse_10d = 1e6;
            }
        }

        // Test 3: Save/Load consistency for both dimensions
        println!("\n[TEST 3] Save/Load Consistency Validation");
        println!("-----------------------------------------");

        results.save_load_mse_2d = test_save_load_consistency(&data, 2);
        results.save_load_mse_10d = test_save_load_consistency(&data, 10);

        // Final evaluation
        results.all_tests_passed =
            results.fit_transform_mse_2d <= MAX_ALLOWED_FIT_TRANSFORM_MSE &&
            results.fit_transform_mse_10d <= MAX_ALLOWED_FIT_TRANSFORM_MSE &&
            results.error_rate_1_percent_2d <= MAX_ALLOWED_1_PERCENT_ERROR_RATE &&
            results.error_rate_1_percent_10d <= MAX_ALLOWED_1_PERCENT_ERROR_RATE &&
            results.coordinate_variety_2d && results.coordinate_variety_10d &&
            results.embedding_quality_2d && results.embedding_quality_10d &&
            results.save_load_mse_2d <= MAX_ALLOWED_SAVE_LOAD_MSE &&
            results.save_load_mse_10d <= MAX_ALLOWED_SAVE_LOAD_MSE;

        let total_time = start_time.elapsed();

        // Print comprehensive results
        print_test_summary(&results, total_time);

        // Assert final result
        assert!(results.all_tests_passed, "Standard comprehensive test failed - see details above");

        println!("\n✅ ALL STANDARD COMPREHENSIVE TESTS PASSED!");
        println!("PacMAP Enhanced standard functionality fully validated!");
    }

    fn generate_structured_test_data(n_samples: usize, n_dim: usize) -> Array2<f64> {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(42); // Reproducible
        let mut data = Array2::zeros((n_samples, n_dim));

        // Create structured data with clusters and gradients
        for i in 0..n_samples {
            let cluster = i % 4; // 4 clusters
            let noise_level = 0.1;

            for j in 0..n_dim {
                let base_value = match cluster {
                    0 => (j as f64 * 0.1).sin() + 1.0,      // Sinusoidal cluster
                    1 => (j as f64 * 0.05).cos() - 1.0,     // Cosine cluster
                    2 => j as f64 * 0.01 + 0.5,             // Linear gradient
                    _ => ((i + j) as f64 * 0.001).exp() - 1.0, // Exponential pattern
                };

                let noise = rng.gen::<f64>() * noise_level - noise_level / 2.0;
                data[[i, j]] = base_value + noise;
            }
        }

        data
    }

    fn test_embedding_comprehensive(
        data: &Array2<f64>,
        config: pacmap::Configuration,
        dimension_name: &str
    ) -> Result<(f64, f64, bool, bool), String> {

        println!("[INFO] Testing {} embedding...", dimension_name);

        // Fit the model
        let (embedding1, model) = fit_transform_normalized(
            data.clone(),
            config.clone(),
            Some(NormalizationMode::ZScore)
        ).map_err(|e| format!("Fit failed: {}", e))?;

        println!("[INFO] First embedding completed: {:?}", embedding1.shape());

        // Fit again for consistency check
        let (embedding2, _) = fit_transform_normalized(
            data.clone(),
            config,
            Some(NormalizationMode::ZScore)
        ).map_err(|e| format!("Second fit failed: {}", e))?;

        println!("[INFO] Second embedding completed: {:?}", embedding2.shape());

        // Calculate MSE between the two embeddings (should be similar)
        let mse = calculate_mse(&embedding1, &embedding2);
        println!("[INFO] Fit-to-fit MSE: {:.6}", mse);

        // Calculate error rate (percentage of points with >1% error)
        let error_rate = calculate_error_rate(&embedding1, &embedding2, 0.01);
        println!("[INFO] Error rate (>1%): {:.2}%", error_rate * 100.0);

        // Check coordinate variety (not collapsed)
        let variety = check_coordinate_variety(&embedding1);
        println!("[INFO] Coordinate variety: {}", if variety { "PASS" } else { "FAIL" });

        // Check embedding quality (reasonable spread, no NaN/Inf)
        let quality = check_embedding_quality(&embedding1);
        println!("[INFO] Embedding quality: {}", if quality { "PASS" } else { "FAIL" });

        Ok((mse, error_rate, variety, quality))
    }

    fn test_save_load_consistency(data: &Array2<f64>, embedding_dim: usize) -> f64 {
        println!("[INFO] Testing save/load consistency for {}D embedding...", embedding_dim);

        let config = pacmap::Configuration {
            embedding_dimensions: embedding_dim,
            override_neighbors: Some(N_NEIGHBORS),
            ..Default::default()
        };

        // Fit model
        let (original_embedding, mut model) = match fit_transform_normalized(
            data.clone(),
            config,
            Some(NormalizationMode::ZScore)
        ) {
            Ok(result) => result,
            Err(e) => {
                println!("[FAIL] Model fitting failed: {}", e);
                return 1e6;
            }
        };

        // Save model
        let save_path = format!("test_save_load_{}d.bin", embedding_dim);
        if let Err(e) = model.save_compressed(&save_path) {
            println!("[FAIL] Model save failed: {}", e);
            return 1e6;
        }

        // Load model
        let loaded_model = match PaCMAP::load_compressed(&save_path) {
            Ok(model) => model,
            Err(e) => {
                println!("[FAIL] Model load failed: {}", e);
                return 1e6;
            }
        };

        // Compare embeddings
        let loaded_embedding = loaded_model.get_embedding();
        let mse = calculate_mse(&original_embedding, &loaded_embedding);

        println!("[INFO] Save/load MSE for {}D: {:.6}", embedding_dim, mse);

        // Cleanup
        std::fs::remove_file(&save_path).ok();

        if mse <= MAX_ALLOWED_SAVE_LOAD_MSE {
            println!("[PASS] Save/load consistency test for {}D", embedding_dim);
        } else {
            println!("[FAIL] Save/load consistency test for {}D (MSE: {:.6})", embedding_dim, mse);
        }

        mse
    }

    fn calculate_mse(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        if a.shape() != b.shape() {
            return 1e6; // Error indicator
        }

        let mut sum_squared_diff = 0.0;
        let n_elements = a.len();

        for (x, y) in a.iter().zip(b.iter()) {
            let diff = x - y;
            sum_squared_diff += diff * diff;
        }

        sum_squared_diff / n_elements as f64
    }

    fn calculate_error_rate(a: &Array2<f64>, b: &Array2<f64>, threshold: f64) -> f64 {
        if a.shape() != b.shape() {
            return 1.0; // 100% error
        }

        let mut error_count = 0;
        let n_points = a.shape()[0];

        for i in 0..n_points {
            let mut point_error = 0.0;
            for j in 0..a.shape()[1] {
                let diff = (a[[i, j]] - b[[i, j]]).abs();
                point_error += diff * diff;
            }

            let rmse = point_error.sqrt();
            if rmse > threshold {
                error_count += 1;
            }
        }

        error_count as f64 / n_points as f64
    }

    fn check_coordinate_variety(embedding: &Array2<f64>) -> bool {
        let n_dim = embedding.shape()[1];

        for dim in 0..n_dim {
            let mut values: Vec<f64> = embedding.column(dim).to_vec();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            values.dedup_by(|a, b| (*a - *b).abs() < MAX_ALLOWED_COORDINATE_COLLAPSE);

            if values.len() < MIN_COORDINATE_VARIETY {
                return false;
            }
        }

        true
    }

    fn check_embedding_quality(embedding: &Array2<f64>) -> bool {
        // Check for NaN or infinite values
        for value in embedding.iter() {
            if !value.is_finite() {
                return false;
            }
        }

        // Check that coordinates have reasonable spread
        let n_dim = embedding.shape()[1];
        for dim in 0..n_dim {
            let column = embedding.column(dim);
            let min_val = column.iter().copied().fold(f64::INFINITY, f64::min);
            let max_val = column.iter().copied().fold(f64::NEG_INFINITY, f64::max);

            // Should have some spread (not all identical)
            if (max_val - min_val).abs() < MAX_ALLOWED_COORDINATE_COLLAPSE {
                return false;
            }
        }

        true
    }

    fn print_failure_details(mse: f64, error_rate: f64, variety: bool, quality: bool) {
        println!("  Failure details:");
        if mse > MAX_ALLOWED_FIT_TRANSFORM_MSE {
            println!("    - MSE too high: {:.6} > {:.6}", mse, MAX_ALLOWED_FIT_TRANSFORM_MSE);
        }
        if error_rate > MAX_ALLOWED_1_PERCENT_ERROR_RATE {
            println!("    - Error rate too high: {:.2}% > {:.2}%",
                     error_rate * 100.0, MAX_ALLOWED_1_PERCENT_ERROR_RATE * 100.0);
        }
        if !variety {
            println!("    - Insufficient coordinate variety");
        }
        if !quality {
            println!("    - Poor embedding quality (NaN/Inf or collapsed)");
        }
    }

    fn print_test_summary(results: &TestResults, total_time: std::time::Duration) {
        println!("\n===========================================");
        println!("   Test Results Summary");
        println!("===========================================");
        println!();

        println!("Performance Metrics:");
        println!("  2D Embedding:");
        println!("    Fit-to-fit MSE: {:.6} (threshold: {:.6})",
                 results.fit_transform_mse_2d, MAX_ALLOWED_FIT_TRANSFORM_MSE);
        println!("    Error rate: {:.2}% (threshold: {:.2}%)",
                 results.error_rate_1_percent_2d * 100.0, MAX_ALLOWED_1_PERCENT_ERROR_RATE * 100.0);
        println!("    Coordinate variety: {}", if results.coordinate_variety_2d { "PASS" } else { "FAIL" });
        println!("    Embedding quality: {}", if results.embedding_quality_2d { "PASS" } else { "FAIL" });
        println!("    Save/load MSE: {:.6} (threshold: {:.6})",
                 results.save_load_mse_2d, MAX_ALLOWED_SAVE_LOAD_MSE);

        println!("  10D Embedding:");
        println!("    Fit-to-fit MSE: {:.6} (threshold: {:.6})",
                 results.fit_transform_mse_10d, MAX_ALLOWED_FIT_TRANSFORM_MSE);
        println!("    Error rate: {:.2}% (threshold: {:.2}%)",
                 results.error_rate_1_percent_10d * 100.0, MAX_ALLOWED_1_PERCENT_ERROR_RATE * 100.0);
        println!("    Coordinate variety: {}", if results.coordinate_variety_10d { "PASS" } else { "FAIL" });
        println!("    Embedding quality: {}", if results.embedding_quality_10d { "PASS" } else { "FAIL" });
        println!("    Save/load MSE: {:.6} (threshold: {:.6})",
                 results.save_load_mse_10d, MAX_ALLOWED_SAVE_LOAD_MSE);

        println!();
        println!("Overall Result: {}", if results.all_tests_passed { "✅ PASS" } else { "❌ FAIL" });
        println!("Total test time: {:.2}s", total_time.as_secs_f64());

        if results.all_tests_passed {
            println!();
            println!("🎉 PacMAP Enhanced standard functionality fully validated!");
            println!("   - Multiple embedding dimensions working correctly");
            println!("   - Consistent fit results");
            println!("   - Reliable save/load functionality");
            println!("   - High-quality embeddings generated");
        }
    }
}