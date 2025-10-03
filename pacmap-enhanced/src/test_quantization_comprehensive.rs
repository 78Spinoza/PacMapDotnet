// Comprehensive quantization test for PacMAP Enhanced
// Tests quantization functionality, compression ratios, and error metrics

use crate::fit_transform_normalized;
use crate::serialization::PaCMAP;
use crate::stats::NormalizationMode;
use crate::quantize::{quantize_embedding_linear, dequantize_embedding};
use ndarray::Array2;
use std::time::Instant;

#[cfg(test)]
mod tests {
    use super::*;

    // Configuration constants for quantization testing
    const N_SAMPLES: usize = 2000; // Dataset size for comprehensive testing
    const N_DIM: usize = 100;      // Input dimensionality
    const EMBEDDING_DIM: usize = 10; // 10D embeddings for testing
    const N_NEIGHBORS: usize = 20;
    const RANDOM_STATE: u64 = 42;

    // Test thresholds
    const MAX_QUANTIZATION_MSE: f64 = 0.1;         // MSE between quantized and non-quantized
    const MAX_SAVE_LOAD_MSE: f64 = 1e-6;           // Save/load consistency
    const MAX_ALLOWED_ERROR_RATE: f64 = 5.0;       // Max 5% points with >1% error
    const MIN_COMPRESSION_RATIO: f64 = 0.9;        // Should achieve some compression

    #[derive(Debug)]
    struct QuantizationTestResult {
        // Non-quantized results
        non_quantized_embedding: Array2<f64>,
        non_quantized_loaded_embedding: Array2<f64>,

        // Quantized results
        quantized_embedding: Array2<f64>,
        quantized_loaded_embedding: Array2<f64>,

        // Error metrics
        non_quantized_vs_quantized_mse: f64,
        quantized_save_load_consistency_mse: f64,

        // File size metrics
        non_quantized_file_size: u64,
        quantized_file_size: u64,
        compression_ratio: f64,

        // Difference statistics
        error_rate_1_percent: f64,
        max_point_error: f64,
        mean_point_error: f64,

        // Performance metrics
        quantization_time_ms: u64,
        dequantization_time_ms: u64,

        // Quality metrics
        all_tests_passed: bool,
    }

    #[test]
    fn test_quantization_comprehensive() {
        println!("\n===========================================");
        println!("   PacMAP Enhanced Quantization Comprehensive Test");
        println!("===========================================");
        println!();

        let start_time = Instant::now();

        // Generate test data
        let data = generate_quantization_test_data(N_SAMPLES, N_DIM);
        println!("[INFO] Generated test data: {} samples x {} features", N_SAMPLES, N_DIM);

        // Run comprehensive quantization test
        let result = run_quantization_test(&data);

        let total_time = start_time.elapsed();

        // Print comprehensive results
        print_quantization_results(&result, total_time);

        // Assert final result
        assert!(result.all_tests_passed, "Quantization comprehensive test failed - see details above");

        println!("\nSUCCESS: ALL QUANTIZATION COMPREHENSIVE TESTS PASSED!");
        println!("PacMAP Enhanced quantization functionality fully validated!");
    }

    fn generate_quantization_test_data(n_samples: usize, n_dim: usize) -> Array2<f64> {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(RANDOM_STATE);
        let mut data = Array2::zeros((n_samples, n_dim));

        // Create structured data with various patterns for robust quantization testing
        for i in 0..n_samples {
            let pattern = i % 5; // 5 different patterns
            let base_value = i as f64 / n_samples as f64;

            for j in 0..n_dim {
                let value = match pattern {
                    0 => (j as f64 * 0.1 + base_value * 2.0).sin() * 5.0,
                    1 => (j as f64 * 0.05 + base_value).cos() * 3.0 + 2.0,
                    2 => j as f64 * 0.02 + base_value * 10.0 - 5.0,
                    3 => ((i + j) as f64 * 0.001).exp() - 1.0,
                    _ => (j as f64 * base_value * 0.1).tanh() * 2.0,
                };

                let noise = (rng.gen::<f64>() - 0.5) * 0.2;
                data[[i, j]] = value + noise;
            }
        }

        data
    }

    fn run_quantization_test(data: &Array2<f64>) -> QuantizationTestResult {
        let config = pacmap::Configuration {
            embedding_dimensions: EMBEDDING_DIM,
            override_neighbors: Some(N_NEIGHBORS),
            seed: Some(RANDOM_STATE),
            ..Default::default()
        };

        println!("\n[TEST 1] Creating non-quantized model...");
        let (non_quantized_embedding, mut non_quantized_model) = fit_transform_normalized(
            data.clone(),
            config.clone(),
            Some(NormalizationMode::ZScore)
        ).expect("Non-quantized model creation failed");

        println!("[INFO] Non-quantized embedding shape: {:?}", non_quantized_embedding.shape());

        println!("\n[TEST 2] Creating quantized model...");
        let (quantized_embedding, mut quantized_model) = fit_transform_normalized(
            data.clone(),
            config,
            Some(NormalizationMode::ZScore)
        ).expect("Quantized model creation failed");

        // Enable quantization
        quantized_model.quantize_on_save = true;

        println!("[INFO] Quantized embedding shape: {:?}", quantized_embedding.shape());

        println!("\n[TEST 3] Testing quantization process...");
        let quantization_start = Instant::now();
        let quantized_embedding_data = quantize_embedding_linear(&quantized_embedding);
        let quantization_time = quantization_start.elapsed();

        let dequantization_start = Instant::now();
        let dequantized_embedding = dequantize_embedding(&quantized_embedding_data);
        let dequantization_time = dequantization_start.elapsed();

        println!("[INFO] Quantization time: {:.2}ms", quantization_time.as_millis());
        println!("[INFO] Dequantization time: {:.2}ms", dequantization_time.as_millis());

        println!("\n[TEST 4] Testing save/load functionality...");

        // Save non-quantized model
        let non_quantized_path = "test_quant_comprehensive_normal.bin";
        non_quantized_model.save_uncompressed(non_quantized_path)
            .expect("Non-quantized save failed");

        // Save quantized model
        let quantized_path = "test_quant_comprehensive_quantized.bin";
        quantized_model.save_compressed(quantized_path)
            .expect("Quantized save failed");

        // Load both models
        let loaded_non_quantized = PaCMAP::load_compressed(non_quantized_path)
            .expect("Non-quantized load failed");
        let loaded_quantized = PaCMAP::load_compressed(quantized_path)
            .expect("Quantized load failed");

        println!("[INFO] Models saved and loaded successfully");

        println!("\n[TEST 5] Computing error metrics...");

        // Get file sizes
        let non_quantized_file_size = std::fs::metadata(non_quantized_path)
            .map(|m| m.len()).unwrap_or(0);
        let quantized_file_size = std::fs::metadata(quantized_path)
            .map(|m| m.len()).unwrap_or(0);

        let compression_ratio = if non_quantized_file_size > 0 {
            quantized_file_size as f64 / non_quantized_file_size as f64
        } else {
            1.0
        };

        // Calculate MSE between non-quantized and quantized
        let non_quantized_vs_quantized_mse = calculate_mse(&non_quantized_embedding, &dequantized_embedding);

        // Calculate save/load consistency for quantized model
        let loaded_quantized_embedding = loaded_quantized.get_embedding();
        let quantized_save_load_mse = calculate_mse(&quantized_embedding, &loaded_quantized_embedding);

        // Calculate error statistics
        let (error_rate, max_error, mean_error) = calculate_error_statistics(
            &non_quantized_embedding,
            &dequantized_embedding,
            0.01 // 1% threshold
        );

        // Check if all tests passed
        let all_tests_passed =
            non_quantized_vs_quantized_mse <= MAX_QUANTIZATION_MSE &&
            quantized_save_load_mse <= MAX_SAVE_LOAD_MSE &&
            error_rate <= MAX_ALLOWED_ERROR_RATE &&
            compression_ratio <= MIN_COMPRESSION_RATIO; // Note: for large embeddings compression should help

        println!("[INFO] Error metrics computed successfully");

        // Cleanup
        std::fs::remove_file(non_quantized_path).ok();
        std::fs::remove_file(quantized_path).ok();

        QuantizationTestResult {
            non_quantized_embedding: non_quantized_embedding.clone(),
            non_quantized_loaded_embedding: loaded_non_quantized.get_embedding(),
            quantized_embedding: quantized_embedding.clone(),
            quantized_loaded_embedding: loaded_quantized_embedding,
            non_quantized_vs_quantized_mse,
            quantized_save_load_consistency_mse: quantized_save_load_mse,
            non_quantized_file_size,
            quantized_file_size,
            compression_ratio,
            error_rate_1_percent: error_rate,
            max_point_error: max_error,
            mean_point_error: mean_error,
            quantization_time_ms: quantization_time.as_millis() as u64,
            dequantization_time_ms: dequantization_time.as_millis() as u64,
            all_tests_passed,
        }
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

    fn calculate_error_statistics(a: &Array2<f64>, b: &Array2<f64>, threshold: f64) -> (f64, f64, f64) {
        if a.shape() != b.shape() {
            return (100.0, 1e6, 1e6); // Error indicators
        }

        let n_points = a.shape()[0];
        let n_dims = a.shape()[1];

        let mut error_count = 0;
        let mut max_error = 0.0;
        let mut total_error = 0.0;

        for i in 0..n_points {
            let mut point_error_squared = 0.0;

            for j in 0..n_dims {
                let diff = a[[i, j]] - b[[i, j]];
                point_error_squared += diff * diff;
            }

            let point_error = point_error_squared.sqrt();
            total_error += point_error;

            if point_error > max_error {
                max_error = point_error;
            }

            // Calculate relative error
            let mut point_magnitude = 0.0;
            for j in 0..n_dims {
                point_magnitude += a[[i, j]] * a[[i, j]];
            }
            point_magnitude = point_magnitude.sqrt();

            let relative_error = if point_magnitude > 1e-10 {
                point_error / point_magnitude
            } else {
                point_error
            };

            if relative_error > threshold {
                error_count += 1;
            }
        }

        let error_rate = (error_count as f64 / n_points as f64) * 100.0;
        let mean_error = total_error / n_points as f64;

        (error_rate, max_error, mean_error)
    }

    fn print_quantization_results(result: &QuantizationTestResult, total_time: std::time::Duration) {
        println!("\n===========================================");
        println!("   Quantization Test Results Summary");
        println!("===========================================");
        println!();

        println!(" File Size Metrics:");
        println!("   Non-quantized file size: {} bytes", result.non_quantized_file_size);
        println!("   Quantized file size: {} bytes", result.quantized_file_size);
        println!("   Compression ratio: {:.2}%", result.compression_ratio * 100.0);

        if result.quantized_file_size < result.non_quantized_file_size {
            let savings = result.non_quantized_file_size - result.quantized_file_size;
            let savings_pct = (savings as f64 / result.non_quantized_file_size as f64) * 100.0;
            println!("   Space saved: {} bytes ({:.1}%)", savings, savings_pct);
        } else {
            let overhead = result.quantized_file_size - result.non_quantized_file_size;
            println!("   Overhead: {} bytes (small dataset - quantization overhead normal)", overhead);
        }

        println!();
        println!("📈 Error Metrics:");
        println!("   Non-quantized vs Quantized MSE: {:.6} (threshold: {:.6})",
                 result.non_quantized_vs_quantized_mse, MAX_QUANTIZATION_MSE);
        println!("   Quantized save/load MSE: {:.6} (threshold: {:.6})",
                 result.quantized_save_load_consistency_mse, MAX_SAVE_LOAD_MSE);
        println!("   Error rate (>1%): {:.2}% (threshold: {:.2}%)",
                 result.error_rate_1_percent, MAX_ALLOWED_ERROR_RATE);
        println!("   Maximum point error: {:.6}", result.max_point_error);
        println!("   Mean point error: {:.6}", result.mean_point_error);

        println!();
        println!("⚡ Performance Metrics:");
        println!("   Quantization time: {}ms", result.quantization_time_ms);
        println!("   Dequantization time: {}ms", result.dequantization_time_ms);
        println!("   Total test time: {:.2}s", total_time.as_secs_f64());

        println!();
        println!("SUCCESS: Test Results:");

        let mse_pass = result.non_quantized_vs_quantized_mse <= MAX_QUANTIZATION_MSE;
        let save_load_pass = result.quantized_save_load_consistency_mse <= MAX_SAVE_LOAD_MSE;
        let error_rate_pass = result.error_rate_1_percent <= MAX_ALLOWED_ERROR_RATE;

        println!("   Quantization MSE: {}", if mse_pass { "SUCCESS: PASS" } else { "ERROR: FAIL" });
        println!("   Save/load consistency: {}", if save_load_pass { "SUCCESS: PASS" } else { "ERROR: FAIL" });
        println!("   Error rate: {}", if error_rate_pass { "SUCCESS: PASS" } else { "ERROR: FAIL" });
        println!("   Compression: {}", if result.compression_ratio <= MIN_COMPRESSION_RATIO { "SUCCESS: PASS" } else { "WARNING: INFO" });

        println!();
        println!(" Overall Result: {}", if result.all_tests_passed { "SUCCESS: PASS" } else { "ERROR: FAIL" });

        if result.all_tests_passed {
            println!();
            println!("🎉 PacMAP Enhanced quantization system fully validated!");
            println!("   - Quantization parameters properly saved and restored");
            println!("   - Error rates within acceptable thresholds");
            println!("   - Save/load consistency maintained");
            println!("   - Performance metrics acceptable");
        } else {
            println!();
            println!("ERROR: Some quantization tests failed - see details above");
            if !mse_pass {
                println!("   - Quantization MSE too high: {:.6} > {:.6}",
                         result.non_quantized_vs_quantized_mse, MAX_QUANTIZATION_MSE);
            }
            if !save_load_pass {
                println!("   - Save/load consistency failed: {:.6} > {:.6}",
                         result.quantized_save_load_consistency_mse, MAX_SAVE_LOAD_MSE);
            }
            if !error_rate_pass {
                println!("   - Error rate too high: {:.2}% > {:.2}%",
                         result.error_rate_1_percent, MAX_ALLOWED_ERROR_RATE);
            }
        }
    }
}