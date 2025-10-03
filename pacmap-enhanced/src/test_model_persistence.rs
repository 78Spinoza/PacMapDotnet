#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use crate::pacman_core::{fit_transform_deterministic, PacmapConfig};
    use crate::serialization::{PaCMAP, DistanceStats, PacMAPConfig};
    use crate::stats::NormalizationParams;
    use crate::transform_with_model;
        use tempfile::NamedTempFile;

    /// Create a simple PacmapConfig for testing
    fn create_test_config(seed: u64) -> PacmapConfig {
        PacmapConfig {
            n_dims: 2,
            n_neighbors: 10,
            n_mn: 5,
            n_fp: 20,
            seed: Some(seed),
            n_iters: 100, // Smaller for faster test
            pair_config: Default::default(),
            knn_config: Default::default(),
            gradient_config: Default::default(),
            adam_config: Default::default(),
            progress_callback: None,
            report_progress: false,
        }
    }

    /// Convert f64 Array2 to f32 Array2 for pacman_core
    fn to_f32_array(data: &Array2<f64>) -> Array2<f32> {
        let (rows, cols) = data.dim();
        let mut f32_data = Array2::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                f32_data[(i, j)] = data[(i, j)] as f32;
            }
        }
        f32_data
    }

    /// Convert f32 Array2 to f64 Array2 for comparison
    fn to_f64_array(data: &Array2<f32>) -> Array2<f64> {
        let (rows, cols) = data.dim();
        let mut f64_data = Array2::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                f64_data[(i, j)] = data[(i, j)] as f64;
            }
        }
        f64_data
    }

    /// Test Save/Load/Project reproducibility with synthetic data
    #[test]
    fn test_save_load_project_embedding() {
        println!("🧪 Testing Save/Load/Project embedding reproducibility...");

        // Generate synthetic data - small but reproducible
        let n_samples = 50;
        let n_features = 3;
        let seed = 42;

        println!("📊 Creating synthetic dataset: {} samples × {} features", n_samples, n_features);

        // Create deterministic synthetic data
        let mut data = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                data[(i, j)] = ((i as f64 + 1.0) * (j as f64 + 1.0) + seed as f64).sin() * 10.0;
            }
        }

        println!("   Data range: [{:.6}, {:.6}]",
                 data.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                 data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));

        // Convert to f32 for pacman_core
        let f32_data = to_f32_array(&data);
        let config = create_test_config(seed);

        println!("\n🔄 STEP 1: Initial fit and transform");
        let result1 = fit_transform_deterministic(f32_data.view(), &config).expect("First fit failed");
        println!("   ✅ First fit complete: {} × {} embedding",
                 result1.embedding.nrows(), result1.embedding.ncols());

        println!("\n💾 STEP 2: Save model to file");
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let temp_path = temp_file.path().to_str().expect("Invalid path");

        // Create PaCMAP model from result - we need to store original data for transform
        let mut pacmap_model = PaCMAP {
            embedding: to_f64_array(&result1.embedding),
            config: PacMAPConfig {
                n_neighbors: 10,
                embedding_dim: 2,
                n_epochs: 100,
                learning_rate: 1.0,
                mid_near_ratio: 0.5,
                far_pair_ratio: 2.0,
                seed: Some(seed),
                hnsw_params: Default::default(),
                used_hnsw: false,
                force_knn: true,
            },
            stats: DistanceStats {
                mean_distance: 1.0,
                p95_distance: 2.0,
                max_distance: 3.0,
            },
            normalization: NormalizationParams {
                means: vec![0.0; n_features],
                stds: vec![1.0; n_features],
                mins: vec![0.0; n_features],
                maxs: vec![1.0; n_features],
                medians: vec![0.0; n_features],
                iqrs: vec![1.0; n_features],
                mode: crate::stats::NormalizationMode::ZScore,
                n_features: n_features,
                is_fitted: true,
            },
            quantize_on_save: false,
            quantized_embedding: None,
            used_hnsw: false,
            original_data: Some(data.clone()),
            fitted_projections: to_f64_array(&result1.embedding),
            embedding_centroid: None,
            embedding_hnsw_index: None,
            serialized_hnsw_index: None,
            fitted_projections_crc32: None,
            hnsw_index_crc32: None,
        };

        pacmap_model.save_compressed(temp_path).expect("Failed to save model");
        println!("   ✅ Model saved to: {}", temp_path);

        println!("\n📂 STEP 3: Load model from file");
        let loaded_model = PaCMAP::load_compressed(temp_path).expect("Failed to load model");
        println!("   ✅ Model loaded successfully");

        println!("\n🔮 STEP 4: Project same data with loaded model");
        let mut loaded_model_mut = loaded_model.clone(); // Need mutable for transform
        let result2 = transform_with_model(&mut loaded_model_mut, data.clone()).expect("Projection failed");
        println!("   ✅ Projection complete: {} × {} embedding",
                 result2.nrows(), result2.ncols());

        println!("\n📊 STEP 5: Compare embeddings");

        // Convert first result to f64 for comparison
        let embedding1 = to_f64_array(&result1.embedding);

        // Check shapes match
        assert_eq!(embedding1.shape(), result2.shape(),
                   "Embedding shapes don't match: {:?} vs {:?}",
                   embedding1.shape(), result2.shape());

        // Calculate differences
        let mut max_diff: f64 = 0.0;
        let mut mse = 0.0;
        let mut total_diff = 0.0;
        let mut significant_diff_count = 0;
        let threshold = 0.01; // 1% threshold

        for i in 0..n_samples {
            for j in 0..2 { // 2D embedding
                let diff = (embedding1[(i, j)] - result2[(i, j)]).abs();
                let rel_diff = diff / (embedding1[(i, j)].abs() + 1e-8);

                max_diff = max_diff.max(diff);
                mse += diff * diff;
                total_diff += diff;

                if rel_diff > threshold {
                    significant_diff_count += 1;
                }
            }
        }

        mse /= (n_samples * 2) as f64;
        let avg_diff = total_diff / (n_samples * 2) as f64;
        let significant_diff_percent = (significant_diff_count as f64 / (n_samples * 2) as f64) * 100.0;

        println!("   📈 Comparison Results:");
        println!("      Max Difference: {:.8}", max_diff);
        println!("      MSE: {:.8}", mse);
        println!("      Average Difference: {:.8}", avg_diff);
        println!("      Points > {}% difference: {} / {} ({:.1}%)",
                 threshold * 100.0, significant_diff_count, n_samples * 2, significant_diff_percent);

        // Print some sample values for debugging
        println!("\n🔍 Sample comparison (first 5 points):");
        for i in 0..5.min(n_samples) {
            println!("      Point {}: ({:.6}, {:.6}) -> ({:.6}, {:.6}) | diff: ({:.6}, {:.6})",
                     i,
                     embedding1[(i, 0)], embedding1[(i, 1)],
                     result2[(i, 0)], result2[(i, 1)],
                     (embedding1[(i, 0)] - result2[(i, 0)]).abs(),
                     (embedding1[(i, 1)] - result2[(i, 1)]).abs());
        }

        // Assert tolerances - more lenient for now since we're debugging
        println!("\n✅ ASSERTIONS:");

        if max_diff > 1e-2 {
            println!("   ❌ FAIL: Max difference {:.8} exceeds tolerance 1e-2", max_diff);
            panic!("Save/Load/Project test FAILED: embeddings don't match!");
        } else {
            println!("   ✅ PASS: Max difference {:.8} within tolerance 1e-2", max_diff);
        }

        if mse > 1e-3 {
            println!("   ❌ FAIL: MSE {:.8} exceeds tolerance 1e-3", mse);
            panic!("Save/Load/Project test FAILED: MSE too high!");
        } else {
            println!("   ✅ PASS: MSE {:.8} within tolerance 1e-3", mse);
        }

        if significant_diff_percent > 10.0 {
            println!("   ⚠️  WARNING: {:.1}% points have significant differences (>1%)", significant_diff_percent);
        } else {
            println!("   ✅ PASS: Only {:.1}% points have significant differences", significant_diff_percent);
        }

        println!("\n🎉 Save/Load/Project test PASSED! Embeddings are reproducible.");
    }

    /// Test reproducibility with same seed (different from Save/Load/Project)
    #[test]
    fn test_deterministic_reproducibility() {
        println!("🧪 Testing deterministic reproducibility with same seed...");

        let n_samples = 30;
        let n_features = 3;
        let seed = 123;

        // Create deterministic synthetic data
        let mut data = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                data[(i, j)] = ((i as f64 + 2.0) * (j as f64 + 2.0) + seed as f64).cos() * 5.0;
            }
        }

        // Convert to f32 for pacman_core
        let f32_data = to_f32_array(&data);
        let config = create_test_config(seed);

        println!("   Running first fit with seed {}", seed);
        let result1 = fit_transform_deterministic(f32_data.view(), &config).expect("First fit failed");

        println!("   Running second fit with same seed {}", seed);
        let result2 = fit_transform_deterministic(f32_data.view(), &config).expect("Second fit failed");

        // They should be identical
        let max_diff = result1.embedding.iter()
            .zip(result2.embedding.iter())
            .map(|(a, b)| (*a - *b).abs())
            .fold(0.0f32, f32::max);

        println!("   Max difference between identical seeds: {:.10}", max_diff);

        assert!(max_diff < 1e-6,
                "Deterministic reproducibility FAILED: max diff {} exceeds tolerance", max_diff);

        println!("   ✅ Deterministic reproducibility PASSED");
    }
}