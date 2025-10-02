use pacmap_enhanced::*;
use ndarray::{Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::time::Instant;
use pacmap::Configuration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== PacMAP Transform Consistency Test (Rust) ===");
    println!("Mode: Direct KNN, No Quantization");
    println!();

    // Create test data
    let n_samples = 1000;
    let n_features = 3;
    let mut rng = rand::thread_rng();

    let data = Array2::random_using((n_samples, n_features), Uniform::new(0.0, 10.0), &mut rng);

    println!("Test data: {} samples × {} features", n_samples, n_features);
    println!();

    // Configuration - use pacmap's standard configuration
    let config = Configuration {
        embedding_dimensions: 2,
        override_neighbors: Some(15),
        seed: Some(42),
        ..Configuration::default()
    };

    // === TEST 1: Fit + Transform Consistency ===
    println!("=== TEST 1: Fit + Transform Consistency ===");

    let start = Instant::now();
    let (fit_embedding, mut model) = fit_transform_normalized_with_progress_and_force_knn(
        data.clone(),
        config.clone(),
        Some(pacmap_enhanced::stats::NormalizationMode::ZScore),
        None,
        true, // force_exact_knn
        false // use_quantization
    )?;
    let fit_duration = start.elapsed();

    println!("✅ Fit completed: {} points ({:.2?})", fit_embedding.shape()[0], fit_duration);

    // Transform the same data
    let start = Instant::now();
    let transform_embedding = transform_with_model(&mut model, data.clone())?;
    let transform_duration = start.elapsed();

    println!("✅ Transform completed: {} points ({:.2?})", transform_embedding.shape()[0], transform_duration);

    // Compare results
    compare_embeddings("Fit vs Transform", &fit_embedding, &transform_embedding)?;
    println!();

    // === TEST 2: Save/Load/Transform Consistency ===
    println!("=== TEST 2: Save/Load/Transform Consistency ===");

    // Save model
    let model_bytes = bincode::serialize(&model)?;
    println!("✅ Model serialized: {} bytes", model_bytes.len());

    // Load model
    let loaded_model: pacmap_enhanced::serialization::PaCMAP = bincode::deserialize(&model_bytes)?;
    println!("✅ Model deserialized");

    // Transform with loaded model
    let start = Instant::now();
    let loaded_transform_embedding = transform_with_model(&mut loaded_model.clone(), data.clone())?;
    let loaded_transform_duration = start.elapsed();

    println!("✅ Loaded Transform completed: {} points ({:.2?})", loaded_transform_embedding.shape()[0], loaded_transform_duration);

    // Compare original vs loaded transform
    compare_embeddings("Original Transform vs Loaded Transform", &transform_embedding, &loaded_transform_embedding)?;
    println!();

    // === TEST 3: Fit Determinism ===
    println!("=== TEST 3: Fit Determinism (same seed) ===");

    let start = Instant::now();
    let (fit_embedding2, _) = fit_transform_normalized_with_progress_and_force_knn(
        data.clone(),
        config.clone(),
        Some(pacmap_enhanced::stats::NormalizationMode::ZScore),
        None,
        true, // force_exact_knn
        false // use_quantization
    )?;
    let fit2_duration = start.elapsed();

    println!("✅ Second Fit completed: {} points ({:.2?})", fit_embedding2.shape()[0], fit2_duration);

    // Compare two fits
    compare_embeddings("Fit1 vs Fit2 (same seed)", &fit_embedding, &fit_embedding2)?;

    Ok(())
}

fn compare_embeddings(
    test_name: &str,
    emb1: &Array2<f64>,
    emb2: &Array2<f64>
) -> Result<(), Box<dyn std::error::Error>> {
    if emb1.shape() != emb2.shape() {
        return Err(format!("Embedding shapes don't match: {:?} vs {:?}", emb1.shape(), emb2.shape()).into());
    }

    let n_samples = emb1.shape()[0];
    let n_dims = emb1.shape()[1];

    let mut max_diff: f64 = 0.0;
    let mut total_diff = 0.0;
    let mut diff_count = 0;
    let mut mse = 0.0;

    for i in 0..n_samples {
        for j in 0..n_dims {
            let diff = (emb1[[i, j]] - emb2[[i, j]]).abs();
            total_diff += diff;
            max_diff = max_diff.max(diff);
            mse += diff * diff;
            if diff > 1e-12 {
                diff_count += 1;
            }
        }
    }

    mse /= (n_samples * n_dims) as f64;

    println!("{} Results:", test_name);
    println!("  Max difference: {:.2e}", max_diff);
    println!("  Average difference: {:.2e}", total_diff / (n_samples * n_dims) as f64);
    println!("  MSE: {:.2e}", mse);
    println!("  Different points: {}/{} ({:.1}%)", diff_count, n_samples * n_dims,
             100.0 * diff_count as f64 / (n_samples * n_dims) as f64);

    if max_diff < 1e-10 {
        println!("✅ PASS: Results are identical");
    } else if max_diff < 1e-6 {
        println!("⚠️  WARNING: Small numerical differences (likely floating point)");
    } else {
        println!("❌ FAIL: Results are significantly different");
    }

    Ok(())
}