use ndarray::Array2;
use pacmap::Configuration;
use std::time::Instant;
use pacmap_enhanced::fit_transform_normalized_with_progress_and_force_knn;
use pacmap_enhanced::serialization::PaCMAP;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("PacMAP Enhanced Example - Synthetic Data Test");

    // Create synthetic test data with 3 clusters (NOT real MNIST)
    let n_samples = 1000;
    let n_features = 784; // High-dimensional synthetic data
    let mut data = Array2::zeros((n_samples, n_features));

    // Generate simple test data with 3 clusters
    for i in 0..n_samples {
        let cluster = i % 3;
        let base_value = cluster as f64 * 0.3;

        for j in 0..n_features {
            data[[i, j]] = base_value + (i + j) as f64 * 0.0001;
        }
    }

    println!("Generated SYNTHETIC test data: {} samples x {} features (3 artificial clusters)", n_samples, n_features);

    // Configure PacMAP
    let config = Configuration {
        embedding_dimensions: 2,
        override_neighbors: Some(15),
        seed: Some(42),
        mid_near_ratio: 0.5,
        far_pair_ratio: 2.0,
        num_iters: (100, 100, 250), // Total 450 epochs
        ..Default::default()
    };

    // Progress callback
    let progress_callback = |phase: &str, current: usize, total: usize, percent: f32, message: &str| {
        println!("[{}] {}/{} ({:.1}%) - {}", phase, current, total, percent, message);
    };

    println!("Starting PacMAP fit_transform...");
    let start = Instant::now();

    let (embedding, mut model) = fit_transform_normalized_with_progress_and_force_knn(
        data,
        config,
        None, // Auto normalization
        Some(Box::new(progress_callback)),
        false, // Use HNSW if available
        false  // No quantization
    )?;

    let duration = start.elapsed();
    println!("Fit time: {:.2}s", duration.as_secs_f64());

    // Print statistics
    println!("Embedding shape: {:?}", embedding.shape());
    println!("Distance stats: mean={:.2}, p95={:.2}, max={:.2}",
             model.stats.mean_distance,
             model.stats.p95_distance,
             model.stats.max_distance);

    // Test model save/load
    println!("Testing model save/load...");
    let model_path = "test_pacmap_model.bin";

    model.save_compressed(model_path)?;
    println!("Model saved to: {}", model_path);

    let loaded_model = PaCMAP::load_compressed(model_path)?;
    println!("Model loaded successfully");

    // Verify embeddings match
    let embedding_diff = embedding.iter()
        .zip(loaded_model.embedding.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    println!("Max embedding difference: {:.6}", embedding_diff);

    if embedding_diff < 1e-10 {
        println!("✅ Save/load test passed!");
    } else {
        println!("⚠️ Save/load difference: {:.6}", embedding_diff);
    }

    // Clean up
    std::fs::remove_file(model_path).ok();

    println!("Example completed successfully!");
    Ok(())
}