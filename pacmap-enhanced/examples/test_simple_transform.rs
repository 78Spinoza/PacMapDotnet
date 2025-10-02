use pacmap_enhanced::*;
use ndarray::{Array2};
use pacmap::Configuration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set verbose mode
    std::env::set_var("PACMAP_VERBOSE", "1");

    println!("=== Simple Transform Test (5 points only) ===");

    // Create simple test data - just 5 points
    let n_samples = 5;
    let n_features = 3;

    let mut data = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            data[[i, j]] = (i * 3 + j) as f64; // Simple deterministic data
        }
    }

    println!("Test data:");
    for i in 0..n_samples {
        println!("  Point {}: [{:.1}, {:.1}, {:.1}]",
                i, data[[i, 0]], data[[i, 1]], data[[i, 2]]);
    }
    println!();

    // Configuration
    let config = Configuration {
        embedding_dimensions: 2,
        override_neighbors: Some(3), // Small for this test
        seed: Some(42),
        ..Configuration::default()
    };

    // Fit and get model
    println!("=== FIT ===");
    let (fit_embedding, mut model) = fit_transform_normalized_with_progress_and_force_knn(
        data.clone(),
        config.clone(),
        Some(pacmap_enhanced::stats::NormalizationMode::ZScore),
        None,
        true, // force_exact_knn
        false // use_quantization
    )?;

    println!("Fit embeddings:");
    for i in 0..n_samples {
        println!("  Point {}: [{:.3}, {:.3}]",
                i, fit_embedding[[i, 0]], fit_embedding[[i, 1]]);
    }
    println!();

    // Transform the same data
    println!("=== TRANSFORM ===");
    let transform_embedding = transform_with_model(&mut model, data.clone())?;

    println!("Transform embeddings:");
    for i in 0..n_samples {
        println!("  Point {}: [{:.3}, {:.3}]",
                i, transform_embedding[[i, 0]], transform_embedding[[i, 1]]);

        let diff = (fit_embedding[[i, 0]] - transform_embedding[[i, 0]]).abs() +
                    (fit_embedding[[i, 1]] - transform_embedding[[i, 1]]).abs();
        println!("  Difference: {:.6}", diff);
    }

    Ok(())
}