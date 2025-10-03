// Simple test to check Save/Load/Project reproducibility without FFI complications
use crate::{fit_transform_normalized_with_progress_and_force_knn, transform_with_model};
use crate::serialization::PaCMAP;
use crate::stats::NormalizationMode;
use ndarray::Array2;
use std::fs;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_reproducibility() {
        println!("\n=== Simple Save/Load/Project Reproducibility Test ===");

        // Create simple test data (10 samples, 3 features)
        let mut data_vec = Vec::new();
        for i in 0..10 {
            for j in 0..3 {
                data_vec.push((i * 3 + j) as f64 * 0.1);
            }
        }
        let data = Array2::from_shape_vec((10, 3), data_vec).unwrap();

        // Create config with KNN forced
        let mut config = pacmap::Configuration::default();
        config.embedding_dimensions = 2;
        config.override_neighbors = Some(5); // Small for speed
        config.seed = Some(42); // Deterministic

        println!("Step 1: Original fit_transform with KNN forced");

        // Step 1: Original fit_transform
        let (original_embedding, mut model) = fit_transform_normalized_with_progress_and_force_knn(
            data.clone(),
            config.clone(),
            Some(NormalizationMode::ZScore),
            None, // No callback
            true, // force_exact_knn
            false, // no quantization
        ).unwrap();

        println!("Original embedding shape: {:?}", original_embedding.shape());
        println!("Original embedding (first 3 points):");
        for i in 0..3.min(original_embedding.shape()[0]) {
            println!("  Point {}: ({:.6}, {:.6})",
                i, original_embedding[[i, 0]], original_embedding[[i, 1]]);
        }

        println!("Step 2: Save model");

        // Step 2: Save model
        let temp_path = "test_temp_model.bin";
        model.save_compressed(temp_path).unwrap();

        println!("Step 3: Load model");

        // Step 3: Load model
        let mut loaded_model = PaCMAP::load_compressed(temp_path).unwrap();

        println!("Step 4: Transform using loaded model (same data)");

        // Step 4: Transform same data using loaded model
        let loaded_embedding = transform_with_model(&mut loaded_model, data.clone()).unwrap();

        println!("Loaded embedding shape: {:?}", loaded_embedding.shape());
        println!("Loaded embedding (first 3 points):");
        for i in 0..3.min(loaded_embedding.shape()[0]) {
            println!("  Point {}: ({:.6}, {:.6})",
                i, loaded_embedding[[i, 0]], loaded_embedding[[i, 1]]);
        }

        println!("Step 5: Compare embeddings");

        // Step 5: Compare embeddings
        assert_eq!(original_embedding.shape(), loaded_embedding.shape(),
            "Embedding shapes should match");

        let mut max_diff = 0.0;
        let mut total_diff = 0.0;
        let mut significant_diff_count = 0;

        for i in 0..original_embedding.shape()[0] {
            for j in 0..original_embedding.shape()[1] {
                let diff = (original_embedding[[i, j]] - loaded_embedding[[i, j]]).abs();
                total_diff += diff;
                if diff > max_diff {
                    max_diff = diff;
                }
                if diff > 0.01 { // Significant difference threshold
                    significant_diff_count += 1;
                }
            }
        }

        let total_elements = original_embedding.shape()[0] * original_embedding.shape()[1];
        let avg_diff = total_diff / total_elements as f64;

        println!("Comparison Results:");
        println!("  Max difference: {:.10}", max_diff);
        println!("  Average difference: {:.10}", avg_diff);
        println!("  Significant differences (>0.01): {}/{}", significant_diff_count, total_elements);

        // Print detailed differences for first few points
        println!("  Detailed differences for first 3 points:");
        for i in 0..3.min(original_embedding.shape()[0]) {
            let diff_x = (original_embedding[[i, 0]] - loaded_embedding[[i, 0]]).abs();
            let diff_y = (original_embedding[[i, 1]] - loaded_embedding[[i, 1]]).abs();
            println!("    Point {}: X diff {:.8}, Y diff {:.8}", i, diff_x, diff_y);
        }

        // Check if embeddings are approximately equal
        const TOLERANCE: f64 = 0.01; // 1% tolerance
        if max_diff < TOLERANCE {
            println!("✅ SUCCESS: Embeddings are reproducible (max diff: {:.10} < {:.10})", max_diff, TOLERANCE);
        } else {
            println!("❌ FAILURE: Embeddings are NOT reproducible (max diff: {:.10} >= {:.10})", max_diff, TOLERANCE);
            println!("This indicates a fundamental issue with the Save/Load/Project pipeline.");
        }

        // Cleanup
        fs::remove_file(temp_path).ok();

        // Assert that embeddings should be very similar
        assert!(
            max_diff < TOLERANCE,
            "Save/Load/Project reproducibility failed: max difference {} exceeds tolerance {}",
            max_diff, TOLERANCE
        );
    }
}