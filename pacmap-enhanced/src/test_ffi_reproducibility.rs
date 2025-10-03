// Test to reproduce Save/Load/Project reproducibility issue using FFI functions
use crate::ffi::*;
use std::ffi::CString;
use tempfile::NamedTempFile;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_save_load_project_reproducibility() {
        println!("\n=== FFI Save/Load/Project Reproducibility Test ===");

        // Create test data (20 samples, 3 features)
        let data: Vec<f64> = (0..60).map(|i| (i as f64) * 0.1).collect();
        let rows = 20;
        let cols = 3;

        // Create config with KNN forced (no HNSW)
        let mut config = pacmap_config_default();
        config.force_exact_knn = true;
        config.seed = 42; // Deterministic

        let mut original_embedding = vec![0.0; rows * config.embedding_dimensions as usize];
        let mut loaded_embedding = vec![0.0; rows * config.embedding_dimensions as usize];

        println!("Step 1: Original fit_transform with KNN forced");

        // Step 1: Original fit_transform
        let original_handle = pacmap_fit_transform_enhanced(
            data.as_ptr(),
            rows as std::os::raw::c_int,
            cols as std::os::raw::c_int,
            config,
            original_embedding.as_mut_ptr(),
            (rows * config.embedding_dimensions as usize) as std::os::raw::c_int,
            None, // No callback for simplicity
        );

        assert!(!original_handle.is_null(), "Original fit_transform should succeed");

        println!("Step 2: Save model");

        // Step 2: Save model
        let temp_file = NamedTempFile::new().unwrap();
        let save_path = CString::new(temp_file.path().to_str().unwrap()).unwrap();
        let save_result = pacmap_save_model_enhanced(original_handle, save_path.as_ptr(), false);
        assert_eq!(save_result, 0, "Model save should succeed");

        println!("Step 3: Load model");

        // Step 3: Load model
        let loaded_handle = pacmap_load_model_enhanced(save_path.as_ptr());
        assert!(!loaded_handle.is_null(), "Model load should succeed");

        println!("Step 4: Transform using loaded model (same data)");

        // Step 4: Transform same data using loaded model
        let transform_result = pacmap_transform(
            loaded_handle,
            data.as_ptr(),
            rows as std::os::raw::c_int,
            cols as std::os::raw::c_int,
            loaded_embedding.as_mut_ptr(),
            (rows * config.embedding_dimensions as usize) as std::os::raw::c_int,
            None, // No callback for simplicity
        );

        assert_eq!(transform_result, 0, "Transform should succeed");

        println!("Step 5: Compare embeddings");

        // Step 5: Compare embeddings
        let mut max_diff = 0.0;
        let mut total_diff = 0.0;
        let mut significant_diff_count = 0;

        for i in 0..original_embedding.len() {
            let diff = (original_embedding[i] - loaded_embedding[i]).abs();
            total_diff += diff;
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > 0.01 { // Significant difference threshold
                significant_diff_count += 1;
            }
        }

        let avg_diff = total_diff / original_embedding.len() as f64;

        println!("Comparison Results:");
        println!("  Original embedding samples:");
        for i in 0..5.min(original_embedding.len() / 2) {
            println!("    Point {}: ({:.6}, {:.6})", i, original_embedding[i*2], original_embedding[i*2+1]);
        }

        println!("  Loaded embedding samples:");
        for i in 0..5.min(loaded_embedding.len() / 2) {
            println!("    Point {}: ({:.6}, {:.6})", i, loaded_embedding[i*2], loaded_embedding[i*2+1]);
        }

        println!("  Max difference: {:.10}", max_diff);
        println!("  Average difference: {:.10}", avg_diff);
        println!("  Significant differences (>0.01): {}/{}", significant_diff_count, original_embedding.len());

        // Print specific differences for first few points
        println!("  Detailed differences for first 5 points:");
        for i in 0..5.min(original_embedding.len() / 2) {
            let diff_x = (original_embedding[i*2] - loaded_embedding[i*2]).abs();
            let diff_y = (original_embedding[i*2+1] - loaded_embedding[i*2+1]).abs();
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
        pacmap_free_model_enhanced(original_handle);
        pacmap_free_model_enhanced(loaded_handle);

        // Assert that embeddings should be very similar
        assert!(
            max_diff < TOLERANCE,
            "Save/Load/Project reproducibility failed: max difference {} exceeds tolerance {}",
            max_diff, TOLERANCE
        );
    }
}