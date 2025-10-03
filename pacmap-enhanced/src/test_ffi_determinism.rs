// Test FFI determinism - reproduce exact C# call pattern in Rust
use crate::{fit_transform_normalized_with_progress_and_force_knn_with_hnsw, Configuration, NormalizationMode, HnswParams};
use ndarray::Array2;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_function_determinism() {
        println!("Testing FFI determinism - calling exact C# function from Rust...");

        // Create exact same data size as C# demo (10,000 mammoth-like samples)
        let mut data_vec = Vec::new();
        for i in 0..10000 { // Same size as C# demo
            for j in 0..3 {
                data_vec.push((i * 3 + j) as f64 * 0.1 + (i as f64 * 0.05));
            }
        }
        let data = Array2::from_shape_vec((10000, 3), data_vec).unwrap();

        // Create exact same configuration as C# demo
        let config = Configuration {
            embedding_dimensions: 2,
            num_iters: (100, 100, 300), // Total = 500 epochs (matches C# demo)
            random_state: Some(42), // Fixed seed
            override_neighbors: Some(15), // Match C# demo
            learning_rate: 1.0, // Match C# demo
            mid_near_ratio: 0.5, // Match C# demo
            far_pair_ratio: 2.0, // Match C# demo
            pair_configuration: crate::PairConfiguration::default(),
        };

        // Create HNSW params like C# demo (same data size)
        let hnsw_params = HnswParams::auto_scale(10000, 3, 15);

        println!("Running first fit using exact C# function...");
        let (embedding1, _model1) = fit_transform_normalized_with_progress_and_force_knn_with_hnsw(
            data.clone(),
            config.clone(),
            Some(NormalizationMode::ZScore),
            None, // No callback
            true, // force_exact_knn = true (like C# knn mode)
            false, // use_quantization = false
            Some(hnsw_params.clone()),
            true // autodetect_hnsw_params = true
        ).unwrap();

        println!("Running second fit using exact C# function...");
        let (embedding2, _model2) = fit_transform_normalized_with_progress_and_force_knn_with_hnsw(
            data.clone(),
            config.clone(),
            Some(NormalizationMode::ZScore),
            None, // No callback
            true, // force_exact_knn = true (like C# knn mode)
            false, // use_quantization = false
            Some(hnsw_params.clone()),
            true // autodetect_hnsw_params = true
        ).unwrap();

        // Compare results
        let max_diff = embedding1.iter().zip(embedding2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        println!("Max difference: {}", max_diff);
        println!("Embedding1 shape: {:?}", embedding1.shape());
        println!("Embedding2 shape: {:?}", embedding2.shape());

        if max_diff < 1e-10 {
            println!("✅ SUCCESS: FFI function is deterministic!");
        } else {
            println!("❌ FAIL: FFI function is not deterministic (max diff: {})", max_diff);

            // Print first few values for debugging
            println!("Embedding1 first 5 values: {:?}", &embedding1.as_slice().unwrap()[..5]);
            println!("Embedding2 first 5 values: {:?}", &embedding2.as_slice().unwrap()[..5]);
        }

        assert!(max_diff < 1e-10, "FFI function should be deterministic");
    }

    #[test]
    fn test_ffi_function_hnsw_determinism() {
        println!("Testing FFI HNSW determinism...");

        // Create exact same data as C# demo (mammoth-like data)
        let mut data_vec = Vec::new();
        for i in 0..100 { // 100 samples should be enough to trigger the issue
            for j in 0..3 {
                data_vec.push((i * 3 + j) as f64 * 0.1 + (i as f64 * 0.05));
            }
        }
        let data = Array2::from_shape_vec((100, 3), data_vec).unwrap();

        // Create exact same configuration as C# demo
        let config = Configuration {
            embedding_dimensions: 2,
            num_iters: (100, 100, 300), // Larger iterations to trigger non-determinism
            random_state: Some(42), // Fixed seed
            override_neighbors: Some(15), // More neighbors
            learning_rate: 1.0,
            mid_near_ratio: 0.5, // Match C# demo
            far_pair_ratio: 2.0, // CRITICAL: Match C# demo
            pair_configuration: crate::PairConfiguration::default(),
        };

        // Create HNSW params like C# demo
        let hnsw_params = HnswParams::auto_scale(100, 3, 15);

        println!("Running first HNSW fit...");
        let (embedding1, _model1) = fit_transform_normalized_with_progress_and_force_knn_with_hnsw(
            data.clone(),
            config.clone(),
            Some(NormalizationMode::ZScore),
            None, // No callback
            false, // force_exact_knn = false (HNSW mode)
            false, // use_quantization = false
            Some(hnsw_params.clone()),
            true // autodetect_hnsw_params = true
        ).unwrap();

        println!("Running second HNSW fit...");
        let (embedding2, _model2) = fit_transform_normalized_with_progress_and_force_knn_with_hnsw(
            data.clone(),
            config.clone(),
            Some(NormalizationMode::ZScore),
            None, // No callback
            false, // force_exact_knn = false (HNSW mode)
            false, // use_quantization = false
            Some(hnsw_params.clone()),
            true // autodetect_hnsw_params = true
        ).unwrap();

        // Compare results
        let max_diff = embedding1.iter().zip(embedding2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        println!("HNSW mode max difference: {}", max_diff);

        if max_diff < 1e-10 {
            println!("✅ SUCCESS: HNSW mode is deterministic!");
        } else {
            println!("❌ FAIL: HNSW mode is not deterministic (max diff: {})", max_diff);

            // Print first few values for debugging
            println!("Embedding1 first 5 values: {:?}", &embedding1.as_slice().unwrap()[..5]);
            println!("Embedding2 first 5 values: {:?}", &embedding2.as_slice().unwrap()[..5]);
        }

        // HNSW might have some non-determinism, but it should be much smaller than what C# shows
        assert!(max_diff < 1e-6, "HNSW function should be mostly deterministic");
    }
}