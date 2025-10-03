// Test integration determinism
use crate::{fit_transform_normalized_with_progress, Configuration, PairConfiguration};
use ndarray::Array2;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_determinism() {
        println!("Testing integration determinism...");

        // Create larger test data to reproduce the non-determinism issue
        let mut data_vec = Vec::new();
        for i in 0..100 { // 100 samples should be enough to trigger the issue
            for j in 0..3 {
                data_vec.push((i * 3 + j) as f64 * 0.1 + (i as f64 * 0.05));
            }
        }
        let data = Array2::from_shape_vec((100, 3), data_vec).unwrap();

        // Create deterministic configuration - match C# demo parameters exactly
        let config = Configuration {
            embedding_dimensions: 2,
            num_iters: (100, 100, 300), // Larger iterations to trigger non-determinism
            random_state: Some(42), // Fixed seed
            override_neighbors: Some(15), // More neighbors
            learning_rate: 1.0,
            mid_near_ratio: 0.5, // Match C# demo
            far_pair_ratio: 2.0, // CRITICAL: Match C# demo - this was 0.05 before!
            pair_configuration: PairConfiguration::default(),
        };

        println!("Running first fit...");
        let (embedding1, _) = fit_transform_normalized_with_progress(
            data.clone(), config.clone(), None, None
        ).unwrap();

        println!("Running second fit...");
        let (embedding2, _) = fit_transform_normalized_with_progress(
            data.clone(), config.clone(), None, None
        ).unwrap();

        // Compare results
        let max_diff = embedding1.iter().zip(embedding2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        println!("Max difference: {}", max_diff);
        println!("Embedding1 shape: {:?}", embedding1.shape());
        println!("Embedding2 shape: {:?}", embedding2.shape());

        if max_diff < 1e-10 {
            println!("✅ SUCCESS: Integration is deterministic!");
        } else {
            println!("❌ FAIL: Integration is not deterministic (max diff: {})", max_diff);

            // Print first few values for debugging
            println!("Embedding1 first 5 values: {:?}", &embedding1.as_slice().unwrap()[..5]);
            println!("Embedding2 first 5 values: {:?}", &embedding2.as_slice().unwrap()[..5]);
        }

        assert!(max_diff < 1e-10, "Integration should be deterministic");
    }
}