// Test integration determinism
use ndarray::array;
use pacmap_enhanced::{fit_transform_normalized_with_progress, Configuration};

fn main() {
    println!("Testing integration determinism...");

    // Create simple test data
    let data = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0]
    ];

    // Create deterministic configuration
    let config = Configuration {
        embedding_dimensions: 2,
        num_iters: (50, 50, 100), // Small for testing
        random_state: Some(42), // Fixed seed
        override_neighbors: Some(2),
        learning_rate: 1.0,
        mid_near_ratio: 0.1,
        far_pair_ratio: 0.05,
        pair_configuration: pacmap_enhanced::PairConfiguration::default(),
    };

    println!("Running first fit...");
    let (embedding1, _) = fit_transform_normalized_with_progress(
        data.clone(), config.clone(), None
    ).unwrap();

    println!("Running second fit...");
    let (embedding2, _) = fit_transform_normalized_with_progress(
        data.clone(), config.clone(), None
    ).unwrap();

    // Compare results
    let max_diff = embedding1.iter().zip(embedding2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    println!("Max difference: {}", max_diff);

    if max_diff < 1e-10 {
        println!("✅ SUCCESS: Integration is deterministic!");
    } else {
        println!("❌ FAIL: Integration is not deterministic (max diff: {})", max_diff);

        // Print first few values for debugging
        println!("Embedding1 first 5 values: {:?}", &embedding1.as_slice().unwrap()[..5]);
        println!("Embedding2 first 5 values: {:?}", &embedding2.as_slice().unwrap()[..5]);
    }
}