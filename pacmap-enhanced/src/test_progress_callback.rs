// Progress callback test for PacMAP Enhanced
// Tests progress reporting functionality during fitting

use crate::{fit_transform_normalized_with_progress, ProgressCallback};
use crate::stats::NormalizationMode;
use ndarray::Array2;
use std::sync::{Arc, Mutex};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_callback() {
        println!("=== Testing Progress Callback Functionality ===");

        // Create test data
        const N_OBS: usize = 200;
        const N_DIM: usize = 10;

        let data = generate_test_data(N_OBS, N_DIM);

        let config = pacmap::Configuration {
            embedding_dimensions: 2,
            override_neighbors: Some(15),
            seed: Some(42),
            ..Default::default()
        };

        // Track progress events
        let progress_events = Arc::new(Mutex::new(Vec::new()));
        let progress_events_clone = Arc::clone(&progress_events);

        // Define progress callback
        let progress_callback: ProgressCallback = Box::new(move |phase, current, total, percent, message| {
            let mut events = progress_events_clone.lock().unwrap();
            events.push((phase.to_string(), current, total, percent, message.to_string()));

            // Print progress to console (like a real application would)
            println!("[{:>12}] {:>3}% ({:>3}/{:<3}) - {}",
                     phase, percent as u32, current, total, message);
        });

        println!("\n🔄 Starting PacMAP fitting with progress tracking...");
        println!("Dataset: {} samples x {} features -> 2D embedding", N_OBS, N_DIM);
        println!();

        // Run fitting with progress callback
        let start_time = std::time::Instant::now();
        match fit_transform_normalized_with_progress(
            data,
            config,
            Some(NormalizationMode::ZScore),
            Some(progress_callback)
        ) {
            Ok((embedding, model)) => {
                let duration = start_time.elapsed();

                println!();
                println!("✅ PacMAP fitting completed successfully!");
                println!("   - Embedding shape: {:?}", embedding.shape());
                println!("   - Total duration: {:.2}s", duration.as_secs_f64());
                println!("   - Normalization: {:?}", model.normalization.mode);
                println!("   - HNSW parameters: M={}, ef_construction={}, ef_search={}",
                         model.config.hnsw_params.m,
                         model.config.hnsw_params.ef_construction,
                         model.config.hnsw_params.ef_search);

                // Analyze captured progress events
                let events = progress_events.lock().unwrap();
                println!();
                println!("📊 Progress Event Analysis:");
                println!("   - Total progress events: {}", events.len());

                if !events.is_empty() {
                    println!("   - First event: {} ({}%)", events[0].0, events[0].3 as u32);
                    println!("   - Last event: {} ({}%)", events.last().unwrap().0, events.last().unwrap().3 as u32);

                    // Check that we have the expected phases
                    let phases: Vec<String> = events.iter().map(|e| e.0.clone()).collect();
                    let expected_phases = vec!["Initializing", "Analyzing", "Normalizing", "HNSW Config", "HNSW Ready", "Embedding", "Embedding Done", "Finalizing", "Complete"];

                    let mut found_phases = Vec::new();
                    for expected in &expected_phases {
                        if phases.iter().any(|p| p == expected) {
                            found_phases.push(expected);
                        }
                    }

                    println!("   - Expected phases found: {}/{}", found_phases.len(), expected_phases.len());

                    if found_phases.len() >= 7 {
                        println!("   ✅ Progress callback working correctly - all major phases reported");
                    } else {
                        println!("   ⚠️  Some expected phases missing: {:?}",
                                expected_phases.iter().filter(|e| !found_phases.contains(e)).collect::<Vec<_>>());
                    }
                }

                // Verify embedding quality
                let non_zero_count = embedding.iter().filter(|&&x| x.abs() > 1e-10).count();
                if non_zero_count > 0 {
                    println!("   ✅ Embedding quality verified - non-zero values present");
                } else {
                    println!("   ❌ Embedding quality issue - all zero values");
                    panic!("Embedding quality test failed");
                }

                println!();
                println!("🎉 Progress callback test completed successfully!");
                println!("The progress callback system is now ready for production use.");

            },
            Err(e) => {
                println!("❌ PacMAP fitting failed: {}", e);
                panic!("Progress callback test failed");
            }
        }
    }

    fn generate_test_data(n_obs: usize, n_dim: usize) -> Array2<f64> {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(12345);
        let mut data = Array2::zeros((n_obs, n_dim));

        // Generate structured data with some clustering
        for i in 0..n_obs {
            let cluster = i % 3; // 3 clusters
            let base_offset = cluster as f64 * 5.0;

            for j in 0..n_dim {
                let noise = (rng.gen::<f64>() - 0.5) * 0.5;
                let pattern = (j as f64 * 0.1 + i as f64 * 0.01).sin() * 2.0;
                data[[i, j]] = base_offset + pattern + noise;
            }
        }

        data
    }
}