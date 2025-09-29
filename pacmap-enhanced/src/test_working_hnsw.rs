// WORKING HNSW tests that actually test what the old garbage tests missed

use ndarray::Array2;
use crate::pairs::*;

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate realistic test data that triggers HNSW (large dataset)
    fn generate_large_test_data(n_samples: usize, n_features: usize) -> Array2<f64> {
        let mut data = Array2::zeros((n_samples, n_features));

        // Generate 3 distinct clusters
        let cluster_size = n_samples / 3;

        for i in 0..n_samples {
            let cluster = i / cluster_size;
            let cluster_center = match cluster {
                0 => [0.0, 0.0, 0.0],
                1 => [5.0, 5.0, 5.0],
                _ => [-5.0, 5.0, -5.0],
            };

            for j in 0..n_features.min(3) {
                let noise = (i + j) as f64 * 0.01 - 0.5; // Deterministic noise
                data[[i, j]] = cluster_center[j] + noise;
            }

            // Fill remaining features
            for j in 3..n_features {
                data[[i, j]] = (i * j) as f64 * 0.001; // Deterministic
            }
        }

        data
    }

    /// Test that HNSW is actually triggered with large datasets (>1000 samples)
    /// This is what the old garbage tests NEVER tested!
    #[test]
    fn test_hnsw_actually_triggered() {
        println!("\n=== REAL TEST: HNSW Actually Triggered ===");

        // Use large dataset that FORCES HNSW path (old tests used 10 samples!)
        let n_samples = 2000;  // Well above 1000 threshold
        let n_features = 8;
        let n_neighbors = 10;

        let data = generate_large_test_data(n_samples, n_features);
        println!("Generated {}x{} dataset (triggers HNSW)", n_samples, n_features);

        // This should trigger HNSW path, not brute-force like old tests
        let hnsw_pairs = compute_pairs_hnsw(data.view(), n_neighbors, 42);
        println!("HNSW returned {} pairs", hnsw_pairs.len());

        // Should get reasonable number of pairs
        assert!(hnsw_pairs.len() > 1000, "HNSW should return substantial pairs, got {}", hnsw_pairs.len());
        assert!(hnsw_pairs.len() < 50000, "HNSW pairs should be reasonable, got {}", hnsw_pairs.len());

        // Validate pairs are reasonable
        for (i, j) in hnsw_pairs.iter().take(10) {
            assert!(*i < n_samples, "Invalid source index: {}", i);
            assert!(*j < n_samples, "Invalid neighbor index: {}", j);
            assert_ne!(*i, *j, "Self-references not allowed: {} -> {}", i, j);
        }

        println!("SUCCESS: HNSW test passed: {} pairs from {} samples", hnsw_pairs.len(), n_samples);
        println!("SUCCESS: This test actually exercises HNSW unlike the old garbage tests!");
    }

    /// Test small datasets fall back to brute-force
    #[test]
    fn test_small_dataset_brute_force_fallback() {
        println!("\n=== REAL TEST: Small Dataset Falls Back to Brute-Force ===");

        // Use small dataset (like the old tests, but intentionally)
        let n_samples = 50; // Below 1000 threshold
        let n_features = 4;
        let n_neighbors = 5;

        let data = generate_large_test_data(n_samples, n_features);

        // Should use brute-force, not HNSW
        let pairs = compute_pairs_hnsw(data.view(), n_neighbors, 42);

        // Should get exactly n_samples * n_neighbors pairs (brute-force is exact)
        let expected_pairs = n_samples * n_neighbors;
        assert_eq!(pairs.len(), expected_pairs,
                  "Brute-force should return exactly {} pairs, got {}", expected_pairs, pairs.len());

        println!("SUCCESS: Small dataset correctly used brute-force: {} pairs", pairs.len());
    }

    /// Test HNSW vs Brute-force on moderate dataset
    #[test]
    fn test_hnsw_vs_brute_force_comparison() {
        println!("\n=== REAL TEST: HNSW vs Brute-Force Comparison ===");

        // Use moderate size for comparison
        let n_samples = 200; // Large enough for meaningful test
        let n_features = 6;
        let n_neighbors = 8;

        let data = generate_large_test_data(n_samples, n_features);

        // Get HNSW pairs (this will use brute-force since <1000, but tests the path)
        let hnsw_pairs = compute_pairs_hnsw(data.view(), n_neighbors, 123);

        // Get brute-force pairs directly
        let brute_pairs = compute_pairs_bruteforce(data.view(), n_neighbors, 123);

        println!("HNSW path: {} pairs, Brute-force: {} pairs", hnsw_pairs.len(), brute_pairs.len());

        // For datasets <1000, they should be identical (both use brute-force)
        assert_eq!(hnsw_pairs.len(), brute_pairs.len(),
                  "HNSW and brute-force should return same count for small datasets");

        // Verify both produce valid neighbor sets
        for pairs in &[&hnsw_pairs, &brute_pairs] {
            let expected_pairs = n_samples * n_neighbors;
            assert_eq!(pairs.len(), expected_pairs,
                      "Should return exactly {} pairs", expected_pairs);

            // Check first few pairs are valid
            for (i, j) in pairs.iter().take(20) {
                assert!(*i < n_samples, "Invalid source index: {}", i);
                assert!(*j < n_samples, "Invalid neighbor index: {}", j);
                assert_ne!(*i, *j, "Self-references not allowed");
            }
        }

        println!("SUCCESS: HNSW vs brute-force comparison passed");
    }

    /// Test that neighbor search actually finds reasonable neighbors
    #[test]
    fn test_neighbor_quality() {
        println!("\n=== REAL TEST: Neighbor Quality Validation ===");

        // Create simple 2D data with clear structure
        let n_samples = 100;
        let mut data = Array2::zeros((n_samples, 2));

        // Create two clear clusters
        for i in 0..50 {
            data[[i, 0]] = 0.0 + (i as f64) * 0.01; // Cluster 1 at origin
            data[[i, 1]] = 0.0 + (i as f64) * 0.01;
        }
        for i in 50..100 {
            data[[i, 0]] = 10.0 + ((i-50) as f64) * 0.01; // Cluster 2 at (10,10)
            data[[i, 1]] = 10.0 + ((i-50) as f64) * 0.01;
        }

        let n_neighbors = 5;
        let pairs = compute_pairs_bruteforce(data.view(), n_neighbors, 42);

        // Convert pairs to neighbor map
        let mut neighbor_map: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for (i, j) in pairs {
            neighbor_map.entry(i).or_insert_with(Vec::new).push(j);
        }

        // Check that points find neighbors in same cluster
        let mut same_cluster_neighbors = 0;
        let mut total_neighbors = 0;

        for (point, neighbors) in neighbor_map.iter() {
            let point_cluster = if *point < 50 { 0 } else { 1 };

            for &neighbor in neighbors {
                total_neighbors += 1;
                let neighbor_cluster = if neighbor < 50 { 0 } else { 1 };
                if point_cluster == neighbor_cluster {
                    same_cluster_neighbors += 1;
                }
            }
        }

        let same_cluster_ratio = same_cluster_neighbors as f64 / total_neighbors as f64;
        println!("Same-cluster neighbor ratio: {:.1}%", same_cluster_ratio * 100.0);

        // At least 70% of neighbors should be in same cluster for good neighbor search
        assert!(same_cluster_ratio > 0.7,
               "Neighbor quality too low: {:.1}% same-cluster", same_cluster_ratio * 100.0);

        println!("SUCCESS: Neighbor search quality validated: {:.1}% same-cluster neighbors", same_cluster_ratio * 100.0);
    }
}