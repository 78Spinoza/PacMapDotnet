// Test module for normalization functionality
use ndarray::Array2;
use crate::stats::{NormalizationParams, NormalizationMode};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zscore_normalization() {
        // Create test data: [[1, 2], [3, 4], [5, 6]]
        let mut data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let original = data.clone();

        // Initialize normalization
        let mut norm = NormalizationParams::new(2, NormalizationMode::ZScore);

        // Fit and transform
        norm.fit_transform(&mut data).unwrap();

        // Check that means are approximately zero
        for j in 0..2 {
            let mean: f64 = data.column(j).iter().sum::<f64>() / 3.0;
            assert!((mean.abs() < 1e-10), "Column {} mean should be ~0, got {}", j, mean);
        }

        // Check that standard deviations are approximately 1
        for j in 0..2 {
            let col = data.column(j);
            let mean = col.iter().sum::<f64>() / 3.0;
            let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 2.0; // n-1 for sample std
            let std = var.sqrt();
            assert!((std - 1.0).abs() < 1e-10, "Column {} std should be ~1, got {}", j, std);
        }

        // Test consistency: apply same normalization to original data
        let mut test_data = original.clone();
        norm.transform(&mut test_data).unwrap();

        // Should get same result
        for i in 0..3 {
            for j in 0..2 {
                assert!((data[[i, j]] - test_data[[i, j]]).abs() < 1e-10,
                        "Normalization inconsistency at [{}, {}]", i, j);
            }
        }

        println!("SUCCESS: Z-score normalization test passed");
    }

    #[test]
    fn test_minmax_normalization() {
        // Create test data with known min/max
        let mut data = Array2::from_shape_vec((3, 2), vec![0.0, 10.0, 5.0, 20.0, 10.0, 30.0]).unwrap();
        let original = data.clone();

        // Initialize normalization
        let mut norm = NormalizationParams::new(2, NormalizationMode::MinMax);

        // Fit and transform
        norm.fit_transform(&mut data).unwrap();

        // Check that data is in [0, 1] range
        for i in 0..3 {
            for j in 0..2 {
                assert!(data[[i, j]] >= 0.0 && data[[i, j]] <= 1.0,
                        "Value at [{}, {}] should be in [0,1], got {}", i, j, data[[i, j]]);
            }
        }

        // Check that min values are 0 and max values are 1
        for j in 0..2 {
            let col = data.column(j);
            let min = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            assert!((min - 0.0).abs() < 1e-10, "Column {} min should be 0, got {}", j, min);
            assert!((max - 1.0).abs() < 1e-10, "Column {} max should be 1, got {}", j, max);
        }

        // Test consistency
        let mut test_data = original.clone();
        norm.transform(&mut test_data).unwrap();

        for i in 0..3 {
            for j in 0..2 {
                assert!((data[[i, j]] - test_data[[i, j]]).abs() < 1e-10,
                        "MinMax normalization inconsistency at [{}, {}]", i, j);
            }
        }

        println!("SUCCESS: MinMax normalization test passed");
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let data1 = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut data2 = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let mut norm = NormalizationParams::new(3, NormalizationMode::ZScore);
        norm.fit(&data1).unwrap();

        // Should fail with dimension mismatch
        let result = norm.transform(&mut data2);
        assert!(result.is_err(), "Should fail with dimension mismatch");

        println!("SUCCESS: Dimension mismatch error test passed");
    }

    #[test]
    fn test_serialization_consistency() {
        use crate::serialization::PaCMAP;
        use crate::stats::compute_distance_stats;

        // Create test data
        let mut data = Array2::from_shape_vec((4, 3), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0
        ]).unwrap();

        // Create normalization parameters
        let mut norm = NormalizationParams::new(3, NormalizationMode::ZScore);
        norm.fit_transform(&mut data).unwrap();

        // Create mock embedding
        let embedding = Array2::from_shape_vec((4, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).unwrap();
        let (mean, p95, max) = compute_distance_stats(&embedding, 42); // Use fixed seed for testing

        // Create PaCMAP model with normalization
        let mut model = PaCMAP {
            embedding: embedding.clone(),
            config: crate::serialization::PacMAPConfig::default(),
            stats: crate::serialization::DistanceStats {
                mean_distance: mean,
                p95_distance: p95,
                max_distance: max
            },
            normalization: norm.clone(),
            quantize_on_save: false,
            quantized_embedding: None,
            original_data: None,
            fitted_projections: embedding.clone(),
            embedding_centroid: None,
            #[cfg(feature = "use_hnsw")]
            hnsw_index: None,
            #[cfg(feature = "use_hnsw")]
            embedding_hnsw_index: None,
            serialized_hnsw_index: None,
            hnsw_index_crc32: None,
            fitted_projections_crc32: None,
        };

        // Test serialization/deserialization
        let temp_file = "test_model.bin";
        model.save_compressed(temp_file).unwrap();
        let loaded_model = PaCMAP::load_compressed(temp_file).unwrap();

        // Verify normalization parameters are preserved
        assert_eq!(loaded_model.normalization.mode, norm.mode);
        assert_eq!(loaded_model.normalization.n_features, norm.n_features);
        assert_eq!(loaded_model.normalization.is_fitted, norm.is_fitted);

        for i in 0..norm.means.len() {
            assert!((loaded_model.normalization.means[i] - norm.means[i]).abs() < 1e-10);
            assert!((loaded_model.normalization.stds[i] - norm.stds[i]).abs() < 1e-10);
        }

        // Clean up
        std::fs::remove_file(temp_file).ok();

        println!("SUCCESS: Serialization consistency test passed");
    }
}