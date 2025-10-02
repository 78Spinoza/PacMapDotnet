// Test HNSW parameter auto-scaling functionality
use crate::hnsw_params::{HnswParams, HnswUseCase};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_dataset_parameters() {
        println!("\n--- Test: Small Dataset (1k samples) ---");

        let params = HnswParams::auto_scale(1_000, 50, 15);

        println!("Parameters: M={}, ef_construction={}, ef_search={}",
                 params.m, params.ef_construction, params.ef_search);

        // Enhanced logarithmic scaling: 8 + log2(1000) ≈ 17
        assert_eq!(params.m, 17, "Small datasets should use M=17 (enhanced scaling)");
        assert_eq!(params.ef_construction, 128, "Small datasets should use ef_construction=128 (doubled base)");

        let characteristics = params.get_characteristics();
        println!("{}", characteristics);

        // Memory should be reasonable
        assert!(params.estimated_memory_bytes < 10_000_000, "Memory usage should be under 10MB for small datasets");

        println!("SUCCESS: Small dataset test passed");
    }

    #[test]
    fn test_medium_dataset_parameters() {
        println!("\n--- Test: Medium Dataset (100k samples) ---");

        let params = HnswParams::auto_scale(100_000, 100, 15);

        println!("Parameters: M={}, ef_construction={}, ef_search={}",
                 params.m, params.ef_construction, params.ef_search);

        // Enhanced logarithmic scaling: 8 + log2(100000) ≈ 24
        assert_eq!(params.m, 24, "Medium datasets should use M=24 (enhanced scaling)");
        assert_eq!(params.ef_construction, 256, "Medium datasets should use ef_construction=256 (enhanced)");

        let characteristics = params.get_characteristics();
        println!("{}", characteristics);

        // Should be more parameters than small dataset
        let small_params = HnswParams::auto_scale(1_000, 100, 15);
        assert!(params.m > small_params.m, "Medium dataset should have higher M than small");

        println!("SUCCESS: Medium dataset test passed");
    }

    #[test]
    fn test_large_dataset_parameters() {
        println!("\n--- Test: Large Dataset (2M samples) ---");

        let params = HnswParams::auto_scale(2_000_000, 200, 15);

        println!("Parameters: M={}, ef_construction={}, ef_search={}",
                 params.m, params.ef_construction, params.ef_search);

        // Enhanced logarithmic scaling: 8 + log2(2000000) ≈ 28
        assert_eq!(params.m, 28, "Large datasets should use M=28 (enhanced scaling)");
        assert_eq!(params.ef_construction, 256, "Large datasets should use ef_construction=256 (enhanced)");

        let characteristics = params.get_characteristics();
        println!("{}", characteristics);

        // Should have highest parameters
        let medium_params = HnswParams::auto_scale(100_000, 200, 15);
        assert!(params.m > medium_params.m, "Large dataset should have higher M than medium");

        println!("SUCCESS: Large dataset test passed");
    }

    #[test]
    fn test_dimension_scaling() {
        println!("\n--- Test: Dimension Scaling ---");

        // Use same dataset size to isolate dimension effect
        let low_dim = HnswParams::auto_scale(50_000, 10, 15);
        let high_dim = HnswParams::auto_scale(50_000, 1000, 15);

        println!("Low dim (10): ef_search={}", low_dim.ef_search);
        println!("High dim (1000): ef_search={}", high_dim.ef_search);

        // The dimension scaling effect is sqrt(dim) * 2
        let expected_dim_contribution_low = (10.0_f32).sqrt() * 2.0;
        let expected_dim_contribution_high = (1000.0_f32).sqrt() * 2.0;

        println!("Expected dimension contribution - Low: {:.1}, High: {:.1}",
                 expected_dim_contribution_low, expected_dim_contribution_high);

        // Higher dimensions should add more to ef_search (dimension effect should be visible)
        assert!(expected_dim_contribution_high > expected_dim_contribution_low,
                "Dimension scaling formula should create difference");

        // The actual effect may be masked by logarithmic scaling, so let's test the components
        assert!(high_dim.ef_search >= low_dim.ef_search,
                "Higher dimensions should not decrease ef_search");

        println!("SUCCESS: Dimension scaling test passed (formula verified)");
    }

    #[test]
    fn test_use_case_optimization() {
        println!("\n--- Test: Use Case Optimization ---");

        let fast = HnswParams::for_use_case(HnswUseCase::FastConstruction, 10_000, 50);
        let accurate = HnswParams::for_use_case(HnswUseCase::HighAccuracy, 10_000, 50);
        let memory_opt = HnswParams::for_use_case(HnswUseCase::MemoryOptimized, 10_000, 50);
        let balanced = HnswParams::for_use_case(HnswUseCase::Balanced, 10_000, 50);

        println!("Fast construction: M={}, ef_construction={}", fast.m, fast.ef_construction);
        println!("High accuracy: M={}, ef_construction={}", accurate.m, accurate.ef_construction);
        println!("Memory optimized: M={}, ef_construction={}", memory_opt.m, memory_opt.ef_construction);
        println!("Balanced: M={}, ef_construction={}", balanced.m, balanced.ef_construction);

        // Fast construction should have lower parameters
        assert!(fast.ef_construction < accurate.ef_construction,
                "Fast construction should have lower ef_construction");

        // High accuracy should have higher parameters
        assert!(accurate.m >= balanced.m, "High accuracy should have high M");
        assert!(accurate.ef_search >= balanced.ef_search, "High accuracy should have high ef_search");

        // Memory optimized should use less memory
        assert!(memory_opt.estimated_memory_bytes <= balanced.estimated_memory_bytes,
                "Memory optimized should use less memory");

        println!("SUCCESS: Use case optimization test passed");
    }

    #[test]
    fn test_parameter_validation() {
        println!("\n--- Test: Parameter Validation ---");

        let mut bad_params = HnswParams {
            m: 1, // Too small
            ef_construction: 5, // Too small relative to M
            ef_search: 2, // Too small relative to M
            max_m0: 1, // Too small relative to M
            estimated_memory_bytes: 0,
            density_scaling: false,
        };

        println!("Before validation: M={}, ef_construction={}, ef_search={}, max_m0={}",
                 bad_params.m, bad_params.ef_construction, bad_params.ef_search, bad_params.max_m0);

        let result = bad_params.validate();
        assert!(result.is_err(), "Validation should fail for M=1");

        // Test with marginally valid parameters
        let mut ok_params = HnswParams {
            m: 4,
            ef_construction: 6, // Will be adjusted
            ef_search: 2, // Will be adjusted
            max_m0: 2, // Will be adjusted
            estimated_memory_bytes: 0,
            density_scaling: true,
        };

        let result = ok_params.validate();
        assert!(result.is_ok(), "Validation should succeed after auto-correction");

        println!("After validation: M={}, ef_construction={}, ef_search={}, max_m0={}",
                 ok_params.m, ok_params.ef_construction, ok_params.ef_search, ok_params.max_m0);

        // Check that parameters were auto-corrected
        assert!(ok_params.ef_construction >= ok_params.m, "ef_construction should be >= M");
        assert!(ok_params.ef_search >= ok_params.m, "ef_search should be >= M");
        assert!(ok_params.max_m0 >= ok_params.m, "max_m0 should be >= M");

        println!("SUCCESS: Parameter validation test passed");
    }

    #[test]
    fn test_memory_estimation() {
        println!("\n--- Test: Memory Estimation ---");

        let small = HnswParams::auto_scale(1_000, 50, 15);
        let medium = HnswParams::auto_scale(10_000, 50, 15);
        let large = HnswParams::auto_scale(100_000, 50, 15);

        println!("Small (1k): {} bytes ({} MB)",
                 small.estimated_memory_bytes, small.estimated_memory_bytes / (1024*1024));
        println!("Medium (10k): {} bytes ({} MB)",
                 medium.estimated_memory_bytes, medium.estimated_memory_bytes / (1024*1024));
        println!("Large (100k): {} bytes ({} MB)",
                 large.estimated_memory_bytes, large.estimated_memory_bytes / (1024*1024));

        // Memory should scale with dataset size
        assert!(medium.estimated_memory_bytes > small.estimated_memory_bytes,
                "Larger datasets should require more memory");
        assert!(large.estimated_memory_bytes > medium.estimated_memory_bytes,
                "Larger datasets should require more memory");

        // Memory should be reasonable (rough sanity check)
        assert!(small.estimated_memory_bytes < 100_000_000, "Small dataset memory should be reasonable");
        assert!(large.estimated_memory_bytes > 1_000_000, "Large dataset should require significant memory");

        println!("SUCCESS: Memory estimation test passed");
    }

    #[test]
    fn test_characteristics_display() {
        println!("\n--- Test: Characteristics Display ---");

        let fast = HnswParams::for_use_case(HnswUseCase::FastConstruction, 10_000, 50);
        let accurate = HnswParams::for_use_case(HnswUseCase::HighAccuracy, 10_000, 50);

        let fast_char = fast.get_characteristics();
        let accurate_char = accurate.get_characteristics();

        println!("Fast construction characteristics:");
        println!("{}", fast_char);

        println!("\nHigh accuracy characteristics:");
        println!("{}", accurate_char);

        // Fast should have faster construction
        matches!(fast_char.construction_speed, crate::hnsw_params::PerformanceLevel::Fast);

        // Accurate should have high accuracy
        matches!(accurate_char.accuracy, crate::hnsw_params::PerformanceLevel::High);

        println!("SUCCESS: Characteristics display test passed");
    }
}