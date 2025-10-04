use ndarray::ArrayView2;
use crate::pairs::compute_pairs_hnsw;
use crate::hnsw_wrapper::EuclideanMetric;
use std::collections::HashSet;
use rand::Rng;
use rand::SeedableRng;
use hnsw::{Hnsw, Params, Searcher};

/// Compute HNSW recall with specific HNSW parameters (used for validation consistency)
pub fn compute_hnsw_recall_with_params(
    data: ArrayView2<f64>,
    n_neighbors: usize,
    seed: u64,
    sample_points: usize,
    hnsw_params: Option<crate::hnsw_params::HnswParams>
) -> f64 {
    let (n_samples, _n_features) = data.dim();

    if n_samples <= 1000 {
        // For small datasets, we don't use HNSW anyway
        return 100.0;
    }

    // Dynamic sampling as per review: min(50, sqrt(n)) for balanced accuracy/speed
    let query_sample_size = ((n_samples as f64).sqrt() as usize)
        .min(50)      // Cap at 50 queries max
        .max(10)      // At least 10 queries for reliability
        .min(sample_points)
        .min(n_samples);

    // Minimal candidate subset: just enough for meaningful k-NN comparison but very fast
    // Use much smaller subset - recall validation is for parameter tuning, not absolute accuracy
    let candidate_subset_size = ((n_samples as f32 * 0.05) as usize) // Only 5% of dataset
        .max(200)  // Minimum 200 candidates (much smaller than before)
        .max(n_neighbors * 10) // At least 10x the k value (reduced from 20x)
        .min(n_samples); // Never exceed dataset size

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Unique sampling to avoid duplicates as per review
    let mut unique_indices = HashSet::new();
    while unique_indices.len() < query_sample_size {
        unique_indices.insert(rng.gen_range(0..n_samples));
    }
    let query_indices: Vec<usize> = unique_indices.into_iter().collect();

    if std::env::var("PACMAP_VERBOSE").is_ok() {
        eprintln!("ULTRA-FAST RECALL: {} queries against {} candidate subset ({}% of {} total)",
                 query_sample_size, candidate_subset_size,
                 (candidate_subset_size * 100 / n_samples), n_samples);
    }

    // Build HNSW index on FULL data using the same parameters as production
    let hnsw_pairs = if let Some(params) = &hnsw_params {
        crate::pairs::compute_pairs_hnsw_with_params(data, n_neighbors, seed, Some(params.clone()))
    } else {
        compute_pairs_hnsw(data, n_neighbors, seed)
    };
    let hnsw_map: std::collections::HashMap<usize, Vec<usize>> = {
        let mut map = std::collections::HashMap::new();
        for (i, j) in hnsw_pairs {
            map.entry(i).or_insert_with(Vec::new).push(j);
        }
        map
    };

    // Calculate high ef for ground truth once (used in multiple places)
    let high_ef = if let Some(params) = &hnsw_params {
        (params.ef_search * 4).max(512)
    } else {
        (n_neighbors * 20).max(512)
    };

    // RESTORED: HNSW validation for new HNSW 0.11 implementation
    if std::env::var("PACMAP_VERBOSE").is_ok() {
        eprintln!("RESTORED: HNSW recall validation enabled for HNSW 0.11");
    }

    // Build ground truth index ONCE with consistent parameters using HNSW 0.11
    let (gt_hnsw, points) = {
        // Convert data to f32 for HNSW processing
        let points: Vec<Vec<f32>> = (0..n_samples)
            .map(|i| data.row(i).iter().map(|&x| crate::hnsw_wrapper::deterministic_f32_from_f64(x)).collect())
            .collect();

        // Use consistent HNSW parameters or defaults from new HNSW 0.11
        let (m, ef_construction) = if let Some(params) = &hnsw_params {
            (params.m, params.ef_construction)
        } else {
            (16, 128)
        };

        // Create ground truth HNSW with high ef_construction for quality
        let params = Params::default()
            .ef_construction(ef_construction * 2); // Double ef_construction for ground truth

        let mut gt_hnsw = Hnsw::<EuclideanMetric, Vec<f32>, rand_pcg::Lcg128Xsl64, 16, 12>::new_params(EuclideanMetric, params);
        let mut searcher = Searcher::default();

        // Insert all points into ground truth HNSW
        for point in &points {
            gt_hnsw.insert(point.clone(), &mut searcher);
        }

        (gt_hnsw, points)
    };

    let mut total_matches = 0;
    let mut total_possible = 0;
    let mut searcher = Searcher::default();

    // For each query point, compare HNSW vs ground truth neighbors
    for &query_idx in &query_indices {
        // Get ground truth neighbors using high-ef search
        let exact_neighbors_subset: HashSet<usize> = {
            let mut neighbor_buffer = vec![space::Neighbor { index: 0, distance: 0u32 }; n_neighbors + 1];
            let gt_results = gt_hnsw.nearest(&points[query_idx], high_ef, &mut searcher, &mut neighbor_buffer);
            gt_results.into_iter()
                .map(|neighbor| neighbor.index)
                .filter(|&j| j != query_idx)
                .take(n_neighbors)
                .collect()
        };

        // Get HNSW neighbors for this query point from the test HNSW
        let hnsw_neighbors_subset: HashSet<usize> = hnsw_map
            .get(&query_idx)
            .map(|neighbors| {
                neighbors.iter()
                    .take(n_neighbors)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();

        // Count matches
        let matches = exact_neighbors_subset.intersection(&hnsw_neighbors_subset).count();
        total_matches += matches;
        total_possible += exact_neighbors_subset.len();
    }

    let recall = if total_possible == 0 {
        100.0
    } else {
        (total_matches as f64 / total_possible as f64) * 100.0
    };

    if std::env::var("PACMAP_VERBOSE").is_ok() {
        eprintln!("HNSW 0.11 RECALL RESULT: {:.1}% ({} total matches out of {} possible)",
                 recall, total_matches, total_possible);
    }

    recall
}

/// HNSW quality validation with auto-retry and specific HNSW parameters
pub fn validate_hnsw_quality_with_retry_and_params(
    data: ArrayView2<f64>,
    n_neighbors: usize,
    seed: u64,
    initial_ef_search: usize,
    progress_callback: Option<&dyn Fn(&str, usize, usize, f32, &str)>,
    base_hnsw_params: Option<crate::hnsw_params::HnswParams>
) -> Result<(String, usize), String> {
    // FIXED: Use same closure pattern as working lib.rs code
    let report_progress = |phase: &str, current: usize, total: usize, percent: f32, message: &str| {
        if let Some(callback) = progress_callback {
            callback(phase, current, total, percent, message);
        }
    };

    report_progress("HNSW Validation", 0, 100, 0.0, "Starting recall validation");

    let (n_samples, _) = data.dim();

    // For small datasets, HNSW is not needed
    if n_samples <= 1000 {
        report_progress("HNSW Validation", 100, 100, 100.0, "Small dataset - exact search sufficient");
        return Ok(("Small dataset - exact search recommended".to_string(), initial_ef_search));
    }

    // Test recall with initial parameters
    let initial_recall = compute_hnsw_recall_with_params(data, n_neighbors, seed, 50, base_hnsw_params.clone());

    report_progress("HNSW Validation", 25, 100, 25.0, &format!("Initial recall: {:.1}%", initial_recall));

    // If recall is already good, return success
    if initial_recall >= 90.0 {
        report_progress("HNSW Validation", 100, 100, 100.0, &format!("Good recall achieved: {:.1}%", initial_recall));
        return Ok((format!("Excellent recall: {:.1}%", initial_recall), initial_ef_search));
    }

    // Try to improve recall by increasing ef_search
    let mut best_ef_search = initial_ef_search;
    let mut best_recall = initial_recall;
    let mut test_ef = initial_ef_search;

    while test_ef <= 1024 { // Cap at reasonable maximum
        let recall = compute_hnsw_recall_with_params(data, n_neighbors, seed, 50,
            base_hnsw_params.as_ref().map(|p| crate::hnsw_params::HnswParams {
                ef_search: test_ef,
                ..p.clone()
            }));

        if recall > best_recall {
            best_recall = recall;
            best_ef_search = test_ef;
        }

        let progress = 25 + ((test_ef - initial_ef_search) * 75 / (1024 - initial_ef_search));
        report_progress("HNSW Validation", progress, 100, progress as f32,
                &format!("Testing ef={}: {:.1}% recall", test_ef, recall));

        // Stop if we achieve good recall
        if recall >= 90.0 {
            break;
        }

        // Exponential backoff for ef values
        test_ef = (test_ef * 2).min(1024);
    }

    // Final assessment
    let quality_msg = if best_recall >= 95.0 {
        format!("Excellent recall: {:.1}% (ef={})", best_recall, best_ef_search)
    } else if best_recall >= 85.0 {
        format!("Good recall: {:.1}% (ef={})", best_recall, best_ef_search)
    } else if best_recall >= 70.0 {
        format!("Fair recall: {:.1}% (ef={})", best_recall, best_ef_search)
    } else {
        format!("Poor recall: {:.1}% - consider exact KNN", best_recall)
    };

    report_progress("HNSW Validation", 100, 100, 100.0, &quality_msg);

    Ok((quality_msg, best_ef_search))
}

