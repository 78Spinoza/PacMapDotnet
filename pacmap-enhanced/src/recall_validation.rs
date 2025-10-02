use ndarray::ArrayView2;
use crate::pairs::compute_pairs_hnsw;
use std::collections::HashSet;
use rand::Rng;
use rand::SeedableRng;

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

    // Build ground truth index ONCE with consistent parameters (major optimization)
    #[cfg(feature = "use_hnsw")]
    let (gt_hnsw, points) = {
        use hnsw_rs::hnsw::Hnsw;
        use hnsw_rs::dist::DistL2;

        let points: Vec<Vec<f32>> = (0..n_samples)
            .map(|i| data.row(i).iter().map(|&x| x as f32).collect())
            .collect();

        // Use consistent HNSW parameters or defaults
        let (m, ef_construction) = if let Some(params) = &hnsw_params {
            (params.m, params.ef_construction)
        } else {
            (16, 128)
        };

        let max_layer = ((n_samples as f32).ln() / (m as f32).ln()).ceil() as usize + 1;
        let max_layer = max_layer.min(32).max(4);

        let gt_hnsw = Hnsw::<f32, DistL2>::new(m, n_samples, max_layer, ef_construction, DistL2{});
        let data_with_id: Vec<(&[f32], usize)> = points.iter().enumerate()
            .map(|(i, p)| (p.as_slice(), i)).collect();

        for (point, i) in data_with_id {
            gt_hnsw.insert((&point.to_vec(), i));
        }

        (gt_hnsw, points)
    };

    let mut total_matches = 0;
    let mut total_possible = 0;

    // For each query point, compare HNSW vs ground truth neighbors
    for &query_idx in &query_indices {
        let _query = data.row(query_idx);

        // OPTIMIZATION: Use pre-built high-ef HNSW as "ground truth" (much faster)
        #[cfg(feature = "use_hnsw")]
        let exact_neighbors_subset: HashSet<usize> = {
            // Query with very high ef for near-exact results (use high_ef calculated above)
            let gt_results = gt_hnsw.search(&points[query_idx], n_neighbors + 1, high_ef);
            gt_results.into_iter()
                .map(|neighbor| neighbor.d_id as usize)
                .filter(|&j| j != query_idx)
                .take(n_neighbors)
                .collect()
        };

        #[cfg(not(feature = "use_hnsw"))]
        let exact_neighbors_subset: HashSet<usize> = {
            // Fallback to subset-based exact computation for non-HNSW builds
            let candidate_indices: Vec<usize> = (0..candidate_subset_size)
                .map(|_| rng.gen_range(0..n_samples))
                .filter(|&i| i != query_idx)
                .collect();

            let mut distances: Vec<(usize, f64)> = Vec::with_capacity(candidate_indices.len());
            for &i in &candidate_indices {
                let target = data.row(i);
                let dist: f64 = query.iter().zip(target.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                distances.push((i, dist));
            }

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.iter()
                .take(n_neighbors.min(distances.len()))
                .map(|(idx, _)| *idx)
                .collect()
        };

        // Get HNSW neighbors for this query point (no filtering needed with HNSW ground truth)
        let hnsw_neighbors_subset: HashSet<usize> = hnsw_map
            .get(&query_idx)
            .map(|neighbors| {
                neighbors.iter()
                    .take(n_neighbors) // Take only the requested number of neighbors
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();

        // Count matches within the candidate subset
        let matches = exact_neighbors_subset.intersection(&hnsw_neighbors_subset).count();
        total_matches += matches;
        total_possible += exact_neighbors_subset.len(); // Use actual found neighbors, not k
    }

    let recall = if total_possible == 0 {
        100.0
    } else {
        (total_matches as f64 / total_possible as f64) * 100.0
    };

    if std::env::var("PACMAP_VERBOSE").is_ok() {
        eprintln!("RECALL RESULT: {:.1}% ({} total matches out of {} possible)",
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
    let (n_samples, _) = data.dim();

    // Large datasets are MORE important to validate, not less!
    // They benefit most from parameter optimization and are most likely to have recall issues

    let mut current_ef_search = initial_ef_search;
    let max_retries = 3;

    for retry in 0..=max_retries {
        // Report progress for each retry attempt
        if let Some(callback) = progress_callback {
            let progress_percent = 35.0 + (retry as f32 / max_retries as f32) * 10.0; // 35-45% range
            callback("HNSW Validation", 35 + retry * 3, 100, progress_percent,
                    &format!("Testing HNSW recall (ef_search={}, attempt {})", current_ef_search, retry + 1));
        }

        // Create HNSW parameters with current ef_search being tested
        let test_hnsw_params = if let Some(mut params) = base_hnsw_params.clone() {
            params.ef_search = current_ef_search; // Use the ef_search being tested
            Some(params)
        } else {
            // Create auto-scaled params but with specific ef_search
            let mut auto_params = crate::hnsw_params::HnswParams::auto_scale(n_samples, data.ncols(), n_neighbors);
            auto_params.ef_search = current_ef_search; // Override with tested value
            Some(auto_params)
        };

        let recall = compute_hnsw_recall_with_params(data, n_neighbors, seed, 50, test_hnsw_params);

        if std::env::var("PACMAP_VERBOSE").is_ok() {
            eprintln!("RECALL TEST: ef_search={}, recall={:.1}% (attempt {})",
                     current_ef_search, recall, retry + 1);
        }

        if recall >= 95.0 {
            if let Some(callback) = progress_callback {
                callback("HNSW Quality", 45, 100, 45.0, &format!("Excellent recall: {:.1}% (ef_search={})", recall, current_ef_search));
            }
            return Ok((format!("Excellent recall: {:.1}% (ef_search={})", recall, current_ef_search), current_ef_search));
        } else if recall >= 90.0 {
            if let Some(callback) = progress_callback {
                callback("HNSW Quality", 45, 100, 45.0, &format!("Good recall: {:.1}% (ef_search={})", recall, current_ef_search));
            }
            return Ok((format!("Good recall: {:.1}% (ef_search={})", recall, current_ef_search), current_ef_search));
        } else if retry < max_retries {
            // Auto-retry with doubled ef_search
            let old_ef_search = current_ef_search;
            current_ef_search = (current_ef_search * 2).min(1024); // Cap at 1024

            if let Some(callback) = progress_callback {
                callback("HNSW Retry", 37 + retry * 2, 100, 37.0 + retry as f32 * 2.0,
                        &format!("Low recall {:.1}%: Boosting ef_search {} -> {} and retrying...", recall, old_ef_search, current_ef_search));
            }

            if std::env::var("PACMAP_VERBOSE").is_ok() {
                eprintln!("LOW RECALL {:.1}%: Boosting ef_search {} -> {} and retrying...",
                         recall, old_ef_search, current_ef_search);
            }
        } else {
            // Final attempt failed
            if let Some(callback) = progress_callback {
                callback("HNSW Error", 45, 100, 45.0, &format!("Poor recall: {:.1}% after {} retries (final ef_search={})", recall, max_retries, current_ef_search));
            }
            return Err(format!("Poor recall: {:.1}% after {} retries (final ef_search={})",
                              recall, max_retries, current_ef_search));
        }
    }

    unreachable!()
}

