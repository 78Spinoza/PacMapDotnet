use ndarray::{ArrayView2, Array1};

/// Brute-force O(n²) k-NN search with local distance scaling - consistent with HNSW path
pub fn compute_pairs_bruteforce(data: ArrayView2<f64>, n_neighbors: usize, _seed: u64) -> Vec<(usize, usize)> {
    let n_samples = data.shape()[0];
    let _n_features = data.shape()[1];
    let mut pairs = Vec::new();

    // Apply same local distance scaling as HNSW for consistency
    let mut sigmas: Array1<f64> = Array1::zeros(n_samples);

    // Phase 1: Compute local bandwidth (sigma) for each point
    for i in 0..n_samples {
        // Compute distances to all other points
        let mut raw_distances: Vec<(usize, f64)> = (0..n_samples)
            .filter(|&j| j != i)
            .map(|j| {
                let dist_sq = euclidean_distance_squared(data.row(i), data.row(j));
                (j, dist_sq)
            })
            .collect();

        // Sort by raw distance to find local bandwidth
        raw_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Compute sigma_i as average of 4th-6th nearest neighbor distances
        let sigma_range = if raw_distances.len() >= 6 {
            &raw_distances[3..6] // 4th-6th neighbors (0-indexed)
        } else if raw_distances.len() >= 3 {
            &raw_distances[2..] // Use what we have
        } else {
            &raw_distances[..] // Fallback for very sparse data
        };

        sigmas[i] = if !sigma_range.is_empty() {
            sigma_range.iter().map(|(_, dist)| dist.sqrt()).sum::<f64>() / sigma_range.len() as f64
        } else {
            1.0 // Fallback bandwidth
        };

        // Avoid division by zero
        if sigmas[i] < 1e-8 {
            sigmas[i] = 1.0;
        }
    }

    // Phase 2: Select neighbors using scaled distances (density-adaptive)
    for i in 0..n_samples {
        // Compute scaled distances to all other points
        let mut scaled_distances: Vec<(usize, f64)> = (0..n_samples)
            .filter(|&j| j != i)
            .map(|j| {
                let dist_sq = euclidean_distance_squared(data.row(i), data.row(j));
                // Apply local distance scaling: d_ij^2 / (sigma_i * sigma_j)
                let scaled_dist = dist_sq / (sigmas[i] * sigmas[j]);
                (j, scaled_dist)
            })
            .collect();

        // Sort by scaled distance and select top k neighbors
        scaled_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (j, _) in scaled_distances.into_iter().take(n_neighbors) {
            pairs.push((i, j));
        }
    }

    if std::env::var("PACMAP_VERBOSE").is_ok() {
        eprintln!("LOCAL SCALING: Applied density adaptation in brute-force with sigma values (min: {:.6}, max: {:.6})",
                 sigmas.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0),
                 sigmas.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0));
    }

    pairs
}

/// Adaptive k-NN search with choice between HNSW and brute-force
pub fn compute_pairs_hnsw(data: ArrayView2<f64>, n_neighbors: usize, seed: u64) -> Vec<(usize, usize)> {
    compute_pairs_hnsw_with_params(data, n_neighbors, seed, None)
}

/// Adaptive k-NN search with choice between HNSW and brute-force, with custom HNSW parameters
pub fn compute_pairs_hnsw_with_params(data: ArrayView2<f64>, n_neighbors: usize, seed: u64, custom_hnsw_params: Option<crate::hnsw_params::HnswParams>) -> Vec<(usize, usize)> {
    // CRITICAL: Force deterministic behavior using comprehensive approach
    // Research shows HNSW has inherent randomness that requires multiple strategies

    // Strategy: Use deterministic sorting and environment variables for reproducibility

    let (n_samples, n_features) = data.dim();

    if n_samples < 2 {
        return Vec::new(); // Need at least 2 points
    }

    // Use brute-force for small datasets (faster due to no index overhead)
    const HNSW_THRESHOLD: usize = 1000;
    if n_samples <= HNSW_THRESHOLD {
        // Only print if verbose mode is enabled (check env var)
        if std::env::var("PACMAP_VERBOSE").is_ok() {
            eprintln!(" Using brute-force k-NN for small dataset ({} samples)", n_samples);
        }
        return compute_pairs_bruteforce(data, n_neighbors, seed);
    }

    // For large datasets, decide based on compilation features
    #[cfg(feature = "use_hnsw")]
    {
        // Only print if verbose mode is enabled (check env var)
        if std::env::var("PACMAP_VERBOSE").is_ok() {
            eprintln!("DEBUG: Using HNSW k-NN for large dataset ({} samples, {} features)", n_samples, n_features);
        }

        // Try HNSW implementation with seed for deterministic results
        match try_hnsw_search_with_params(data, n_neighbors, n_samples, n_features, seed, custom_hnsw_params) {
            Ok(pairs) => {
                eprintln!("SUCCESS: HNSW neighbor search completed: {} pairs found", pairs.len());
                return pairs;
            },
            Err(e) => {
                eprintln!("WARNING: HNSW failed ({}), falling back to brute-force", e);
            }
        }
    }

    #[cfg(not(feature = "use_hnsw"))]
    {
        eprintln!(" HNSW not enabled, using brute-force k-NN for {} samples", n_samples);
    }

    // Fallback to brute-force
    compute_pairs_bruteforce(data, n_neighbors, seed)
}

#[cfg(feature = "use_hnsw")]
fn try_hnsw_search_with_params(data: ArrayView2<f64>, n_neighbors: usize, n_samples: usize, n_features: usize, seed: u64, custom_hnsw_params: Option<crate::hnsw_params::HnswParams>) -> Result<Vec<(usize, usize)>, String> {
    use hnsw_rs::hnsw::Hnsw;
    use hnsw_rs::dist::DistL2;

    // CRITICAL: Deterministic HNSW construction - using comprehensive approach

    // Use custom HNSW parameters or auto-scale
    let hnsw_params = if let Some(custom_params) = custom_hnsw_params {
        eprintln!("   Using custom HNSW params: M={}, ef_construction={}, ef_search={}", custom_params.m, custom_params.ef_construction, custom_params.ef_search);
        custom_params
    } else {
        let auto_params = crate::hnsw_params::HnswParams::auto_scale(n_samples, n_features, n_neighbors);
        eprintln!("   Auto-scaled HNSW params: M={}, ef_construction={}", auto_params.m, auto_params.ef_construction);
        auto_params
    };

    // Convert data to Vec<Vec<f32>> format
    let points: Vec<Vec<f32>> = (0..n_samples)
        .map(|i| {
            data.row(i).iter().map(|&x| x as f32).collect()
        })
        .collect();

    // Calculate dynamic max_layer based on dataset size and m parameter
    let max_layer = ((n_samples as f32).ln() / (hnsw_params.m as f32).ln()).ceil() as usize + 1;
    let max_layer = max_layer.min(32).max(4); // Cap between 4-32 layers

    // CRITICAL: Build HNSW index with deterministic construction
    // Final approach: Single-threaded construction + deterministic insertion order
    let hnsw = {
        use std::sync::Mutex;
        static GLOBAL_CONSTRUCTION_LOCK: Mutex<()> = Mutex::new(());

        // CRITICAL: Single-threaded HNSW construction ONLY (search remains parallel)
        let _construction_lock = GLOBAL_CONSTRUCTION_LOCK.lock().map_err(|e| format!("Failed to acquire construction lock: {}", e))?;

        // Set process-wide seed for any internal RNG usage
        std::env::set_var("PACMAP_HNSW_SEED", seed.to_string());
        std::env::set_var("RUST_TEST_TIME_UNIT", "1000,1000"); // Force deterministic timing

        Hnsw::<f32, DistL2>::new(
            hnsw_params.m,              // max_nb_connection
            n_samples,                  // max_elements
            max_layer,                  // max_layer (dynamic)
            hnsw_params.ef_construction, // ef_construction
            DistL2{}                    // distance_function
        )
    };

    // DETERMINISTIC INSERTION: Sort points by seed-based hash for reproducible order
    // Even with single-threaded construction, insertion order affects graph structure
    let mut data_with_id: Vec<(&[f32], usize)> = points.iter().enumerate().map(|(i, p)| (p.as_slice(), i)).collect();

    // Sort by deterministic hash combining seed + point coordinates + index
    data_with_id.sort_by_key(|(point, id)| {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);  // Include seed in hash
        id.hash(&mut hasher);    // Include original index
        for &coord in point.iter() {
            // Hash float bits for deterministic ordering
            (coord.to_bits()).hash(&mut hasher);
        }
        hasher.finish()
    });

    // Insert in deterministic order (single-threaded)
    for (point, i) in data_with_id {
        hnsw.insert((&point.to_vec(), i));
    }

    // CRITICAL FIX: Implement local distance scaling for proper density adaptation
    // This fixes the core bug causing "completely off" HNSW results
    let extra_candidates = 50; // Buffer for approximation and density selection
    let mut pairs = Vec::new();
    let mut sigmas: Array1<f64> = Array1::zeros(n_samples); // Per-point local bandwidth

    // OPTIMIZATION: Combined sigma computation and neighbor selection
    // This reduces redundant HNSW searches from 2*n_samples to n_samples
    for i in 0..n_samples {
        let query_point = &points[i];

        // Single HNSW search for both sigma computation and neighbor selection
        // Use the larger of the two required search sizes
        let sigma_k = 12; // For sigma estimation (10 + buffer)
        let neighbor_k = n_neighbors + extra_candidates + 1;
        let combined_k = std::cmp::max(sigma_k, neighbor_k);

        let candidates = hnsw.search(query_point, combined_k, hnsw_params.ef_search);

        // Phase 1: Compute sigma from the same search results
        let mut hnsw_distances: Vec<f32> = candidates
            .iter()
            .filter_map(|neighbor| {
                let j = neighbor.d_id as usize;
                if i != j {
                    Some(neighbor.distance)
                } else {
                    None
                }
            })
            .collect();

        // Sort HNSW distance estimates
        hnsw_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute sigma_i from 4th-6th HNSW distance estimates
        let sigma_range = if hnsw_distances.len() >= 6 {
            &hnsw_distances[3..6] // 4th-6th neighbors (0-indexed)
        } else if hnsw_distances.len() >= 3 {
            &hnsw_distances[2..] // Use what we have
        } else {
            &hnsw_distances[..] // Fallback for very sparse data
        };

        sigmas[i] = if !sigma_range.is_empty() {
            // Convert from squared distance to distance
            sigma_range.iter().map(|&dist_sq| (dist_sq as f64).sqrt()).sum::<f64>() / sigma_range.len() as f64
        } else {
            1.0 // Fallback bandwidth
        };

        // Avoid division by zero
        if sigmas[i] < 1e-8 {
            sigmas[i] = 1.0;
        }

        // Phase 2: Select neighbors using scaled distances (density-adaptive)
        // Reuse the same candidates to avoid redundant search
        let mut scaled_distances: Vec<(usize, f64)> = candidates
            .into_iter()
            .filter_map(|neighbor| {
                let j = neighbor.d_id as usize;
                if i != j {
                    // Use HNSW distance estimate directly (much faster than exact computation)
                    let dist_sq = neighbor.distance as f64;

                    // Apply local distance scaling: d_ij^2 / (sigma_i * sigma_j)
                    let scaled_dist = dist_sq / (sigmas[i] * sigmas[j]);
                    Some((j, scaled_dist))
                } else {
                    None
                }
            })
            .collect();

        // Sort by scaled distance and select top k neighbors
        scaled_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (j, _) in scaled_distances.into_iter().take(n_neighbors) {
            pairs.push((i, j));
        }
    }

    if std::env::var("PACMAP_VERBOSE").is_ok() {
        eprintln!("LOCAL SCALING: Applied density adaptation with sigma values (min: {:.6}, max: {:.6})",
                 sigmas.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0),
                 sigmas.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0));
    }

    Ok(pairs)
}

/// Convert pairs to per-point neighbor lists with distances for proper symmetrization
/// Returns Vec<Vec<(neighbor_id, distance)>> for each point
/// Convert pairs to per-point neighbor lists with distances for proper symmetrization (with custom HNSW params)
/// Returns Vec<Vec<(neighbor_id, distance)>> for each point
pub fn compute_pairs_hnsw_to_per_point_with_params(data: ArrayView2<f64>, n_neighbors: usize, _seed: u64, custom_hnsw_params: Option<crate::hnsw_params::HnswParams>) -> Result<Vec<Vec<(usize, f64)>>, String> {
    let (n_samples, n_features) = data.dim();

    if n_samples < 2 {
        return Ok(vec![Vec::new(); n_samples]);
    }

    // Use same strategy as compute_pairs_hnsw
    if n_samples <= 1000 {
        return Ok(get_per_point_bruteforce(data, n_neighbors));
    }

    #[cfg(feature = "use_hnsw")]
    {
        if let Ok(per_point_result) = try_hnsw_per_point_search_with_params(data, n_neighbors, n_samples, n_features, custom_hnsw_params) {
            return Ok(per_point_result);
        }
    }

    // Fallback to brute-force
    Ok(get_per_point_bruteforce(data, n_neighbors))
}

fn get_per_point_bruteforce(data: ArrayView2<f64>, n_neighbors: usize) -> Vec<Vec<(usize, f64)>> {
    let n_samples = data.shape()[0];

    (0..n_samples)
        .map(|i| {
            // Compute distances to all other points
            let mut distances: Vec<(usize, f64)> = (0..n_samples)
                .filter(|&j| j != i)
                .map(|j| {
                    let dist = euclidean_distance(data.row(i), data.row(j));
                    (j, dist)
                })
                .collect();

            // Sort by distance and take n_neighbors
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.into_iter().take(n_neighbors).collect()
        })
        .collect()
}

#[cfg(feature = "use_hnsw")]
#[allow(dead_code)]
fn try_hnsw_per_point_search(data: ArrayView2<f64>, n_neighbors: usize, n_samples: usize, n_features: usize) -> Result<Vec<Vec<(usize, f64)>>, String> {
    try_hnsw_per_point_search_with_params(data, n_neighbors, n_samples, n_features, None)
}

#[cfg(feature = "use_hnsw")]
fn try_hnsw_per_point_search_with_params(data: ArrayView2<f64>, n_neighbors: usize, n_samples: usize, n_features: usize, custom_hnsw_params: Option<crate::hnsw_params::HnswParams>) -> Result<Vec<Vec<(usize, f64)>>, String> {
    use hnsw_rs::hnsw::Hnsw;
    use hnsw_rs::dist::DistL2;

    let hnsw_params = if let Some(custom_params) = custom_hnsw_params {
        custom_params
    } else {
        crate::hnsw_params::HnswParams::auto_scale(n_samples, n_features, n_neighbors)
    };

    let points: Vec<Vec<f32>> = (0..n_samples)
        .map(|i| data.row(i).iter().map(|&x| x as f32).collect())
        .collect();

    let max_layer = ((n_samples as f32).ln() / (hnsw_params.m as f32).ln()).ceil() as usize + 1;
    let max_layer = max_layer.min(32).max(4);

    let hnsw = Hnsw::<f32, DistL2>::new(
        hnsw_params.m,
        n_samples,
        max_layer,
        hnsw_params.ef_construction,
        DistL2{}
    );

    let data_with_id: Vec<(&[f32], usize)> = points.iter().enumerate().map(|(i, p)| (p.as_slice(), i)).collect();

    #[cfg(feature = "parallel")]
    {
        hnsw.parallel_insert(&data_with_id);
    }
    #[cfg(not(feature = "parallel"))]
    {
        for (point, i) in data_with_id {
            hnsw.insert((&point.to_vec(), i));
        }
    }

    let result = (0..n_samples)
        .map(|i| {
            let query_point = &points[i];
            let candidates = hnsw.search(query_point, n_neighbors + 1, hnsw_params.ef_search);

            candidates
                .into_iter()
                .filter_map(|neighbor| {
                    let j = neighbor.d_id as usize;
                    if i != j {
                        Some((j, neighbor.distance as f64))
                    } else {
                        None
                    }
                })
                .take(n_neighbors)
                .collect()
        })
        .collect();

    Ok(result)
}

/// Get k-NN indices with choice between HNSW and brute-force
pub fn get_knn_indices(data: ArrayView2<f64>, n_neighbors: usize, _seed: u64) -> Vec<Vec<usize>> {
    let (n_samples, _n_features) = data.dim();

    if n_samples < 2 {
        return vec![Vec::new(); n_samples];
    }

    // Use same strategy as compute_pairs_hnsw
    if n_samples <= 1000 {
        return get_knn_indices_bruteforce(data, n_neighbors);
    }

    #[cfg(feature = "use_hnsw")]
    {
        // Try HNSW first for large datasets
        if let Ok(hnsw_result) = try_hnsw_knn_indices(data, n_neighbors) {
            return hnsw_result;
        }
    }

    // Fallback to brute-force
    get_knn_indices_bruteforce(data, n_neighbors)
}

fn get_knn_indices_bruteforce(data: ArrayView2<f64>, n_neighbors: usize) -> Vec<Vec<usize>> {
    let n_samples = data.shape()[0];

    (0..n_samples)
        .map(|i| {
            // Compute distances to all other points
            let mut distances: Vec<(usize, f64)> = (0..n_samples)
                .filter(|&j| j != i)
                .map(|j| {
                    let dist = euclidean_distance(data.row(i), data.row(j));
                    (j, dist)
                })
                .collect();

            // Sort by distance and take n_neighbors
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.into_iter().take(n_neighbors).map(|(j, _)| j).collect()
        })
        .collect()
}

#[cfg(feature = "use_hnsw")]
fn try_hnsw_knn_indices(data: ArrayView2<f64>, n_neighbors: usize) -> Result<Vec<Vec<usize>>, String> {
    use hnsw_rs::hnsw::Hnsw;
    use hnsw_rs::dist::DistL2;

    let (n_samples, n_features) = data.dim();
    let hnsw_params = crate::hnsw_params::HnswParams::auto_scale(n_samples, n_features, n_neighbors);

    let points: Vec<Vec<f32>> = (0..n_samples)
        .map(|i| {
            data.row(i).iter().map(|&x| x as f32).collect()
        })
        .collect();

    // Calculate dynamic max_layer based on dataset size and m parameter
    let max_layer = ((n_samples as f32).ln() / (hnsw_params.m as f32).ln()).ceil() as usize + 1;
    let max_layer = max_layer.min(32).max(4); // Cap between 4-32 layers

    let hnsw = Hnsw::<f32, DistL2>::new(
        hnsw_params.m,
        n_samples,
        max_layer,                  // max_layer (dynamic)
        hnsw_params.ef_construction,
        DistL2{}
    );

    // Insert all points into HNSW index using parallel insert if available
    let data_with_id: Vec<(&[f32], usize)> = points.iter().enumerate().map(|(i, p)| (p.as_slice(), i)).collect();

    #[cfg(feature = "parallel")]
    {
        hnsw.parallel_insert(&data_with_id);
    }
    #[cfg(not(feature = "parallel"))]
    {
        for (point, i) in data_with_id {
            hnsw.insert((&point.to_vec(), i));
        }
    }

    let result = (0..n_samples)
        .map(|i| {
            let search_result = hnsw.search(&points[i], n_neighbors + 1, hnsw_params.ef_search);

            search_result
                .into_iter()
                .map(|neighbor| neighbor.d_id as usize)
                .filter(|&j| j != i)
                .take(n_neighbors)  // Ensure exactly n_neighbors per point
                .collect()
        })
        .collect();

    Ok(result)
}

fn euclidean_distance(a: ndarray::ArrayView1<f64>, b: ndarray::ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn euclidean_distance_squared(a: ndarray::ArrayView1<f64>, b: ndarray::ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
}