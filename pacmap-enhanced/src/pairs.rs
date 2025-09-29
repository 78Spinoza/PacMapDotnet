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
        raw_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

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
        scaled_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for (j, _) in scaled_distances.into_iter().take(n_neighbors) {
            pairs.push((i, j));
        }
    }

    if std::env::var("PACMAP_VERBOSE").is_ok() {
        eprintln!("✅ LOCAL SCALING: Applied density adaptation in brute-force with sigma values (min: {:.6}, max: {:.6})",
                 sigmas.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0),
                 sigmas.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0));
    }

    pairs
}

/// Adaptive k-NN search with choice between HNSW and brute-force
pub fn compute_pairs_hnsw(data: ArrayView2<f64>, n_neighbors: usize, seed: u64) -> Vec<(usize, usize)> {
    let (n_samples, _n_features) = data.dim();

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

        // Try HNSW implementation
        match try_hnsw_search(data, n_neighbors, n_samples, n_features) {
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
fn try_hnsw_search(data: ArrayView2<f64>, n_neighbors: usize, n_samples: usize, n_features: usize) -> Result<Vec<(usize, usize)>, String> {
    use hnsw_rs::hnsw::Hnsw;
    use hnsw_rs::dist::DistL2;

    // Auto-scale HNSW parameters
    let hnsw_params = crate::hnsw_params::HnswParams::auto_scale(n_samples, n_features, n_neighbors);
    eprintln!("   HNSW params: M={}, ef_construction={}", hnsw_params.m, hnsw_params.ef_construction);

    // Convert data to Vec<Vec<f32>> format
    let points: Vec<Vec<f32>> = (0..n_samples)
        .map(|i| {
            data.row(i).iter().map(|&x| x as f32).collect()
        })
        .collect();

    // Calculate dynamic max_layer based on dataset size and m parameter
    let max_layer = ((n_samples as f32).ln() / (hnsw_params.m as f32).ln()).ceil() as usize + 1;
    let max_layer = max_layer.min(32).max(4); // Cap between 4-32 layers

    // Build HNSW index with L2 distance
    let hnsw = Hnsw::<f32, DistL2>::new(
        hnsw_params.m,              // max_nb_connection
        n_samples,                  // max_elements
        max_layer,                  // max_layer (dynamic)
        hnsw_params.ef_construction, // ef_construction
        DistL2{}                    // distance_function
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

    // CRITICAL FIX: Implement local distance scaling for proper density adaptation
    // This fixes the core bug causing "completely off" HNSW results
    let extra_candidates = 50; // Buffer for approximation and density selection
    let mut pairs = Vec::new();
    let mut sigmas: Array1<f64> = Array1::zeros(n_samples); // Per-point local bandwidth

    // Phase 1: Compute local bandwidth (sigma) for each point
    // Sigma_i = average distance to 4th-6th nearest neighbors (density adaptation)
    for i in 0..n_samples {
        let query_point = &points[i];
        let candidates = hnsw.search(query_point, n_neighbors + extra_candidates + 1, hnsw_params.ef_search);

        // Extract raw distances and neighbor IDs
        let mut raw_distances: Vec<(usize, f64)> = candidates
            .into_iter()
            .filter_map(|neighbor| {
                let j = neighbor.d_id as usize;
                if i != j {
                    // Compute raw Euclidean distance (squared for efficiency)
                    let mut dist_sq = 0.0;
                    for k in 0..n_features {
                        let diff = data[[i, k]] - data[[j, k]];
                        dist_sq += diff * diff;
                    }
                    Some((j, dist_sq))
                } else {
                    None
                }
            })
            .collect();

        // Sort by raw distance to find local bandwidth
        raw_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

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
        let query_point = &points[i];
        let candidates = hnsw.search(query_point, n_neighbors + extra_candidates + 1, hnsw_params.ef_search);

        // Extract and scale all candidate distances
        let mut scaled_distances: Vec<(usize, f64)> = candidates
            .into_iter()
            .filter_map(|neighbor| {
                let j = neighbor.d_id as usize;
                if i != j {
                    // Compute raw distance squared
                    let mut dist_sq = 0.0;
                    for k in 0..n_features {
                        let diff = data[[i, k]] - data[[j, k]];
                        dist_sq += diff * diff;
                    }

                    // Apply local distance scaling: d_ij^2 / (sigma_i * sigma_j)
                    let scaled_dist = dist_sq / (sigmas[i] * sigmas[j]);
                    Some((j, scaled_dist))
                } else {
                    None
                }
            })
            .collect();

        // Sort by scaled distance and select top k neighbors
        scaled_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for (j, _) in scaled_distances.into_iter().take(n_neighbors) {
            pairs.push((i, j));
        }
    }

    if std::env::var("PACMAP_VERBOSE").is_ok() {
        eprintln!("✅ LOCAL SCALING: Applied density adaptation with sigma values (min: {:.6}, max: {:.6})",
                 sigmas.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0),
                 sigmas.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0));
    }

    Ok(pairs)
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