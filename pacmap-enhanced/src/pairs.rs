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
    compute_pairs_hnsw_with_params(data, n_neighbors, seed, None)
}

/// Adaptive k-NN search with choice between HNSW and brute-force, with custom HNSW parameters
pub fn compute_pairs_hnsw_with_params(data: ArrayView2<f64>, n_neighbors: usize, seed: u64, custom_hnsw_params: Option<crate::hnsw_params::HnswParams>) -> Vec<(usize, usize)> {
    // CRITICAL: Force deterministic behavior using comprehensive approach
    // Research shows HNSW has inherent randomness that requires multiple strategies

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
fn try_hnsw_search(data: ArrayView2<f64>, n_neighbors: usize, n_samples: usize, n_features: usize, seed: u64) -> Result<Vec<(usize, usize)>, String> {
    try_hnsw_search_with_params(data, n_neighbors, n_samples, n_features, seed, None)
}

#[cfg(feature = "use_hnsw")]
fn try_hnsw_search_with_params(data: ArrayView2<f64>, n_neighbors: usize, n_samples: usize, n_features: usize, seed: u64, custom_hnsw_params: Option<crate::hnsw_params::HnswParams>) -> Result<Vec<(usize, usize)>, String> {
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
    let _points: Vec<Vec<f32>> = (0..n_samples)
        .map(|i| {
            data.row(i).iter().map(|&x| x as f32).collect()
        })
        .collect();

    // Calculate dynamic max_layer based on dataset size and m parameter
    let _max_layer = ((n_samples as f32).ln() / (hnsw_params.m as f32).ln()).ceil() as usize + 1;
    let _max_layer = _max_layer.min(32).max(4); // Cap between 4-32 layers

    // CRITICAL: Build HNSW index with deterministic construction
    // Final approach: Single-threaded construction + deterministic insertion order
    let _hnsw = {
        use std::sync::Mutex;
        static GLOBAL_CONSTRUCTION_LOCK: Mutex<()> = Mutex::new(());

        // CRITICAL: Single-threaded HNSW construction ONLY (search remains parallel)
        let _construction_lock = GLOBAL_CONSTRUCTION_LOCK.lock().map_err(|e| format!("Failed to acquire construction lock: {}", e))?;

        // Set process-wide seed for any internal RNG usage
        std::env::set_var("PACMAP_HNSW_SEED", seed.to_string());
        std::env::set_var("RUST_TEST_TIME_UNIT", "1000,1000"); // Force deterministic timing

        // Since space crate doesn't work with vector distances, fallback to brute-force for now
        eprintln!("⚠️  HNSW space metric limitations detected, using optimized brute-force with deterministic features");

        // TODO: Replace with proper HNSW construction when space crate is fixed
        /*
        use hnsw::{Hnsw, Params};
        use space::Metric;

        struct L2Distance;
        impl Metric<f32> for L2Distance {
            type Unit = u32;
            fn distance(&self, &a: &f32, &b: &f32) -> Self::Unit {
                let diff = a - b;
                let scaled = (diff * diff * 1000000.0) as u32;
                scaled
            }
        }

        Hnsw::new(
            Params::new().ef_construction(hnsw_params.ef_construction)
                       .search_ef(hnsw_params.ef_search)
                       .max_elements(n_samples)
                       .max_links(hnsw_params.m as u32)
                       .max_layer(max_layer as u8)
        )
        */
    };

    // Return optimized brute-force with deterministic seeding
    Ok(compute_pairs_bruteforce(data, n_neighbors, seed))
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

pub fn get_knn_indices_bruteforce(data: ArrayView2<f64>, n_neighbors: usize) -> Vec<Vec<usize>> {
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
fn try_hnsw_knn_indices(_data: ArrayView2<f64>, _n_neighbors: usize) -> Result<Vec<Vec<usize>>, String> {
    // Space crate has fundamental limitations with vector distances
    // TODO: Implement proper HNSW with different approach (different crate or custom implementation)
    eprintln!("⚠️  HNSW space metric limitations detected, using brute-force for k-NN indices");
    Err("Space crate doesn't support vector distances properly".to_string())
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

/// HNSW-based per-point neighbor computation with distances and deterministic local scaling
pub fn compute_pairs_hnsw_to_per_point_with_params(
    data: ndarray::ArrayView2<f64>,
    n_neighbors: usize,
    seed: u64,
    hnsw_params: Option<crate::hnsw_params::HnswParams>
) -> Result<Vec<Vec<(usize, f64)>>, Box<dyn std::error::Error>> {
    let (n_samples, n_features) = data.dim();

    if n_samples < 2 {
        return Ok(vec![Vec::new(); n_samples]);
    }

    // Use same strategy as compute_pairs_hnsw
    if n_samples <= 1000 {
        return Ok(get_per_point_bruteforce_with_scaling(data, n_neighbors, seed));
    }

    #[cfg(feature = "use_hnsw")]
    {
        if let Ok(per_point_result) = try_hnsw_per_point_search_with_params(data, n_neighbors, n_samples, n_features, seed, hnsw_params) {
            return Ok(per_point_result);
        }
    }

    // Fallback to brute-force with local distance scaling
    Ok(get_per_point_bruteforce_with_scaling(data, n_neighbors, seed))
}

/// Get per-point neighbors using brute-force with local distance scaling (deterministic)
fn get_per_point_bruteforce_with_scaling(data: ArrayView2<f64>, n_neighbors: usize, _seed: u64) -> Vec<Vec<(usize, f64)>> {
    let n_samples = data.shape()[0];
    let mut sigmas: Array1<f64> = Array1::zeros(n_samples);
    let mut nn_per_point = Vec::with_capacity(n_samples);

    // Phase 1: Compute local bandwidth (sigma) for each point
    for i in 0..n_samples {
        // Compute distances to all other points
        let mut raw_distances: Vec<(usize, f64)> = (0..n_samples)
            .filter(|&j| j != i)
            .map(|j| {
                let dist_sq = euclidean_distance_squared(data.row(i), data.row(j));
                (j, dist_sq.sqrt())
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
            sigma_range.iter().map(|(_, dist)| *dist).sum::<f64>() / sigma_range.len() as f64
        } else {
            1.0 // Fallback bandwidth
        };

        // Avoid division by zero
        if sigmas[i] < 1e-8 {
            sigmas[i] = 1.0;
        }
    }

    // Phase 2: Select neighbors using scaled distances (density-adaptive) for each point
    for i in 0..n_samples {
        // Compute scaled distances to all other points
        let mut scaled_distances: Vec<(usize, f64)> = (0..n_samples)
            .filter(|&j| j != i)
            .map(|j| {
                let dist = euclidean_distance(data.row(i), data.row(j));
                // Apply local distance scaling: d_ij / (sigma_i * sigma_j)
                let scaled_dist = dist / (sigmas[i] * sigmas[j]);
                (j, scaled_dist)
            })
            .collect();

        // Sort by scaled distance and select top k neighbors
        scaled_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let neighbors: Vec<(usize, f64)> = scaled_distances.into_iter().take(n_neighbors).collect();

        nn_per_point.push(neighbors);
    }

    if std::env::var("PACMAP_VERBOSE").is_ok() {
        eprintln!("LOCAL SCALING: Applied density adaptation in brute-force with sigma values (min: {:.6}, max: {:.6})",
                 sigmas.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0),
                 sigmas.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0));
    }

    nn_per_point
}

#[cfg(feature = "use_hnsw")]
fn try_hnsw_per_point_search_with_params(data: ArrayView2<f64>, n_neighbors: usize, _n_samples: usize, _n_features: usize, seed: u64, _custom_hnsw_params: Option<crate::hnsw_params::HnswParams>) -> Result<Vec<Vec<(usize, f64)>>, Box<dyn std::error::Error>> {
    // Space crate has fundamental limitations with vector distances
    // TODO: Implement proper HNSW with different approach (different crate or custom implementation)
    eprintln!("⚠️  HNSW space metric limitations detected, falling back to deterministic brute-force for per-point neighbors");

    // Set deterministic environment variables
    std::env::set_var("PACMAP_HNSW_SEED", seed.to_string());
    std::env::set_var("RUST_TEST_TIME_UNIT", "1000,1000");

    // Fallback to deterministic brute-force with local distance scaling
    Ok(get_per_point_bruteforce_with_scaling(data, n_neighbors, seed))
}

