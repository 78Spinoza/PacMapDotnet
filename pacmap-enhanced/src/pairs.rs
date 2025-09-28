use ndarray::ArrayView2;
use crate::hnsw_params::HnswParams;

/// Brute-force O(n²) k-NN search - always available, good for small datasets
pub fn compute_pairs_bruteforce(data: ArrayView2<f64>, n_neighbors: usize, _seed: u64) -> Vec<(usize, usize)> {
    let n_samples = data.shape()[0];
    let mut pairs = Vec::new();

    for i in 0..n_samples {
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

        for (j, _) in distances.into_iter().take(n_neighbors) {
            pairs.push((i, j));
        }
    }

    pairs
}

/// Adaptive k-NN search with choice between HNSW and brute-force
pub fn compute_pairs_hnsw(data: ArrayView2<f64>, n_neighbors: usize, seed: u64) -> Vec<(usize, usize)> {
    let (n_samples, n_features) = data.dim();

    if n_samples < 2 {
        return Vec::new(); // Need at least 2 points
    }

    // Use brute-force for small datasets (faster due to no index overhead)
    const HNSW_THRESHOLD: usize = 1000;
    if n_samples <= HNSW_THRESHOLD {
        // Only print if verbose mode is enabled (check env var)
        if std::env::var("PACMAP_VERBOSE").is_ok() {
            eprintln!("🔍 Using brute-force k-NN for small dataset ({} samples)", n_samples);
        }
        return compute_pairs_bruteforce(data, n_neighbors, seed);
    }

    // For large datasets, decide based on compilation features
    #[cfg(feature = "use_hnsw")]
    {
        // Only print if verbose mode is enabled (check env var)
        if std::env::var("PACMAP_VERBOSE").is_ok() {
            eprintln!("🔧 Using HNSW k-NN for large dataset ({} samples, {} features)", n_samples, n_features);
        }

        // Try HNSW implementation
        match try_hnsw_search(data, n_neighbors, n_samples, n_features) {
            Ok(pairs) => {
                eprintln!("✅ HNSW neighbor search completed: {} pairs found", pairs.len());
                return pairs;
            },
            Err(e) => {
                eprintln!("⚠️ HNSW failed ({}), falling back to brute-force", e);
            }
        }
    }

    #[cfg(not(feature = "use_hnsw"))]
    {
        eprintln!("🔍 HNSW not enabled, using brute-force k-NN for {} samples", n_samples);
    }

    // Fallback to brute-force
    compute_pairs_bruteforce(data, n_neighbors, seed)
}

#[cfg(feature = "use_hnsw")]
fn try_hnsw_search(data: ArrayView2<f64>, n_neighbors: usize, n_samples: usize, n_features: usize) -> Result<Vec<(usize, usize)>, String> {
    use hnsw_rs::hnsw::Hnsw;
    use hnsw_rs::dist::DistL2;

    // Auto-scale HNSW parameters
    let hnsw_params = HnswParams::auto_scale(n_samples, n_features, n_neighbors);
    eprintln!("   HNSW params: M={}, ef_construction={}", hnsw_params.m, hnsw_params.ef_construction);

    // Convert data to Vec<Vec<f32>> format
    let points: Vec<Vec<f32>> = (0..n_samples)
        .map(|i| {
            data.row(i).iter().map(|&x| x as f32).collect()
        })
        .collect();

    // Build HNSW index with L2 distance
    let hnsw = Hnsw::<f32, DistL2>::new(
        hnsw_params.m,              // max_nb_connection
        n_samples,                  // max_elements
        16,                         // max_layer
        hnsw_params.ef_construction, // ef_construction
        DistL2{}                    // distance_function
    );

    // Insert all points into HNSW index
    for (i, point) in points.iter().enumerate() {
        hnsw.insert((point, i));
    }

    let mut pairs = Vec::new();

    // For each point, find its k nearest neighbors using HNSW
    for i in 0..n_samples {
        let query_point = &points[i];
        let search_result = hnsw.search(query_point, n_neighbors + 1, hnsw_params.ef_search);

        let mut neighbors_added = 0;
        for neighbor in search_result {
            // Extract the point ID and convert to usize
            let j = neighbor.p_id.0 as usize;
            if i != j && neighbors_added < n_neighbors {
                pairs.push((i, j));
                neighbors_added += 1;
            }
        }
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
    let hnsw_params = HnswParams::auto_scale(n_samples, n_features, n_neighbors);

    let points: Vec<Vec<f32>> = (0..n_samples)
        .map(|i| {
            data.row(i).iter().map(|&x| x as f32).collect()
        })
        .collect();

    let hnsw = Hnsw::<f32, DistL2>::new(
        hnsw_params.m,
        n_samples,
        16,
        hnsw_params.ef_construction,
        DistL2{}
    );

    for (i, point) in points.iter().enumerate() {
        hnsw.insert((point, i));
    }

    let result = (0..n_samples)
        .map(|i| {
            let search_result = hnsw.search(&points[i], n_neighbors + 1, hnsw_params.ef_search);

            search_result
                .into_iter()
                .map(|neighbor| neighbor.p_id.0 as usize)
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