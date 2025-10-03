use crate::serialization::{HnswSerializedParams};

/// Wrapper for HNSW index that can be serialized/deserialized
#[derive(Clone, Debug)]
pub struct HnswIndexWrapper {
    pub points: Vec<Vec<f32>>,
    pub params: HnswSerializedParams,
    pub is_loaded: bool,
}

impl HnswIndexWrapper {
    pub fn new(points: Vec<Vec<f32>>, params: HnswSerializedParams) -> Self {
        Self {
            points,
            params,
            is_loaded: true,
        }
    }

    pub fn search_neighbors(&self, query: &[f32], k: usize, _ef_search: usize) -> Vec<(usize, f32)> {
        if !self.is_loaded {
            return Vec::new();
        }

        // TEMPORARY: Brute-force search until proper HNSW reconstruction is implemented
        let mut distances: Vec<(usize, f32)> = self.points
            .iter()
            .enumerate()
            .map(|(i, point)| {
                let dist: f32 = point
                    .iter()
                    .zip(query.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt();
                (i, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.into_iter().take(k).collect()
    }

    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    pub fn dimension(&self) -> usize {
        self.points.first().map(|p| p.len()).unwrap_or(0)
    }
}