use half::f16;
use ndarray::Array2;
use serde::{Serialize, Deserialize};

/// Quantization parameters for embedding compression
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct QuantizationParams {
    /// Minimum value in the original embedding
    pub min_value: f64,
    /// Maximum value in the original embedding
    pub max_value: f64,
    /// Scale factor for quantization
    pub scale: f64,
    /// Zero point for quantization
    pub zero_point: f64,
    /// Whether centroids were used for clustering-based quantization
    pub use_centroids: bool,
    /// Cluster centroids for k-means based quantization (optional)
    pub centroids: Option<Vec<f64>>,
}

/// Quantized embedding with its parameters
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct QuantizedEmbedding {
    pub data: Array2<f16>,
    pub params: QuantizationParams,
}

/// Simple linear quantization with proper parameter tracking
pub fn quantize_embedding_linear(embedding: &Array2<f64>) -> QuantizedEmbedding {
    let flat = embedding.as_slice().unwrap();
    let min_val = flat.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = flat.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Simple f16 conversion without artificial scaling/clamping
    // f16 can handle range ±65504 with good precision around 0
    let quantized = embedding.mapv(|x| f16::from_f64(x));

    QuantizedEmbedding {
        data: quantized,
        params: QuantizationParams {
            min_value: min_val,
            max_value: max_val,
            scale: 1.0,      // No scaling applied
            zero_point: 0.0, // No zero point offset
            use_centroids: false,
            centroids: None,
        },
    }
}

/// K-means based quantization with centroids
#[allow(dead_code)]
pub fn quantize_embedding_kmeans(embedding: &Array2<f64>, k: usize) -> QuantizedEmbedding {
    let flat = embedding.as_slice().unwrap();

    // Simple k-means clustering for demonstration
    let mut centroids = Vec::new();
    let min_val = flat.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = flat.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Initialize centroids evenly across the range
    for i in 0..k {
        let centroid = min_val + (max_val - min_val) * (i as f64) / ((k - 1) as f64);
        centroids.push(centroid);
    }

    // Assign each value to nearest centroid and quantize
    let quantized = embedding.mapv(|x| {
        let mut best_idx = 0;
        let mut best_dist = (x - centroids[0]).abs();

        for (i, &centroid) in centroids.iter().enumerate().skip(1) {
            let dist = (x - centroid).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        f16::from_f64(best_idx as f64)
    });

    QuantizedEmbedding {
        data: quantized,
        params: QuantizationParams {
            min_value: min_val,
            max_value: max_val,
            scale: 1.0,
            zero_point: 0.0,
            use_centroids: true,
            centroids: Some(centroids),
        },
    }
}

/// Dequantize embedding using stored parameters
pub fn dequantize_embedding(quantized: &QuantizedEmbedding) -> Array2<f64> {
    if quantized.params.use_centroids {
        // Centroid-based dequantization
        if let Some(ref centroids) = quantized.params.centroids {
            quantized.data.mapv(|x| {
                let idx = x.to_f64() as usize;
                if idx < centroids.len() {
                    centroids[idx]
                } else {
                    0.0
                }
            })
        } else {
            // Fallback to linear if centroids missing
            quantized.data.mapv(|x| x.to_f64())
        }
    } else {
        // Simple f16 to f64 conversion (no scaling needed for new approach)
        quantized.data.mapv(|x| x.to_f64())
    }
}

/// Legacy function for backward compatibility
pub fn quantize_embedding(embedding: &Array2<f64>) -> Array2<f16> {
    quantize_embedding_linear(embedding).data
}
