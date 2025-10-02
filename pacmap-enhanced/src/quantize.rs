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
    /// Scale factor for quantization (currently always 1.0 for direct f16 conversion)
    pub scale: f64,
    /// Zero point for quantization (currently always 0.0 for direct f16 conversion)
    pub zero_point: f64,
    /// Whether centroids are used for k-means quantization
    #[serde(default)]
    pub use_centroids: bool,
    /// Cluster centroids for k-means quantization (None for linear quantization)
    #[serde(default)]
    pub centroids: Option<Vec<f64>>,
}

/// Quantized embedding with its parameters
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct QuantizedEmbedding {
    /// Quantized data - f16 for linear quantization, u8/u16 indices for k-means
    pub data: Array2<f16>,
    /// K-means indices stored separately for better compression (when use_centroids=true)
    #[serde(default)]
    pub indices: Option<Array2<u8>>,
    pub params: QuantizationParams,
}

/// Simple linear quantization with MSE validation
pub fn quantize_embedding_linear(embedding: &Array2<f64>) -> QuantizedEmbedding {
    let flat = embedding.as_slice().unwrap();
    let min_val = flat.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = flat.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Simple f16 conversion without artificial scaling/clamping
    // f16 can handle range ±65504 with good precision around 0
    let quantized = embedding.mapv(|x| f16::from_f64(x));

    let result = QuantizedEmbedding {
        data: quantized,
        indices: None, // Linear quantization doesn't use indices
        params: QuantizationParams {
            min_value: min_val,
            max_value: max_val,
            scale: 1.0,      // No scaling applied
            zero_point: 0.0, // No zero point offset
            use_centroids: false,
            centroids: None,
        },
    };

    // Validate quantization quality with MSE check
    validate_quantization_mse(&result, embedding);

    result
}

/// K-means based quantization with adaptive centroids for optimal compression

/// Validate quantization quality using Mean Squared Error
fn validate_quantization_mse(quantized: &QuantizedEmbedding, original: &Array2<f64>) {
    // Dequantize and calculate MSE
    let dequantized = dequantize_embedding(quantized);

    if dequantized.shape() != original.shape() {
        eprintln!("WARNING: Quantization validation: Shape mismatch - original: {:?}, dequantized: {:?}",
                  original.shape(), dequantized.shape());
        return;
    }

    let mse = original.iter()
        .zip(dequantized.iter())
        .map(|(&orig, &deq)| (orig - deq).powi(2))
        .sum::<f64>() / original.len() as f64;

    let rmse = mse.sqrt();
    let max_val = original.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_val = original.iter().copied().fold(f64::INFINITY, f64::min);
    let data_range = max_val - min_val;
    let relative_rmse = rmse / data_range * 100.0;

    println!("Quantization Quality Assessment:");
    println!("   - MSE: {:.6}", mse);
    println!("   - RMSE: {:.6}", rmse);
    println!("   - Relative RMSE: {:.2}%", relative_rmse);
    println!("   - Data range: [{:.6}, {:.6}]", min_val, max_val);

    // Quality thresholds
    if relative_rmse < 0.1 {
        println!("   EXCELLENT: Excellent quantization quality");
    } else if relative_rmse < 0.5 {
        println!("   GOOD: Good quantization quality");
    } else if relative_rmse < 1.0 {
        println!("   ACCEPTABLE: Acceptable quantization quality");
    } else {
        println!("   POOR: Poor quantization quality - consider alternative approach");
    }
}

/// Dequantize embedding using stored parameters
pub fn dequantize_embedding(quantized: &QuantizedEmbedding) -> Array2<f64> {
    if quantized.params.use_centroids {
        // K-means centroid-based dequantization using u8 indices
        if let (Some(ref centroids), Some(ref indices)) = (&quantized.params.centroids, &quantized.indices) {
            indices.mapv(|idx| {
                let idx = idx as usize;
                if idx < centroids.len() {
                    centroids[idx]
                } else {
                    panic!("Quantization error: centroid index {} out of bounds ({} centroids available). This indicates data corruption or invalid quantization parameters.",
                           idx, centroids.len());
                }
            })
        } else {
            // Fallback to linear if centroids or indices missing
            quantized.data.mapv(|x| x.to_f64())
        }
    } else {
        // Simple f16 to f64 conversion for linear quantization
        quantized.data.mapv(|x| x.to_f64())
    }
}

/// Legacy function for backward compatibility
pub fn quantize_embedding(embedding: &Array2<f64>) -> Array2<f16> {
    quantize_embedding_linear(embedding).data
}
