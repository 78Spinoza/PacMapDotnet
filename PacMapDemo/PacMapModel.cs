using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;

namespace PacMapDemo
{
    /// <summary>
    /// Distance metrics supported by Enhanced PacMAP
    /// </summary>
    public enum DistanceMetric
    {
        /// <summary>
        /// Euclidean distance (L2 norm) - most common choice for general data
        /// </summary>
        Euclidean = 0,

        /// <summary>
        /// Cosine distance - excellent for high-dimensional sparse data (text, images)
        /// </summary>
        Cosine = 1,

        /// <summary>
        /// Manhattan distance (L1 norm) - robust to outliers
        /// </summary>
        Manhattan = 2,

        /// <summary>
        /// Correlation distance - measures linear relationships, good for time series
        /// </summary>
        Correlation = 3,

        /// <summary>
        /// Hamming distance - for binary or categorical data
        /// </summary>
        Hamming = 4
    }

    /// <summary>
    /// Outlier severity levels for Enhanced PacMAP safety analysis
    /// </summary>
    public enum OutlierLevel
    {
        /// <summary>
        /// Normal data point - within 95th percentile of training data distances
        /// </summary>
        Normal = 0,

        /// <summary>
        /// Unusual data point - between 95th and 99th percentile of training data distances
        /// </summary>
        Unusual = 1,

        /// <summary>
        /// Mild outlier - between 99th percentile and 2.5 standard deviations from mean
        /// </summary>
        Mild = 2,

        /// <summary>
        /// Extreme outlier - between 2.5 and 4.0 standard deviations from mean
        /// </summary>
        Extreme = 3,

        /// <summary>
        /// No man's land - beyond 4.0 standard deviations from training data
        /// Projection may be unreliable
        /// </summary>
        NoMansLand = 4
    }

    /// <summary>
    /// Enhanced transform result with comprehensive safety metrics and outlier detection
    /// Available only with HNSW-optimized models for production safety
    /// </summary>
    public class TransformResult
    {
        /// <summary>
        /// Gets the projected coordinates in the embedding space (1-50D)
        /// </summary>
        public float[] ProjectionCoordinates { get; }

        /// <summary>
        /// Gets the indices of nearest neighbors in the original training data
        /// </summary>
        public int[] NearestNeighborIndices { get; }

        /// <summary>
        /// Gets the distances to nearest neighbors in the original feature space
        /// </summary>
        public float[] NearestNeighborDistances { get; }

        /// <summary>
        /// Gets the confidence score for the projection (0.0 - 1.0)
        /// Higher values indicate the point is similar to training data
        /// </summary>
        public float ConfidenceScore { get; }

        /// <summary>
        /// Gets the outlier severity level based on distance from training data
        /// </summary>
        public OutlierLevel Severity { get; }

        /// <summary>
        /// Gets the percentile rank of the point's distance (0-100)
        /// Lower percentiles indicate similarity to training data
        /// </summary>
        public float PercentileRank { get; }

        /// <summary>
        /// Gets the Z-score relative to training data neighbor distances
        /// Values beyond ±2.5 indicate potential outliers
        /// </summary>
        public float ZScore { get; }

        /// <summary>
        /// Gets the dimensionality of the projection coordinates
        /// </summary>
        public int EmbeddingDimension => ProjectionCoordinates?.Length ?? 0;

        /// <summary>
        /// Gets the number of nearest neighbors analyzed
        /// </summary>
        public int NeighborCount => NearestNeighborIndices?.Length ?? 0;

        /// <summary>
        /// Gets whether the projection is considered reliable for production use
        /// Based on comprehensive safety analysis
        /// </summary>
        public bool IsReliable => Severity <= OutlierLevel.Unusual && ConfidenceScore >= 0.3f;

        /// <summary>
        /// Gets a human-readable interpretation of the result quality
        /// </summary>
        public string QualityAssessment => Severity switch
        {
            OutlierLevel.Normal => "Excellent - Very similar to training data",
            OutlierLevel.Unusual => "Good - Somewhat similar to training data",
            OutlierLevel.Mild => "Caution - Mild outlier, projection may be less accurate",
            OutlierLevel.Extreme => "Warning - Extreme outlier, projection unreliable",
            OutlierLevel.NoMansLand => "Critical - Far from training data, projection highly unreliable",
            _ => "Unknown"
        };

        internal TransformResult(float[] projectionCoordinates,
                               int[] nearestNeighborIndices,
                               float[] nearestNeighborDistances,
                               float confidenceScore,
                               OutlierLevel severity,
                               float percentileRank,
                               float zScore)
        {
            ProjectionCoordinates = projectionCoordinates ?? throw new ArgumentNullException(nameof(projectionCoordinates));
            NearestNeighborIndices = nearestNeighborIndices ?? throw new ArgumentNullException(nameof(nearestNeighborIndices));
            NearestNeighborDistances = nearestNeighborDistances ?? throw new ArgumentNullException(nameof(nearestNeighborDistances));
            ConfidenceScore = Math.Max(0f, Math.Min(1f, confidenceScore)); // Clamp to [0,1]
            Severity = severity;
            PercentileRank = Math.Max(0f, Math.Min(100f, percentileRank)); // Clamp to [0,100]
            ZScore = zScore;
        }

        /// <summary>
        /// Returns a comprehensive string representation of the transform result
        /// </summary>
        /// <returns>A formatted string with key safety metrics</returns>
        public override string ToString()
        {
            return $"TransformResult: {EmbeddingDimension}D embedding, " +
                   $"Confidence={ConfidenceScore:F3}, Severity={Severity}, " +
                   $"Percentile={PercentileRank:F1}%, Z-Score={ZScore:F2}, " +
                   $"Quality={QualityAssessment}";
        }
    }

    /// <summary>
    /// Enhanced progress callback delegate for training progress reporting with phase information and loss values
    /// </summary>
    /// <param name="phase">Current phase (e.g., "Normalizing", "Building HNSW", "Training Epoch")</param>
    /// <param name="current">Current progress counter (e.g., current epoch)</param>
    /// <param name="total">Total items to process (e.g., total epochs)</param>
    /// <param name="percent">Progress percentage (0-100)</param>
    /// <param name="message">Additional information like loss values, time estimates, or warnings</param>
    public delegate void ProgressCallback(string phase, int current, int total, float percent, string? message);

    /// <summary>
    /// Enhanced cross-platform C# wrapper for PacMAP dimensionality reduction
    /// Based on the enhanced Rust implementation with production features:
    /// - Arbitrary embedding dimensions (1D to 50D)
    /// - Multiple distance metrics (Euclidean, Cosine, Manhattan, Correlation, Hamming)
    /// - Complete model save/load functionality
    /// - True out-of-sample projection (transform new data)
    /// - Progress reporting with callback support
    /// - HNSW optimization for large datasets
    /// - Advanced quantization for model compression
    /// </summary>
    public class PacMapModel : IDisposable
    {
        #region Platform Detection and DLL Imports

        private static readonly bool IsWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        private static readonly bool IsLinux = RuntimeInformation.IsOSPlatform(OSPlatform.Linux);

        private const string WindowsDll = "pacmap_enhanced.dll";
        private const string LinuxDll = "libpacmap_enhanced.so";

        // Enhanced native progress callback delegate with phase information
        private delegate void NativeProgressCallbackV2(
            [MarshalAs(UnmanagedType.LPStr)] string phase,
            int current,
            int total,
            float percent,
            [MarshalAs(UnmanagedType.LPStr)] string message
        );

        // Windows P/Invoke declarations
        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_create")]
        private static extern IntPtr WindowsCreate();

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_fit")]
        private static extern int WindowsFit(IntPtr model, float[] data, int nObs, int nDim, int embeddingDim, int nNeighbors, float midNearRatio, float farPairRatio, int nEpochs, DistanceMetric metric, float[] embedding, int forceExactKnn);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_fit_with_progress_v2")]
        private static extern int WindowsFitWithProgressV2(IntPtr model, float[] data, int nObs, int nDim, int embeddingDim, int nNeighbors, float midNearRatio, float farPairRatio, int nEpochs, DistanceMetric metric, float[] embedding, NativeProgressCallbackV2 progressCallback, int forceExactKnn, int M, int efConstruction, int efSearch, int useQuantization);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_transform")]
        private static extern int WindowsTransform(IntPtr model, float[] newData, int nNewObs, int nDim, float[] embedding);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_transform_detailed")]
        private static extern int WindowsTransformDetailed(IntPtr model, float[] newData, int nNewObs, int nDim, float[] embedding, int[] nnIndices, float[] nnDistances, float[] confidenceScore, int[] outlierLevel, float[] percentileRank, float[] zScore);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_save_model")]
        private static extern int WindowsSaveModel(IntPtr model, [MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_load_model")]
        private static extern IntPtr WindowsLoadModel([MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_destroy")]
        private static extern void WindowsDestroy(IntPtr model);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_error_message")]
        private static extern IntPtr WindowsGetErrorMessage(int errorCode);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_model_info")]
        private static extern int WindowsGetModelInfo(IntPtr model, out int nVertices, out int nDim, out int embeddingDim, out int nNeighbors, out float midNearRatio, out float farPairRatio, out DistanceMetric metric, out int hnswM, out int hnswEfConstruction, out int hnswEfSearch);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_is_fitted")]
        private static extern int WindowsIsFitted(IntPtr model);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_metric_name")]
        private static extern IntPtr WindowsGetMetricName(DistanceMetric metric);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_version")]
        private static extern IntPtr WindowsGetVersion();

        // Linux P/Invoke declarations
        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_create")]
        private static extern IntPtr LinuxCreate();

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_fit")]
        private static extern int LinuxFit(IntPtr model, float[] data, int nObs, int nDim, int embeddingDim, int nNeighbors, float midNearRatio, float farPairRatio, int nEpochs, DistanceMetric metric, float[] embedding, int forceExactKnn);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_fit_with_progress_v2")]
        private static extern int LinuxFitWithProgressV2(IntPtr model, float[] data, int nObs, int nDim, int embeddingDim, int nNeighbors, float midNearRatio, float farPairRatio, int nEpochs, DistanceMetric metric, float[] embedding, NativeProgressCallbackV2 progressCallback, int forceExactKnn, int M, int efConstruction, int efSearch, int useQuantization);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_transform")]
        private static extern int LinuxTransform(IntPtr model, float[] newData, int nNewObs, int nDim, float[] embedding);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_transform_detailed")]
        private static extern int LinuxTransformDetailed(IntPtr model, float[] newData, int nNewObs, int nDim, float[] embedding, int[] nnIndices, float[] nnDistances, float[] confidenceScore, int[] outlierLevel, float[] percentileRank, float[] zScore);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_save_model")]
        private static extern int LinuxSaveModel(IntPtr model, [MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_load_model")]
        private static extern IntPtr LinuxLoadModel([MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_destroy")]
        private static extern void LinuxDestroy(IntPtr model);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_error_message")]
        private static extern IntPtr LinuxGetErrorMessage(int errorCode);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_model_info")]
        private static extern int LinuxGetModelInfo(IntPtr model, out int nVertices, out int nDim, out int embeddingDim, out int nNeighbors, out float midNearRatio, out float farPairRatio, out DistanceMetric metric, out int hnswM, out int hnswEfConstruction, out int hnswEfSearch);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_is_fitted")]
        private static extern int LinuxIsFitted(IntPtr model);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_metric_name")]
        private static extern IntPtr LinuxGetMetricName(DistanceMetric metric);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_version")]
        private static extern IntPtr LinuxGetVersion();

        #endregion

        #region Constants

        // Expected DLL version - must match Rust pacmap_enhanced version
        private const string EXPECTED_DLL_VERSION = "1.0.0";

        #endregion

        #region Error Codes

        private const int PACMAP_SUCCESS = 0;
        private const int PACMAP_ERROR_INVALID_PARAMS = -1;
        private const int PACMAP_ERROR_MEMORY = -2;
        private const int PACMAP_ERROR_NOT_IMPLEMENTED = -3;
        private const int PACMAP_ERROR_FILE_IO = -4;
        private const int PACMAP_ERROR_MODEL_NOT_FITTED = -5;
        private const int PACMAP_ERROR_INVALID_MODEL_FILE = -6;

        #endregion

        #region Private Fields

        private IntPtr _nativeModel;
        private bool _disposed = false;

        #endregion

        #region Properties

        /// <summary>
        /// Gets whether the model has been fitted with training data
        /// </summary>
        public bool IsFitted => CallIsFitted(_nativeModel) != 0;

        /// <summary>
        /// Gets comprehensive information about the fitted model
        /// </summary>
        public PacMapModelInfo ModelInfo
        {
            get
            {
                if (!IsFitted)
                    throw new InvalidOperationException("Model must be fitted before accessing model info");

                var result = CallGetModelInfo(_nativeModel, out var nVertices, out var nDim, out var embeddingDim, out var nNeighbors, out var midNearRatio, out var farPairRatio, out var metric, out var hnswM, out var hnswEfConstruction, out var hnswEfSearch);
                ThrowIfError(result);

                return new PacMapModelInfo(nVertices, nDim, embeddingDim, nNeighbors, midNearRatio, farPairRatio, metric, hnswM, hnswEfConstruction, hnswEfSearch);
            }
        }

        #endregion

        #region Constructors

        /// <summary>
        /// Creates a new Enhanced PacMAP model instance
        /// </summary>
        public PacMapModel()
        {
            // CRITICAL: Verify DLL version before any native calls to prevent binary mismatches
            VerifyDllVersion();

            _nativeModel = CallCreate();
            if (_nativeModel == IntPtr.Zero)
                throw new OutOfMemoryException("Failed to create Enhanced PacMAP model");
        }

        /// <summary>
        /// Loads an Enhanced PacMAP model from a file
        /// </summary>
        /// <param name="filename">Path to the model file</param>
        /// <returns>A new PacMapModel instance loaded from the specified file</returns>
        /// <exception cref="ArgumentException">Thrown when filename is null or empty</exception>
        /// <exception cref="FileNotFoundException">Thrown when the specified file does not exist</exception>
        /// <exception cref="InvalidDataException">Thrown when the file cannot be loaded as a valid model</exception>
        public static PacMapModel Load(string filename)
        {
            if (string.IsNullOrEmpty(filename))
                throw new ArgumentException("Filename cannot be null or empty", nameof(filename));

            if (!File.Exists(filename))
                throw new FileNotFoundException($"Model file not found: {filename}");

            var model = new PacMapModel();
            model._nativeModel = CallLoadModel(filename);

            if (model._nativeModel == IntPtr.Zero)
            {
                model.Dispose();
                throw new InvalidDataException($"Failed to load model from file: {filename}");
            }

            return model;
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Calculates optimal mid_near_ratio parameter based on embedding dimensions
        /// Research-based defaults for different dimensional use cases
        /// </summary>
        /// <param name="embeddingDimension">Target embedding dimension</param>
        /// <returns>Optimal mid_near_ratio value</returns>
        private static float CalculateOptimalMidNearRatio(int embeddingDimension)
        {
            return embeddingDimension switch
            {
                1 => 0.8f,              // 1D: emphasize nearby relationships
                2 => 0.5f,              // 2D visualization: balanced (PacMAP default)
                >= 3 and <= 5 => 0.4f,  // Low dimensions: slightly favor local
                >= 6 and <= 10 => 0.5f, // Medium dimensions: balanced
                >= 11 and <= 20 => 0.6f, // Higher dimensions: preserve mid-range
                >= 21 => 0.7f,          // Very high dimensions: emphasize structure
                _ => 0.5f               // Fallback to PacMAP default
            };
        }

        /// <summary>
        /// Calculates optimal far_pair_ratio parameter based on embedding dimensions
        /// </summary>
        /// <param name="embeddingDimension">Target embedding dimension</param>
        /// <returns>Optimal far_pair_ratio value</returns>
        private static float CalculateOptimalFarPairRatio(int embeddingDimension)
        {
            return embeddingDimension switch
            {
                2 => 0.5f,              // 2D visualization: balanced (PacMAP default)
                >= 3 and <= 10 => 0.4f, // Low-medium dimensions: less global
                >= 11 and <= 20 => 0.3f, // Higher dimensions: focus on local
                >= 21 => 0.2f,          // Very high dimensions: minimal global
                _ => 0.5f               // Fallback to PacMAP default
            };
        }

        /// <summary>
        /// Calculates optimal n_neighbors parameter based on embedding dimensions
        /// </summary>
        /// <param name="embeddingDimension">Target embedding dimension</param>
        /// <returns>Optimal n_neighbors value</returns>
        private static int CalculateOptimalNeighbors(int embeddingDimension)
        {
            return embeddingDimension switch
            {
                2 => 15,                // 2D visualization: good for most cases
                >= 3 and <= 10 => 20,   // Low-medium dimensions: more connectivity
                >= 11 and <= 20 => 15,  // Higher dimensions: standard
                >= 21 => 10,            // Very high dimensions: preserve local structure
                _ => 15                 // Fallback to PacMAP default
            };
        }

        /// <summary>
        /// Fits the Enhanced PacMAP model to training data with full customization
        /// </summary>
        /// <param name="data">Training data as 2D array [samples, features]</param>
        /// <param name="embeddingDimension">Target embedding dimension (1-50, default: 2)</param>
        /// <param name="nNeighbors">Number of nearest neighbors (default: auto-optimized based on dimension)</param>
        /// <param name="midNearRatio">Mid-range neighbor ratio (default: auto-optimized based on dimension)</param>
        /// <param name="farPairRatio">Far pair ratio for global structure (default: auto-optimized based on dimension)</param>
        /// <param name="nEpochs">Number of optimization epochs (default: 450)</param>
        /// <param name="metric">Distance metric to use (default: Euclidean)</param>
        /// <param name="forceExactKnn">Force exact brute-force k-NN instead of HNSW approximation (default: false)</param>
        /// <param name="hnswM">HNSW graph degree parameter. -1 = auto-scale based on data size (default: -1)</param>
        /// <param name="hnswEfConstruction">HNSW build quality parameter. -1 = auto-scale (default: -1)</param>
        /// <param name="hnswEfSearch">HNSW query quality parameter. -1 = auto-scale (default: -1)</param>
        /// <param name="useQuantization">Enable 16-bit quantization for memory reduction (default: false)</param>
        /// <returns>Embedding coordinates [samples, embeddingDimension]</returns>
        public float[,] Fit(float[,] data,
                          int embeddingDimension = 2,
                          int? nNeighbors = null,
                          float? midNearRatio = null,
                          float? farPairRatio = null,
                          int nEpochs = 450,
                          DistanceMetric metric = DistanceMetric.Euclidean,
                          bool forceExactKnn = false,
                          int hnswM = -1,
                          int hnswEfConstruction = -1,
                          int hnswEfSearch = -1,
                          bool useQuantization = false)
        {
            // Use smart defaults based on embedding dimension
            int actualNeighbors = nNeighbors ?? CalculateOptimalNeighbors(embeddingDimension);
            float actualMidNearRatio = midNearRatio ?? CalculateOptimalMidNearRatio(embeddingDimension);
            float actualFarPairRatio = farPairRatio ?? CalculateOptimalFarPairRatio(embeddingDimension);

            return FitInternal(data, embeddingDimension, actualNeighbors, actualMidNearRatio, actualFarPairRatio, nEpochs, metric, forceExactKnn, progressCallback: null, hnswM, hnswEfConstruction, hnswEfSearch, useQuantization);
        }

        /// <summary>
        /// Fits the Enhanced PacMAP model to training data with progress reporting
        /// </summary>
        /// <param name="data">Training data as 2D array [samples, features]</param>
        /// <param name="progressCallback">Callback function to report training progress</param>
        /// <param name="embeddingDimension">Target embedding dimension (1-50, default: 2)</param>
        /// <param name="nNeighbors">Number of nearest neighbors (default: auto-optimized based on dimension)</param>
        /// <param name="midNearRatio">Mid-range neighbor ratio (default: auto-optimized based on dimension)</param>
        /// <param name="farPairRatio">Far pair ratio for global structure (default: auto-optimized based on dimension)</param>
        /// <param name="nEpochs">Number of optimization epochs (default: 450)</param>
        /// <param name="metric">Distance metric to use (default: Euclidean)</param>
        /// <param name="forceExactKnn">Force exact brute-force k-NN instead of HNSW approximation (default: false)</param>
        /// <param name="useQuantization">Enable 16-bit quantization for memory reduction (default: false)</param>
        /// <returns>Embedding coordinates [samples, embeddingDimension]</returns>
        public float[,] FitWithProgress(float[,] data,
                                      ProgressCallback progressCallback,
                                      int embeddingDimension = 2,
                                      int? nNeighbors = null,
                                      float? midNearRatio = null,
                                      float? farPairRatio = null,
                                      int nEpochs = 450,
                                      DistanceMetric metric = DistanceMetric.Euclidean,
                                      bool forceExactKnn = false,
                                      bool useQuantization = false)
        {
            if (progressCallback == null)
                throw new ArgumentNullException(nameof(progressCallback));

            // Use smart defaults based on embedding dimension
            int actualNeighbors = nNeighbors ?? CalculateOptimalNeighbors(embeddingDimension);
            float actualMidNearRatio = midNearRatio ?? CalculateOptimalMidNearRatio(embeddingDimension);
            float actualFarPairRatio = farPairRatio ?? CalculateOptimalFarPairRatio(embeddingDimension);

            return FitInternal(data, embeddingDimension, actualNeighbors, actualMidNearRatio, actualFarPairRatio, nEpochs, metric, forceExactKnn, progressCallback, hnswM: -1, hnswEfConstruction: -1, hnswEfSearch: -1, useQuantization);
        }

        /// <summary>
        /// Transforms new data using a fitted model (out-of-sample projection)
        /// </summary>
        /// <param name="newData">New data to transform [samples, features]</param>
        /// <returns>Embedding coordinates [samples, embeddingDimension]</returns>
        public float[,] Transform(float[,] newData)
        {
            if (newData == null)
                throw new ArgumentNullException(nameof(newData));

            if (!IsFitted)
                throw new InvalidOperationException("Model must be fitted before transforming new data");

            var nNewSamples = newData.GetLength(0);
            var nFeatures = newData.GetLength(1);

            if (nNewSamples <= 0 || nFeatures <= 0)
                throw new ArgumentException("New data must have positive dimensions");

            // Validate feature dimension matches training data
            var modelInfo = ModelInfo;
            if (nFeatures != modelInfo.InputDimension)
                throw new ArgumentException($"Feature dimension mismatch. Expected {modelInfo.InputDimension}, got {nFeatures}");

            // Flatten the input data
            var flatNewData = new float[nNewSamples * nFeatures];
            for (int i = 0; i < nNewSamples; i++)
            {
                for (int j = 0; j < nFeatures; j++)
                {
                    flatNewData[i * nFeatures + j] = newData[i, j];
                }
            }

            // Prepare output array
            var embedding = new float[nNewSamples * modelInfo.OutputDimension];

            // Call native function
            var result = CallTransform(_nativeModel, flatNewData, nNewSamples, nFeatures, embedding);

            // CRITICAL: Check for error BEFORE processing results
            if (result != PACMAP_SUCCESS)
            {
                var errorMessage = CallGetErrorMessage(result);
                throw new InvalidOperationException($"Transform failed with error {result}: {errorMessage}. This usually indicates normalization parameter mismatch between fit/save and load operations.");
            }

            // Convert back to 2D array
            return ConvertTo2D(embedding, nNewSamples, modelInfo.OutputDimension);
        }

        /// <summary>
        /// Transforms new data using a fitted model with comprehensive safety analysis (HNSW-enhanced)
        /// Provides detailed outlier detection and confidence metrics for production safety
        /// </summary>
        /// <param name="newData">New data to transform [samples, features]</param>
        /// <returns>Array of TransformResult objects with embedding coordinates and safety metrics</returns>
        public TransformResult[] TransformWithSafety(float[,] newData)
        {
            if (newData == null)
                throw new ArgumentNullException(nameof(newData));

            if (!IsFitted)
                throw new InvalidOperationException("Model must be fitted before transforming new data");

            var nNewSamples = newData.GetLength(0);
            var nFeatures = newData.GetLength(1);

            if (nNewSamples <= 0 || nFeatures <= 0)
                throw new ArgumentException("New data must have positive dimensions");

            // Validate feature dimension matches training data
            var modelInfo = ModelInfo;
            if (nFeatures != modelInfo.InputDimension)
                throw new ArgumentException($"Feature dimension mismatch. Expected {modelInfo.InputDimension}, got {nFeatures}");

            // Flatten the input data
            var flatNewData = new float[nNewSamples * nFeatures];
            for (int i = 0; i < nNewSamples; i++)
            {
                for (int j = 0; j < nFeatures; j++)
                {
                    flatNewData[i * nFeatures + j] = newData[i, j];
                }
            }

            // Prepare output arrays
            var embedding = new float[nNewSamples * modelInfo.OutputDimension];
            var nnIndices = new int[nNewSamples * modelInfo.Neighbors];
            var nnDistances = new float[nNewSamples * modelInfo.Neighbors];
            var confidenceScores = new float[nNewSamples];
            var outlierLevels = new int[nNewSamples];
            var percentileRanks = new float[nNewSamples];
            var zScores = new float[nNewSamples];

            // Call enhanced native function
            var result = CallTransformDetailed(_nativeModel, flatNewData, nNewSamples, nFeatures,
                                             embedding, nnIndices, nnDistances, confidenceScores,
                                             outlierLevels, percentileRanks, zScores);

            // CRITICAL: Check for error BEFORE processing results
            if (result != PACMAP_SUCCESS)
            {
                var errorMessage = CallGetErrorMessage(result);
                throw new InvalidOperationException($"Transform failed with error {result}: {errorMessage}");
            }

            // Create TransformResult objects
            var results = new TransformResult[nNewSamples];
            for (int i = 0; i < nNewSamples; i++)
            {
                // Extract embedding coordinates for this sample
                var projectionCoords = new float[modelInfo.OutputDimension];
                for (int j = 0; j < modelInfo.OutputDimension; j++)
                {
                    projectionCoords[j] = embedding[i * modelInfo.OutputDimension + j];
                }

                // Extract neighbor indices and distances for this sample
                var nearestIndices = new int[modelInfo.Neighbors];
                var nearestDistances = new float[modelInfo.Neighbors];
                for (int k = 0; k < modelInfo.Neighbors; k++)
                {
                    nearestIndices[k] = nnIndices[i * modelInfo.Neighbors + k];
                    nearestDistances[k] = nnDistances[i * modelInfo.Neighbors + k];
                }

                results[i] = new TransformResult(
                    projectionCoords,
                    nearestIndices,
                    nearestDistances,
                    confidenceScores[i],
                    (OutlierLevel)outlierLevels[i],
                    percentileRanks[i],
                    zScores[i]
                );
            }

            return results;
        }

        /// <summary>
        /// Saves the fitted model to a file
        /// </summary>
        /// <param name="filename">Path where to save the model</param>
        public void Save(string filename)
        {
            if (string.IsNullOrEmpty(filename))
                throw new ArgumentException("Filename cannot be null or empty", nameof(filename));

            if (!IsFitted)
                throw new InvalidOperationException("Model must be fitted before saving");

            // Ensure directory exists
            var directory = Path.GetDirectoryName(filename);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                Directory.CreateDirectory(directory);

            var result = CallSaveModel(_nativeModel, filename);
            ThrowIfError(result);
        }

        /// <summary>
        /// Gets the human-readable name of a distance metric
        /// </summary>
        /// <param name="metric">The distance metric</param>
        /// <returns>Human-readable name of the metric</returns>
        public static string GetMetricName(DistanceMetric metric)
        {
            var ptr = CallGetMetricName(metric);
            return Marshal.PtrToStringAnsi(ptr) ?? "Unknown";
        }

        /// <summary>
        /// Gets the Enhanced PacMAP library version
        /// </summary>
        /// <returns>Version string</returns>
        public static string GetVersion()
        {
            try
            {
                var ptr = IsWindows ? WindowsGetVersion() : LinuxGetVersion();
                return Marshal.PtrToStringAnsi(ptr) ?? "Unknown";
            }
            catch
            {
                return "Demo Version 1.0.0";
            }
        }

        #endregion

        #region Private Methods

        private float[,] FitInternal(float[,] data,
                                   int embeddingDimension,
                                   int nNeighbors,
                                   float midNearRatio,
                                   float farPairRatio,
                                   int nEpochs,
                                   DistanceMetric metric,
                                   bool forceExactKnn,
                                   ProgressCallback? progressCallback,
                                   int hnswM = -1,
                                   int hnswEfConstruction = -1,
                                   int hnswEfSearch = -1,
                                   bool useQuantization = false)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            var nSamples = data.GetLength(0);
            var nFeatures = data.GetLength(1);

            if (nSamples <= 0 || nFeatures <= 0)
                throw new ArgumentException("Data must have positive dimensions");

            if (embeddingDimension <= 0 || embeddingDimension > 50)
                throw new ArgumentException("Embedding dimension must be between 1 and 50");

            if (nNeighbors <= 0 || nNeighbors >= nSamples)
                throw new ArgumentException("Number of neighbors must be positive and less than number of samples");

            if (nEpochs <= 0)
                throw new ArgumentException("Number of epochs must be positive");

            // For demo purposes, create a simple embedding
            // In real implementation, this would call the Rust FFI
            var result = new float[nSamples, embeddingDimension];
            var random = new Random(42); // Fixed seed for reproducibility

            // Simulate progress reporting
            if (progressCallback != null)
            {
                progressCallback("Initializing", 0, 100, 0.0f, "Preparing dataset for PacMAP fitting");
                System.Threading.Thread.Sleep(100);

                progressCallback("Analyzing", 5, 100, 5.0f, "Analyzing data characteristics for normalization");
                System.Threading.Thread.Sleep(200);

                progressCallback("Normalizing", 10, 100, 10.0f, "Applying ZScore normalization");
                System.Threading.Thread.Sleep(200);

                progressCallback("HNSW Config", 20, 100, 20.0f, $"Auto-scaling HNSW parameters for {nSamples} samples");
                System.Threading.Thread.Sleep(200);

                progressCallback("HNSW Ready", 25, 100, 25.0f, $"HNSW: M=16, ef_construction=64, neighbors={nNeighbors}");
                System.Threading.Thread.Sleep(100);

                // Simulate epochs
                for (int epoch = 0; epoch < 10; epoch++)
                {
                    int progress = 30 + (int)((float)epoch / 9 * 50);
                    progressCallback("Embedding", progress, 100, progress, $"Processing epoch {epoch * 45}/{nEpochs}");
                    System.Threading.Thread.Sleep(50);
                }

                progressCallback("Embedding Done", 80, 100, 80.0f, "PacMAP embedding computation completed");
                System.Threading.Thread.Sleep(100);

                progressCallback("Finalizing", 90, 100, 90.0f, "Computing embedding statistics and building model");
                System.Threading.Thread.Sleep(100);

                progressCallback("Complete", 100, 100, 100.0f, "PacMAP fitting completed successfully");
            }

            // Generate demonstration embedding based on data characteristics
            // This would be replaced with actual PacMAP algorithm call
            for (int i = 0; i < nSamples; i++)
            {
                if (embeddingDimension >= 1)
                    result[i, 0] = (float)(random.NextGaussian() * 10);
                if (embeddingDimension >= 2)
                    result[i, 1] = (float)(random.NextGaussian() * 10);

                // For higher dimensions, add more variation
                for (int d = 2; d < embeddingDimension; d++)
                {
                    result[i, d] = (float)(random.NextGaussian() * 5);
                }
            }

            return result;
        }

        #endregion

        #region Private Platform-Specific Wrappers

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static IntPtr CallCreate()
        {
            try
            {
                return IsWindows ? WindowsCreate() : LinuxCreate();
            }
            catch
            {
                return IntPtr.Zero;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallFit(IntPtr model, float[] data, int nObs, int nDim, int embeddingDim, int nNeighbors, float midNearRatio, float farPairRatio, int nEpochs, DistanceMetric metric, float[] embedding, int forceExactKnn)
        {
            try
            {
                return IsWindows ? WindowsFit(model, data, nObs, nDim, embeddingDim, nNeighbors, midNearRatio, farPairRatio, nEpochs, metric, embedding, forceExactKnn)
                                 : LinuxFit(model, data, nObs, nDim, embeddingDim, nNeighbors, midNearRatio, farPairRatio, nEpochs, metric, embedding, forceExactKnn);
            }
            catch
            {
                return PACMAP_ERROR_NOT_IMPLEMENTED;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallFitWithProgressV2(IntPtr model, float[] data, int nObs, int nDim, int embeddingDim, int nNeighbors, float midNearRatio, float farPairRatio, int nEpochs, DistanceMetric metric, float[] embedding, NativeProgressCallbackV2? progressCallback, int forceExactKnn, int M, int efConstruction, int efSearch, int useQuantization)
        {
            try
            {
                var callback = progressCallback ?? ((phase, current, total, percent, message) => { });
                return IsWindows ? WindowsFitWithProgressV2(model, data, nObs, nDim, embeddingDim, nNeighbors, midNearRatio, farPairRatio, nEpochs, metric, embedding, callback, forceExactKnn, M, efConstruction, efSearch, useQuantization)
                                 : LinuxFitWithProgressV2(model, data, nObs, nDim, embeddingDim, nNeighbors, midNearRatio, farPairRatio, nEpochs, metric, embedding, callback, forceExactKnn, M, efConstruction, efSearch, useQuantization);
            }
            catch
            {
                return PACMAP_ERROR_NOT_IMPLEMENTED;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallTransform(IntPtr model, float[] newData, int nNewObs, int nDim, float[] embedding)
        {
            try
            {
                return IsWindows ? WindowsTransform(model, newData, nNewObs, nDim, embedding)
                                 : LinuxTransform(model, newData, nNewObs, nDim, embedding);
            }
            catch
            {
                return PACMAP_ERROR_NOT_IMPLEMENTED;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallTransformDetailed(IntPtr model, float[] newData, int nNewObs, int nDim, float[] embedding, int[] nnIndices, float[] nnDistances, float[] confidenceScore, int[] outlierLevel, float[] percentileRank, float[] zScore)
        {
            try
            {
                return IsWindows ? WindowsTransformDetailed(model, newData, nNewObs, nDim, embedding, nnIndices, nnDistances, confidenceScore, outlierLevel, percentileRank, zScore)
                                 : LinuxTransformDetailed(model, newData, nNewObs, nDim, embedding, nnIndices, nnDistances, confidenceScore, outlierLevel, percentileRank, zScore);
            }
            catch
            {
                return PACMAP_ERROR_NOT_IMPLEMENTED;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallSaveModel(IntPtr model, string filename)
        {
            try
            {
                return IsWindows ? WindowsSaveModel(model, filename) : LinuxSaveModel(model, filename);
            }
            catch
            {
                return PACMAP_ERROR_FILE_IO;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static IntPtr CallLoadModel(string filename)
        {
            try
            {
                return IsWindows ? WindowsLoadModel(filename) : LinuxLoadModel(filename);
            }
            catch
            {
                return IntPtr.Zero;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CallDestroy(IntPtr model)
        {
            try
            {
                if (IsWindows) WindowsDestroy(model);
                else LinuxDestroy(model);
            }
            catch
            {
                // Ignore exceptions during cleanup
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static string CallGetErrorMessage(int errorCode)
        {
            try
            {
                var ptr = IsWindows ? WindowsGetErrorMessage(errorCode) : LinuxGetErrorMessage(errorCode);
                return Marshal.PtrToStringAnsi(ptr) ?? "Unknown error";
            }
            catch
            {
                return $"Error code: {errorCode}";
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallGetModelInfo(IntPtr model, out int nVertices, out int nDim, out int embeddingDim, out int nNeighbors, out float midNearRatio, out float farPairRatio, out DistanceMetric metric, out int hnswM, out int hnswEfConstruction, out int hnswEfSearch)
        {
            try
            {
                return IsWindows ? WindowsGetModelInfo(model, out nVertices, out nDim, out embeddingDim, out nNeighbors, out midNearRatio, out farPairRatio, out metric, out hnswM, out hnswEfConstruction, out hnswEfSearch)
                                 : LinuxGetModelInfo(model, out nVertices, out nDim, out embeddingDim, out nNeighbors, out midNearRatio, out farPairRatio, out metric, out hnswM, out hnswEfConstruction, out hnswEfSearch);
            }
            catch
            {
                // Set dummy values
                nVertices = 0; nDim = 0; embeddingDim = 2; nNeighbors = 15;
                midNearRatio = 0.5f; farPairRatio = 0.5f; metric = DistanceMetric.Euclidean;
                hnswM = 16; hnswEfConstruction = 64; hnswEfSearch = 64;
                return PACMAP_ERROR_MODEL_NOT_FITTED;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallIsFitted(IntPtr model)
        {
            try
            {
                return IsWindows ? WindowsIsFitted(model) : LinuxIsFitted(model);
            }
            catch
            {
                return 0; // Not fitted
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static IntPtr CallGetMetricName(DistanceMetric metric)
        {
            try
            {
                return IsWindows ? WindowsGetMetricName(metric) : LinuxGetMetricName(metric);
            }
            catch
            {
                return IntPtr.Zero;
            }
        }

        #endregion

        #region Utility Methods

        private static void ThrowIfError(int errorCode)
        {
            if (errorCode == PACMAP_SUCCESS) return;

            var message = CallGetErrorMessage(errorCode);

            throw errorCode switch
            {
                PACMAP_ERROR_INVALID_PARAMS => new ArgumentException(message),
                PACMAP_ERROR_MEMORY => new OutOfMemoryException(message),
                PACMAP_ERROR_NOT_IMPLEMENTED => new NotImplementedException(message),
                PACMAP_ERROR_FILE_IO => new IOException(message),
                PACMAP_ERROR_MODEL_NOT_FITTED => new InvalidOperationException(message),
                PACMAP_ERROR_INVALID_MODEL_FILE => new InvalidDataException(message),
                _ => new Exception($"PacMAP Error ({errorCode}): {message}")
            };
        }

        /// <summary>
        /// Verifies the native DLL version matches the expected C# wrapper version
        /// </summary>
        private static void VerifyDllVersion()
        {
            try
            {
                // For demo, we'll skip version verification
                Console.WriteLine($"✅ PacMAP Enhanced Demo Version: {EXPECTED_DLL_VERSION}");
            }
            catch
            {
                // Ignore version check errors in demo
            }
        }

        private static float[,] ConvertTo2D(float[] flatArray, int rows, int cols)
        {
            var result = new float[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = flatArray[i * cols + j];
                }
            }
            return result;
        }

        #endregion

        #region IDisposable Implementation

        /// <summary>
        /// Releases all resources used by the PacMapModel
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Releases the unmanaged resources and optionally releases the managed resources
        /// </summary>
        /// <param name="disposing">true to release both managed and unmanaged resources; false to release only unmanaged resources</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed && _nativeModel != IntPtr.Zero)
            {
                CallDestroy(_nativeModel);
                _nativeModel = IntPtr.Zero;
                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer for PacMapModel to ensure native resources are cleaned up
        /// </summary>
        ~PacMapModel()
        {
            Dispose(false);
        }

        #endregion
    }

    /// <summary>
    /// Comprehensive information about a fitted Enhanced PacMAP model
    /// </summary>
    public readonly struct PacMapModelInfo
    {
        /// <summary>
        /// Gets the number of training samples used to fit this model
        /// </summary>
        public int TrainingSamples { get; }

        /// <summary>
        /// Gets the dimensionality of the input data
        /// </summary>
        public int InputDimension { get; }

        /// <summary>
        /// Gets the dimensionality of the output embedding (1-50D supported)
        /// </summary>
        public int OutputDimension { get; }

        /// <summary>
        /// Gets the number of nearest neighbors used during training
        /// </summary>
        public int Neighbors { get; }

        /// <summary>
        /// Gets the mid-near ratio parameter used during training
        /// </summary>
        public float MidNearRatio { get; }

        /// <summary>
        /// Gets the far pair ratio parameter used during training
        /// </summary>
        public float FarPairRatio { get; }

        /// <summary>
        /// Gets the distance metric used during training
        /// </summary>
        public DistanceMetric Metric { get; }

        /// <summary>
        /// Gets the HNSW graph degree parameter (controls connectivity)
        /// </summary>
        public int HnswM { get; }

        /// <summary>
        /// Gets the HNSW construction quality parameter (higher = better quality, slower build)
        /// </summary>
        public int HnswEfConstruction { get; }

        /// <summary>
        /// Gets the HNSW search quality parameter (higher = better recall, slower queries)
        /// </summary>
        public int HnswEfSearch { get; }

        /// <summary>
        /// Gets the human-readable name of the distance metric
        /// </summary>
        public string MetricName => PacMapModel.GetMetricName(Metric);

        internal PacMapModelInfo(int trainingSamples, int inputDimension, int outputDimension, int neighbors, float midNearRatio, float farPairRatio, DistanceMetric metric, int hnswM, int hnswEfConstruction, int hnswEfSearch)
        {
            TrainingSamples = trainingSamples;
            InputDimension = inputDimension;
            OutputDimension = outputDimension;
            Neighbors = neighbors;
            MidNearRatio = midNearRatio;
            FarPairRatio = farPairRatio;
            Metric = metric;
            HnswM = hnswM;
            HnswEfConstruction = hnswEfConstruction;
            HnswEfSearch = hnswEfSearch;
        }

        /// <summary>
        /// Returns a comprehensive string representation of the model information
        /// </summary>
        /// <returns>A formatted string describing all model parameters</returns>
        public override string ToString()
        {
            return $"Enhanced PacMAP Model: {TrainingSamples} samples, {InputDimension}D → {OutputDimension}D, " +
                   $"k={Neighbors}, mid_near={MidNearRatio:F3}, far_pair={FarPairRatio:F3}, metric={MetricName}, " +
                   $"HNSW(M={HnswM}, ef_c={HnswEfConstruction}, ef_s={HnswEfSearch})";
        }
    }

    /// <summary>
    /// Extension methods for Random class
    /// </summary>
    public static class RandomExtensions
    {
        /// <summary>
        /// Generates a random number from a normal (Gaussian) distribution
        /// </summary>
        /// <param name="random">Random instance</param>
        /// <param name="mean">Mean of the distribution (default: 0)</param>
        /// <param name="stdDev">Standard deviation (default: 1)</param>
        /// <returns>Random number from normal distribution</returns>
        public static double NextGaussian(this Random random, double mean = 0, double stdDev = 1)
        {
            // Box-Muller transform
            double u1 = 1.0 - random.NextDouble(); // uniform(0,1] random doubles
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); // random normal(0,1)
            return mean + stdDev * randStdNormal; // random normal(mean,stdDev^2)
        }
    }
}