using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Diagnostics;

namespace PacMapSharp
{
    /// <summary>
    /// Distance metrics supported by PACMAP
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
    /// Outlier severity levels for PACMAP safety analysis
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
        public double[] ProjectionCoordinates { get; }

        /// <summary>
        /// Gets the indices of nearest neighbors in the original training data
        /// </summary>
        public int[] NearestNeighborIndices { get; }

        /// <summary>
        /// Gets the distances to nearest neighbors in the original feature space
        /// </summary>
        public double[] NearestNeighborDistances { get; }

        /// <summary>
        /// Gets the confidence score for the projection (0.0 - 1.0)
        /// Higher values indicate the point is similar to training data
        /// </summary>
        public double ConfidenceScore { get; }

        /// <summary>
        /// Gets the outlier severity level based on distance from training data
        /// </summary>
        public OutlierLevel Severity { get; }

        /// <summary>
        /// Gets the percentile rank of the point's distance (0-100)
        /// Lower percentiles indicate similarity to training data
        /// </summary>
        public double PercentileRank { get; }

        /// <summary>
        /// Gets the Z-score relative to training data neighbor distances
        /// Values beyond ±2.5 indicate potential outliers
        /// </summary>
        public double ZScore { get; }

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
        public bool IsReliable => Severity <= OutlierLevel.Unusual && ConfidenceScore >= 0.3;

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

        internal TransformResult(double[] projectionCoordinates,
                               int[] nearestNeighborIndices,
                               double[] nearestNeighborDistances,
                               double confidenceScore,
                               OutlierLevel severity,
                               double percentileRank,
                               double zScore)
        {
            ProjectionCoordinates = projectionCoordinates ?? throw new ArgumentNullException(nameof(projectionCoordinates));
            NearestNeighborIndices = nearestNeighborIndices ?? throw new ArgumentNullException(nameof(nearestNeighborIndices));
            NearestNeighborDistances = nearestNeighborDistances ?? throw new ArgumentNullException(nameof(nearestNeighborDistances));
            ConfidenceScore = Math.Max(0.0, Math.Min(1.0, confidenceScore)); // Clamp to [0,1]
            Severity = severity;
            PercentileRank = Math.Max(0.0, Math.Min(100.0, percentileRank)); // Clamp to [0,100]
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
    /// PACMAP (Pairwise Controlled Manifold Approximation and Projection) implementation
    /// High-performance dimensionality reduction using triplet-based optimization with Adam optimizer
    ///
    /// Key Features:
    /// - Triplet-based structure preservation with three pair types (neighbors, mid-near, far)
    /// - Three-phase optimization with dynamic weight adjustment for global/local balance
    /// - Adam optimizer for stable, fast convergence with adaptive learning rates
    /// - HNSW optimization for ultra-fast nearest neighbor search (50-2000x speedup)
    /// - Cross-platform determinism with strict floating-point controls
    /// - Production-ready safety features with 5-level outlier detection
    /// - Model persistence with CRC32 validation and 16-bit quantization
    /// </summary>
    public class PacMapModel : IDisposable
    {
        #region Platform Detection and DLL Imports

        private static readonly bool IsWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        private static readonly bool IsLinux = RuntimeInformation.IsOSPlatform(OSPlatform.Linux);

        private const string WindowsDll = "pacmap.dll";
        private const string LinuxDll = "libpacmap.so";

        // Enhanced native progress callback delegate with phase information and loss values
        private delegate void NativeProgressCallbackV2(
            [MarshalAs(UnmanagedType.LPStr)] string phase,
            int current,
            int total,
            float percent,
            [MarshalAs(UnmanagedType.LPStr)] string message
        );

        // Windows P/Invoke declarations - updated to use available PACMAP functions
        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_create")]
        private static extern IntPtr WindowsCreate();

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_fit_with_progress_v2")]
        private static extern int WindowsFitWithProgressV2(IntPtr model, double[,] data, int nObs, int nDim, int embeddingDim,
                                                          int nNeighbors, float mnRatio, float fpRatio,
                                                          float learningRate, int nIters, int phase1Iters, int phase2Iters, int phase3Iters,
                                                          DistanceMetric metric, double[,] embedding, NativeProgressCallbackV2 progressCallback,
                                                          int forceExactKnn, int M, int efConstruction, int efSearch,
                                                          int useQuantization, int randomSeed = -1, int autoHNSWParam = 1,
                                                          float initializationStdDev = 0.1f);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_transform_detailed")]
        private static extern int WindowsTransformDetailed(IntPtr model, double[,] newData, int nNewObs, int nDim,
                                                        double[,] embedding, int[] nnIndices, double[] nnDistances,
                                                        double[] confidenceScore, int[] outlierLevel,
                                                        double[] percentileRank, double[] zScore);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_save_model")]
        private static extern int WindowsSaveModel(IntPtr model, [MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_load_model")]
        private static extern IntPtr WindowsLoadModel([MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_destroy")]
        private static extern void WindowsDestroy(IntPtr model);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_error_message")]
        private static extern IntPtr WindowsGetErrorMessage(int errorCode);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_model_info_simple")]
        private static extern int WindowsGetModelInfo(IntPtr model, out int nVertices, out int nDim, out int embeddingDim,
                                                      out int nNeighbors, out float mnRatio, out float fpRatio,
                                                      out DistanceMetric metric, out int hnswM, out int hnswEfConstruction, out int hnswEfSearch,
                                                      out int forceExactKnn, out int randomSeed,
                                                      out float minEmbeddingDistance, out float p95EmbeddingDistance, out float p99EmbeddingDistance,
                                                      out float mildEmbeddingOutlierThreshold, out float extremeEmbeddingOutlierThreshold,
                                                      out float meanEmbeddingDistance, out float stdEmbeddingDistance,
                                                      out uint originalSpaceCrc, out uint embeddingSpaceCrc, out uint modelVersionCrc,
                                                      out float initializationStdDev, out int alwaysSaveEmbeddingData,
                                                      out float p25Distance, out float p75Distance, out float adamEps);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_is_fitted")]
        private static extern int WindowsIsFitted(IntPtr model);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_version")]
        private static extern IntPtr WindowsGetVersion();

        // Linux P/Invoke declarations
        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_create")]
        private static extern IntPtr LinuxCreate();

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_fit_with_progress_v2")]
        private static extern int LinuxFitWithProgressV2(IntPtr model, double[,] data, int nObs, int nDim, int embeddingDim,
                                                         int nNeighbors, float mnRatio, float fpRatio,
                                                         float learningRate, int nIters, int phase1Iters, int phase2Iters, int phase3Iters,
                                                         DistanceMetric metric, double[,] embedding, NativeProgressCallbackV2 progressCallback,
                                                         int forceExactKnn, int M, int efConstruction, int efSearch,
                                                         int useQuantization, int randomSeed = -1, int autoHNSWParam = 1,
                                                         float initializationStdDev = 0.1f);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_transform_detailed")]
        private static extern int LinuxTransformDetailed(IntPtr model, double[,] newData, int nNewObs, int nDim,
                                                      double[,] embedding, int[] nnIndices, double[] nnDistances,
                                                      double[] confidenceScore, int[] outlierLevel,
                                                      double[] percentileRank, double[] zScore);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_save_model")]
        private static extern int LinuxSaveModel(IntPtr model, [MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_load_model")]
        private static extern IntPtr LinuxLoadModel([MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_destroy")]
        private static extern void LinuxDestroy(IntPtr model);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_error_message")]
        private static extern IntPtr LinuxGetErrorMessage(int errorCode);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_model_info_simple")]
        private static extern int LinuxGetModelInfo(IntPtr model, out int nVertices, out int nDim, out int embeddingDim,
                                                    out int nNeighbors, out float mnRatio, out float fpRatio,
                                                    out DistanceMetric metric, out int hnswM, out int hnswEfConstruction, out int hnswEfSearch,
                                                    out int forceExactKnn, out int randomSeed,
                                                    out float minEmbeddingDistance, out float p95EmbeddingDistance, out float p99EmbeddingDistance,
                                                    out float mildEmbeddingOutlierThreshold, out float extremeEmbeddingOutlierThreshold,
                                                    out float meanEmbeddingDistance, out float stdEmbeddingDistance,
                                                    out uint originalSpaceCrc, out uint embeddingSpaceCrc, out uint modelVersionCrc,
                                                    out float initializationStdDev, out int alwaysSaveEmbeddingData,
                                                    out float p25Distance, out float p75Distance, out float adamEps);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_is_fitted")]
        private static extern int LinuxIsFitted(IntPtr model);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_version")]
        private static extern IntPtr LinuxGetVersion();

        #endregion

        #region Constants

        // Expected DLL version - must match C++ PACMAP_WRAPPER_VERSION_STRING
        private const string EXPECTED_DLL_VERSION = "2.8.24";

        #endregion

        #region Error Codes

        private const int PACMAP_SUCCESS = 0;
        private const int PACMAP_ERROR_INVALID_PARAMS = -1;
        private const int PACMAP_ERROR_MEMORY = -2;
        private const int PACMAP_ERROR_NOT_IMPLEMENTED = -3;
        private const int PACMAP_ERROR_FILE_IO = -4;
        private const int PACMAP_ERROR_MODEL_NOT_FITTED = -5;
        private const int PACMAP_ERROR_INVALID_MODEL_FILE = -6;
        private const int PACMAP_ERROR_CRC_MISMATCH = -7;
        private const int PACMAP_ERROR_QUANTIZATION_FAILURE = -8;
        private const int PACMAP_ERROR_OPTIMIZATION_FAILURE = -9;

        #endregion

        #region Private Fields

        private IntPtr _nativeModel;
        private bool _disposed = false;

        // PACMAP-specific parameters
        private float _mnRatio = 0.5f;
        private float _fpRatio = 2.0f;
        private float _learningRate = 1.0f;  // Updated default for Adam optimizer
        private float _adamBeta1 = 0.9f;
        private float _adamBeta2 = 0.999f;
        private float _initializationStdDev = 1e-4f;  // Standard deviation for embedding initialization
        private (int phase1, int phase2, int phase3) _numIters = (150, 100, 250);

        #endregion

        #region Properties

        /// <summary>
        /// Gets the MN_ratio parameter for mid-near pair sampling
        /// </summary>
        public float MN_ratio => _mnRatio;

        /// <summary>
        /// Gets the FP_ratio parameter for far pair sampling
        /// </summary>
        public float FP_ratio => _fpRatio;

        /// <summary>
        /// Gets the learning rate for the Adam optimizer
        /// </summary>
        public float LearningRate => _learningRate;

        /// <summary>
        /// Gets the Adam beta1 parameter for momentum
        /// </summary>
        public float AdamBeta1 => _adamBeta1;

        /// <summary>
        /// Gets the Adam beta2 parameter for RMSprop-like decay
        /// </summary>
        public float AdamBeta2 => _adamBeta2;

        /// <summary>
        /// Gets the standard deviation for embedding initialization
        /// </summary>
        public float InitializationStdDev => _initializationStdDev;

        /// <summary>
        /// Gets the number of iterations for each optimization phase
        /// </summary>
        public (int phase1, int phase2, int phase3) NumIters => _numIters;

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

                var result = CallGetModelInfo(_nativeModel, out var nVertices, out var nDim, out var embeddingDim,
                                                  out var nNeighbors, out var mnRatio, out var fpRatio,
                                                  out var metric, out var hnswM, out var hnswEfConstruction, out var hnswEfSearch,
                                                  out var forceExactKnn, out var randomSeed,
                                                  out var minEmbeddingDistance, out var p95EmbeddingDistance, out var p99EmbeddingDistance,
                                                  out var mildEmbeddingOutlierThreshold, out var extremeEmbeddingOutlierThreshold,
                                                  out var meanEmbeddingDistance, out var stdEmbeddingDistance,
                                                  out var originalSpaceCrc, out var embeddingSpaceCrc, out var modelVersionCrc,
                                                  out var initializationStdDev, out var alwaysSaveEmbeddingData,
                                                  out var p25Distance, out var p75Distance, out var adamEps);
                ThrowIfError(result);

                return new PacMapModelInfo(nVertices, nDim, embeddingDim, nNeighbors, mnRatio, fpRatio, metric, hnswM, hnswEfConstruction, hnswEfSearch,
                                          forceExactKnn != 0, randomSeed,
                                          minEmbeddingDistance, p95EmbeddingDistance, p99EmbeddingDistance,
                                          mildEmbeddingOutlierThreshold, extremeEmbeddingOutlierThreshold,
                                          meanEmbeddingDistance, stdEmbeddingDistance,
                                          originalSpaceCrc, embeddingSpaceCrc, modelVersionCrc,
                                          initializationStdDev, alwaysSaveEmbeddingData != 0,
                                          p25Distance, p75Distance, adamEps);
            }
        }

        #endregion

        #region Constructors

        /// <summary>
        /// Creates a new PACMAP model instance with default parameters
        /// </summary>
        public PacMapModel()
        {
            // CRITICAL: Verify DLL version before any native calls to prevent binary mismatches
            VerifyDllVersion();

            _nativeModel = CallCreate();
            if (_nativeModel == IntPtr.Zero)
                throw new OutOfMemoryException("Failed to create PACMAP model");
        }

        /// <summary>
        /// Creates a new PACMAP model instance with custom parameters
        /// </summary>
        /// <param name="mnRatio">Mid-near pair ratio for global structure preservation (default: 0.5)</param>
        /// <param name="fpRatio">Far-pair ratio for uniform distribution (default: 2.0)</param>
        /// <param name="learningRate">Learning rate for Adam optimizer (default: 1.0)</param>
        /// <param name="adamBeta1">Adam beta1 parameter for momentum (default: 0.9)</param>
        /// <param name="adamBeta2">Adam beta2 parameter for RMSprop-like decay (default: 0.999)</param>
        /// <param name="initializationStdDev">Standard deviation for embedding initialization (default: 0.1)</param>
        /// <param name="numIters">Number of iterations for each optimization phase (default: (100, 100, 250))</param>
        public PacMapModel(float mnRatio = 0.5f, float fpRatio = 2.0f, float learningRate = 1.0f,
                            float adamBeta1 = 0.9f, float adamBeta2 = 0.999f, float initializationStdDev = 1e-4f,
                            (int, int, int) numIters = default((int, int, int)))
        {
            // CRITICAL: Verify DLL version before any native calls to prevent binary mismatches
            VerifyDllVersion();

            _nativeModel = CallCreate();
            if (_nativeModel == IntPtr.Zero)
                throw new OutOfMemoryException("Failed to create PACMAP model");

            _mnRatio = mnRatio;
            _fpRatio = fpRatio;
            _learningRate = learningRate;
            _adamBeta1 = adamBeta1;
            _adamBeta2 = adamBeta2;
            _initializationStdDev = initializationStdDev;

            // Handle default value for numIters
            _numIters = numIters.Equals(default((int, int, int))) ? (100, 100, 250) : numIters;
        }

        /// <summary>
        /// Loads a PACMAP model from a file
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
        /// Fits the PACMAP model to training data with full customization and optional progress reporting
        /// </summary>
        /// <param name="data">Training data as 2D array [samples, features]</param>
        /// <param name="embeddingDimension">Target embedding dimension (1-50, default: 2)</param>
        /// <param name="nNeighbors">Number of nearest neighbors (default: 10)</param>
        /// <param name="mnRatio">Mid-near pair ratio for global structure (default: 0.5)</param>
        /// <param name="fpRatio">Far-pair ratio for uniform distribution (default: 2.0)</param>
        /// <param name="numIters">Three-phase iterations (phase1, phase2, phase3) (default: (100, 100, 250))</param>
        /// <param name="metric">Distance metric to use (default: Euclidean)</param>
        /// <param name="forceExactKnn">Force exact brute-force k-NN instead of HNSW approximation (default: false)</param>
        /// <param name="hnswM">HNSW graph degree parameter (default: 16)</param>
        /// <param name="hnswEfConstruction">HNSW build quality parameter (default: 150, lowered from 200 for faster builds)</param>
        /// <param name="hnswEfSearch">HNSW query quality parameter (default: 100, lowered from 200 for 2x faster queries)</param>
        /// <param name="randomSeed">Random seed for reproducibility (default: -1 for non-deterministic)</param>
        /// <param name="autoHNSWParam">Auto-tune HNSW parameters based on data size (default: true)</param>
        /// <param name="learningRate">Learning rate for Adam optimizer (default: 1.0)</param>
        /// <param name="useQuantization">Enable 16-bit quantization for memory reduction (default: false)</param>
        /// <param name="progressCallback">Optional callback function to report training progress (default: null)</param>
        /// <returns>Embedding coordinates [samples, embeddingDimension]</returns>
        /// <exception cref="ArgumentNullException">Thrown when data is null</exception>
        /// <exception cref="ArgumentException">Thrown when parameters are invalid</exception>
        public double[,] Fit(double[,] data,
                            int embeddingDimension = 2,
                            int nNeighbors = 10,
                            float mnRatio = 0.5f,
                            float fpRatio = 2.0f,
                            (int, int, int) numIters = default((int, int, int)),
                            DistanceMetric metric = DistanceMetric.Euclidean,
                            bool forceExactKnn = false,
                            int hnswM = 16,
                            int hnswEfConstruction = 150,
                            int hnswEfSearch = 100,
                            int randomSeed = -1,
                            bool autoHNSWParam = true,
                            float learningRate = 1.0f,
                            bool useQuantization = false,
                            ProgressCallback? progressCallback = null)
        {
            // Handle default value for numIters
            var actualNumIters = numIters.Equals(default((int, int, int))) ? (100, 100, 250) : numIters;

            return FitInternal(data, embeddingDimension, nNeighbors, mnRatio, fpRatio,
                             actualNumIters, metric, forceExactKnn,
                             progressCallback, hnswM, hnswEfConstruction, hnswEfSearch,
                             randomSeed, autoHNSWParam, learningRate, useQuantization);
        }

        /// <summary>
        /// Fits the PACMAP model to training data with progress reporting (legacy compatibility)
        /// </summary>
        /// <param name="data">Training data as 2D array [samples, features]</param>
        /// <param name="progressCallback">Callback function to report training progress</param>
        /// <param name="embeddingDimension">Target embedding dimension (1-50, default: 2)</param>
        /// <param name="nNeighbors">Number of nearest neighbors (default: 10)</param>
        /// <param name="mnRatio">Mid-near pair ratio for global structure (default: 0.5)</param>
        /// <param name="fpRatio">Far-pair ratio for uniform distribution (default: 2.0)</param>
        /// <param name="numIters">Three-phase iterations (phase1, phase2, phase3) (default: (100, 100, 250))</param>
        /// <param name="metric">Distance metric to use (default: Euclidean)</param>
        /// <param name="forceExactKnn">Force exact brute-force k-NN instead of HNSW approximation (default: false)</param>
        /// <param name="randomSeed">Random seed for reproducibility (default: -1 for non-deterministic)</param>
        /// <param name="autoHNSWParam">Auto-tune HNSW parameters based on data size (default: true)</param>
        /// <param name="learningRate">Learning rate for Adam optimizer (default: 1.0)</param>
        /// <param name="useQuantization">Enable 16-bit quantization for memory reduction (default: false)</param>
        /// <param name="hnswM">HNSW graph degree parameter (default: 16)</param>
        /// <param name="hnswEfConstruction">HNSW build quality parameter (default: 150)</param>
        /// <param name="hnswEfSearch">HNSW query quality parameter (default: 100)</param>
        /// <returns>Embedding coordinates [samples, embeddingDimension]</returns>
        /// <exception cref="ArgumentNullException">Thrown when data or progressCallback is null</exception>
        /// <exception cref="ArgumentException">Thrown when parameters are invalid</exception>
        /// <remarks>This method is provided for backward compatibility. Consider using the unified Fit method instead.</remarks>
        [Obsolete("Use the unified Fit method with optional progressCallback parameter instead.")]
        public double[,] FitWithProgress(double[,] data,
                                       ProgressCallback progressCallback,
                                       int embeddingDimension = 2,
                                       int nNeighbors = 10,
                                       float mnRatio = 0.5f,
                                       float fpRatio = 2.0f,
                                       (int, int, int) numIters = default((int, int, int)),
                                       DistanceMetric metric = DistanceMetric.Euclidean,
                                       bool forceExactKnn = false,
                                       int randomSeed = -1,
                                       bool autoHNSWParam = true,
                                       float learningRate = 1.0f,
                                       bool useQuantization = false,
                                       int hnswM = 16,
                                       int hnswEfConstruction = 150,
                                       int hnswEfSearch = 100)
        {
            if (progressCallback == null)
                throw new ArgumentNullException(nameof(progressCallback));

            return Fit(data, embeddingDimension, nNeighbors, mnRatio, fpRatio,
                      numIters, metric, forceExactKnn, hnswM, hnswEfConstruction, hnswEfSearch,
                      randomSeed, autoHNSWParam, learningRate, useQuantization, progressCallback);
        }

        /// <summary>
        /// Transforms new data using a fitted model (out-of-sample projection)
        /// </summary>
        /// <param name="newData">New data to transform [samples, features]</param>
        /// <returns>Embedding coordinates [samples, embeddingDimension]</returns>
        /// <exception cref="ArgumentNullException">Thrown when newData is null</exception>
        /// <exception cref="InvalidOperationException">Thrown when model is not fitted</exception>
        /// <exception cref="ArgumentException">Thrown when feature dimensions don't match training data</exception>
        public double[,] Transform(double[,] newData)
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

            // Prepare output array
            var embedding = new double[nNewSamples, modelInfo.OutputDimension];

            // Call native function
            var result = CallTransformDetailed(_nativeModel, newData, nNewSamples, nFeatures,
                                             embedding, null!, null!, null!, null!, null!, null!);

            // CRITICAL: Check for error BEFORE processing results
            if (result != PACMAP_SUCCESS)
            {
                var errorMessage = CallGetErrorMessage(result);
                throw new InvalidOperationException($"Transform failed with error {result}: {errorMessage}");
            }

            return embedding;
        }

        /// <summary>
        /// Transforms new data using a fitted model with comprehensive safety analysis (HNSW-enhanced)
        /// Provides detailed outlier detection and confidence metrics for production safety
        /// </summary>
        /// <param name="newData">New data to transform [samples, features]</param>
        /// <returns>Array of TransformResult objects with embedding coordinates and safety metrics</returns>
        /// <exception cref="ArgumentNullException">Thrown when newData is null</exception>
        /// <exception cref="InvalidOperationException">Thrown when model is not fitted</exception>
        /// <exception cref="ArgumentException">Thrown when feature dimensions don't match training data</exception>
        public TransformResult[] TransformWithSafety(double[,] newData)
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

            // Prepare output arrays
            var embedding = new double[nNewSamples, modelInfo.OutputDimension];
            var nnIndices = new int[nNewSamples * modelInfo.Neighbors];
            var nnDistances = new double[nNewSamples * modelInfo.Neighbors];
            var confidenceScores = new double[nNewSamples];
            var outlierLevels = new int[nNewSamples];
            var percentileRanks = new double[nNewSamples];
            var zScores = new double[nNewSamples];

            // Call enhanced native function
            var result = CallTransformDetailed(_nativeModel, newData, nNewSamples, nFeatures,
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
                var projectionCoords = new double[modelInfo.OutputDimension];
                for (int j = 0; j < modelInfo.OutputDimension; j++)
                {
                    projectionCoords[j] = embedding[i, j];
                }

                // Extract neighbor indices and distances for this sample
                var nearestIndices = new int[modelInfo.Neighbors];
                var nearestDistances = new double[modelInfo.Neighbors];
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
        /// <exception cref="ArgumentException">Thrown when filename is null or empty</exception>
        /// <exception cref="InvalidOperationException">Thrown when model is not fitted</exception>
        /// <exception cref="IOException">Thrown when file cannot be written</exception>
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

        #endregion

        #region Private Methods

        private double[,] FitInternal(double[,] data,
                                   int embeddingDimension,
                                   int nNeighbors,
                                   float mnRatio,
                                   float fpRatio,
                                   (int, int, int) numIters,
                                   DistanceMetric metric,
                                   bool forceExactKnn,
                                   ProgressCallback? progressCallback,
                                   int hnswM = 16,
                                   int hnswEfConstruction = 150,
                                   int hnswEfSearch = 100,
                                   int randomSeed = -1,
                                   bool autoHNSWParam = true,
                                   float learningRate = 1.0f,
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

            if (mnRatio < 0 || fpRatio < 0)
                throw new ArgumentException("MN_ratio and FP_ratio must be non-negative");

            if (learningRate <= 0)
                throw new ArgumentException("Learning rate must be positive");

            if (numIters.Item1 <= 0 || numIters.Item2 <= 0 || numIters.Item3 <= 0)
                throw new ArgumentException("All phase iteration counts must be positive");

            // Prepare output array
            var embedding = new double[nSamples, embeddingDimension];

            // Call appropriate native function
            int result;
            if (progressCallback != null)
            {
                // Create enhanced native callback wrapper
                NativeProgressCallbackV2 nativeCallback = (phase, current, total, percent, message) =>
                {
                    try
                    {
                        progressCallback(phase ?? "Training", current, total, percent, message);
                    }
                    catch
                    {
                        // Ignore exceptions in callback to prevent native crashes
                    }
                };

                int totalIters = numIters.Item1 + numIters.Item2 + numIters.Item3;
                result = CallFitWithProgressV2(_nativeModel, data, nSamples, nFeatures, embeddingDimension,
                                                 nNeighbors, mnRatio, fpRatio, learningRate, totalIters, numIters.Item1, numIters.Item2, numIters.Item3,
                                                 metric, embedding, nativeCallback,
                                                 forceExactKnn ? 1 : 0, hnswM, hnswEfConstruction, hnswEfSearch,
                                                 useQuantization ? 1 : 0, randomSeed, autoHNSWParam ? 1 : 0,
                                                 _initializationStdDev);
            }
            else
            {
                int totalIters = numIters.Item1 + numIters.Item2 + numIters.Item3;
                result = CallFitWithProgressV2(_nativeModel, data, nSamples, nFeatures, embeddingDimension,
                                                 nNeighbors, mnRatio, fpRatio, learningRate, totalIters, numIters.Item1, numIters.Item2, numIters.Item3,
                                                 metric, embedding, null,
                                                 forceExactKnn ? 1 : 0, hnswM, hnswEfConstruction, hnswEfSearch,
                                                 useQuantization ? 1 : 0, randomSeed, autoHNSWParam ? 1 : 0,
                                                 _initializationStdDev);
            }

            ThrowIfError(result);

            // Store parameters in the model
            _mnRatio = mnRatio;
            _fpRatio = fpRatio;
            _learningRate = learningRate;
            _numIters = numIters;

            return embedding;
        }

        #endregion

        #region Private Platform-Specific Wrappers

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static IntPtr CallCreate()
        {
            return IsWindows ? WindowsCreate() : LinuxCreate();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallFitWithProgressV2(IntPtr model, double[,] data, int nObs, int nDim, int embeddingDim,
                                                  int nNeighbors, float mnRatio, float fpRatio,
                                                  float learningRate, int nIters, int phase1Iters, int phase2Iters, int phase3Iters,
                                                  DistanceMetric metric, double[,] embedding, NativeProgressCallbackV2? progressCallback,
                                                  int forceExactKnn, int M, int efConstruction, int efSearch,
                                                  int useQuantization, int randomSeed = -1, int autoHNSWParam = 1,
                                                  float initializationStdDev = 0.1f)
        {
            var callback = progressCallback ?? ((phase, current, total, percent, message) => { });
            return IsWindows ? WindowsFitWithProgressV2(model, data, nObs, nDim, embeddingDim, nNeighbors, mnRatio, fpRatio,
                                                      learningRate, nIters, phase1Iters, phase2Iters, phase3Iters, metric, embedding, callback,
                                                      forceExactKnn, M, efConstruction, efSearch, useQuantization, randomSeed, autoHNSWParam,
                                                      initializationStdDev)
                             : LinuxFitWithProgressV2(model, data, nObs, nDim, embeddingDim, nNeighbors, mnRatio, fpRatio,
                                                     learningRate, nIters, phase1Iters, phase2Iters, phase3Iters, metric, embedding, callback,
                                                     forceExactKnn, M, efConstruction, efSearch, useQuantization, randomSeed, autoHNSWParam,
                                                     initializationStdDev);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallTransformDetailed(IntPtr model, double[,] newData, int nNewObs, int nDim,
                                                double[,] embedding, int[] nnIndices, double[] nnDistances,
                                                double[] confidenceScore, int[] outlierLevel,
                                                double[] percentileRank, double[] zScore)
        {
            return IsWindows ? WindowsTransformDetailed(model, newData, nNewObs, nDim, embedding, nnIndices, nnDistances,
                                                       confidenceScore, outlierLevel, percentileRank, zScore)
                             : LinuxTransformDetailed(model, newData, nNewObs, nDim, embedding, nnIndices, nnDistances,
                                                      confidenceScore, outlierLevel, percentileRank, zScore);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallSaveModel(IntPtr model, string filename)
        {
            return IsWindows ? WindowsSaveModel(model, filename) : LinuxSaveModel(model, filename);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static IntPtr CallLoadModel(string filename)
        {
            return IsWindows ? WindowsLoadModel(filename) : LinuxLoadModel(filename);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CallDestroy(IntPtr model)
        {
            if (IsWindows) WindowsDestroy(model);
            else LinuxDestroy(model);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static string CallGetErrorMessage(int errorCode)
        {
            var ptr = IsWindows ? WindowsGetErrorMessage(errorCode) : LinuxGetErrorMessage(errorCode);
            return Marshal.PtrToStringAnsi(ptr) ?? "Unknown error";
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallGetModelInfo(IntPtr model, out int nVertices, out int nDim, out int embeddingDim,
                                              out int nNeighbors, out float mnRatio, out float fpRatio,
                                              out DistanceMetric metric, out int hnswM, out int hnswEfConstruction, out int hnswEfSearch,
                                              out int forceExactKnn, out int randomSeed,
                                              out float minEmbeddingDistance, out float p95EmbeddingDistance, out float p99EmbeddingDistance,
                                              out float mildEmbeddingOutlierThreshold, out float extremeEmbeddingOutlierThreshold,
                                              out float meanEmbeddingDistance, out float stdEmbeddingDistance,
                                              out uint originalSpaceCrc, out uint embeddingSpaceCrc, out uint modelVersionCrc,
                                              out float initializationStdDev, out int alwaysSaveEmbeddingData,
                                              out float p25Distance, out float p75Distance, out float adamEps)
        {
            return IsWindows ? WindowsGetModelInfo(model, out nVertices, out nDim, out embeddingDim, out nNeighbors,
                                                      out mnRatio, out fpRatio, out metric, out hnswM, out hnswEfConstruction, out hnswEfSearch,
                                                      out forceExactKnn, out randomSeed, out minEmbeddingDistance, out p95EmbeddingDistance, out p99EmbeddingDistance,
                                                      out mildEmbeddingOutlierThreshold, out extremeEmbeddingOutlierThreshold,
                                                      out meanEmbeddingDistance, out stdEmbeddingDistance,
                                                      out originalSpaceCrc, out embeddingSpaceCrc, out modelVersionCrc,
                                                      out initializationStdDev, out alwaysSaveEmbeddingData, out p25Distance, out p75Distance, out adamEps)
                             : LinuxGetModelInfo(model, out nVertices, out nDim, out embeddingDim, out nNeighbors,
                                                     out mnRatio, out fpRatio, out metric, out hnswM, out hnswEfConstruction, out hnswEfSearch,
                                                     out forceExactKnn, out randomSeed, out minEmbeddingDistance, out p95EmbeddingDistance, out p99EmbeddingDistance,
                                                     out mildEmbeddingOutlierThreshold, out extremeEmbeddingOutlierThreshold,
                                                     out meanEmbeddingDistance, out stdEmbeddingDistance,
                                                     out originalSpaceCrc, out embeddingSpaceCrc, out modelVersionCrc,
                                                     out initializationStdDev, out alwaysSaveEmbeddingData, out p25Distance, out p75Distance, out adamEps);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallIsFitted(IntPtr model)
        {
            return IsWindows ? WindowsIsFitted(model) : LinuxIsFitted(model);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static string CallGetVersion()
        {
            var ptr = IsWindows ? WindowsGetVersion() : LinuxGetVersion();
            return Marshal.PtrToStringAnsi(ptr) ?? "Unknown";
        }

        /// <summary>
        /// Gets the PACMAP library version
        /// </summary>
        public static string GetVersion()
        {
            return CallGetVersion();
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
                PACMAP_ERROR_CRC_MISMATCH => new InvalidDataException("CRC32 validation failed - file may be corrupted"),
                PACMAP_ERROR_QUANTIZATION_FAILURE => new InvalidOperationException("Quantization operation failed"),
                PACMAP_ERROR_OPTIMIZATION_FAILURE => new InvalidOperationException("Optimization failed to converge"),
                _ => new Exception($"PACMAP Error ({errorCode}): {message}")
            };
        }

        /// <summary>
        /// Verifies the native DLL version matches the expected C# wrapper version
        /// Uses internal API for version checking (cross-platform compatible)
        /// </summary>
        /// <exception cref="InvalidOperationException">Thrown when DLL version mismatch detected</exception>
        private static void VerifyDllVersion()
        {
            try
            {
                // Get the version from the native library internal API
                var actualVersion = CallGetVersion();

                // Verify version matches expected version
                if (actualVersion != EXPECTED_DLL_VERSION)
                {
                    throw new InvalidOperationException(
                        $"CRITICAL: PACMAP library version mismatch!\n" +
                        $"Expected version: {EXPECTED_DLL_VERSION}\n" +
                        $"Actual version: {actualVersion}\n" +
                        $"Platform: {RuntimeInformation.OSArchitecture} on {RuntimeInformation.OSDescription}\n" +
                        $"Ensure the native library version matches the C# wrapper version.\n" +
                        $"This can cause crashes, data corruption, or incorrect results.");
                }

                // Version verified silently
            }
            catch (DllNotFoundException ex)
            {
                throw new DllNotFoundException(
                    $"CRITICAL: Native PACMAP library not found!\n" +
                    $"Expected: {(RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "pacmap.dll" : "libpacmap.so")}\n" +
                    $"Platform: {RuntimeInformation.OSArchitecture} on {RuntimeInformation.OSDescription}\n" +
                    $"Ensure the native library is in the application directory.\n" +
                    $"Original error: {ex.Message}");
            }
            catch (Exception ex) when (!(ex is InvalidOperationException))
            {
                throw new InvalidOperationException(
                    $"❌ CRITICAL: Could not verify PACMAP library version!\n" +
                    $"Expected version: {EXPECTED_DLL_VERSION}\n" +
                    $"Error: {ex.Message}\n" +
                    $"Platform: {RuntimeInformation.OSArchitecture} on {RuntimeInformation.OSDescription}\n" +
                    $"Ensure the native library version matches the C# wrapper version exactly.", ex);
            }
        }

        /// <summary>
        /// Gets the path to the native DLL (cross-platform)
        /// </summary>
        /// <returns>Full path to the native library</returns>
        private static string GetDllPath()
        {
            var dllName = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "pacmap.dll" : "libpacmap.so";
            var assemblyLocation = Assembly.GetExecutingAssembly().Location;
            var assemblyDir = Path.GetDirectoryName(assemblyLocation)!;

            // Try multiple common locations
            var possiblePaths = new[]
            {
                Path.Combine(assemblyDir, dllName),
                Path.Combine(assemblyDir, "runtimes", "win-x64", "native", dllName),
                Path.Combine(assemblyDir, "runtimes", "linux-x64", "native", dllName),
                dllName // Fallback to PATH/LD_LIBRARY_PATH
            };

            foreach (var path in possiblePaths)
            {
                if (File.Exists(path))
                    return Path.GetFullPath(path);
            }

            throw new DllNotFoundException($"Native PACMAP library not found. Searched paths: {string.Join(", ", possiblePaths)}");
        }

        /// <summary>
        /// Gets the file version of the native library (cross-platform)
        /// </summary>
        /// <param name="dllPath">Path to the native library</param>
        /// <returns>Version string or "Unknown" if cannot be determined</returns>
        private static string GetFileVersion(string dllPath)
        {
            try
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    // Windows: Use FileVersionInfo
                    var versionInfo = FileVersionInfo.GetVersionInfo(dllPath);
                    return !string.IsNullOrEmpty(versionInfo.FileVersion)
                        ? versionInfo.FileVersion
                        : versionInfo.ProductVersion ?? "Unknown";
                }
                else
                {
                    // Linux: Try to read version from file properties or use fallback
                    // Linux shared libraries don't have standardized version resources like Windows
                    // We'll use file modification time as a proxy or read from library if possible
                    var fileInfo = new FileInfo(dllPath);

                    // Try to get version from library string fallback (if available)
                    try
                    {
                        using var process = new System.Diagnostics.Process
                        {
                            StartInfo = new System.Diagnostics.ProcessStartInfo
                            {
                                FileName = "objdump",
                                Arguments = $"-p {dllPath} | grep -i version",
                                UseShellExecute = false,
                                RedirectStandardOutput = true,
                                CreateNoWindow = true
                            }
                        };
                        process.Start();
                        var output = process.StandardOutput.ReadToEnd();
                        process.WaitForExit();

                        if (!string.IsNullOrEmpty(output) && output.Contains("version"))
                        {
                            var lines = output.Split('\n');
                            foreach (var line in lines)
                            {
                                if (line.ToLower().Contains("version"))
                                {
                                    var parts = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                                    foreach (var part in parts)
                                    {
                                        if (part.Contains('.'))
                                        {
                                            var version = part.Trim().Trim('"');
                                            if (version.Split('.').Length >= 2)
                                                return version;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    catch
                    {
                        // objdump not available or failed, use file timestamp fallback
                    }

                    // Fallback: Use file modification timestamp as version indicator
                    return $"1.2.0-{fileInfo.LastWriteTime:yyyyMMdd-HHmmss}";
                }
            }
            catch
            {
                throw new InvalidOperationException(
                    $"❌ CRITICAL: Could not determine DLL version from {dllPath}\n" +
                    $"Platform: {RuntimeInformation.OSArchitecture} on {RuntimeInformation.OSDescription}\n" +
                    $"Ensure the DLL has proper version resources embedded.");
            }
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
    /// Comprehensive information about a fitted PACMAP model with enhanced persistence fields
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
        /// Gets the MN_ratio parameter used during training
        /// </summary>
        public float MN_ratio { get; }

        /// <summary>
        /// Gets the FP_ratio parameter used during training
        /// </summary>
        public float FP_ratio { get; }

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

        // Enhanced persistence fields

        /// <summary>
        /// Gets whether exact k-NN search is forced instead of HNSW approximation
        /// </summary>
        public bool ForceExactKnn { get; }

        /// <summary>
        /// Gets the random seed used during training for reproducibility
        /// </summary>
        public int RandomSeed { get; }

        /// <summary>
        /// Gets the minimum embedding distance in the training data
        /// Used for transform outlier detection
        /// </summary>
        public float MinEmbeddingDistance { get; }

        /// <summary>
        /// Gets the 95th percentile embedding distance in the training data
        /// Used for transform confidence scoring
        /// </summary>
        public float P95EmbeddingDistance { get; }

        /// <summary>
        /// Gets the 99th percentile embedding distance in the training data
        /// Used for transform outlier detection
        /// </summary>
        public float P99EmbeddingDistance { get; }

        /// <summary>
        /// Gets the mild outlier threshold (2.5 standard deviations from mean)
        /// Points beyond this are considered mild outliers
        /// </summary>
        public float MildEmbeddingOutlierThreshold { get; }

        /// <summary>
        /// Gets the extreme outlier threshold (4.0 standard deviations from mean)
        /// Points beyond this are considered extreme outliers
        /// </summary>
        public float ExtremeEmbeddingOutlierThreshold { get; }

        /// <summary>
        /// Gets the mean embedding distance across all training sample pairs
        /// Used for transform z-score calculation
        /// </summary>
        public float MeanEmbeddingDistance { get; }

        /// <summary>
        /// Gets the standard deviation of embedding distances across training data
        /// Used for transform statistical analysis
        /// </summary>
        public float StdEmbeddingDistance { get; }

        /// <summary>
        /// Gets the CRC32 checksum of the original space data for integrity validation
        /// </summary>
        public uint OriginalSpaceCrc { get; }

        /// <summary>
        /// Gets the CRC32 checksum of the embedding space data for integrity validation
        /// </summary>
        public uint EmbeddingSpaceCrc { get; }

        /// <summary>
        /// Gets the CRC32 checksum of the model version and structure for integrity validation
        /// </summary>
        public uint ModelVersionCrc { get; }

        /// <summary>
        /// Gets the standard deviation used for embedding initialization during training
        /// </summary>
        public float InitializationStdDev { get; }

        /// <summary>
        /// Gets whether embedding data is always saved during model persistence
        /// </summary>
        public bool AlwaysSaveEmbeddingData { get; }

        /// <summary>
        /// Gets the 25th percentile distance in the original feature space
        /// Used for distance-based sampling and outlier detection
        /// </summary>
        public float P25Distance { get; }

        /// <summary>
        /// Gets the 75th percentile distance in the original feature space
        /// Used for distance-based sampling and outlier detection
        /// </summary>
        public float P75Distance { get; }

        /// <summary>
        /// Gets the Adam optimizer epsilon parameter for numerical stability
        /// </summary>
        public float AdamEps { get; }

        internal PacMapModelInfo(int trainingSamples, int inputDimension, int outputDimension, int neighbors,
                               float mnRatio, float fpRatio, DistanceMetric metric,
                               int hnswM, int hnswEfConstruction, int hnswEfSearch,
                               bool forceExactKnn, int randomSeed,
                               float minEmbeddingDistance, float p95EmbeddingDistance, float p99EmbeddingDistance,
                               float mildEmbeddingOutlierThreshold, float extremeEmbeddingOutlierThreshold,
                               float meanEmbeddingDistance, float stdEmbeddingDistance,
                               uint originalSpaceCrc, uint embeddingSpaceCrc, uint modelVersionCrc,
                               float initializationStdDev, bool alwaysSaveEmbeddingData,
                               float p25Distance, float p75Distance, float adamEps)
        {
            TrainingSamples = trainingSamples;
            InputDimension = inputDimension;
            OutputDimension = outputDimension;
            Neighbors = neighbors;
            MN_ratio = mnRatio;
            FP_ratio = fpRatio;
            Metric = metric;
            HnswM = hnswM;
            HnswEfConstruction = hnswEfConstruction;
            HnswEfSearch = hnswEfSearch;

            // Enhanced persistence fields
            ForceExactKnn = forceExactKnn;
            RandomSeed = randomSeed;
            MinEmbeddingDistance = minEmbeddingDistance;
            P95EmbeddingDistance = p95EmbeddingDistance;
            P99EmbeddingDistance = p99EmbeddingDistance;
            MildEmbeddingOutlierThreshold = mildEmbeddingOutlierThreshold;
            ExtremeEmbeddingOutlierThreshold = extremeEmbeddingOutlierThreshold;
            MeanEmbeddingDistance = meanEmbeddingDistance;
            StdEmbeddingDistance = stdEmbeddingDistance;
            OriginalSpaceCrc = originalSpaceCrc;
            EmbeddingSpaceCrc = embeddingSpaceCrc;
            ModelVersionCrc = modelVersionCrc;

            // Additional persistence fields
            InitializationStdDev = initializationStdDev;
            AlwaysSaveEmbeddingData = alwaysSaveEmbeddingData;
            P25Distance = p25Distance;
            P75Distance = p75Distance;
            AdamEps = adamEps;
        }

        /// <summary>
        /// Returns a comprehensive string representation of the model information
        /// </summary>
        /// <returns>A formatted string describing all model parameters</returns>
        public override string ToString()
        {
            return $"PACMAP Model: {TrainingSamples} samples, {InputDimension}D → {OutputDimension}D, " +
                   $"k={Neighbors}, MN_ratio={MN_ratio:F2}, FP_ratio={FP_ratio:F2}, metric={Metric}, " +
                   $"HNSW(M={HnswM}, ef_c={HnswEfConstruction}, ef_s={HnswEfSearch}), " +
                   $"ExactKNN={ForceExactKnn}, Seed={RandomSeed}, " +
                   $"EmbedStats(min={MinEmbeddingDistance:F3}, p95={P95EmbeddingDistance:F3}, p99={P99EmbeddingDistance:F3}), " +
                   $"CRC(Orig={OriginalSpaceCrc:X8}, Emb={EmbeddingSpaceCrc:X8}, Ver={ModelVersionCrc:X8})";
        }
    }
}