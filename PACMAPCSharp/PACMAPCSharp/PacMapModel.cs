using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;

namespace UMAPuwotSharp
{
    /// <summary>
    /// Distance metrics supported by Enhanced UMAP
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
    /// Outlier detection levels for safety analysis
    /// </summary>
    public enum OutlierLevel
    {
        /// <summary>
        /// Normal data point within expected distribution
        /// </summary>
        Normal = 0,

        /// <summary>
        /// Mild outlier - slightly outside expected range
        /// </summary>
        Mild = 1,

        /// <summary>
        /// Moderate outlier - clearly outside expected range
        /// </summary>
        Moderate = 2,

        /// <summary>
        /// Severe outlier - very far from expected distribution
        /// </summary>
        Severe = 3,

        /// <summary>
        /// Extreme outlier - potentially anomalous or erroneous data
        /// </summary>
        Extreme = 4
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
    /// Enhanced cross-platform C# wrapper for PACMAP dimensionality reduction
    /// Based on PACMAP algorithm with HNSW optimization:
    /// - Arbitrary embedding dimensions (1D to 50D)
    /// - Multiple distance metrics (Euclidean, Cosine, Manhattan, Correlation, Hamming)
    /// - Complete model save/load functionality with conditional storage
    /// - True out-of-sample projection (transform new data)
    /// - Progress reporting with callback support
    /// - HNSW mode: Memory efficient (saves indices, not raw data)
    /// - Exact KNN mode: Saves raw training data for exact neighbor search
    /// </summary>
    public class TransformResult
    {
        /// <summary>
        /// Embedding coordinates for this sample
        /// </summary>
        public float[] Embedding { get; }

        /// <summary>
        /// Indices of nearest neighbors
        /// </summary>
        public int[] NearestIndices { get; }

        /// <summary>
        /// Distances to nearest neighbors
        /// </summary>
        public float[] NearestDistances { get; }

        /// <summary>
        /// Confidence score (0-1, higher = more confident)
        /// </summary>
        public float Confidence { get; }

        /// <summary>
        /// Outlier level classification
        /// </summary>
        public OutlierLevel OutlierLevel { get; }

        /// <summary>
        /// Percentile rank among training data
        /// </summary>
        public float PercentileRank { get; }

        /// <summary>
        /// Z-score indicating how many standard deviations from mean
        /// </summary>
        public float ZScore { get; }

        public TransformResult(float[] embedding, int[] nearestIndices, float[] nearestDistances,
            float confidence, OutlierLevel outlierLevel, float percentileRank, float zScore)
        {
            Embedding = embedding;
            NearestIndices = nearestIndices;
            NearestDistances = nearestDistances;
            Confidence = confidence;
            OutlierLevel = outlierLevel;
            PercentileRank = percentileRank;
            ZScore = zScore;
        }
    }

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

        // Windows P/Invoke declarations
        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_create")]
        private static extern IntPtr WindowsCreate();

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_fit")]
        private static extern int WindowsFit(IntPtr model, float[] data, int nObs, int nDim, int embeddingDim, int nNeighbors, float mnRatio, float fpRatio, float learningRate, int nIters, int phase1Iters, int phase2Iters, int phase3Iters, DistanceMetric metric, float[] embedding, int forceExactKnn);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_fit_with_progress_v2")]
        private static extern int WindowsFitWithProgressV2(IntPtr model, float[] data, int nObs, int nDim, int embeddingDim, int nNeighbors, float mnRatio, float fpRatio, float learningRate, int nIters, int phase1Iters, int phase2Iters, int phase3Iters, DistanceMetric metric, float[] embedding, NativeProgressCallbackV2 progressCallback, int forceExactKnn, int M, int efConstruction, int efSearch, int useQuantization, int randomSeed = -1, int autoHNSWParam = 1);

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

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_model_info_simple")]
        private static extern int WindowsGetModelInfo(IntPtr model, out int nSamples, out int nFeatures, out int nComponents, out int nNeighbors, out float mnRatio, out float fpRatio, out DistanceMetric metric, out int hnswM, out int hnswEfConstruction, out int hnswEfSearch);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_is_fitted")]
        private static extern int WindowsIsFitted(IntPtr model);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_metric_name")]
        private static extern IntPtr WindowsGetMetricName(DistanceMetric metric);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_set_global_callback")]
        private static extern void WindowsSetGlobalCallback(NativeProgressCallbackV2 callback);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_clear_global_callback")]
        private static extern void WindowsClearGlobalCallback();

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_version")]
        private static extern IntPtr WindowsGetVersion();

        // Linux P/Invoke declarations
        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_create")]
        private static extern IntPtr LinuxCreate();

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_fit")]
        private static extern int LinuxFit(IntPtr model, float[] data, int nObs, int nDim, int embeddingDim, int nNeighbors, float mnRatio, float fpRatio, float learningRate, int nIters, int phase1Iters, int phase2Iters, int phase3Iters, DistanceMetric metric, float[] embedding, int forceExactKnn);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_fit_with_progress_v2")]
        private static extern int LinuxFitWithProgressV2(IntPtr model, float[] data, int nObs, int nDim, int embeddingDim, int nNeighbors, float mnRatio, float fpRatio, float learningRate, int nIters, int phase1Iters, int phase2Iters, int phase3Iters, DistanceMetric metric, float[] embedding, NativeProgressCallbackV2 progressCallback, int forceExactKnn, int M, int efConstruction, int efSearch, int useQuantization, int randomSeed = -1, int autoHNSWParam = 1);

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

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_model_info_simple")]
        private static extern int LinuxGetModelInfo(IntPtr model, out int nSamples, out int nFeatures, out int nComponents, out int nNeighbors, out float mnRatio, out float fpRatio, out DistanceMetric metric, out int hnswM, out int hnswEfConstruction, out int hnswEfSearch);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_is_fitted")]
        private static extern int LinuxIsFitted(IntPtr model);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_metric_name")]
        private static extern IntPtr LinuxGetMetricName(DistanceMetric metric);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_set_global_callback")]
        private static extern void LinuxSetGlobalCallback(NativeProgressCallbackV2 callback);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_clear_global_callback")]
        private static extern void LinuxClearGlobalCallback();

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_version")]
        private static extern IntPtr LinuxGetVersion();

        #endregion

        #region Constants

        // Expected DLL version - must match C++ PACMAP version
        private const string EXPECTED_DLL_VERSION = "1.0.0-PACMAP-HNSW-Optimized";

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

                var result = CallGetModelInfo(_nativeModel, out var nSamples, out var nFeatures, out var nComponents, out var nNeighbors, out var mnRatio, out var fpRatio, out var metric, out var hnswM, out var hnswEfConstruction, out var hnswEfSearch);
                ThrowIfError(result);

                return new PacMapModelInfo(nSamples, nFeatures, nComponents, nNeighbors, mnRatio, fpRatio, metric, hnswM, hnswEfConstruction, hnswEfSearch);
            }
        }

        #endregion

        #region Constructors

        /// <summary>
        /// Creates a new Enhanced PACMAP model instance
        /// </summary>
        public PacMapModel()
        {
            // CRITICAL: Verify DLL version before any native calls to prevent binary mismatches
            VerifyDllVersion();

            _nativeModel = CallCreate();
            if (_nativeModel == IntPtr.Zero)
                throw new OutOfMemoryException("Failed to create Enhanced PACMAP model");
        }

        /// <summary>
        /// Loads an Enhanced PACMAP model from a file
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
        /// Fits the Enhanced PACMAP model to training data with full customization
        /// </summary>
        /// <param name="data">Training data as 2D array [samples, features]</param>
        /// <param name="embeddingDimension">Target embedding dimension (1-50, default: 2)</param>
        /// <param name="nNeighbors">Number of nearest neighbors (default: 10)</param>
        /// <param name="MNRatio">Medium-near pair ratio (default: 2.0)</param>
        /// <param name="FPRatio">Far-pair ratio (default: 1.0)</param>
        /// <param name="learningRate">Learning rate for optimization (default: 1.0)</param>
        /// <param name="nIters">Number of optimization iterations (default: 100)</param>
        /// <param name="phase1Iters">Phase 1 iterations (default: 100)</param>
        /// <param name="phase2Iters">Phase 2 iterations (default: 100)</param>
        /// <param name="phase3Iters">Phase 3 iterations (default: 100)</param>
        /// <param name="metric">Distance metric to use (default: Euclidean)</param>
        /// <param name="forceExactKnn">Force exact brute-force k-NN instead of HNSW approximation (default: false). Use for validation or small datasets.</param>
        /// <param name="hnswM">HNSW graph degree parameter. -1 = auto-scale based on data size (default: -1)</param>
        /// <param name="hnswEfConstruction">HNSW build quality parameter. -1 = auto-scale (default: -1)</param>
        /// <param name="hnswEfSearch">HNSW query quality parameter. -1 = auto-scale (default: -1)</param>
        /// <param name="useQuantization">Enable 16-bit quantization for memory reduction (default: false)</param>
        /// <param name="randomSeed">Random seed for reproducibility (default: -1 for random)</param>
        /// <param name="autoHNSWParam">Auto-tune HNSW parameters (default: true)</param>
        /// <returns>Embedding coordinates [samples, embeddingDimension]</returns>
        /// <exception cref="ArgumentNullException">Thrown when data is null</exception>
        /// <exception cref="ArgumentException">Thrown when parameters are invalid</exception>
        public float[,] Fit(float[,] data,
                          int embeddingDimension = 2,
                          int nNeighbors = 10,
                          float MNRatio = 2.0f,
                          float FPRatio = 1.0f,
                          float learningRate = 1.0f,
                          int nIters = 100,
                          int phase1Iters = 100,
                          int phase2Iters = 100,
                          int phase3Iters = 100,
                          DistanceMetric metric = DistanceMetric.Euclidean,
                          bool forceExactKnn = false,
                          int hnswM = -1,
                          int hnswEfConstruction = -1,
                          int hnswEfSearch = -1,
                          bool useQuantization = false,
                          int randomSeed = -1,
                          bool autoHNSWParam = true)
        {
            return FitInternal(data, embeddingDimension, nNeighbors, MNRatio, FPRatio, learningRate,
                             nIters, phase1Iters, phase2Iters, phase3Iters, metric, forceExactKnn,
                             progressCallback: null, hnswM, hnswEfConstruction, hnswEfSearch,
                             useQuantization, randomSeed, autoHNSWParam);
        }

        /// <summary>
        /// Fits the Enhanced PACMAP model to training data with progress reporting
        /// </summary>
        /// <param name="data">Training data as 2D array [samples, features]</param>
        /// <param name="progressCallback">Callback function to report training progress</param>
        /// <param name="embeddingDimension">Target embedding dimension (1-50, default: 2)</param>
        /// <param name="nNeighbors">Number of nearest neighbors (default: 10)</param>
        /// <param name="MNRatio">Medium-near pair ratio (default: 2.0)</param>
        /// <param name="FPRatio">Far-pair ratio (default: 1.0)</param>
        /// <param name="learningRate">Learning rate for optimization (default: 1.0)</param>
        /// <param name="nIters">Number of optimization iterations (default: 100)</param>
        /// <param name="phase1Iters">Phase 1 iterations (default: 100)</param>
        /// <param name="phase2Iters">Phase 2 iterations (default: 100)</param>
        /// <param name="phase3Iters">Phase 3 iterations (default: 100)</param>
        /// <param name="metric">Distance metric to use (default: Euclidean)</param>
        /// <param name="forceExactKnn">Force exact brute-force k-NN instead of HNSW approximation (default: false)</param>
        /// <param name="useQuantization">Enable 16-bit quantization for memory reduction (default: false)</param>
        /// <param name="randomSeed">Random seed for reproducibility (default: -1 for random)</param>
        /// <param name="autoHNSWParam">Auto-tune HNSW parameters (default: true)</param>
        /// <returns>Embedding coordinates [samples, embeddingDimension]</returns>
        /// <exception cref="ArgumentNullException">Thrown when data or progressCallback is null</exception>
        /// <exception cref="ArgumentException">Thrown when parameters are invalid</exception>
        public float[,] FitWithProgress(float[,] data,
                                      ProgressCallback progressCallback,
                                      int embeddingDimension = 2,
                                      int nNeighbors = 10,
                                      float MNRatio = 2.0f,
                                      float FPRatio = 1.0f,
                                      float learningRate = 1.0f,
                                      int nIters = 100,
                                      int phase1Iters = 100,
                                      int phase2Iters = 100,
                                      int phase3Iters = 100,
                                      DistanceMetric metric = DistanceMetric.Euclidean,
                                      bool forceExactKnn = false,
                                      bool useQuantization = false,
                                      int randomSeed = -1,
                                      bool autoHNSWParam = true)
        {
            if (progressCallback == null)
                throw new ArgumentNullException(nameof(progressCallback));

            return FitInternal(data, embeddingDimension, nNeighbors, MNRatio, FPRatio, learningRate,
                             nIters, phase1Iters, phase2Iters, phase3Iters, metric, forceExactKnn,
                             progressCallback, hnswM: -1, hnswEfConstruction: -1, hnswEfSearch: -1,
                             useQuantization, randomSeed, autoHNSWParam);
        }

        /// <summary>
        /// Transforms new data using a fitted model (out-of-sample projection)
        /// </summary>
        /// <param name="newData">New data to transform [samples, features]</param>
        /// <returns>Embedding coordinates [samples, embeddingDimension]</returns>
        /// <exception cref="ArgumentNullException">Thrown when newData is null</exception>
        /// <exception cref="InvalidOperationException">Thrown when model is not fitted</exception>
        /// <exception cref="ArgumentException">Thrown when feature dimensions don't match training data</exception>
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
        /// <exception cref="ArgumentNullException">Thrown when newData is null</exception>
        /// <exception cref="InvalidOperationException">Thrown when model is not fitted</exception>
        /// <exception cref="ArgumentException">Thrown when feature dimensions don't match training data</exception>
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
                throw new InvalidOperationException($"Transform failed with error {result}: {errorMessage}. This usually indicates normalization parameter mismatch between fit/save and load operations.");
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

        #endregion

        #region Private Methods

        private float[,] FitInternal(float[,] data,
                                   int embeddingDimension,
                                   int nNeighbors,
                                   float mnRatio,
                                   float fpRatio,
                                   float learningRate,
                                   int nIters,
                                   int phase1Iters,
                                   int phase2Iters,
                                   int phase3Iters,
                                   DistanceMetric metric,
                                   bool forceExactKnn,
                                   ProgressCallback? progressCallback,
                                   int hnswM = -1,
                                   int hnswEfConstruction = -1,
                                   int hnswEfSearch = -1,
                                   bool useQuantization = false,
                                   int randomSeed = -1,
                                   bool autoHNSWParam = true)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            var nSamples = data.GetLength(0);
            var nFeatures = data.GetLength(1);

            if (nSamples <= 0 || nFeatures <= 0)
                throw new ArgumentException("Data must have positive dimensions");

            if (embeddingDimension <= 0 || embeddingDimension > 50)
                throw new ArgumentException("Embedding dimension must be between 1 and 50 (includes 27D support)");

            if (nNeighbors <= 0 || nNeighbors >= nSamples)
                throw new ArgumentException("Number of neighbors must be positive and less than number of samples");

            if (mnRatio <= 0)
                throw new ArgumentException("MN ratio must be positive");

            if (fpRatio <= 0)
                throw new ArgumentException("FP ratio must be positive");

            if (learningRate <= 0)
                throw new ArgumentException("Learning rate must be positive");

            if (nIters <= 0)
                throw new ArgumentException("Number of iterations must be positive");

            if (phase1Iters <= 0 || phase2Iters <= 0 || phase3Iters <= 0)
                throw new ArgumentException("Phase iterations must be positive");

            // Flatten the input data
            var flatData = new float[nSamples * nFeatures];
            for (int i = 0; i < nSamples; i++)
            {
                for (int j = 0; j < nFeatures; j++)
                {
                    flatData[i * nFeatures + j] = data[i, j];
                }
            }

            // Prepare output array
            var embedding = new float[nSamples * embeddingDimension];

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

                result = CallFitWithProgressV2(_nativeModel, flatData, nSamples, nFeatures, embeddingDimension, nNeighbors, mnRatio, fpRatio, learningRate, nIters, phase1Iters, phase2Iters, phase3Iters, metric, embedding, nativeCallback, forceExactKnn ? 1 : 0, hnswM, hnswEfConstruction, hnswEfSearch, useQuantization ? 1 : 0, randomSeed, autoHNSWParam ? 1 : 0);
            }
            else
            {
                // Always use the unified PACMAP function (even without progress callback)
                result = CallFitWithProgressV2(_nativeModel, flatData, nSamples, nFeatures, embeddingDimension, nNeighbors, mnRatio, fpRatio, learningRate, nIters, phase1Iters, phase2Iters, phase3Iters, metric, embedding, null, forceExactKnn ? 1 : 0, hnswM, hnswEfConstruction, hnswEfSearch, useQuantization ? 1 : 0, randomSeed, autoHNSWParam ? 1 : 0);
            }

            ThrowIfError(result);

            // Convert back to 2D array
            return ConvertTo2D(embedding, nSamples, embeddingDimension);
        }

        #endregion

        #region Private Platform-Specific Wrappers

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static IntPtr CallCreate()
        {
            return IsWindows ? WindowsCreate() : LinuxCreate();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallFit(IntPtr model, float[] data, int nObs, int nDim, int embeddingDim, int nNeighbors, float mnRatio, float fpRatio, float learningRate, int nIters, int phase1Iters, int phase2Iters, int phase3Iters, DistanceMetric metric, float[] embedding, int forceExactKnn)
        {
            return IsWindows ? WindowsFit(model, data, nObs, nDim, embeddingDim, nNeighbors, mnRatio, fpRatio, learningRate, nIters, phase1Iters, phase2Iters, phase3Iters, metric, embedding, forceExactKnn)
                             : LinuxFit(model, data, nObs, nDim, embeddingDim, nNeighbors, mnRatio, fpRatio, learningRate, nIters, phase1Iters, phase2Iters, phase3Iters, metric, embedding, forceExactKnn);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallFitWithProgressV2(IntPtr model, float[] data, int nObs, int nDim, int embeddingDim, int nNeighbors, float mnRatio, float fpRatio, float learningRate, int nIters, int phase1Iters, int phase2Iters, int phase3Iters, DistanceMetric metric, float[] embedding, NativeProgressCallbackV2? progressCallback, int forceExactKnn, int M, int efConstruction, int efSearch, int useQuantization, int randomSeed = -1, int autoHNSWParam = 1)
        {
            // Use null-coalescing to provide a default no-op callback if progressCallback is null
            var callback = progressCallback ?? ((phase, current, total, percent, message) => { });
            return IsWindows ? WindowsFitWithProgressV2(model, data, nObs, nDim, embeddingDim, nNeighbors, mnRatio, fpRatio, learningRate, nIters, phase1Iters, phase2Iters, phase3Iters, metric, embedding, callback, forceExactKnn, M, efConstruction, efSearch, useQuantization, randomSeed, autoHNSWParam)
                             : LinuxFitWithProgressV2(model, data, nObs, nDim, embeddingDim, nNeighbors, mnRatio, fpRatio, learningRate, nIters, phase1Iters, phase2Iters, phase3Iters, metric, embedding, callback, forceExactKnn, M, efConstruction, efSearch, useQuantization, randomSeed, autoHNSWParam);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallTransform(IntPtr model, float[] newData, int nNewObs, int nDim, float[] embedding)
        {
            return IsWindows ? WindowsTransform(model, newData, nNewObs, nDim, embedding)
                             : LinuxTransform(model, newData, nNewObs, nDim, embedding);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallTransformDetailed(IntPtr model, float[] newData, int nNewObs, int nDim, float[] embedding, int[] nnIndices, float[] nnDistances, float[] confidenceScore, int[] outlierLevel, float[] percentileRank, float[] zScore)
        {
            return IsWindows ? WindowsTransformDetailed(model, newData, nNewObs, nDim, embedding, nnIndices, nnDistances, confidenceScore, outlierLevel, percentileRank, zScore)
                             : LinuxTransformDetailed(model, newData, nNewObs, nDim, embedding, nnIndices, nnDistances, confidenceScore, outlierLevel, percentileRank, zScore);
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
        private static int CallGetModelInfo(IntPtr model, out int nSamples, out int nFeatures, out int nComponents, out int nNeighbors, out float mnRatio, out float fpRatio, out DistanceMetric metric, out int hnswM, out int hnswEfConstruction, out int hnswEfSearch)
        {
            return IsWindows ? WindowsGetModelInfo(model, out nSamples, out nFeatures, out nComponents, out nNeighbors, out mnRatio, out fpRatio, out metric, out hnswM, out hnswEfConstruction, out hnswEfSearch)
                             : LinuxGetModelInfo(model, out nSamples, out nFeatures, out nComponents, out nNeighbors, out mnRatio, out fpRatio, out metric, out hnswM, out hnswEfConstruction, out hnswEfSearch);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallIsFitted(IntPtr model)
        {
            return IsWindows ? WindowsIsFitted(model) : LinuxIsFitted(model);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static IntPtr CallGetMetricName(DistanceMetric metric)
        {
            return IsWindows ? WindowsGetMetricName(metric) : LinuxGetMetricName(metric);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CallSetGlobalCallback(NativeProgressCallbackV2 callback)
        {
            if (IsWindows) WindowsSetGlobalCallback(callback);
            else LinuxSetGlobalCallback(callback);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CallClearGlobalCallback()
        {
            if (IsWindows) WindowsClearGlobalCallback();
            else LinuxClearGlobalCallback();
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
                PACMAP_ERROR_CRC_MISMATCH => new InvalidDataException(message),
                _ => new Exception($"PACMAP Error ({errorCode}): {message}")
            };
        }

        /// <summary>
        /// Verifies the native DLL version matches the expected C# wrapper version
        /// </summary>
        /// <exception cref="InvalidOperationException">Thrown when DLL version mismatch detected</exception>
        private static void VerifyDllVersion()
        {
            try
            {
                // Get version string from native DLL
                IntPtr versionPtr = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                    ? WindowsGetVersion()
                    : LinuxGetVersion();

                if (versionPtr == IntPtr.Zero)
                {
                    throw new InvalidOperationException(
                        $"❌ CRITICAL: Failed to get DLL version. This indicates a binary mismatch or corrupted DLL.\n" +
                        $"Expected version: {EXPECTED_DLL_VERSION}\n" +
                        $"Please ensure the correct pacmap.dll/libpacmap.so is in the application directory.");
                }

                string actualVersion = Marshal.PtrToStringAnsi(versionPtr) ?? "unknown";

                if (actualVersion != EXPECTED_DLL_VERSION)
                {
                    throw new InvalidOperationException(
                        $"❌ CRITICAL DLL VERSION MISMATCH DETECTED!\n" +
                        $"Expected C# wrapper version: {EXPECTED_DLL_VERSION}\n" +
                        $"Actual native DLL version:    {actualVersion}\n" +
                        $"\n" +
                        $"This mismatch can cause:\n" +
                        $"• Pipeline inconsistencies\n" +
                        $"• Save/load failures\n" +
                        $"• Memory corruption\n" +
                        $"\n" +
                        $"SOLUTION: Rebuild the native library or copy the correct DLL version.\n" +
                        $"Platform: {(RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "Windows" : "Linux")}");
                }

                // Success - log the version match for debugging
                Console.WriteLine($"✅ DLL Version Check PASSED: {actualVersion}");
            }
            catch (DllNotFoundException ex)
            {
                throw new DllNotFoundException(
                    $"❌ CRITICAL: Native PACMAP library not found!\n" +
                    $"Expected: {(RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "pacmap.dll" : "libpacmap.so")}\n" +
                    $"Platform: {RuntimeInformation.OSArchitecture} on {RuntimeInformation.OSDescription}\n" +
                    $"Ensure the native library is in the application directory.\n" +
                    $"Original error: {ex.Message}");
            }
            catch (EntryPointNotFoundException ex)
            {
                throw new InvalidOperationException(
                    $"❌ CRITICAL: DLL is missing version function!\n" +
                    $"This indicates an old or incompatible DLL version.\n" +
                    $"Expected version: {EXPECTED_DLL_VERSION}\n" +
                    $"Original error: {ex.Message}");
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

        #region Global Callback Management

        /// <summary>
        /// Sets a global callback for all UMAP operations to receive warnings, errors, and status messages.
        /// This callback will receive ALL warnings, errors, and status messages from any UMAP operation.
        /// </summary>
        /// <param name="callback">Enhanced progress callback that receives phase, progress, and message information</param>
        public static void SetGlobalCallback(ProgressCallback callback)
        {
            if (callback != null)
            {
                var nativeCallback = new NativeProgressCallbackV2((phase, current, total, percent, message) =>
                {
                    try
                    {
                        callback(phase ?? "Unknown", current, total, percent, message);
                    }
                    catch
                    {
                        // Ignore exceptions in callback to prevent native crashes
                    }
                });

                // Keep reference to prevent garbage collection
                _globalCallback = nativeCallback;
                CallSetGlobalCallback(nativeCallback);
            }
            else
            {
                CallClearGlobalCallback();
                _globalCallback = null;
            }
        }

        /// <summary>
        /// Clears the global callback for UMAP operations
        /// </summary>
        public static void ClearGlobalCallback()
        {
            CallClearGlobalCallback();
            _globalCallback = null;
        }

        private static NativeProgressCallbackV2? _globalCallback;

        #endregion

        #region IDisposable Implementation

        /// <summary>
        /// Releases all resources used by the UMapModel
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
        /// Finalizer for UMapModel to ensure native resources are cleaned up
        /// </summary>
        ~PacMapModel()
        {
            Dispose(false);
        }

        #endregion
    }

    /// <summary>
    /// Comprehensive information about a fitted Enhanced PACMAP model
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
        /// Gets the medium-near pair ratio used during training
        /// </summary>
        public float MnRatio { get; }

        /// <summary>
        /// Gets the far-pair ratio used during training
        /// </summary>
        public float FpRatio { get; }

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

        internal PacMapModelInfo(int trainingSamples, int inputDimension, int outputDimension, int neighbors, float mnRatio, float fpRatio, DistanceMetric metric, int hnswM, int hnswEfConstruction, int hnswEfSearch)
        {
            TrainingSamples = trainingSamples;
            InputDimension = inputDimension;
            OutputDimension = outputDimension;
            Neighbors = neighbors;
            MnRatio = mnRatio;
            FpRatio = fpRatio;
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
            return $"Enhanced PACMAP Model: {TrainingSamples} samples, {InputDimension}D → {OutputDimension}D, " +
                   $"k={Neighbors}, MN_ratio={MnRatio:F3}, FP_ratio={FpRatio:F3}, metric={MetricName}, " +
                   $"HNSW(M={HnswM}, ef_c={HnswEfConstruction}, ef_s={HnswEfSearch})";
        }
    }
}