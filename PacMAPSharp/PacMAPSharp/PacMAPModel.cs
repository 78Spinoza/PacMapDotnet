using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;

namespace PacMAPSharp
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
        /// Angular distance - similar to cosine but with different normalization
        /// </summary>
        Angular = 2,

        /// <summary>
        /// Manhattan distance (L1 norm) - robust to outliers
        /// </summary>
        Manhattan = 3
    }

    /// <summary>
    /// Outlier severity levels for Enhanced PacMAP safety analysis
    /// </summary>
    public enum OutlierLevel
    {
        /// <summary>
        /// Normal data point - within training data distribution
        /// </summary>
        Normal = 0,

        /// <summary>
        /// Unusual data point - outside normal range but acceptable
        /// </summary>
        Unusual = 1,

        /// <summary>
        /// Outlier - significantly different from training data
        /// </summary>
        Outlier = 2,

        /// <summary>
        /// Extreme outlier - very far from training distribution
        /// </summary>
        Extreme = 3
    }

    /// <summary>
    /// Enhanced transform result with comprehensive safety metrics and outlier detection
    /// Available with statistical analysis for production safety
    /// </summary>
    public class TransformResult
    {
        /// <summary>
        /// Gets the projected coordinates in the embedding space
        /// </summary>
        public float[] ProjectionCoordinates { get; }

        /// <summary>
        /// Gets the distance to nearest training samples
        /// </summary>
        public float Distance { get; }

        /// <summary>
        /// Gets whether this sample is considered an outlier
        /// </summary>
        public bool IsOutlier { get; }

        /// <summary>
        /// Gets the outlier severity level
        /// </summary>
        public OutlierLevel Severity { get; }

        /// <summary>
        /// Gets the confidence score for the projection (0.0 - 1.0)
        /// </summary>
        public float ConfidenceScore { get; }

        /// <summary>
        /// Gets the dimensionality of the projection coordinates
        /// </summary>
        public int EmbeddingDimension => ProjectionCoordinates?.Length ?? 0;

        /// <summary>
        /// Gets whether the projection is considered reliable for production use
        /// </summary>
        public bool IsReliable => !IsOutlier && ConfidenceScore >= 0.7f;

        /// <summary>
        /// Gets a human-readable interpretation of the result quality
        /// </summary>
        public string QualityAssessment => Severity switch
        {
            OutlierLevel.Normal => "Excellent - Similar to training data",
            OutlierLevel.Unusual => "Good - Within acceptable range",
            OutlierLevel.Outlier => "Caution - Outside normal range",
            OutlierLevel.Extreme => "Warning - Very different from training data",
            _ => "Unknown"
        };

        internal TransformResult(float[] projectionCoordinates, float distance, bool isOutlier, OutlierLevel severity, float confidenceScore)
        {
            ProjectionCoordinates = projectionCoordinates ?? throw new ArgumentNullException(nameof(projectionCoordinates));
            Distance = distance;
            IsOutlier = isOutlier;
            Severity = severity;
            ConfidenceScore = Math.Max(0f, Math.Min(1f, confidenceScore));
        }

        /// <summary>
        /// Returns a comprehensive string representation of the transform result
        /// </summary>
        public override string ToString()
        {
            return $"TransformResult: {EmbeddingDimension}D embedding, " +
                   $"Distance={Distance:F3}, Outlier={IsOutlier}, " +
                   $"Confidence={ConfidenceScore:F3}, Quality={QualityAssessment}";
        }
    }

    /// <summary>
    /// Enhanced progress callback delegate for training progress reporting
    /// </summary>
    /// <param name="phase">Current phase (e.g., "Neighbor Search", "Optimization")</param>
    /// <param name="current">Current progress counter</param>
    /// <param name="total">Total items to process</param>
    /// <param name="percent">Progress percentage (0-100)</param>
    /// <param name="message">Additional information like time estimates or warnings</param>
    public delegate void ProgressCallback(string phase, int current, int total, float percent, string? message);

    /// <summary>
    /// Enhanced cross-platform C# wrapper for PacMAP dimensionality reduction
    /// Based on the high-performance Rust pacmap crate with enhanced features:
    /// - HNSW-accelerated neighbor search (50-100x faster)
    /// - 16-bit quantization for 85-95% file size reduction
    /// - Statistical normalization with saved parameters
    /// - Distance-based outlier detection for production safety
    /// - Complete model save/load functionality
    /// - Real-time progress reporting
    /// </summary>
    public class PacMAPModel : IDisposable
    {
        #region Platform Detection and DLL Imports

        private static readonly bool IsWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        private static readonly bool IsLinux = RuntimeInformation.IsOSPlatform(OSPlatform.Linux);
        private static readonly bool IsMacOS = RuntimeInformation.IsOSPlatform(OSPlatform.OSX);

        private const string WindowsDll = "pacmap_enhanced.dll";
        private const string LinuxDll = "libpacmap_enhanced.so";
        private const string MacOSDll = "libpacmap_enhanced.dylib";

        // Native progress callback delegate
        private delegate void NativeProgressCallback(
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
        private static extern int WindowsFit(IntPtr model, float[] data, int nObs, int nDim, int embeddingDim, int nNeighbors, DistanceMetric metric, float[] embedding, int useHNSW, int useQuantization);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_fit_with_progress")]
        private static extern int WindowsFitWithProgress(IntPtr model, float[] data, int nObs, int nDim, int embeddingDim, int nNeighbors, DistanceMetric metric, float[] embedding, NativeProgressCallback progressCallback, int useHNSW, int hnswM, int hnswEfConstruction, int hnswEfSearch, int useQuantization);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_transform")]
        private static extern int WindowsTransform(IntPtr model, float[] newData, int nNewObs, int nDim, float[] embedding);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_transform_with_stats")]
        private static extern int WindowsTransformWithStats(IntPtr model, float[] newData, int nNewObs, int nDim, float[] embedding, float[] distances, int[] outlierFlags, float[] confidenceScores);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_save_model")]
        private static extern int WindowsSaveModel(IntPtr model, [MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_load_model")]
        private static extern IntPtr WindowsLoadModel([MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_destroy")]
        private static extern void WindowsDestroy(IntPtr model);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_error_message")]
        private static extern IntPtr WindowsGetErrorMessage(int errorCode);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_model_info")]
        private static extern int WindowsGetModelInfo(IntPtr model, out int nVertices, out int nDim, out int embeddingDim, out int nNeighbors, out DistanceMetric metric, out int useHNSW, out int hnswM, out int hnswEfConstruction, out int hnswEfSearch);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_is_fitted")]
        private static extern int WindowsIsFitted(IntPtr model);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_version")]
        private static extern IntPtr WindowsGetVersion();

        // Linux P/Invoke declarations
        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_create")]
        private static extern IntPtr LinuxCreate();

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_fit")]
        private static extern int LinuxFit(IntPtr model, float[] data, int nObs, int nDim, int embeddingDim, int nNeighbors, DistanceMetric metric, float[] embedding, int useHNSW, int useQuantization);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_fit_with_progress")]
        private static extern int LinuxFitWithProgress(IntPtr model, float[] data, int nObs, int nDim, int embeddingDim, int nNeighbors, DistanceMetric metric, float[] embedding, NativeProgressCallback progressCallback, int useHNSW, int hnswM, int hnswEfConstruction, int hnswEfSearch, int useQuantization);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_transform")]
        private static extern int LinuxTransform(IntPtr model, float[] newData, int nNewObs, int nDim, float[] embedding);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_transform_with_stats")]
        private static extern int LinuxTransformWithStats(IntPtr model, float[] newData, int nNewObs, int nDim, float[] embedding, float[] distances, int[] outlierFlags, float[] confidenceScores);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_save_model")]
        private static extern int LinuxSaveModel(IntPtr model, [MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_load_model")]
        private static extern IntPtr LinuxLoadModel([MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_destroy")]
        private static extern void LinuxDestroy(IntPtr model);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_error_message")]
        private static extern IntPtr LinuxGetErrorMessage(int errorCode);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_model_info")]
        private static extern int LinuxGetModelInfo(IntPtr model, out int nVertices, out int nDim, out int embeddingDim, out int nNeighbors, out DistanceMetric metric, out int useHNSW, out int hnswM, out int hnswEfConstruction, out int hnswEfSearch);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_is_fitted")]
        private static extern int LinuxIsFitted(IntPtr model);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_version")]
        private static extern IntPtr LinuxGetVersion();

        // macOS P/Invoke declarations
        [DllImport(MacOSDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_create")]
        private static extern IntPtr MacOSCreate();

        [DllImport(MacOSDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_fit")]
        private static extern int MacOSFit(IntPtr model, float[] data, int nObs, int nDim, int embeddingDim, int nNeighbors, DistanceMetric metric, float[] embedding, int useHNSW, int useQuantization);

        [DllImport(MacOSDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_fit_with_progress")]
        private static extern int MacOSFitWithProgress(IntPtr model, float[] data, int nObs, int nDim, int embeddingDim, int nNeighbors, DistanceMetric metric, float[] embedding, NativeProgressCallback progressCallback, int useHNSW, int hnswM, int hnswEfConstruction, int hnswEfSearch, int useQuantization);

        [DllImport(MacOSDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_transform")]
        private static extern int MacOSTransform(IntPtr model, float[] newData, int nNewObs, int nDim, float[] embedding);

        [DllImport(MacOSDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_transform_with_stats")]
        private static extern int MacOSTransformWithStats(IntPtr model, float[] newData, int nNewObs, int nDim, float[] embedding, float[] distances, int[] outlierFlags, float[] confidenceScores);

        [DllImport(MacOSDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_save_model")]
        private static extern int MacOSSaveModel(IntPtr model, [MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(MacOSDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_load_model")]
        private static extern IntPtr MacOSLoadModel([MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(MacOSDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_destroy")]
        private static extern void MacOSDestroy(IntPtr model);

        [DllImport(MacOSDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_error_message")]
        private static extern IntPtr MacOSGetErrorMessage(int errorCode);

        [DllImport(MacOSDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_model_info")]
        private static extern int MacOSGetModelInfo(IntPtr model, out int nVertices, out int nDim, out int embeddingDim, out int nNeighbors, out DistanceMetric metric, out int useHNSW, out int hnswM, out int hnswEfConstruction, out int hnswEfSearch);

        [DllImport(MacOSDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_is_fitted")]
        private static extern int MacOSIsFitted(IntPtr model);

        [DllImport(MacOSDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_version")]
        private static extern IntPtr MacOSGetVersion();

        #endregion

        #region Constants

        // Expected DLL version - must match Rust library version
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
        public PacMAPModelInfo ModelInfo
        {
            get
            {
                if (!IsFitted)
                    throw new InvalidOperationException("Model must be fitted before accessing model info");

                var result = CallGetModelInfo(_nativeModel, out var nVertices, out var nDim, out var embeddingDim, out var nNeighbors, out var metric, out var useHNSW, out var hnswM, out var hnswEfConstruction, out var hnswEfSearch);
                ThrowIfError(result);

                return new PacMAPModelInfo(nVertices, nDim, embeddingDim, nNeighbors, metric, useHNSW != 0, hnswM, hnswEfConstruction, hnswEfSearch);
            }
        }

        #endregion

        #region Constructors

        /// <summary>
        /// Creates a new Enhanced PacMAP model instance
        /// </summary>
        public PacMAPModel()
        {
            // Verify DLL version before any native calls
            VerifyDllVersion();

            _nativeModel = CallCreate();
            if (_nativeModel == IntPtr.Zero)
                throw new OutOfMemoryException("Failed to create Enhanced PacMAP model");
        }

        /// <summary>
        /// Loads an Enhanced PacMAP model from a file
        /// </summary>
        /// <param name="filename">Path to the model file</param>
        /// <returns>A new PacMAPModel instance loaded from the specified file</returns>
        public static PacMAPModel LoadModel(string filename)
        {
            if (string.IsNullOrEmpty(filename))
                throw new ArgumentException("Filename cannot be null or empty", nameof(filename));

            if (!File.Exists(filename))
                throw new FileNotFoundException($"Model file not found: {filename}");

            var model = new PacMAPModel();
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
        /// Fits the Enhanced PacMAP model to training data
        /// </summary>
        /// <param name="data">Training data as 2D array [samples, features]</param>
        /// <param name="embeddingDimension">Target embedding dimension (default: 2)</param>
        /// <param name="nNeighbors">Number of nearest neighbors (default: 10)</param>
        /// <param name="metric">Distance metric to use (default: Euclidean)</param>
        /// <param name="useHNSW">Enable HNSW acceleration (default: true)</param>
        /// <param name="useQuantization">Enable 16-bit quantization (default: false)</param>
        /// <returns>Embedding coordinates [samples, embeddingDimension]</returns>
        public float[,] Fit(float[,] data,
                          int embeddingDimension = 2,
                          int nNeighbors = 10,
                          DistanceMetric metric = DistanceMetric.Euclidean,
                          bool useHNSW = true,
                          bool useQuantization = false)
        {
            return FitInternal(data, embeddingDimension, nNeighbors, metric, useHNSW, null, -1, -1, -1, useQuantization);
        }

        /// <summary>
        /// Fits the Enhanced PacMAP model to training data with progress reporting
        /// </summary>
        /// <param name="data">Training data as 2D array [samples, features]</param>
        /// <param name="progressCallback">Callback function to report training progress</param>
        /// <param name="embeddingDimension">Target embedding dimension (default: 2)</param>
        /// <param name="nNeighbors">Number of nearest neighbors (default: 10)</param>
        /// <param name="metric">Distance metric to use (default: Euclidean)</param>
        /// <param name="useHNSW">Enable HNSW acceleration (default: true)</param>
        /// <param name="hnswM">HNSW graph degree parameter (default: auto)</param>
        /// <param name="hnswEfConstruction">HNSW build quality parameter (default: auto)</param>
        /// <param name="hnswEfSearch">HNSW query quality parameter (default: auto)</param>
        /// <param name="useQuantization">Enable 16-bit quantization (default: false)</param>
        /// <returns>Embedding coordinates [samples, embeddingDimension]</returns>
        public float[,] FitWithProgress(float[,] data,
                                      ProgressCallback progressCallback,
                                      int embeddingDimension = 2,
                                      int nNeighbors = 10,
                                      DistanceMetric metric = DistanceMetric.Euclidean,
                                      bool useHNSW = true,
                                      int hnswM = -1,
                                      int hnswEfConstruction = -1,
                                      int hnswEfSearch = -1,
                                      bool useQuantization = false)
        {
            if (progressCallback == null)
                throw new ArgumentNullException(nameof(progressCallback));

            return FitInternal(data, embeddingDimension, nNeighbors, metric, useHNSW, progressCallback, hnswM, hnswEfConstruction, hnswEfSearch, useQuantization);
        }

        /// <summary>
        /// Transforms new data using a fitted model
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
            ThrowIfError(result);

            // Convert back to 2D array
            return ConvertTo2D(embedding, nNewSamples, modelInfo.OutputDimension);
        }

        /// <summary>
        /// Transforms new data using a fitted model with comprehensive statistical analysis
        /// </summary>
        /// <param name="newData">New data to transform [samples, features]</param>
        /// <returns>Array of TransformResult objects with embedding coordinates and statistics</returns>
        public TransformResult[] TransformWithStatistics(float[,] newData)
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
            var distances = new float[nNewSamples];
            var outlierFlags = new int[nNewSamples];
            var confidenceScores = new float[nNewSamples];

            // Call enhanced native function
            var result = CallTransformWithStats(_nativeModel, flatNewData, nNewSamples, nFeatures,
                                              embedding, distances, outlierFlags, confidenceScores);
            ThrowIfError(result);

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

                var isOutlier = outlierFlags[i] != 0;
                var severity = isOutlier ? (distances[i] > 2.0f ? OutlierLevel.Extreme : OutlierLevel.Outlier) : OutlierLevel.Normal;

                results[i] = new TransformResult(
                    projectionCoords,
                    distances[i],
                    isOutlier,
                    severity,
                    confidenceScores[i]
                );
            }

            return results;
        }

        /// <summary>
        /// Saves the fitted model to a file
        /// </summary>
        /// <param name="filename">Path where to save the model</param>
        public void SaveModel(string filename)
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

        private float[,] FitInternal(float[,] data,
                                   int embeddingDimension,
                                   int nNeighbors,
                                   DistanceMetric metric,
                                   bool useHNSW,
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
                // Create native callback wrapper
                NativeProgressCallback nativeCallback = (phase, current, total, percent, message) =>
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

                result = CallFitWithProgress(_nativeModel, flatData, nSamples, nFeatures, embeddingDimension, nNeighbors, metric, embedding, nativeCallback, useHNSW ? 1 : 0, hnswM, hnswEfConstruction, hnswEfSearch, useQuantization ? 1 : 0);
            }
            else
            {
                result = CallFit(_nativeModel, flatData, nSamples, nFeatures, embeddingDimension, nNeighbors, metric, embedding, useHNSW ? 1 : 0, useQuantization ? 1 : 0);
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
            return IsWindows ? WindowsCreate() : IsLinux ? LinuxCreate() : MacOSCreate();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallFit(IntPtr model, float[] data, int nObs, int nDim, int embeddingDim, int nNeighbors, DistanceMetric metric, float[] embedding, int useHNSW, int useQuantization)
        {
            return IsWindows ? WindowsFit(model, data, nObs, nDim, embeddingDim, nNeighbors, metric, embedding, useHNSW, useQuantization)
                             : IsLinux ? LinuxFit(model, data, nObs, nDim, embeddingDim, nNeighbors, metric, embedding, useHNSW, useQuantization)
                             : MacOSFit(model, data, nObs, nDim, embeddingDim, nNeighbors, metric, embedding, useHNSW, useQuantization);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallFitWithProgress(IntPtr model, float[] data, int nObs, int nDim, int embeddingDim, int nNeighbors, DistanceMetric metric, float[] embedding, NativeProgressCallback progressCallback, int useHNSW, int hnswM, int hnswEfConstruction, int hnswEfSearch, int useQuantization)
        {
            return IsWindows ? WindowsFitWithProgress(model, data, nObs, nDim, embeddingDim, nNeighbors, metric, embedding, progressCallback, useHNSW, hnswM, hnswEfConstruction, hnswEfSearch, useQuantization)
                             : IsLinux ? LinuxFitWithProgress(model, data, nObs, nDim, embeddingDim, nNeighbors, metric, embedding, progressCallback, useHNSW, hnswM, hnswEfConstruction, hnswEfSearch, useQuantization)
                             : MacOSFitWithProgress(model, data, nObs, nDim, embeddingDim, nNeighbors, metric, embedding, progressCallback, useHNSW, hnswM, hnswEfConstruction, hnswEfSearch, useQuantization);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallTransform(IntPtr model, float[] newData, int nNewObs, int nDim, float[] embedding)
        {
            return IsWindows ? WindowsTransform(model, newData, nNewObs, nDim, embedding)
                             : IsLinux ? LinuxTransform(model, newData, nNewObs, nDim, embedding)
                             : MacOSTransform(model, newData, nNewObs, nDim, embedding);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallTransformWithStats(IntPtr model, float[] newData, int nNewObs, int nDim, float[] embedding, float[] distances, int[] outlierFlags, float[] confidenceScores)
        {
            return IsWindows ? WindowsTransformWithStats(model, newData, nNewObs, nDim, embedding, distances, outlierFlags, confidenceScores)
                             : IsLinux ? LinuxTransformWithStats(model, newData, nNewObs, nDim, embedding, distances, outlierFlags, confidenceScores)
                             : MacOSTransformWithStats(model, newData, nNewObs, nDim, embedding, distances, outlierFlags, confidenceScores);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallSaveModel(IntPtr model, string filename)
        {
            return IsWindows ? WindowsSaveModel(model, filename) : IsLinux ? LinuxSaveModel(model, filename) : MacOSSaveModel(model, filename);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static IntPtr CallLoadModel(string filename)
        {
            return IsWindows ? WindowsLoadModel(filename) : IsLinux ? LinuxLoadModel(filename) : MacOSLoadModel(filename);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CallDestroy(IntPtr model)
        {
            if (IsWindows) WindowsDestroy(model);
            else if (IsLinux) LinuxDestroy(model);
            else MacOSDestroy(model);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static string CallGetErrorMessage(int errorCode)
        {
            var ptr = IsWindows ? WindowsGetErrorMessage(errorCode) : IsLinux ? LinuxGetErrorMessage(errorCode) : MacOSGetErrorMessage(errorCode);
            return Marshal.PtrToStringAnsi(ptr) ?? "Unknown error";
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallGetModelInfo(IntPtr model, out int nVertices, out int nDim, out int embeddingDim, out int nNeighbors, out DistanceMetric metric, out int useHNSW, out int hnswM, out int hnswEfConstruction, out int hnswEfSearch)
        {
            return IsWindows ? WindowsGetModelInfo(model, out nVertices, out nDim, out embeddingDim, out nNeighbors, out metric, out useHNSW, out hnswM, out hnswEfConstruction, out hnswEfSearch)
                             : IsLinux ? LinuxGetModelInfo(model, out nVertices, out nDim, out embeddingDim, out nNeighbors, out metric, out useHNSW, out hnswM, out hnswEfConstruction, out hnswEfSearch)
                             : MacOSGetModelInfo(model, out nVertices, out nDim, out embeddingDim, out nNeighbors, out metric, out useHNSW, out hnswM, out hnswEfConstruction, out hnswEfSearch);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallIsFitted(IntPtr model)
        {
            return IsWindows ? WindowsIsFitted(model) : IsLinux ? LinuxIsFitted(model) : MacOSIsFitted(model);
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
                // Get version string from native DLL
                IntPtr versionPtr = IsWindows ? WindowsGetVersion() : IsLinux ? LinuxGetVersion() : MacOSGetVersion();

                if (versionPtr == IntPtr.Zero)
                {
                    throw new InvalidOperationException(
                        $"❌ CRITICAL: Failed to get DLL version. This indicates a binary mismatch or corrupted library.\n" +
                        $"Expected version: {EXPECTED_DLL_VERSION}\n" +
                        $"Please ensure the correct native library is in the application directory.");
                }

                string actualVersion = Marshal.PtrToStringAnsi(versionPtr) ?? "unknown";

                if (actualVersion != EXPECTED_DLL_VERSION)
                {
                    throw new InvalidOperationException(
                        $"❌ CRITICAL LIBRARY VERSION MISMATCH DETECTED!\n" +
                        $"Expected C# wrapper version: {EXPECTED_DLL_VERSION}\n" +
                        $"Actual native library version: {actualVersion}\n" +
                        $"\n" +
                        $"This mismatch can cause:\n" +
                        $"• API inconsistencies\n" +
                        $"• Save/load failures\n" +
                        $"• Memory corruption\n" +
                        $"\n" +
                        $"SOLUTION: Rebuild the native library or copy the correct version.\n" +
                        $"Platform: {RuntimeInformation.OSDescription}");
                }

                // Success - log the version match for debugging
                Console.WriteLine($"✅ Library Version Check PASSED: {actualVersion}");
            }
            catch (DllNotFoundException ex)
            {
                var expectedLib = IsWindows ? "pacmap_enhanced.dll" : IsLinux ? "libpacmap_enhanced.so" : "libpacmap_enhanced.dylib";
                throw new DllNotFoundException(
                    $"❌ CRITICAL: Native PacMAP library not found!\n" +
                    $"Expected: {expectedLib}\n" +
                    $"Platform: {RuntimeInformation.OSArchitecture} on {RuntimeInformation.OSDescription}\n" +
                    $"Ensure the native library is in the application directory.\n" +
                    $"Original error: {ex.Message}");
            }
            catch (EntryPointNotFoundException ex)
            {
                throw new InvalidOperationException(
                    $"❌ CRITICAL: Library is missing version function!\n" +
                    $"This indicates an old or incompatible library version.\n" +
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

        #region IDisposable Implementation

        /// <summary>
        /// Releases all resources used by the PacMAPModel
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Releases the unmanaged resources and optionally releases the managed resources
        /// </summary>
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
        /// Finalizer for PacMAPModel to ensure native resources are cleaned up
        /// </summary>
        ~PacMAPModel()
        {
            Dispose(false);
        }

        #endregion
    }

    /// <summary>
    /// Comprehensive information about a fitted Enhanced PacMAP model
    /// </summary>
    public readonly struct PacMAPModelInfo
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
        /// Gets the dimensionality of the output embedding
        /// </summary>
        public int OutputDimension { get; }

        /// <summary>
        /// Gets the number of nearest neighbors used during training
        /// </summary>
        public int Neighbors { get; }

        /// <summary>
        /// Gets the distance metric used during training
        /// </summary>
        public DistanceMetric Metric { get; }

        /// <summary>
        /// Gets whether HNSW acceleration was used
        /// </summary>
        public bool UseHNSW { get; }

        /// <summary>
        /// Gets the HNSW graph degree parameter
        /// </summary>
        public int HnswM { get; }

        /// <summary>
        /// Gets the HNSW construction quality parameter
        /// </summary>
        public int HnswEfConstruction { get; }

        /// <summary>
        /// Gets the HNSW search quality parameter
        /// </summary>
        public int HnswEfSearch { get; }

        /// <summary>
        /// Gets the human-readable name of the distance metric
        /// </summary>
        public string MetricName => Metric.ToString();

        internal PacMAPModelInfo(int trainingSamples, int inputDimension, int outputDimension, int neighbors, DistanceMetric metric, bool useHNSW, int hnswM, int hnswEfConstruction, int hnswEfSearch)
        {
            TrainingSamples = trainingSamples;
            InputDimension = inputDimension;
            OutputDimension = outputDimension;
            Neighbors = neighbors;
            Metric = metric;
            UseHNSW = useHNSW;
            HnswM = hnswM;
            HnswEfConstruction = hnswEfConstruction;
            HnswEfSearch = hnswEfSearch;
        }

        /// <summary>
        /// Returns a comprehensive string representation of the model information
        /// </summary>
        public override string ToString()
        {
            return $"Enhanced PacMAP Model: {TrainingSamples} samples, {InputDimension}D → {OutputDimension}D, " +
                   $"k={Neighbors}, metric={MetricName}, " +
                   $"HNSW={UseHNSW}" + (UseHNSW ? $"(M={HnswM}, ef_c={HnswEfConstruction}, ef_s={HnswEfSearch})" : "");
        }
    }
}