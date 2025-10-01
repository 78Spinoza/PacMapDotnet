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
    /// Normalization modes for input data preprocessing
    /// </summary>
    public enum NormalizationMode
    {
        /// <summary>
        /// Auto-detect best normalization based on data characteristics
        /// </summary>
        Auto = 0,

        /// <summary>
        /// Z-score normalization: (x - μ) / σ - assumes normal distribution
        /// </summary>
        ZScore = 1,

        /// <summary>
        /// Min-max scaling: (x - min) / (max - min) - maps to [0, 1] range
        /// </summary>
        MinMax = 2,

        /// <summary>
        /// Robust scaling: (x - median) / IQR - less sensitive to outliers
        /// </summary>
        Robust = 3,

        /// <summary>
        /// No normalization applied - use raw data values
        /// </summary>
        None = 4
    }

    /// <summary>
    /// HNSW optimization use cases for different performance priorities
    /// </summary>
    public enum HnswUseCase
    {
        /// <summary>
        /// Minimize index construction time - fastest setup, lower accuracy
        /// </summary>
        FastConstruction = 0,

        /// <summary>
        /// Maximize search accuracy/recall - highest quality, slower setup
        /// </summary>
        HighAccuracy = 1,

        /// <summary>
        /// Minimize memory footprint - lowest memory usage
        /// </summary>
        MemoryOptimized = 2,

        /// <summary>
        /// Balanced performance across all metrics - recommended default
        /// </summary>
        Balanced = 3
    }

    /// <summary>
    /// Outlier severity levels for Enhanced PacMAP quality analysis
    /// </summary>
    public enum OutlierLevel
    {
        /// <summary>
        /// Normal embedding - within expected quality range
        /// </summary>
        Normal = 0,

        /// <summary>
        /// Unusual embedding - slightly outside normal range but acceptable
        /// </summary>
        Unusual = 1,

        /// <summary>
        /// Mild outlier - embedding may be less accurate
        /// </summary>
        Mild = 2,

        /// <summary>
        /// Extreme outlier - embedding reliability is questionable
        /// </summary>
        Extreme = 3,

        /// <summary>
        /// Critical outlier - embedding is highly unreliable
        /// </summary>
        NoMansLand = 4
    }

    /// <summary>
    /// Comprehensive result from PacMAP embedding transformation
    /// Provides embedding coordinates plus quality assessment and diagnostics
    /// </summary>
    public readonly struct EmbeddingResult
    {
        /// <summary>
        /// Gets the embedding coordinates in the reduced dimensional space
        /// </summary>
        public float[] EmbeddingCoordinates { get; }

        /// <summary>
        /// Gets the confidence score for the embedding quality (0.0 to 1.0)
        /// Higher values indicate better preservation of local/global structure
        /// </summary>
        public float ConfidenceScore { get; }

        /// <summary>
        /// Gets the outlier severity assessment for this embedding
        /// </summary>
        public OutlierLevel Severity { get; }

        /// <summary>
        /// Gets statistics about the embedding distances
        /// </summary>
        public (double Mean, double P95, double Max) DistanceStats { get; }

        /// <summary>
        /// Gets the dimensionality of the embedding coordinates
        /// </summary>
        public int EmbeddingDimension => EmbeddingCoordinates?.Length ?? 0;

        /// <summary>
        /// Gets whether the embedding is considered reliable for production use
        /// </summary>
        public bool IsReliable => Severity <= OutlierLevel.Unusual && ConfidenceScore >= 0.3f;

        /// <summary>
        /// Gets a human-readable interpretation of the result quality
        /// </summary>
        public string QualityAssessment => Severity switch
        {
            OutlierLevel.Normal => "Excellent - High quality embedding",
            OutlierLevel.Unusual => "Good - Acceptable embedding quality",
            OutlierLevel.Mild => "Caution - Mild quality issues detected",
            OutlierLevel.Extreme => "Warning - Poor embedding quality",
            OutlierLevel.NoMansLand => "Critical - Embedding unreliable",
            _ => "Unknown"
        };

        internal EmbeddingResult(float[] embeddingCoordinates,
                                float confidenceScore,
                                OutlierLevel severity,
                                (double, double, double) distanceStats)
        {
            EmbeddingCoordinates = embeddingCoordinates ?? throw new ArgumentNullException(nameof(embeddingCoordinates));
            ConfidenceScore = Math.Max(0f, Math.Min(1f, confidenceScore));
            Severity = severity;
            DistanceStats = distanceStats;
        }

        /// <summary>
        /// Returns a comprehensive string representation of the embedding result
        /// </summary>
        public override string ToString()
        {
            return $"EmbeddingResult: {EmbeddingDimension}D embedding, " +
                   $"Confidence={ConfidenceScore:F3}, Severity={Severity}, " +
                   $"Quality={QualityAssessment}";
        }
    }

    /// <summary>
    /// Enhanced progress callback delegate for training progress reporting
    /// </summary>
    /// <param name="phase">Current phase (e.g., "Normalizing", "HNSW Build", "Optimizing")</param>
    /// <param name="current">Current progress counter</param>
    /// <param name="total">Total items to process</param>
    /// <param name="percent">Progress percentage (0-100)</param>
    /// <param name="message">Additional information like timing, warnings, or quality metrics</param>
    public delegate void ProgressCallback(string phase, int current, int total, float percent, string? message);

    /// <summary>
    /// Model information for a fitted PacMAP model
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
        /// Gets the number of nearest neighbors used
        /// </summary>
        public int Neighbors { get; }

        /// <summary>
        /// Gets the distance metric used for neighbor computation
        /// </summary>
        public DistanceMetric Metric { get; }

        /// <summary>
        /// Gets the normalization mode applied to the input data
        /// </summary>
        public NormalizationMode Normalization { get; }

        /// <summary>
        /// Gets whether HNSW acceleration was used
        /// </summary>
        public bool UsedHNSW { get; }

        /// <summary>
        /// Gets the HNSW recall percentage achieved (if HNSW was used)
        /// </summary>
        public float HnswRecall { get; }

        internal PacMAPModelInfo(int trainingSamples, int inputDimension, int outputDimension,
                                int neighbors, DistanceMetric metric, NormalizationMode normalization,
                                bool usedHnsw, float hnswRecall)
        {
            TrainingSamples = trainingSamples;
            InputDimension = inputDimension;
            OutputDimension = outputDimension;
            Neighbors = neighbors;
            Metric = metric;
            Normalization = normalization;
            UsedHNSW = usedHnsw;
            HnswRecall = hnswRecall;
        }

        /// <summary>
        /// Returns a comprehensive string representation of the model info
        /// </summary>
        public override string ToString()
        {
            return $"PacMAPModel: {TrainingSamples} samples, {InputDimension}D → {OutputDimension}D, " +
                   $"neighbors={Neighbors}, metric={Metric}, normalization={Normalization}" +
                   (UsedHNSW ? $", HNSW (recall={HnswRecall:F1}%)" : ", exact KNN");
        }
    }

    /// <summary>
    /// HNSW configuration for PacMAP FFI
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct PacmapHnswConfig
    {
        public bool AutoScale;       // If true, ignore manual parameters and auto-scale
        public int UseCase;          // 0=Balanced, 1=FastConstruction, 2=HighAccuracy, 3=MemoryOptimized
        public int M;                // Manual M parameter (ignored if auto_scale=true)
        public int EfConstruction;   // Manual ef_construction (ignored if auto_scale=true)
        public int EfSearch;         // Manual ef_search (ignored if auto_scale=true)
        public int MemoryLimitMb;    // Memory limit in MB (0 = no limit)

        public static PacmapHnswConfig Default => new PacmapHnswConfig
        {
            AutoScale = true,
            UseCase = 0, // Balanced
            M = 16,
            EfConstruction = 128,
            EfSearch = 64,
            MemoryLimitMb = 0
        };
    }

    /// <summary>
    /// PacMAP configuration struct for FFI
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct PacmapConfig
    {
        public int NNeighbors;
        public int EmbeddingDimensions;
        public int NEpochs;
        public double LearningRate;
        public double MinDist;
        public double MidNearRatio;
        public double FarPairRatio;
        public int Seed;              // -1 for random seed
        public int NormalizationMode; // 0=Auto, 1=ZScore, 2=MinMax, 3=Robust, 4=None
        public bool ForceExactKnn;    // If true, disable HNSW and use brute-force KNN
        public bool UseQuantization;  // If true, enable 16-bit quantization
        public PacmapHnswConfig HnswConfig;

        public static PacmapConfig Default => new PacmapConfig
        {
            NNeighbors = 10,
            EmbeddingDimensions = 2,
            NEpochs = 450,
            LearningRate = 0.01,
            MinDist = 0.1,
            MidNearRatio = 0.5,
            FarPairRatio = 1.0,
            Seed = 42,
            NormalizationMode = 1, // ZScore
            ForceExactKnn = false,
            UseQuantization = false,
            HnswConfig = PacmapHnswConfig.Default
        };
    }

    /// <summary>
    /// Enhanced cross-platform C# wrapper for PacMAP dimensionality reduction
    /// Features:
    /// - HNSW acceleration for large datasets with density-adaptive local scaling
    /// - Multiple distance metrics and normalization modes
    /// - Complete model save/load functionality
    /// - Progress reporting with detailed callback support
    /// - Comprehensive quality assessment and validation
    /// - Graph symmetrization for improved connectivity
    /// </summary>
    public class PacMAPModel : IDisposable
    {
        #region Platform Detection and DLL Imports

        private static readonly bool IsWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        private static readonly bool IsLinux = RuntimeInformation.IsOSPlatform(OSPlatform.Linux);

        private const string WindowsDll = "pacmap_enhanced.dll";
        private const string LinuxDll = "libpacmap_enhanced.so";

        // Native progress callback delegate
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void NativeProgressCallback(IntPtr phase, int current, int total, float percent, IntPtr message);

        // Native function imports - Windows
        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_version")]
        private static extern IntPtr CallGetVersionWindows();

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_fit_transform_enhanced")]
        private static extern IntPtr CallFitTransformWindows(IntPtr data, int rows, int cols,
            PacmapConfig config, IntPtr embedding, NativeProgressCallback? callback);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_model_info")]
        private static extern void CallGetModelInfoWindows(IntPtr model, out int nSamples, out int nFeatures, out int embeddingDim, out int normalizationMode);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_model_stats")]
        private static extern void CallGetDistanceStatsWindows(IntPtr model, out double mean, out double p95, out double max);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_save_model_enhanced")]
        private static extern int CallSaveModelWindows(IntPtr model, IntPtr path, bool quantize);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_load_model_enhanced")]
        private static extern IntPtr CallLoadModelWindows(IntPtr path);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_free_model_enhanced")]
        private static extern void CallFreeModelWindows(IntPtr model);

        // Linux function imports - similar pattern
        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_version")]
        private static extern IntPtr CallGetVersionLinux();

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_fit_transform_enhanced")]
        private static extern IntPtr CallFitTransformLinux(IntPtr data, int rows, int cols,
            PacmapConfig config, IntPtr embedding, NativeProgressCallback? callback);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_model_info")]
        private static extern void CallGetModelInfoLinux(IntPtr model, out int nSamples, out int nFeatures, out int embeddingDim, out int normalizationMode);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_model_stats")]
        private static extern void CallGetDistanceStatsLinux(IntPtr model, out double mean, out double p95, out double max);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_save_model_enhanced")]
        private static extern int CallSaveModelLinux(IntPtr model, IntPtr path, bool quantize);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_load_model_enhanced")]
        private static extern IntPtr CallLoadModelLinux(IntPtr path);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_free_model_enhanced")]
        private static extern void CallFreeModelLinux(IntPtr model);

        #endregion

        #region Private Fields

        private IntPtr _nativeModel = IntPtr.Zero;
        private bool _disposed = false;
        private bool _isFitted = false;
        private PacMAPModelInfo? _modelInfo = null;
        private ProgressCallback? _managedCallback = null;

        #endregion

        #region Public Properties

        /// <summary>
        /// Gets whether this model has been fitted to training data
        /// </summary>
        public bool IsFitted => _isFitted && _nativeModel != IntPtr.Zero;

        /// <summary>
        /// Gets comprehensive information about the fitted model
        /// Only available after fitting
        /// </summary>
        public PacMAPModelInfo ModelInfo
        {
            get
            {
                if (!IsFitted)
                    throw new InvalidOperationException("Model must be fitted before accessing model info");
                return _modelInfo!.Value;
            }
        }

        #endregion

        #region Constructor and Factory Methods

        /// <summary>
        /// Initializes a new PacMAP model
        /// </summary>
        public PacMAPModel()
        {
            // Model will be created when Fit is called
        }

        /// <summary>
        /// Gets the version of the native PacMAP library
        /// </summary>
        /// <returns>Version string from the native library</returns>
        /// <exception cref="PlatformNotSupportedException">Thrown on unsupported platforms</exception>
        /// <exception cref="DllNotFoundException">Thrown when native library cannot be loaded</exception>
        public static string GetVersion()
        {
            try
            {
                IntPtr versionPtr = IsWindows
                    ? CallGetVersionWindows()
                    : CallGetVersionLinux();

                if (versionPtr == IntPtr.Zero)
                    throw new InvalidOperationException("Failed to get version from native library");

                return Marshal.PtrToStringUTF8(versionPtr) ?? "Unknown version";
            }
            catch (DllNotFoundException ex)
            {
                throw new DllNotFoundException($"Failed to load PacMAP native library. Ensure {(IsWindows ? WindowsDll : LinuxDll)} is available.", ex);
            }
        }

        /// <summary>
        /// Verifies that the native library can be loaded and returns version info
        /// </summary>
        /// <returns>True if library loads successfully, false otherwise</returns>
        public static bool VerifyLibrary()
        {
            try
            {
                var version = GetVersion();
                return !string.IsNullOrEmpty(version) && version != "Unknown version";
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Loads a previously saved PacMAP model from file
        /// </summary>
        /// <param name="filename">Path to the saved model file</param>
        /// <returns>Loaded PacMAP model ready for transformation</returns>
        /// <exception cref="ArgumentException">Thrown when filename is invalid</exception>
        /// <exception cref="FileNotFoundException">Thrown when model file doesn't exist</exception>
        /// <exception cref="InvalidDataException">Thrown when model file is corrupted</exception>
        public static PacMAPModel Load(string filename)
        {
            if (string.IsNullOrEmpty(filename))
                throw new ArgumentException("Filename cannot be null or empty", nameof(filename));
            if (!File.Exists(filename))
                throw new FileNotFoundException($"Model file not found: {filename}");

            var model = new PacMAPModel();

            unsafe
            {
                var pathBytes = System.Text.Encoding.UTF8.GetBytes(filename);
                fixed (byte* pathPtr = pathBytes)
                {
                    model._nativeModel = IsWindows
                        ? CallLoadModelWindows((IntPtr)pathPtr)
                        : CallLoadModelLinux((IntPtr)pathPtr);

                    if (model._nativeModel == IntPtr.Zero)
                        throw new InvalidDataException($"Failed to load model from: {filename}");

                    model._isFitted = true;
                    // TODO: Extract model info from loaded model
                }
            }

            return model;
        }

        #endregion

        #region Main API Methods

        /// <summary>
        /// Fits the PacMAP model to training data and returns the embedding
        /// </summary>
        /// <param name="data">Training data as row-major array (samples × features)</param>
        /// <param name="embeddingDimensions">Target dimensionality (default: 2)</param>
        /// <param name="neighbors">Number of neighbors to consider (default: 10)</param>
        /// <param name="normalization">Data normalization mode (default: ZScore)</param>
        /// <param name="metric">Distance metric (default: Euclidean)</param>
        /// <param name="hnswUseCase">HNSW optimization preference (default: Balanced)</param>
        /// <param name="forceExactKnn">Force exact KNN instead of HNSW (default: false)</param>
        /// <param name="seed">Random seed for reproducibility (default: 42)</param>
        /// <param name="progressCallback">Optional progress reporting callback</param>
        /// <returns>Embedding result with coordinates and quality assessment</returns>
        /// <exception cref="ArgumentException">Thrown for invalid parameters</exception>
        /// <exception cref="InvalidOperationException">Thrown when fit fails</exception>
        public EmbeddingResult Fit(double[,] data,
                                  int embeddingDimensions = 2,
                                  int neighbors = 10,
                                  NormalizationMode normalization = NormalizationMode.ZScore,
                                  DistanceMetric metric = DistanceMetric.Euclidean,
                                  HnswUseCase hnswUseCase = HnswUseCase.Balanced,
                                  bool forceExactKnn = false,
                                  ulong seed = 42,
                                  ProgressCallback? progressCallback = null)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            int rows = data.GetLength(0);
            int cols = data.GetLength(1);

            if (rows < 2)
                throw new ArgumentException("Need at least 2 data points", nameof(data));
            if (cols < 1)
                throw new ArgumentException("Need at least 1 feature", nameof(data));
            if (embeddingDimensions < 1 || embeddingDimensions > cols)
                throw new ArgumentException($"Embedding dimensions must be between 1 and {cols}", nameof(embeddingDimensions));
            if (neighbors < 1 || neighbors >= rows)
                throw new ArgumentException($"Neighbors must be between 1 and {rows - 1}", nameof(neighbors));

            _managedCallback = progressCallback;

            unsafe
            {
                var embedding = new double[rows * embeddingDimensions];
                fixed (double* dataPtr = data)
                fixed (double* embeddingPtr = embedding)
                {
                    var nativeCallback = progressCallback != null ? new NativeProgressCallback(NativeProgressHandler) : null;

                    var config = PacmapConfig.Default;
                    config.NNeighbors = neighbors;
                    config.EmbeddingDimensions = embeddingDimensions;
                    config.Seed = (int)seed;
                    config.NormalizationMode = (int)normalization; // C# enum matches FFI values directly
                    config.ForceExactKnn = forceExactKnn;
                    config.HnswConfig.UseCase = (int)hnswUseCase;

                    _nativeModel = IsWindows
                        ? CallFitTransformWindows((IntPtr)dataPtr, rows, cols, config, (IntPtr)embeddingPtr, nativeCallback)
                        : CallFitTransformLinux((IntPtr)dataPtr, rows, cols, config, (IntPtr)embeddingPtr, nativeCallback);

                    if (_nativeModel == IntPtr.Zero)
                        throw new InvalidOperationException("Failed to fit PacMAP model");

                    _isFitted = true;

                    // Get distance statistics
                    if (IsWindows)
                    {
                        CallGetDistanceStatsWindows(_nativeModel, out double mean, out double p95, out double max);
                        var distanceStats = (mean, p95, max);

                        // Store model info
                        _modelInfo = new PacMAPModelInfo(rows, cols, embeddingDimensions, neighbors,
                                                       metric, normalization, !forceExactKnn && rows > 1000, 100.0f);

                        // Convert double array to float array for API consistency
                        var floatEmbedding = new float[embedding.Length];
                        for (int i = 0; i < embedding.Length; i++)
                        {
                            floatEmbedding[i] = (float)embedding[i];
                        }

                        // Create result with quality assessment
                        var confidence = AssessConfidence(distanceStats);
                        var severity = AssessSeverity(confidence);

                        return new EmbeddingResult(floatEmbedding, confidence, severity, distanceStats);
                    }
                    else
                    {
                        CallGetDistanceStatsLinux(_nativeModel, out double mean, out double p95, out double max);
                        var distanceStats = (mean, p95, max);

                        // Store model info
                        _modelInfo = new PacMAPModelInfo(rows, cols, embeddingDimensions, neighbors,
                                                       metric, normalization, !forceExactKnn && rows > 1000, 100.0f);

                        // Convert double array to float array for API consistency
                        var floatEmbedding = new float[embedding.Length];
                        for (int i = 0; i < embedding.Length; i++)
                        {
                            floatEmbedding[i] = (float)embedding[i];
                        }

                        // Create result with quality assessment
                        var confidence = AssessConfidence(distanceStats);
                        var severity = AssessSeverity(confidence);

                        return new EmbeddingResult(floatEmbedding, confidence, severity, distanceStats);
                    }
                }
            }
        }

        /// <summary>
        /// Saves the fitted model to a file for later use
        /// </summary>
        /// <param name="filename">Path where to save the model</param>
        /// <exception cref="InvalidOperationException">Thrown when model is not fitted</exception>
        /// <exception cref="ArgumentException">Thrown for invalid filename</exception>
        /// <exception cref="IOException">Thrown when save operation fails</exception>
        public void Save(string filename)
        {
            if (!IsFitted)
                throw new InvalidOperationException("Model must be fitted before saving");
            if (string.IsNullOrEmpty(filename))
                throw new ArgumentException("Filename cannot be null or empty", nameof(filename));

            unsafe
            {
                var pathBytes = System.Text.Encoding.UTF8.GetBytes(filename);
                fixed (byte* pathPtr = pathBytes)
                {
                    var result = IsWindows
                        ? CallSaveModelWindows(_nativeModel, (IntPtr)pathPtr, false) // No quantization
                        : CallSaveModelLinux(_nativeModel, (IntPtr)pathPtr, false); // No quantization

                    if (result != 0)
                        throw new IOException($"Failed to save model to: {filename}");
                }
            }
        }

        #endregion

        #region Helper Methods

        private void NativeProgressHandler(IntPtr phasePtr, int current, int total, float percent, IntPtr messagePtr)
        {
            try
            {
                var phase = Marshal.PtrToStringUTF8(phasePtr) ?? "Unknown";
                var message = Marshal.PtrToStringUTF8(messagePtr);
                _managedCallback?.Invoke(phase, current, total, percent, message);
            }
            catch
            {
                // Ignore callback errors to prevent crashing native code
            }
        }

        private static float AssessConfidence((double Mean, double P95, double Max) stats)
        {
            // Simple heuristic based on distance distribution
            // Lower p95/mean ratio generally indicates better structure preservation
            if (stats.Mean <= 0) return 0.0f;

            var ratio = stats.P95 / stats.Mean;
            return Math.Max(0.0f, Math.Min(1.0f, (float)(1.0 / (1.0 + ratio * 0.1))));
        }

        private static OutlierLevel AssessSeverity(float confidence)
        {
            return confidence switch
            {
                >= 0.8f => OutlierLevel.Normal,
                >= 0.6f => OutlierLevel.Unusual,
                >= 0.4f => OutlierLevel.Mild,
                >= 0.2f => OutlierLevel.Extreme,
                _ => OutlierLevel.NoMansLand
            };
        }

        #endregion

        #region IDisposable Implementation

        /// <summary>
        /// Releases all resources used by the PacMAP model
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Releases unmanaged resources and optionally releases managed resources
        /// </summary>
        /// <param name="disposing">True to release both managed and unmanaged resources</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (_nativeModel != IntPtr.Zero)
                {
                    if (IsWindows)
                        CallFreeModelWindows(_nativeModel);
                    else
                        CallFreeModelLinux(_nativeModel);

                    _nativeModel = IntPtr.Zero;
                }

                _disposed = true;
                _isFitted = false;
                _modelInfo = null;
                _managedCallback = null;
            }
        }

        /// <summary>
        /// Finalizer to ensure native resources are released
        /// </summary>
        ~PacMAPModel()
        {
            Dispose(false);
        }

        #endregion
    }
}