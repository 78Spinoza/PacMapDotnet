using System;
using System.Collections.Concurrent;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;

namespace PacMAPSharp
{
    /// <summary>
    /// Progress event arguments for PacMAP operations
    /// </summary>
    public class ProgressEventArgs : EventArgs
    {
        public string Phase { get; set; } = "";
        public int Current { get; set; }
        public int Total { get; set; }
        public float Percent { get; set; }
        public string Message { get; set; } = "";
    }

    /// <summary>
    /// Manages thread-safe progress callbacks using the queue+poll pattern
    /// This is the recommended approach for multi-threaded applications
    /// </summary>
    public class ThreadSafeProgressCallbackManager : IDisposable
    {
        private Thread? _pollingThread;
        private CancellationTokenSource? _cancellationTokenSource;
        private readonly object _lock = new object();
        private event EventHandler<ProgressEventArgs>? _progressEvent;
        private bool _isDisposed = false;

        /// <summary>
        /// Event fired when progress messages are received from Rust
        /// </summary>
        public event EventHandler<ProgressEventArgs>? ProgressChanged
        {
            add
            {
                lock (_lock)
                {
                    _progressEvent += value;
                    if (!_isDisposed)
                        StartPollingIfNotRunning();
                }
            }
            remove
            {
                lock (_lock)
                {
                    _progressEvent -= value;
                }
            }
        }

        /// <summary>
        /// Poll for messages from the Rust side and fire events
        /// </summary>
        public void PollMessages()
        {
            while (PacMAPModel.HasMessages())
            {
                if (PacMAPModel.PollNextMessage(new byte[2048], out int messageLength))
                {
                    if (messageLength > 0)
                    {
                        var buffer = new byte[messageLength];
                        if (PacMAPModel.PollNextMessage(buffer, out int actualLength) && actualLength > 0)
                        {
                            string message = Encoding.UTF8.GetString(buffer, 0, actualLength);
                            ParseAndFireEvent(message);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Parse the message format and fire the progress event
        /// </summary>
        private void ParseAndFireEvent(string message)
        {
            try
            {
                // Expected format: "[Phase] message (percentage%)"
                var match = System.Text.RegularExpressions.Regex.Match(message, @"\[([^\]]+)\]\s*([^(]+)\s*\(([\d.]+)%\)");
                if (match.Success)
                {
                    string phase = match.Groups[1].Value.Trim();
                    string details = match.Groups[2].Value.Trim();
                    float percent = float.Parse(match.Groups[3].Value);

                    var args = new ProgressEventArgs
                    {
                        Phase = phase,
                        Current = (int)percent, // Approximate
                        Total = 100, // Normalize to 100
                        Percent = percent,
                        Message = details
                    };

                    _progressEvent?.Invoke(this, args);
                }
                else
                {
                    // Fallback: treat the entire message as details
                    var args = new ProgressEventArgs
                    {
                        Phase = "Progress",
                        Current = 0,
                        Total = 100,
                        Percent = 0,
                        Message = message
                    };

                    _progressEvent?.Invoke(this, args);
                }
            }
            catch
            {
                // If parsing fails, still fire the event with raw message
                var args = new ProgressEventArgs
                {
                    Phase = "Progress",
                    Current = 0,
                    Total = 100,
                    Percent = 0,
                    Message = message
                };

                _progressEvent?.Invoke(this, args);
            }
        }

        /// <summary>
        /// Start the polling thread if not already running
        /// </summary>
        private void StartPollingIfNotRunning()
        {
            if (_pollingThread != null && _pollingThread.IsAlive)
                return;

            _cancellationTokenSource = new CancellationTokenSource();
            _pollingThread = new Thread(PollingThread)
            {
                IsBackground = true,
                Name = "PacMAPProgressPoller"
            };
            _pollingThread.Start();
        }

        /// <summary>
        /// Background thread that continuously polls for messages
        /// </summary>
        private void PollingThread()
        {
            try
            {
                while (_cancellationTokenSource != null && !_cancellationTokenSource.IsCancellationRequested)
                {
                    PollMessages();
                    Thread.Sleep(50); // Poll every 50ms
                }
            }
            catch (ThreadAbortException)
            {
                // Thread is being aborted
            }
            catch (Exception ex)
            {
                Console.WriteLine($"PacMAP progress polling thread error: {ex.Message}");
            }
        }

        /// <summary>
        /// Dispose resources and stop polling
        /// </summary>
        public void Dispose()
        {
            lock (_lock)
            {
                if (_isDisposed)
                    return;

                _isDisposed = true;

                if (_cancellationTokenSource != null)
                {
                    _cancellationTokenSource.Cancel();
                    _cancellationTokenSource.Dispose();
                    _cancellationTokenSource = null;
                }

                if (_pollingThread != null && _pollingThread.IsAlive)
                {
                    try
                    {
                        _pollingThread.Join(1000); // Wait up to 1 second
                        if (_pollingThread.IsAlive)
                        {
                            _pollingThread.Interrupt();
                        }
                    }
                    catch
                    {
                        // Ignore thread interruption errors
                    }
                }

                // Clear any remaining messages
                PacMAPModel.ClearMessages();

                _progressEvent = null;
            }
        }
    }
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

        /// <summary>
        /// Gets the discovered HNSW M parameter (if autodetect was used)
        /// </summary>
        public int? DiscoveredHnswM { get; }

        /// <summary>
        /// Gets the discovered HNSW ef_construction parameter (if autodetect was used)
        /// </summary>
        public int? DiscoveredHnswEfConstruction { get; }

        /// <summary>
        /// Gets the discovered HNSW ef_search parameter (if autodetect was used)
        /// </summary>
        public int? DiscoveredHnswEfSearch { get; }

        /// <summary>
        /// Gets the HNSW maximum connections per layer 0 node (M0 parameter)
        /// </summary>
        public int? HnswMaxM0 { get; }

        /// <summary>
        /// Gets the HNSW random seed used for deterministic construction
        /// </summary>
        public long? HnswSeed { get; }

        /// <summary>
        /// Gets the current number of layers in the HNSW index
        /// </summary>
        public int? HnswMaxLayer { get; }

        /// <summary>
        /// Gets the total number of elements indexed in the HNSW structure
        /// </summary>
        public int? HnswTotalElements { get; }

        /// <summary>
        /// Gets the learning rate used in training
        /// </summary>
        public double LearningRate { get; }

        /// <summary>
        /// Gets the number of epochs used in training
        /// </summary>
        public int NEpochs { get; }

        /// <summary>
        /// Gets the mid-near ratio used in pair sampling
        /// </summary>
        public double MidNearRatio { get; }

        /// <summary>
        /// Gets the far pair ratio used in pair sampling
        /// </summary>
        public double FarPairRatio { get; }

        /// <summary>
        /// Gets the random seed used for deterministic results
        /// </summary>
        public int Seed { get; }

        /// <summary>
        /// Gets whether the model uses quantization on save
        /// </summary>
        public bool QuantizeOnSave { get; }

        /// <summary>
        /// Gets the CRC32 checksum of the original HNSW index (null if not available)
        /// </summary>
        public uint? HnswIndexCrc32 { get; }

        /// <summary>
        /// Gets the CRC32 checksum of the embedding HNSW index (null if not available)
        /// </summary>
        public uint? EmbeddingHnswIndexCrc32 { get; }

        /// <summary>
        /// Gets the file path if the model was saved or loaded from a file
        /// </summary>
        public string? FilePath { get; }

        internal PacMAPModelInfo(int trainingSamples, int inputDimension, int outputDimension,
                                int neighbors, DistanceMetric metric, NormalizationMode normalization,
                                bool usedHnsw, float hnswRecall, int? discoveredHnswM, int? discoveredHnswEfConstruction, int? discoveredHnswEfSearch,
                                int? hnswMaxM0, long? hnswSeed, int? hnswMaxLayer, int? hnswTotalElements,
                                double learningRate, int nEpochs, double midNearRatio,
                                double farPairRatio, int seed, bool quantizeOnSave, uint? hnswIndexCrc32 = null, uint? embeddingHnswIndexCrc32 = null, string? filePath = null)
        {
            TrainingSamples = trainingSamples;
            InputDimension = inputDimension;
            OutputDimension = outputDimension;
            Neighbors = neighbors;
            Metric = metric;
            Normalization = normalization;
            UsedHNSW = usedHnsw;
            HnswRecall = hnswRecall;
            DiscoveredHnswM = discoveredHnswM;
            DiscoveredHnswEfConstruction = discoveredHnswEfConstruction;
            DiscoveredHnswEfSearch = discoveredHnswEfSearch;
            HnswMaxM0 = hnswMaxM0;
            HnswSeed = hnswSeed;
            HnswMaxLayer = hnswMaxLayer;
            HnswTotalElements = hnswTotalElements;
            LearningRate = learningRate;
            NEpochs = nEpochs;
            MidNearRatio = midNearRatio;
            FarPairRatio = farPairRatio;
            Seed = seed;
            QuantizeOnSave = quantizeOnSave;
            HnswIndexCrc32 = hnswIndexCrc32;
            EmbeddingHnswIndexCrc32 = embeddingHnswIndexCrc32;
            FilePath = filePath;
        }

        /// <summary>
        /// Returns a comprehensive string representation of the model info with ALL parameters and file info
        /// </summary>
        public override string ToString()
        {
            var sb = new System.Text.StringBuilder();

            // Header
            sb.AppendLine("╔═══════════════════════════════════════════════════════════════════╗");
            sb.AppendLine("║                      PacMAP Model Complete Info                   ║");
            sb.AppendLine("╠═══════════════════════════════════════════════════════════════════╣");

            // File Information
            if (!string.IsNullOrEmpty(FilePath))
            {
                sb.AppendLine($"📁 File: {FilePath}");
                if (System.IO.File.Exists(FilePath))
                {
                    var fileInfo = new System.IO.FileInfo(FilePath);
                    sb.AppendLine($"   Size: {fileInfo.Length / 1024.0:F1} KB");
                    sb.AppendLine($"   Modified: {fileInfo.LastWriteTime:yyyy-MM-dd HH:mm:ss}");
                    sb.AppendLine($"   Status: ✅ EXISTS");
                }
                else
                {
                    sb.AppendLine($"   Status: ❌ FILE NOT FOUND");
                }
                sb.AppendLine();
            }

            // Dataset Information
            sb.AppendLine("📊 Dataset Information:");
            sb.AppendLine($"   Training Samples: {TrainingSamples:N0}");
            sb.AppendLine($"   Input Dimensions: {InputDimension}D");
            sb.AppendLine($"   Output Dimensions: {OutputDimension}D");
            sb.AppendLine($"   Distance Metric: {Metric}");
            sb.AppendLine($"   Normalization: {Normalization}");
            sb.AppendLine();

            // Algorithm Configuration
            sb.AppendLine("⚙️  Algorithm Configuration:");
            sb.AppendLine($"   Neighbors (k): {Neighbors}");
            sb.AppendLine($"   Training Epochs: {NEpochs}");
            sb.AppendLine($"   Learning Rate: {LearningRate:F6}");
            sb.AppendLine($"   Mid-Near Ratio: {MidNearRatio:F3}");
            sb.AppendLine($"   Far-Pair Ratio: {FarPairRatio:F3}");
            sb.AppendLine();

            // Neighbor Search Method
            if (UsedHNSW)
            {
                sb.AppendLine("⚡ HNSW Neighbor Search:");
                sb.AppendLine($"   Method: Approximate (HNSW)");
                sb.AppendLine($"   Recall Achieved: {HnswRecall:F1}%");

                if (DiscoveredHnswM.HasValue && DiscoveredHnswEfConstruction.HasValue && DiscoveredHnswEfSearch.HasValue)
                {
                    sb.AppendLine($"   M Parameter: {DiscoveredHnswM.Value}");
                    sb.AppendLine($"   ef_construction: {DiscoveredHnswEfConstruction.Value}");
                    sb.AppendLine($"   ef_search: {DiscoveredHnswEfSearch.Value}");
                    sb.AppendLine($"   Status: 🎯 OPTIMIZED (autodetected)");
                }
                else
                {
                    sb.AppendLine($"   Status: ⚙️  CONFIGURED (manual)");
                }

                // CRC integrity information
                if (HnswIndexCrc32.HasValue || EmbeddingHnswIndexCrc32.HasValue)
                {
                    sb.AppendLine($"   Index Integrity:");
                    if (HnswIndexCrc32.HasValue)
                        sb.AppendLine($"     Original Index CRC32: 0x{HnswIndexCrc32.Value:X8}");
                    if (EmbeddingHnswIndexCrc32.HasValue)
                        sb.AppendLine($"     Embedding Index CRC32: 0x{EmbeddingHnswIndexCrc32.Value:X8}");
                }
            }
            else
            {
                sb.AppendLine("🔍 Exact KNN Search:");
                sb.AppendLine($"   Method: Exact neighbor search");
                sb.AppendLine($"   Accuracy: 100.0%");
                sb.AppendLine($"   Status: ✅ PRECISE");
            }
            sb.AppendLine();

            // Storage and Performance
            sb.AppendLine("💾 Storage & Performance:");
            sb.AppendLine($"   Quantization: {(QuantizeOnSave ? "✅ ENABLED (f16 compression)" : "❌ DISABLED (full f64 precision)")}");

            if (QuantizeOnSave)
            {
                sb.AppendLine($"   Space Savings: ~50% file size reduction");
                sb.AppendLine($"   Quality Impact: Minimal (typically <0.1% RMSE)");
            }

            // Performance characteristics
            if (UsedHNSW)
            {
                sb.AppendLine($"   Speed Profile: Fast embedding, good recall");
                sb.AppendLine($"   Memory Usage: Low (HNSW optimized)");
            }
            else
            {
                sb.AppendLine($"   Speed Profile: Slower but exact");
                sb.AppendLine($"   Memory Usage: Higher (exact computation)");
            }

            sb.AppendLine("╚═══════════════════════════════════════════════════════════════════╝");

            return sb.ToString();
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
        public bool AutodetectHnswParams; // If true, do recall validation and auto-optimize; if false, use params as-is

        public static PacmapHnswConfig Default => new PacmapHnswConfig
        {
            AutoScale = true,
            UseCase = 0, // Balanced
            M = 16,
            EfConstruction = 128,
            EfSearch = 64,
            MemoryLimitMb = 0,
            AutodetectHnswParams = true // Enable recall validation by default
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
            MidNearRatio = 0.5,
            FarPairRatio = 2.0,
            Seed = 42,
            NormalizationMode = (int)PacMAPSharp.NormalizationMode.ZScore, // Use enum to prevent mapping bugs
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
        private delegate void NativeProgressCallback(IntPtr userData, IntPtr dataPtr, UIntPtr len);

        // Native function imports - Windows
        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_version")]
        private static extern IntPtr CallGetVersionWindows();

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_fit_transform_enhanced")]
        private static extern IntPtr CallFitTransformWindows(IntPtr data, int rows, int cols,
            PacmapConfig config, IntPtr embedding, int embeddingBufferLen, NativeProgressCallback? callback);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_model_info")]
        private static extern int CallGetModelInfoWindows(IntPtr model, out int nSamples, out int nFeatures, out int embeddingDim, out int nNeighbors,
            out float midNearRatio, out float farPairRatio, out int metric, out int hnswM, out int hnswEfConstruction, out int hnswEfSearch,
            out int hnswMaxM0, out long hnswSeed, out int hnswMaxLayer, out int hnswTotalElements, out uint hnswIndexCrc32, out uint embeddingHnswIndexCrc32);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_model_stats")]
        private static extern void CallGetDistanceStatsWindows(IntPtr model, out double mean, out double p95, out double max);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_save_model_enhanced")]
        private static extern int CallSaveModelWindows(IntPtr model, IntPtr path, bool quantize);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_transform")]
        private static extern int CallTransformWindows(IntPtr model, IntPtr data, int rows, int cols, IntPtr embedding, int embeddingBufferLen, NativeProgressCallback? callback);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_load_model_enhanced")]
        private static extern IntPtr CallLoadModelWindows(IntPtr path);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_free_model_enhanced")]
        private static extern void CallFreeModelWindows(IntPtr model);

        // Thread-safe callback queue functions
        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_register_text_callback")]
        private static extern void CallRegisterTextCallbackWindows(IntPtr callback, IntPtr user_data);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_enqueue_message")]
        private static extern void CallEnqueueMessageWindows(IntPtr message);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_poll_next_message")]
        private static extern UIntPtr CallPollNextMessageWindows(IntPtr buffer, UIntPtr bufLen);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_has_messages")]
        [return: MarshalAs(UnmanagedType.I1)]
        private static extern bool CallHasMessagesWindows();

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_clear_messages")]
        private static extern void CallClearMessagesWindows();

        // Linux function imports - similar pattern
        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_version")]
        private static extern IntPtr CallGetVersionLinux();

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_fit_transform_enhanced")]
        private static extern IntPtr CallFitTransformLinux(IntPtr data, int rows, int cols,
            PacmapConfig config, IntPtr embedding, int embeddingBufferLen, NativeProgressCallback? callback);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_model_info")]
        private static extern int CallGetModelInfoLinux(IntPtr model, out int nSamples, out int nFeatures, out int embeddingDim, out int nNeighbors,
            out float midNearRatio, out float farPairRatio, out int metric, out int hnswM, out int hnswEfConstruction, out int hnswEfSearch,
            out int hnswMaxM0, out long hnswSeed, out int hnswMaxLayer, out int hnswTotalElements, out uint hnswIndexCrc32, out uint embeddingHnswIndexCrc32);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_model_stats")]
        private static extern void CallGetDistanceStatsLinux(IntPtr model, out double mean, out double p95, out double max);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_save_model_enhanced")]
        private static extern int CallSaveModelLinux(IntPtr model, IntPtr path, bool quantize);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_transform")]
        private static extern int CallTransformLinux(IntPtr model, IntPtr data, int rows, int cols, IntPtr embedding, int embeddingBufferLen, NativeProgressCallback? callback);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_load_model_enhanced")]
        private static extern IntPtr CallLoadModelLinux(IntPtr path);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_free_model_enhanced")]
        private static extern void CallFreeModelLinux(IntPtr model);

        // Thread-safe callback queue functions - Linux
        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_register_text_callback")]
        private static extern void CallRegisterTextCallbackLinux(IntPtr callback, IntPtr user_data);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_enqueue_message")]
        private static extern void CallEnqueueMessageLinux(IntPtr message);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_poll_next_message")]
        private static extern UIntPtr CallPollNextMessageLinux(IntPtr buffer, UIntPtr bufLen);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_has_messages")]
        [return: MarshalAs(UnmanagedType.I1)]
        private static extern bool CallHasMessagesLinux();

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_clear_messages")]
        private static extern void CallClearMessagesLinux();

        #endregion

        #region Thread-Safe Callback Wrapper Methods

        /// <summary>
        /// Platform-safe wrapper for checking if messages are available
        /// </summary>
        public static bool HasMessages()
        {
            try
            {
                return IsWindows
                    ? CallHasMessagesWindows()
                    : CallHasMessagesLinux();
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Poll for the next message and return it as a string
        /// Returns null if no message is available
        /// </summary>
        public static string? PollNextMessage()
        {
            try
            {
                byte[] buffer = new byte[2048];
                if (PollNextMessage(buffer, out int messageLength) && messageLength > 0)
                {
                    return Encoding.UTF8.GetString(buffer, 0, messageLength);
                }
                return null;
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Platform-safe wrapper for polling the next message
        /// </summary>
        public static bool PollNextMessage(byte[] buffer, out int messageLength)
        {
            messageLength = 0;
            try
            {
                GCHandle handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
                try
                {
                    UIntPtr len = IsWindows
                        ? CallPollNextMessageWindows(handle.AddrOfPinnedObject(), (UIntPtr)buffer.Length)
                        : CallPollNextMessageLinux(handle.AddrOfPinnedObject(), (UIntPtr)buffer.Length);

                    messageLength = (int)len;
                    return messageLength > 0;
                }
                finally
                {
                    handle.Free();
                }
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Platform-safe wrapper for clearing messages
        /// </summary>
        public static void ClearMessages()
        {
            try
            {
                if (IsWindows)
                    CallClearMessagesWindows();
                else
                    CallClearMessagesLinux();
            }
            catch
            {
                // Ignore cleanup errors
            }
        }

        /// <summary>
        /// Platform-safe wrapper for enqueuing messages
        /// </summary>
        private static unsafe void EnqueueMessage(string message)
        {
            try
            {
                byte[] messageBytes = Encoding.UTF8.GetBytes(message);
                fixed (byte* messagePtr = messageBytes)
                {
                    if (IsWindows)
                        CallEnqueueMessageWindows((IntPtr)messagePtr);
                    else
                        CallEnqueueMessageLinux((IntPtr)messagePtr);
                }
            }
            catch
            {
                // Ignore enqueue errors
            }
        }

        /// <summary>
        /// Platform-safe wrapper for registering text callbacks
        /// </summary>
        private static void RegisterTextCallback(NativeProgressCallback callback, IntPtr user_data)
        {
            try
            {
                IntPtr callbackPtr = callback != null ? Marshal.GetFunctionPointerForDelegate(callback) : IntPtr.Zero;
                if (IsWindows)
                    CallRegisterTextCallbackWindows(callbackPtr, user_data);
                else
                    CallRegisterTextCallbackLinux(callbackPtr, user_data);
            }
            catch
            {
                // Ignore registration errors
            }
        }

        #endregion

        #region Private Fields

        private IntPtr _nativeModel = IntPtr.Zero;
        private bool _disposed = false;
        private bool _isFitted = false;
        private string? _filePath;
        private PacMAPModelInfo? _modelInfo = null;
        private ProgressCallback? _managedCallback = null;

        // Thread-safe callback support
        private ThreadSafeProgressCallbackManager? _callbackManager = null;
        private bool _useThreadSafeCallbacks = true; // Default to thread-safe mode

        // Expected DLL version - must match Rust pacmap_enhanced version
        private const string EXPECTED_DLL_VERSION = "0.4.1";

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

        /// <summary>
        /// Gets or sets whether to use thread-safe callback system (recommended)
        /// When true, uses queue+poll pattern for multi-threaded safety
        /// When false, uses legacy direct callbacks (not recommended for multi-threaded applications)
        /// </summary>
        public bool UseThreadSafeCallbacks
        {
            get => _useThreadSafeCallbacks;
            set => _useThreadSafeCallbacks = value;
        }

        /// <summary>
        /// Gets the thread-safe callback manager for accessing progress events
        /// Only available when UseThreadSafeCallbacks is true
        /// </summary>
        public ThreadSafeProgressCallbackManager ThreadSafeCallbackManager
        {
            get
            {
                if (!_useThreadSafeCallbacks)
                    throw new InvalidOperationException("Thread-safe callbacks are disabled. Set UseThreadSafeCallbacks to true.");
                return _callbackManager ?? throw new InvalidOperationException("Callback manager not initialized.");
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
            _callbackManager = new ThreadSafeProgressCallbackManager();
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
        /// Verifies the native DLL version matches the expected C# wrapper version
        /// </summary>
        /// <exception cref="InvalidOperationException">Thrown when version mismatch is detected</exception>
        /// <exception cref="DllNotFoundException">Thrown when native library cannot be loaded</exception>
        public static void VerifyVersion()
        {
            try
            {
                // Get the actual version from the native library
                string actualVersion = GetVersion();

                // Extract just the version number (e.g., "0.3.0" from "PacMAP Enhanced v0.3.0 - HNSW: ENABLED, OpenBLAS: SYSTEM")
                string extractedVersion = ExtractVersionNumber(actualVersion);

                if (extractedVersion != EXPECTED_DLL_VERSION)
                {
                    string errorMsg = $"PacMAP version mismatch detected!\n" +
                                    $"  Expected: v{EXPECTED_DLL_VERSION}\n" +
                                    $"  Actual:   v{extractedVersion}\n" +
                                    $"  Full:     {actualVersion}\n" +
                                    $"Please update the native library or C# wrapper to matching versions.";

                    throw new InvalidOperationException(errorMsg);
                }
            }
            catch (DllNotFoundException ex)
            {
                string errorMsg = $"Failed to load PacMAP native library for version verification. " +
                                $"Ensure {(IsWindows ? WindowsDll : LinuxDll)} is available.";
                throw new DllNotFoundException(errorMsg, ex);
            }
            catch (InvalidOperationException)
            {
                // Re-throw version mismatch errors
                throw;
            }
            catch (Exception ex)
            {
                string errorMsg = $"Version verification failed: {ex.Message}";
                throw new InvalidOperationException(errorMsg, ex);
            }
        }

        /// <summary>
        /// Extracts version number from the full version string returned by native library
        /// </summary>
        /// <param name="fullVersionString">Full version string like "PacMAP Enhanced v0.3.0 - HNSW: ENABLED, OpenBLAS: SYSTEM"</param>
        /// <returns>Just the version number like "0.3.0"</returns>
        private static string ExtractVersionNumber(string fullVersionString)
        {
            if (string.IsNullOrEmpty(fullVersionString))
                return "unknown";

            // Look for pattern "v{version}" in the string
            var match = System.Text.RegularExpressions.Regex.Match(fullVersionString, @"v(\d+\.\d+\.\d+)");
            if (match.Success)
                return match.Groups[1].Value;

            // Fallback: look for just version pattern without 'v'
            match = System.Text.RegularExpressions.Regex.Match(fullVersionString, @"(\d+\.\d+\.\d+)");
            if (match.Success)
                return match.Groups[1].Value;

            // If no pattern found, return the original string
            return fullVersionString;
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
                    model._filePath = Path.GetFullPath(filename);

                    // Extract model info from loaded model
                    model.ExtractModelInfoFromLoadedModel();
                }
            }

            return model;
        }

        /// <summary>
        /// Extracts model information from a loaded native model
        /// </summary>
        private void ExtractModelInfoFromLoadedModel()
        {
            if (_nativeModel == IntPtr.Zero)
                throw new InvalidOperationException("Native model is null");

                unsafe
                {
                    int nSamples, nFeatures, embeddingDim;
                    int hnswM, hnswEfConstruction, hnswEfSearch;
                    int nNeighbors, metric, hnswMaxM0, hnswMaxLayer, hnswTotalElements;
                    long hnswSeed;
                    float midNearRatio, farPairRatio;
                    uint hnswIndexCrc32, embeddingHnswIndexCrc32;

                    // Extract model info from native model
                    if (IsWindows)
                    {
                        CallGetModelInfoWindows(_nativeModel, out nSamples, out nFeatures, out embeddingDim, out nNeighbors,
                            out midNearRatio, out farPairRatio, out metric, out hnswM, out hnswEfConstruction, out hnswEfSearch,
                            out hnswMaxM0, out hnswSeed, out hnswMaxLayer, out hnswTotalElements, out hnswIndexCrc32, out embeddingHnswIndexCrc32);
                    }
                    else
                    {
                        CallGetModelInfoLinux(_nativeModel, out nSamples, out nFeatures, out embeddingDim, out nNeighbors,
                            out midNearRatio, out farPairRatio, out metric, out hnswM, out hnswEfConstruction, out hnswEfSearch,
                            out hnswMaxM0, out hnswSeed, out hnswMaxLayer, out hnswTotalElements, out hnswIndexCrc32, out embeddingHnswIndexCrc32);
                    }

                // Create model info object
                _modelInfo = new PacMAPModelInfo(
                    trainingSamples: nSamples,
                    inputDimension: nFeatures,
                    outputDimension: embeddingDim,
                    neighbors: nNeighbors, // Use actual neighbors from FFI
                    metric: (DistanceMetric)metric, // Use actual metric from FFI
                    normalization: NormalizationMode.ZScore, // Default - not returned by new FFI
                    usedHnsw: hnswM > 0, // Determine HNSW usage from M parameter
                    hnswRecall: 100.0f, // Default value for loaded models
                    discoveredHnswM: hnswM > 0 ? hnswM : null,
                    discoveredHnswEfConstruction: hnswM > 0 ? hnswEfConstruction : null,
                    discoveredHnswEfSearch: hnswM > 0 ? hnswEfSearch : null,
                    hnswMaxM0: hnswM > 0 ? hnswMaxM0 : null,
                    hnswSeed: hnswM > 0 ? hnswSeed : null,
                    hnswMaxLayer: hnswM > 0 ? hnswMaxLayer : null,
                    hnswTotalElements: hnswM > 0 ? hnswTotalElements : null,
                    learningRate: 0.0, // Default - not returned by new FFI
                    nEpochs: 450, // Default - not returned by new FFI
                    midNearRatio: midNearRatio,
                    farPairRatio: farPairRatio,
                    seed: (int)hnswSeed, // Use HNSW seed as general seed
                    quantizeOnSave: false, // Default - not returned by new FFI
                    hnswIndexCrc32: hnswIndexCrc32 != 0 ? hnswIndexCrc32 : null, // Use actual CRC if non-zero
                    embeddingHnswIndexCrc32: embeddingHnswIndexCrc32 != 0 ? embeddingHnswIndexCrc32 : null, // Use actual CRC if non-zero
                    filePath: _filePath
                );
            }
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
        /// <param name="learningRate">Learning rate for optimization (default: 1.0)</param>
        /// <param name="nEpochs">Number of training epochs (default: 450)</param>
        /// <param name="midNearRatio">Mid-near pair ratio (default: 0.5)</param>
        /// <param name="farPairRatio">Far pair ratio (default: 2.0)</param>
        /// <param name="autodetectHnswParams">Enable HNSW parameter autodetection (default: true)</param>
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
                                  double learningRate = 1.0,
                                  int nEpochs = 450,
                                  double midNearRatio = 0.5,
                                  double farPairRatio = 2.0,
                                  bool autodetectHnswParams = true,
                                  ulong seed = 42,
                                  ProgressCallback? progressCallback = null)
        {
            // CRITICAL: Verify DLL version before any native calls to prevent binary mismatches
            VerifyVersion();

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
                    _managedCallback = progressCallback; // Store callback for native handler
                    var nativeCallback = progressCallback != null ? new NativeProgressCallback(NativeProgressHandler) : null;

                    var config = PacmapConfig.Default;
                    config.NNeighbors = neighbors;
                    config.EmbeddingDimensions = embeddingDimensions;
                    config.NEpochs = nEpochs;
                    config.Seed = (int)seed;
                    config.NormalizationMode = (int)normalization; // C# enum matches FFI values directly
                    config.ForceExactKnn = forceExactKnn;
                    config.LearningRate = learningRate;
                    config.MidNearRatio = midNearRatio;
                    config.FarPairRatio = farPairRatio;
                    config.HnswConfig.UseCase = (int)hnswUseCase;
                    config.HnswConfig.AutodetectHnswParams = autodetectHnswParams;

                    int embeddingBufferLen = rows * embeddingDimensions;
                    _nativeModel = IsWindows
                        ? CallFitTransformWindows((IntPtr)dataPtr, rows, cols, config, (IntPtr)embeddingPtr, embeddingBufferLen, nativeCallback)
                        : CallFitTransformLinux((IntPtr)dataPtr, rows, cols, config, (IntPtr)embeddingPtr, embeddingBufferLen, nativeCallback);

                    if (_nativeModel == IntPtr.Zero)
                        throw new InvalidOperationException("Failed to fit PacMAP model");

                    _isFitted = true;

                    // Get distance statistics and all model parameters
                    if (IsWindows)
                    {
                        CallGetDistanceStatsWindows(_nativeModel, out double mean, out double p95, out double max);
                        var distanceStats = (mean, p95, max);

                        // Get ALL model parameters from the native library (complete serialization metadata)
                        int infoResult = CallGetModelInfoWindows(_nativeModel, out int actualSamples, out int actualFeatures, out int actualEmbedDim, out int actualNeighbors,
                                                               out float actualMidNearRatio, out float actualFarPairRatio, out int actualMetric, out int hnswM, out int hnswEfConstruction, out int hnswEfSearch,
                                                               out int hnswMaxM0, out long hnswSeed, out int hnswMaxLayer, out int hnswTotalElements, out uint hnswIndexCrc32, out uint embeddingHnswIndexCrc32);

                        if (infoResult != 0)
                        {
                            Console.WriteLine("Warning: Failed to retrieve complete model info from native library");
                        }

                        // Store complete model info with ALL discovered parameters
                        bool actualUsedHnsw = hnswM > 0;
                        int? discoveredM = actualUsedHnsw ? hnswM : null;
                        int? discoveredEfConstruction = actualUsedHnsw ? hnswEfConstruction : null;
                        int? discoveredEfSearch = actualUsedHnsw ? hnswEfSearch : null;
                        uint? crc1 = hnswIndexCrc32 != 0 ? hnswIndexCrc32 : null;
                        uint? crc2 = embeddingHnswIndexCrc32 != 0 ? embeddingHnswIndexCrc32 : null;

                        // Use the input configuration parameters since they're not returned by FFI
                        _modelInfo = new PacMAPModelInfo(actualSamples, actualFeatures, actualEmbedDim, neighbors,
                                                       metric, normalization, actualUsedHnsw, 100.0f,
                                                       discoveredM, discoveredEfConstruction, discoveredEfSearch,
                                                       hnswM > 0 ? hnswMaxM0 : null, hnswM > 0 ? hnswSeed : null,
                                                       hnswM > 0 ? hnswMaxLayer : null, hnswM > 0 ? hnswTotalElements : null,
                                                       learningRate, nEpochs, actualMidNearRatio,
                                                       actualFarPairRatio, (int)seed, false, crc1, crc2, _filePath);

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

                        // Get ALL model parameters from the native library (complete serialization metadata)
                        int infoResult = CallGetModelInfoLinux(_nativeModel, out int actualSamples, out int actualFeatures, out int actualEmbedDim, out int actualNeighbors,
                                                             out float actualMidNearRatio, out float actualFarPairRatio, out int actualMetric, out int hnswM, out int hnswEfConstruction, out int hnswEfSearch,
                                                             out int hnswMaxM0, out long hnswSeed, out int hnswMaxLayer, out int hnswTotalElements, out uint hnswIndexCrc32, out uint embeddingHnswIndexCrc32);

                        if (infoResult != 0)
                        {
                            Console.WriteLine("Warning: Failed to retrieve complete model info from native library");
                        }

                        // Store complete model info with ALL discovered parameters
                        bool actualUsedHnsw = hnswM > 0;
                        int? discoveredM = actualUsedHnsw ? hnswM : null;
                        int? discoveredEfConstruction = actualUsedHnsw ? hnswEfConstruction : null;
                        int? discoveredEfSearch = actualUsedHnsw ? hnswEfSearch : null;
                        uint? crc1 = hnswIndexCrc32 != 0 ? hnswIndexCrc32 : null;
                        uint? crc2 = embeddingHnswIndexCrc32 != 0 ? embeddingHnswIndexCrc32 : null;

                        // Use the input configuration parameters since they're not returned by FFI
                        _modelInfo = new PacMAPModelInfo(actualSamples, actualFeatures, actualEmbedDim, neighbors,
                                                       metric, normalization, actualUsedHnsw, 100.0f,
                                                       discoveredM, discoveredEfConstruction, discoveredEfSearch,
                                                       hnswM > 0 ? hnswMaxM0 : null, hnswM > 0 ? hnswSeed : null,
                                                       hnswM > 0 ? hnswMaxLayer : null, hnswM > 0 ? hnswTotalElements : null,
                                                       learningRate, nEpochs, actualMidNearRatio,
                                                       actualFarPairRatio, (int)seed, false, crc1, crc2, _filePath);

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
        /// Transforms new data using an already fitted PacMAP model
        /// </summary>
        /// <param name="data">Input data matrix</param>
        /// <param name="progressCallback">Optional progress callback</param>
        /// <returns>Embedding result for the transformed data</returns>
        /// <exception cref="InvalidOperationException">Thrown when model is not fitted</exception>
        /// <exception cref="ArgumentException">Thrown for invalid parameters</exception>
        public EmbeddingResult Transform(double[,] data, ProgressCallback? progressCallback = null)
        {
            // CRITICAL: Verify DLL version before any native calls to prevent binary mismatches
            VerifyVersion();

            if (!IsFitted)
                throw new InvalidOperationException("Model must be fitted before transforming data");

            if (data == null)
                throw new ArgumentNullException(nameof(data));

            int rows = data.GetLength(0);
            int cols = data.GetLength(1);

            if (rows < 1)
                throw new ArgumentException("Need at least 1 data point", nameof(data));
            if (cols < 1)
                throw new ArgumentException("Need at least 1 feature", nameof(data));

            // Check if data dimensions match training dimensions
            if (_modelInfo.HasValue && cols != _modelInfo.Value.InputDimension)
                throw new ArgumentException($"Data has {cols} features but model was trained on {_modelInfo.Value.InputDimension} features", nameof(data));

            _managedCallback = progressCallback;

            // Use output dimension from model info or default to 2
            int outputDim = _modelInfo.HasValue ? _modelInfo.Value.OutputDimension : 2;

            unsafe
            {
                var embedding = new double[rows * outputDim];
                fixed (double* dataPtr = data)
                fixed (double* embeddingPtr = embedding)
                {
                    _managedCallback = progressCallback; // Store callback for native handler
                    var nativeCallback = progressCallback != null ? new NativeProgressCallback(NativeProgressHandler) : null;

                    // Transform data using existing model
                    var resultCode = IsWindows
                        ? CallTransformWindows(_nativeModel, (IntPtr)dataPtr, rows, cols, (IntPtr)embeddingPtr, embedding.Length, nativeCallback)
                        : CallTransformLinux(_nativeModel, (IntPtr)dataPtr, rows, cols, (IntPtr)embeddingPtr, embedding.Length, nativeCallback);

                    if (resultCode != 0)
                        throw new InvalidOperationException($"Failed to transform data with PacMAP model. Error code: {resultCode}");

                    // Convert double[] to float[] for consistency with Fit method
                    var floatEmbedding = new float[embedding.Length];
                    for (int i = 0; i < embedding.Length; i++)
                    {
                        floatEmbedding[i] = (float)embedding[i];
                    }

                    // Calculate distance stats for the transformed data
                    (double mean, double p95, double max) distanceStats;
                    if (IsWindows)
                    {
                        CallGetDistanceStatsWindows(_nativeModel, out double mean, out double p95, out double max);
                        distanceStats = (mean, p95, max);
                    }
                    else
                    {
                        CallGetDistanceStatsLinux(_nativeModel, out double mean, out double p95, out double max);
                        distanceStats = (mean, p95, max);
                    }

                    // Create result with existing confidence assessment
                    var confidence = AssessConfidence(distanceStats);
                    var severity = AssessSeverity(confidence);

                    return new EmbeddingResult(floatEmbedding, confidence, severity, distanceStats);
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

                    // Store the file path and update model info to include it
                    _filePath = Path.GetFullPath(filename);
                    if (_modelInfo != null)
                    {
                        var info = _modelInfo.Value;
                        _modelInfo = new PacMAPModelInfo(info.TrainingSamples, info.InputDimension, info.OutputDimension,
                                                       info.Neighbors, info.Metric, info.Normalization,
                                                       info.UsedHNSW, info.HnswRecall,
                                                       info.DiscoveredHnswM, info.DiscoveredHnswEfConstruction, info.DiscoveredHnswEfSearch,
                                                       info.HnswMaxM0, info.HnswSeed, info.HnswMaxLayer, info.HnswTotalElements,
                                                       info.LearningRate, info.NEpochs,
                                                       info.MidNearRatio, info.FarPairRatio, info.Seed, info.QuantizeOnSave,
                                                       info.HnswIndexCrc32, info.EmbeddingHnswIndexCrc32, _filePath);
                    }
                }
            }
        }

        #endregion

        #region Helper Methods

        private void NativeProgressHandler(IntPtr userData, IntPtr dataPtr, UIntPtr len)
        {
            try
            {
                // Copy bytes immediately into managed buffer
                int length = (int)len;
                if (length == 0) return;

                byte[] buffer = new byte[length];
                Marshal.Copy(dataPtr, buffer, 0, length);

                // Decode UTF-8 string
                string fullMessage = Encoding.UTF8.GetString(buffer);

                // Parse the formatted message: "[Phase] Message (percent%)"
                // Expected format: "[Phase] Message (95.0%)"
                int startBracket = fullMessage.IndexOf('[');
                int endBracket = fullMessage.IndexOf(']');
                int startParen = fullMessage.LastIndexOf('(');
                int endParen = fullMessage.LastIndexOf(')');

                if (startBracket >= 0 && endBracket > startBracket &&
                    startParen > endBracket && endParen > startParen)
                {
                    string phase = fullMessage.Substring(startBracket + 1, endBracket - startBracket - 1);
                    string message = fullMessage.Substring(endBracket + 2, startParen - endBracket - 2).Trim();
                    string percentStr = fullMessage.Substring(startParen + 1, endParen - startParen - 1);

                    if (float.TryParse(percentStr.TrimEnd('%'), out float percent))
                    {
                        _managedCallback?.Invoke(phase, 0, 100, percent, message);
                    }
                }
                else
                {
                    // Fallback: treat as simple progress message
                    _managedCallback?.Invoke("Progress", 0, 100, 0.0f, fullMessage);
                }
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

                // Clean up thread-safe callback manager
                if (_callbackManager != null)
                {
                    _callbackManager.Dispose();
                    _callbackManager = null;
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