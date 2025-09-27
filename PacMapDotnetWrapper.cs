// C# Wrapper for PacMAP Enhanced with HNSW Auto-scaling
// Demonstrates integration with the enhanced FFI interface

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace PacMapDotnet
{
    /// <summary>
    /// Progress callback delegate for C# integration
    /// Matches the native PacmapProgressCallback signature
    /// </summary>
    /// <param name="phase">Current phase name</param>
    /// <param name="current">Current progress counter</param>
    /// <param name="total">Total items to process</param>
    /// <param name="percent">Progress percentage (0-100)</param>
    /// <param name="message">Additional message or null</param>
    public delegate void ProgressCallback(IntPtr phase, int current, int total, float percent, IntPtr message);

    /// <summary>
    /// HNSW configuration structure
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct HnswConfig
    {
        [MarshalAs(UnmanagedType.I1)]
        public bool AutoScale;          // If true, auto-scale parameters based on dataset
        public int UseCase;             // 0=Balanced, 1=FastConstruction, 2=HighAccuracy, 3=MemoryOptimized
        public int M;                   // Manual M parameter (ignored if AutoScale=true)
        public int EfConstruction;      // Manual ef_construction (ignored if AutoScale=true)
        public int EfSearch;            // Manual ef_search (ignored if AutoScale=true)
        public int MemoryLimitMb;       // Memory limit in MB (0 = no limit)

        /// <summary>
        /// Create HNSW configuration with auto-scaling enabled
        /// </summary>
        public static HnswConfig AutoScale(HnswUseCase useCase = HnswUseCase.Balanced, int memoryLimitMb = 0)
        {
            return new HnswConfig
            {
                AutoScale = true,
                UseCase = (int)useCase,
                M = 16,
                EfConstruction = 128,
                EfSearch = 64,
                MemoryLimitMb = memoryLimitMb
            };
        }

        /// <summary>
        /// Create HNSW configuration with manual parameters
        /// </summary>
        public static HnswConfig Manual(int m, int efConstruction, int efSearch, int memoryLimitMb = 0)
        {
            return new HnswConfig
            {
                AutoScale = false,
                UseCase = (int)HnswUseCase.Balanced,
                M = m,
                EfConstruction = efConstruction,
                EfSearch = efSearch,
                MemoryLimitMb = memoryLimitMb
            };
        }
    }

    /// <summary>
    /// HNSW use case optimization targets
    /// </summary>
    public enum HnswUseCase
    {
        Balanced = 0,           // Balanced performance across all metrics
        FastConstruction = 1,   // Minimize index construction time
        HighAccuracy = 2,       // Maximize search accuracy/recall
        MemoryOptimized = 3     // Minimize memory footprint
    }

    /// <summary>
    /// Normalization mode options
    /// </summary>
    public enum NormalizationMode
    {
        Auto = 0,      // Auto-detect best normalization
        ZScore = 1,    // Z-score normalization (mean=0, std=1)
        MinMax = 2,    // Min-max normalization (range 0-1)
        Robust = 3,    // Robust normalization (median/IQR)
        None = 4       // No normalization
    }

    /// <summary>
    /// Main PacMAP configuration structure
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct PacMapConfig
    {
        public int NNeighbors;
        public int EmbeddingDimensions;
        public int NEpochs;
        public double LearningRate;
        public double MinDist;
        public double MidNearRatio;
        public double FarPairRatio;
        public int Seed;                    // -1 for random seed
        public int NormalizationMode;       // See NormalizationMode enum
        public HnswConfig HnswConfig;

        /// <summary>
        /// Create default PacMAP configuration
        /// </summary>
        public static PacMapConfig Default()
        {
            return new PacMapConfig
            {
                NNeighbors = 10,
                EmbeddingDimensions = 2,
                NEpochs = 450,
                LearningRate = 1.0,
                MinDist = 0.1,
                MidNearRatio = 0.5,
                FarPairRatio = 0.5,
                Seed = -1,
                NormalizationMode = (int)NormalizationMode.Auto,
                HnswConfig = HnswConfig.AutoScale()
            };
        }
    }

    /// <summary>
    /// Model information structure
    /// </summary>
    public struct ModelInfo
    {
        public int NSamples;
        public int NFeatures;
        public int EmbeddingDim;
        public NormalizationMode NormalizationMode;
        public int HnswM;
        public int HnswEfConstruction;
        public int HnswEfSearch;
        public int MemoryUsageMb;
    }

    /// <summary>
    /// Native method imports from PacMAP Enhanced library
    /// </summary>
    internal static class NativeMethods
    {
        private const string DllName = "pacmap_enhanced";

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern PacMapConfig pacmap_config_default();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern HnswConfig pacmap_hnsw_config_for_use_case(int useCase);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr pacmap_fit_transform_enhanced(
            double[] data,
            int rows,
            int cols,
            PacMapConfig config,
            double[] embedding,
            ProgressCallback callback);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int pacmap_transform(
            IntPtr handle,
            double[] data,
            int rows,
            int cols,
            double[] embedding,
            ProgressCallback callback);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int pacmap_get_model_info(
            IntPtr handle,
            out int nSamples,
            out int nFeatures,
            out int embeddingDim,
            out int normalizationMode,
            out int hnswM,
            out int hnswEfConstruction,
            out int hnswEfSearch,
            out int memoryUsageMb);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int pacmap_save_model_enhanced(IntPtr handle, [MarshalAs(UnmanagedType.LPStr)] string path, bool quantize);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr pacmap_load_model_enhanced([MarshalAs(UnmanagedType.LPStr)] string path);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void pacmap_free_model_enhanced(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr pacmap_get_version();
    }

    /// <summary>
    /// High-level C# wrapper for PacMAP Enhanced
    /// </summary>
    public class PacMapModel : IDisposable
    {
        private IntPtr _handle = IntPtr.Zero;
        private bool _disposed = false;

        /// <summary>
        /// Event fired during fitting/transform operations to report progress
        /// </summary>
        public event EventHandler<ProgressEventArgs> ProgressChanged;

        /// <summary>
        /// Get PacMAP Enhanced library version
        /// </summary>
        public static string GetVersion()
        {
            IntPtr versionPtr = NativeMethods.pacmap_get_version();
            return Marshal.PtrToStringAnsi(versionPtr) ?? "Unknown";
        }

        /// <summary>
        /// Fit PacMAP model to data and return embedding
        /// </summary>
        /// <param name="data">Input data matrix (row-major)</param>
        /// <param name="config">PacMAP configuration</param>
        /// <returns>2D embedding</returns>
        public double[,] FitTransform(double[,] data, PacMapConfig? config = null)
        {
            if (data == null) throw new ArgumentNullException(nameof(data));

            int rows = data.GetLength(0);
            int cols = data.GetLength(1);
            var flatData = new double[rows * cols];
            var actualConfig = config ?? PacMapConfig.Default();

            // Flatten input data (row-major)
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    flatData[i * cols + j] = data[i, j];
                }
            }

            // Prepare output embedding
            var embedding = new double[rows * actualConfig.EmbeddingDimensions];

            // Create progress callback
            ProgressCallback progressCallback = (phase, current, total, percent, message) =>
            {
                string phaseStr = Marshal.PtrToStringAnsi(phase) ?? "Unknown";
                string messageStr = message != IntPtr.Zero ? Marshal.PtrToStringAnsi(message) : "";

                ProgressChanged?.Invoke(this, new ProgressEventArgs
                {
                    Phase = phaseStr,
                    Current = current,
                    Total = total,
                    Percent = percent,
                    Message = messageStr
                });
            };

            // Call native fit function
            _handle = NativeMethods.pacmap_fit_transform_enhanced(
                flatData,
                rows,
                cols,
                actualConfig,
                embedding,
                progressCallback);

            if (_handle == IntPtr.Zero)
            {
                throw new InvalidOperationException("PacMAP fitting failed");
            }

            // Convert flat embedding back to 2D array
            var result = new double[rows, actualConfig.EmbeddingDimensions];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < actualConfig.EmbeddingDimensions; j++)
                {
                    result[i, j] = embedding[i * actualConfig.EmbeddingDimensions + j];
                }
            }

            return result;
        }

        /// <summary>
        /// Transform new data using fitted model
        /// </summary>
        /// <param name="data">New data to transform</param>
        /// <returns>Transformed embedding</returns>
        public double[,] Transform(double[,] data)
        {
            if (_handle == IntPtr.Zero) throw new InvalidOperationException("Model not fitted");
            if (data == null) throw new ArgumentNullException(nameof(data));

            int rows = data.GetLength(0);
            int cols = data.GetLength(1);
            var flatData = new double[rows * cols];

            // Get model info to determine embedding dimensions
            var modelInfo = GetModelInfo();
            if (cols != modelInfo.NFeatures)
            {
                throw new ArgumentException($"Feature count mismatch: expected {modelInfo.NFeatures}, got {cols}");
            }

            // Flatten input data
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    flatData[i * cols + j] = data[i, j];
                }
            }

            // Prepare output embedding
            var embedding = new double[rows * modelInfo.EmbeddingDim];

            // Create progress callback
            ProgressCallback progressCallback = (phase, current, total, percent, message) =>
            {
                string phaseStr = Marshal.PtrToStringAnsi(phase) ?? "Unknown";
                string messageStr = message != IntPtr.Zero ? Marshal.PtrToStringAnsi(message) : "";

                ProgressChanged?.Invoke(this, new ProgressEventArgs
                {
                    Phase = phaseStr,
                    Current = current,
                    Total = total,
                    Percent = percent,
                    Message = messageStr
                });
            };

            // Call native transform function
            int result = NativeMethods.pacmap_transform(
                _handle,
                flatData,
                rows,
                cols,
                embedding,
                progressCallback);

            if (result != 0)
            {
                throw new InvalidOperationException($"Transform failed with error code: {result}");
            }

            // Convert flat embedding back to 2D array
            var resultArray = new double[rows, modelInfo.EmbeddingDim];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < modelInfo.EmbeddingDim; j++)
                {
                    resultArray[i, j] = embedding[i * modelInfo.EmbeddingDim + j];
                }
            }

            return resultArray;
        }

        /// <summary>
        /// Get information about the fitted model
        /// </summary>
        /// <returns>Model information</returns>
        public ModelInfo GetModelInfo()
        {
            if (_handle == IntPtr.Zero) throw new InvalidOperationException("Model not fitted");

            int result = NativeMethods.pacmap_get_model_info(
                _handle,
                out int nSamples,
                out int nFeatures,
                out int embeddingDim,
                out int normalizationMode,
                out int hnswM,
                out int hnswEfConstruction,
                out int hnswEfSearch,
                out int memoryUsageMb);

            if (result != 0)
            {
                throw new InvalidOperationException("Failed to get model info");
            }

            return new ModelInfo
            {
                NSamples = nSamples,
                NFeatures = nFeatures,
                EmbeddingDim = embeddingDim,
                NormalizationMode = (NormalizationMode)normalizationMode,
                HnswM = hnswM,
                HnswEfConstruction = hnswEfConstruction,
                HnswEfSearch = hnswEfSearch,
                MemoryUsageMb = memoryUsageMb
            };
        }

        /// <summary>
        /// Save model to file
        /// </summary>
        /// <param name="path">File path</param>
        /// <param name="quantize">Whether to use quantization for smaller file size</param>
        public void Save(string path, bool quantize = false)
        {
            if (_handle == IntPtr.Zero) throw new InvalidOperationException("Model not fitted");
            if (string.IsNullOrEmpty(path)) throw new ArgumentException("Path cannot be null or empty");

            int result = NativeMethods.pacmap_save_model_enhanced(_handle, path, quantize);
            if (result != 0)
            {
                throw new InvalidOperationException($"Failed to save model: error code {result}");
            }
        }

        /// <summary>
        /// Load model from file
        /// </summary>
        /// <param name="path">File path</param>
        /// <returns>Loaded PacMAP model</returns>
        public static PacMapModel Load(string path)
        {
            if (string.IsNullOrEmpty(path)) throw new ArgumentException("Path cannot be null or empty");

            IntPtr handle = NativeMethods.pacmap_load_model_enhanced(path);
            if (handle == IntPtr.Zero)
            {
                throw new InvalidOperationException("Failed to load model");
            }

            return new PacMapModel { _handle = handle };
        }

        /// <summary>
        /// Dispose of native resources
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero)
                {
                    NativeMethods.pacmap_free_model_enhanced(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
            }
        }

        ~PacMapModel()
        {
            Dispose(false);
        }
    }

    /// <summary>
    /// Progress event arguments
    /// </summary>
    public class ProgressEventArgs : EventArgs
    {
        public string Phase { get; set; } = "";
        public int Current { get; set; }
        public int Total { get; set; }
        public float Percent { get; set; }
        public string Message { get; set; } = "";
    }
}