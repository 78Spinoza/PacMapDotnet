// Real PacMAP Wrapper using actual Rust FFI
using System;
using System.Runtime.InteropServices;

namespace PacMapDemo
{
    /// <summary>
    /// HNSW configuration for FFI
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct HnswConfig
    {
        public bool auto_scale;
        public int use_case;        // 0=Balanced, 1=FastConstruction, 2=HighAccuracy, 3=MemoryOptimized
        public int m;
        public int ef_construction;
        public int ef_search;
        public int memory_limit_mb;

        public static HnswConfig Default => new HnswConfig
        {
            auto_scale = true,
            use_case = 0,    // Balanced
            m = 16,
            ef_construction = 128,
            ef_search = 64,
            memory_limit_mb = 0
        };
    }

    /// <summary>
    /// PacMAP configuration for FFI
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct PacmapConfig
    {
        public int n_neighbors;
        public int embedding_dimensions;
        public int n_epochs;
        public double learning_rate;
        public double min_dist;
        public double mid_near_ratio;
        public double far_pair_ratio;
        public int seed;
        public int normalization_mode;  // 0=Auto, 1=ZScore, 2=MinMax, 3=Robust, 4=None
        public bool force_exact_knn;    // If true, disable HNSW and use brute-force KNN
        public HnswConfig hnsw_config;

        public static PacmapConfig Default => new PacmapConfig
        {
            n_neighbors = 10,
            embedding_dimensions = 2,
            n_epochs = 450,
            learning_rate = 1.0,
            min_dist = 0.1,
            mid_near_ratio = 0.5,
            far_pair_ratio = 0.5,
            seed = -1,
            normalization_mode = 0,  // Auto
            force_exact_knn = false, // Use HNSW by default
            hnsw_config = HnswConfig.Default
        };
    }

    /// <summary>
    /// Progress callback delegate
    /// </summary>
    public delegate void PacmapProgressCallback(
        IntPtr phase,
        int current,
        int total,
        float percent,
        IntPtr message
    );

    /// <summary>
    /// Real PacMAP wrapper using actual Rust FFI
    /// </summary>
    public class RealPacMapModel : IDisposable
    {
        private const string WindowsDll = "pacmap_enhanced.dll";
        private const string LinuxDll = "libpacmap_enhanced.so";

        private static bool IsWindows => RuntimeInformation.IsOSPlatform(OSPlatform.Windows);

        // FFI imports
        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_version")]
        private static extern IntPtr WindowsGetVersion();

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_version")]
        private static extern IntPtr LinuxGetVersion();

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_fit_transform_enhanced")]
        private static extern IntPtr WindowsFitTransform(
            double[] data,
            int rows,
            int cols,
            PacmapConfig config,
            double[] embedding,
            PacmapProgressCallback callback
        );

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_fit_transform_enhanced")]
        private static extern IntPtr LinuxFitTransform(
            double[] data,
            int rows,
            int cols,
            PacmapConfig config,
            double[] embedding,
            PacmapProgressCallback callback
        );

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_free_model_enhanced")]
        private static extern void WindowsFreeModel(IntPtr handle);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_free_model_enhanced")]
        private static extern void LinuxFreeModel(IntPtr handle);

        private IntPtr _modelHandle = IntPtr.Zero;
        private bool _disposed = false;

        /// <summary>
        /// Event fired during operations to report progress
        /// </summary>
        public event EventHandler<ProgressEventArgs>? ProgressChanged;

        /// <summary>
        /// Get PacMAP version
        /// </summary>
        public static string GetVersion()
        {
            try
            {
                var versionPtr = IsWindows ? WindowsGetVersion() : LinuxGetVersion();
                return versionPtr != IntPtr.Zero ? Marshal.PtrToStringAnsi(versionPtr) ?? "Unknown" : "Unknown";
            }
            catch
            {
                return "Unknown";
            }
        }

        /// <summary>
        /// Fit PacMAP model and return 2D embedding using real Rust implementation
        /// </summary>
        public double[,] FitTransform(double[,] data, int neighbors = 15, int seed = 42, bool forceExactKnn = false, bool useQuantization = true)
        {
            return FitTransform(data, neighbors, seed, forceExactKnn, useQuantization, 0.5, 2.0, 450);
        }

        /// <summary>
        /// Fit PacMAP model and return 2D embedding with full hyperparameter control
        /// </summary>
        public double[,] FitTransform(double[,] data, int neighbors = 15, int seed = 42, bool forceExactKnn = false, bool useQuantization = true,
                                     double midNearRatio = 0.5, double farPairRatio = 2.0, int numIters = 450)
        {
            if (data == null) throw new ArgumentNullException(nameof(data));

            int rows = data.GetLength(0);
            int cols = data.GetLength(1);

            Console.WriteLine("🔧 DETAILED PARAMETER BREAKDOWN:");
            Console.WriteLine($"   ┌─ Data shape: {rows} samples × {cols} features");
            Console.WriteLine($"   ├─ Neighbors: {neighbors}");
            Console.WriteLine($"   ├─ Embedding dimensions: 2");
            Console.WriteLine($"   ├─ Force exact KNN: {forceExactKnn} {(forceExactKnn ? "(HNSW DISABLED)" : "(HNSW ENABLED)")}");
            Console.WriteLine($"   ├─ Use quantization: {useQuantization} {(useQuantization ? "(ENABLED)" : "(DISABLED)")}");
            Console.WriteLine($"   ├─ Distance metric: Euclidean");
            Console.WriteLine($"   ├─ Random seed: {seed}");
            Console.WriteLine($"   ├─ Epochs: {numIters}");
            Console.WriteLine($"   ├─ Mid-near ratio: {midNearRatio}");
            Console.WriteLine($"   └─ Far-pair ratio: {farPairRatio}");
            Console.WriteLine();

            // Configure PacMAP
            var config = PacmapConfig.Default;
            config.n_neighbors = neighbors;
            config.seed = seed;
            config.mid_near_ratio = midNearRatio;
            config.far_pair_ratio = farPairRatio;
            config.n_epochs = numIters;
            config.force_exact_knn = forceExactKnn;  // Pass forceExactKnn to Rust

            if (forceExactKnn)
            {
                // When forcing exact KNN, don't configure HNSW at all - let Rust handle it
                config.hnsw_config.auto_scale = false;
                Console.WriteLine("🔧 C# DEBUG: force_exact_knn=true, HNSW config disabled");
            }
            else
            {
                // Only configure HNSW when not forcing exact KNN
                config.hnsw_config.auto_scale = true;
                config.hnsw_config.ef_search = 1;
            }

            // Convert 2D array to 1D for FFI
            var flatData = new double[rows * cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    flatData[i * cols + j] = data[i, j];
                }
            }

            // Prepare output array
            var flatEmbedding = new double[rows * 2];

            // Create progress callback
            PacmapProgressCallback progressCallback = (phasePtr, current, total, percent, messagePtr) =>
            {
                try
                {
                    var phase = phasePtr != IntPtr.Zero ? Marshal.PtrToStringAnsi(phasePtr) ?? "Unknown" : "Unknown";
                    var message = messagePtr != IntPtr.Zero ? Marshal.PtrToStringAnsi(messagePtr) ?? "" : "";

                    ProgressChanged?.Invoke(this, new ProgressEventArgs
                    {
                        Phase = phase,
                        Current = current,
                        Total = total,
                        Percent = percent,
                        Message = message
                    });
                }
                catch
                {
                    // Ignore callback errors
                }
            };

            Console.WriteLine("🚀 Calling real PacMAP Rust implementation...");

            // Call the real Rust function
            _modelHandle = IsWindows
                ? WindowsFitTransform(flatData, rows, cols, config, flatEmbedding, progressCallback)
                : LinuxFitTransform(flatData, rows, cols, config, flatEmbedding, progressCallback);

            if (_modelHandle == IntPtr.Zero)
            {
                throw new InvalidOperationException("PacMAP fit_transform failed");
            }

            // Convert back to 2D array
            var result = new double[rows, 2];
            for (int i = 0; i < rows; i++)
            {
                result[i, 0] = flatEmbedding[i * 2];
                result[i, 1] = flatEmbedding[i * 2 + 1];
            }

            Console.WriteLine("✅ Real PacMAP completed successfully!");
            return result;
        }

        /// <summary>
        /// Get model information
        /// </summary>
        public ModelInfo GetModelInfo()
        {
            return new ModelInfo
            {
                NSamples = 0,  // Not available from Rust FFI
                NFeatures = 0, // Not available from Rust FFI
                EmbeddingDim = 2,
                MemoryUsageMb = 0  // Not available from Rust FFI
            };
        }

        /// <summary>
        /// Save model (not implemented in current Rust FFI)
        /// </summary>
        public void Save(string path, bool quantize = false)
        {
            Console.WriteLine($"💾 Model save not implemented in current FFI: {path}");
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (_modelHandle != IntPtr.Zero)
                {
                    try
                    {
                        if (IsWindows)
                            WindowsFreeModel(_modelHandle);
                        else
                            LinuxFreeModel(_modelHandle);
                    }
                    catch
                    {
                        // Ignore disposal errors
                    }
                    _modelHandle = IntPtr.Zero;
                }
                _disposed = true;
            }
        }
    }
}