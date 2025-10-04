using System;
using System.Runtime.InteropServices;

public class SimpleSeedTest
{
    // Import the PacMAP fit transform function directly
    [DllImport("pacmap_enhanced.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern int fit_transform_hnsw_with_params(
        IntPtr data,
        int rows,
        int cols,
        IntPtr embedding,
        int embedding_buffer_len,
        PacmapConfig config,
        IntPtr callback,
        IntPtr user_data
    );

    [StructLayout(LayoutKind.Sequential)]
    public struct PacmapConfig
    {
        public int n_neighbors;
        public int n_components;
        public int pn_neighbors;
        public int distance_metric;
        public int normalization_mode;
        public int hnsw_use_case;
        public int callback_mode;
        public int seed;
        public int num_iterations;
        public float learning_rate;
        public float apply_decay;
        public int verbose;
    }

    public static void Main()
    {
        Console.WriteLine("🔍 Direct FFI Seed Test");
        Console.WriteLine("=========================");

        // Create simple test data (10 points, 3 dimensions)
        int rows = 10, cols = 3;
        double[] data = new double[rows * cols];
        double[] embedding = new double[rows * 2]; // 2D embedding

        // Fill with deterministic data
        Random rand = new Random(42);
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = rand.NextDouble();
        }

        Console.WriteLine($"📊 Created {rows}×{cols} test data");

        // Set up config with explicit seed
        var config = new PacmapConfig
        {
            n_neighbors = 5,
            n_components = 2,
            pn_neighbors = 5,
            distance_metric = 0, // Euclidean
            normalization_mode = 0, // Standard
            hnsw_use_case = 0, // Default
            callback_mode = 1, // Thread-safe
            seed = 42, // EXPLICIT SEED
            num_iterations = 100,
            learning_rate = 1.0f,
            apply_decay = 1.0f,
            verbose = 1
        };

        Console.WriteLine($"🚀 Calling PacMAP with seed={config.seed}");

        // Pin data and call the function
        GCHandle dataHandle = GCHandle.Alloc(data, GCHandleType.Pinned);
        GCHandle embedHandle = GCHandle.Alloc(embedding, GCHandleType.Pinned);

        try
        {
            IntPtr dataPtr = dataHandle.AddrOfPinnedObject();
            IntPtr embedPtr = embedHandle.AddrOfPinnedObject();

            int result = fit_transform_hnsw_with_params(
                dataPtr, rows, cols, embedPtr, embedding.Length,
                config, IntPtr.Zero, IntPtr.Zero
            );

            Console.WriteLine($"✅ Result code: {result}");

            if (result == 0)
            {
                Console.WriteLine($"📋 First embedding point: ({embedding[0]:F6}, {embedding[1]:F6})");
                Console.WriteLine($"📋 Second embedding point: ({embedding[2]:F6}, {embedding[3]:F6})");
            }
            else
            {
                Console.WriteLine($"❌ Error code: {result}");
            }
        }
        finally
        {
            dataHandle.Free();
            embedHandle.Free();
        }
    }
}