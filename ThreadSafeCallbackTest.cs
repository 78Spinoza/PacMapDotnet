using System;
using System.Threading;
using System.Threading.Tasks;
using PacMapDemo;

namespace ThreadSafeCallbackTest
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("🚀 Testing Thread-Safe Callback System");
            Console.WriteLine("=======================================");
            Console.WriteLine();

            // Test 1: Thread-safe callbacks with automatic events
            Console.WriteLine("📋 Test 1: Thread-Safe Callbacks (Automatic Events)");
            Console.WriteLine("--------------------------------------------------");
            await TestThreadSafeCallbacksAutomatic();
            Console.WriteLine();

            // Test 2: Legacy callbacks for comparison
            Console.WriteLine("📋 Test 2: Legacy Callbacks (For Comparison)");
            Console.WriteLine("--------------------------------------------");
            await TestLegacyCallbacks();
            Console.WriteLine();

            Console.WriteLine("✅ All tests completed!");
        }

        /// <summary>
        /// Test thread-safe callbacks using automatic event handling
        /// </summary>
        static async Task TestThreadSafeCallbacksAutomatic()
        {
            using var model = new SimplePacMapModel(useThreadSafeCallbacks: true);

            // Subscribe to progress events
            model.ProgressChanged += (sender, args) =>
            {
                Console.WriteLine($"[Event] {args.Phase}: {args.Message} ({args.Percent:F1}%)");
            };

            Console.WriteLine("🔄 Creating sample data...");
            var data = CreateSampleData(50, 3);

            Console.WriteLine("🚀 Running PacMAP with thread-safe callbacks (automatic events)...");
            var embedding = model.FitTransform(data, neighbors: 10, seed: 42);

            Console.WriteLine($"✅ Completed! Embedding shape: {embedding.GetLength(0)} × {embedding.GetLength(1)}");
        }

        /// <summary>
        /// Test legacy callbacks for comparison
        /// </summary>
        static async Task TestLegacyCallbacks()
        {
            using var model = new SimplePacMapModel(useThreadSafeCallbacks: false);

            // Subscribe to progress events
            model.ProgressChanged += (sender, args) =>
            {
                Console.WriteLine($"[Legacy] {args.Phase}: {args.Message} ({args.Percent:F1}%)");
            };

            Console.WriteLine("🔄 Creating sample data...");
            var data = CreateSampleData(50, 3);

            Console.WriteLine("🚀 Running PacMAP with legacy callbacks...");
            var embedding = model.FitTransform(data, neighbors: 10, seed: 42);

            Console.WriteLine($"✅ Completed! Embedding shape: {embedding.GetLength(0)} × {embedding.GetLength(1)}");
        }

        /// <summary>
        /// Create sample data for testing
        /// </summary>
        static double[,] CreateSampleData(int rows, int cols)
        {
            var random = new Random(42);
            var data = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    // Create some interesting patterns
                    data[i, j] = Math.Sin(i * 0.1) * Math.Cos(j * 0.2) + random.NextDouble() * 0.1;
                }
            }

            return data;
        }
    }
}