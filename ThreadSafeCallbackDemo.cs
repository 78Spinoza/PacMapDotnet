using System;
using System.Threading;
using System.Threading.Tasks;
using PacMapDemo;

namespace ThreadSafeCallbackDemo
{
    /// <summary>
    /// Demonstration of the new thread-safe callback system
    /// Shows both automatic event-based and manual polling approaches
    /// </summary>
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("🚀 PacMAP Thread-Safe Callback System Demo");
            Console.WriteLine("==========================================");
            Console.WriteLine();

            // Test 1: Thread-safe callbacks with automatic event handling
            Console.WriteLine("📋 Test 1: Thread-Safe Callbacks (Automatic Events)");
            Console.WriteLine("--------------------------------------------------");
            await TestThreadSafeCallbacksAutomatic();
            Console.WriteLine();

            // Test 2: Thread-safe callbacks with manual polling
            Console.WriteLine("📋 Test 2: Thread-Safe Callbacks (Manual Polling)");
            Console.WriteLine("-------------------------------------------------");
            await TestThreadSafeCallbacksManual();
            Console.WriteLine();

            // Test 3: Legacy callbacks for comparison
            Console.WriteLine("📋 Test 3: Legacy Callbacks (For Comparison)");
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
            var data = CreateSampleData(100, 10);

            Console.WriteLine("🚀 Running PacMAP with thread-safe callbacks (automatic events)...");
            var embedding = model.FitTransform(data, neighbors: 10, seed: 42);

            Console.WriteLine($"✅ Completed! Embedding shape: {embedding.GetLength(0)} × {embedding.GetLength(1)}");
        }

        /// <summary>
        /// Test thread-safe callbacks using manual polling
        /// </summary>
        static async Task TestThreadSafeCallbacksManual()
        {
            using var model = new SimplePacMapModel(useThreadSafeCallbacks: true);

            // Start manual polling
            using var cts = new CancellationTokenSource();

            // Create a background task to poll and display messages
            var pollingTask = Task.Run(async () =>
            {
                while (!cts.Token.IsCancellationRequested)
                {
                    if (model.HasPendingMessages())
                    {
                        var message = model.PollSingleMessage();
                        if (!string.IsNullOrEmpty(message))
                        {
                            Console.WriteLine($"[Manual Poll] {message}");
                        }
                    }
                    await Task.Delay(50, cts.Token);
                }
            });

            Console.WriteLine("🔄 Creating sample data...");
            var data = CreateSampleData(100, 10);

            Console.WriteLine("🚀 Running PacMAP with thread-safe callbacks (manual polling)...");
            var embedding = model.FitTransform(data, neighbors: 10, seed: 42);

            // Wait a bit for any remaining messages
            await Task.Delay(500);

            // Stop polling
            cts.Cancel();
            try
            {
                await pollingTask;
            }
            catch (OperationCanceledException)
            {
                // Expected
            }

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
            var data = CreateSampleData(100, 10);

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