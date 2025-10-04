using System;
using PacMapDemo;

namespace SeedTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("🔍 Testing Seed Propagation");
            Console.WriteLine("===============================");

            // Create simple test data
            var data = new double[50, 3];
            var random = new Random(42);

            for (int i = 0; i < 50; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    data[i, j] = random.NextDouble();
                }
            }

            Console.WriteLine("📊 Created test data: 50 × 3");

            // Test with fixed seed
            Console.WriteLine("🚀 Running PacMAP with seed=42...");

            using var model = new SimplePacMapModel(useThreadSafeCallbacks: true);
            model.ProgressChanged += (sender, e) => {
                Console.WriteLine($"  Progress: {e.Percent:F1}% - {e.Message}");
            };

            var embedding = model.FitTransform(data, neighbors: 10, seed: 42);

            Console.WriteLine($"✅ Completed! Embedding shape: {embedding.GetLength(0)} × {embedding.GetLength(1)}");
            Console.WriteLine($"📋 First point: ({embedding[0, 0]:F6}, {embedding[0, 1]:F6})");
        }
    }
}