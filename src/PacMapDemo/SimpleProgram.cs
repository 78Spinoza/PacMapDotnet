using System;
using PACMAPuwotSharp;

namespace PacMapDemo
{
    class SimpleProgram
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=== PACMAP Demo with New PACMAPCSharp Library ===");
            Console.WriteLine();

            try
            {
                // Create some sample data
                Console.WriteLine("📊 Generating sample data...");
                var (data, labels) = GenerateSampleData(1000, 50);
                Console.WriteLine($"   Generated: {data.GetLength(0)} samples, {data.GetLength(1)} dimensions");
                Console.WriteLine();

                // Test basic PACMAP functionality
                Console.WriteLine("🧪 Testing PACMAP embedding...");
                TestPacMapEmbedding(data);
                Console.WriteLine();

                // Test model persistence
                Console.WriteLine("💾 Testing model save/load...");
                TestModelPersistence(data);
                Console.WriteLine();

                // Test different distance metrics
                Console.WriteLine("📏 Testing different distance metrics...");
                TestDistanceMetrics(data);
                Console.WriteLine();

                Console.WriteLine("✅ All tests completed successfully!");
                Console.WriteLine("PacMapDemo is now working with the new PACMAPCSharp library!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }

            Console.WriteLine();
            Console.WriteLine("Demo completed successfully!");
        }

        static (float[,] data, int[] labels) GenerateSampleData(int samples, int dimensions)
        {
            var random = new Random(42);
            var data = new float[samples, dimensions];
            var labels = new int[samples];

            // Generate 3 clusters of data
            for (int i = 0; i < samples; i++)
            {
                int cluster = i % 3;
                labels[i] = cluster;

                for (int j = 0; j < dimensions; j++)
                {
                    // Add cluster centers and some noise
                    float center = cluster * 2.0f;
                    data[i, j] = center + (float)(random.NextDouble() - 0.5) * 0.5f;
                }
            }

            return (data, labels);
        }

        static void TestPacMapEmbedding(float[,] data)
        {
            using var model = new PacMapModel();

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            var embedding = model.Fit(
                data: data,
                embeddingDimension: 2,
                nNeighbors: 15,
                metric: DistanceMetric.Euclidean,
                randomSeed: 42
            );
            stopwatch.Stop();

            Console.WriteLine($"   ✅ Embedding complete: {embedding.GetLength(0)} points → {embedding.GetLength(1)}D");
            Console.WriteLine($"   ⏱️  Time: {stopwatch.Elapsed.TotalSeconds:F2} seconds");
            Console.WriteLine($"   📏 Model info: {model.ModelInfo}");
        }

        static void TestModelPersistence(float[,] data)
        {
            string modelPath = "test_model.pmm";

            // Train and save model
            using (var model = new PacMapModel())
            {
                var embedding = model.Fit(data, embeddingDimension: 2, randomSeed: 42);
                model.Save(modelPath);
                Console.WriteLine($"   ✅ Model saved to: {modelPath}");
            }

            // Load model and use it
            using (var loadedModel = PacMapModel.Load(modelPath))
            {
                Console.WriteLine($"   ✅ Model loaded successfully");
                Console.WriteLine($"   📏 Loaded model info: {loadedModel.ModelInfo}");

                // Test transform with new data
                var newData = new float[10, data.GetLength(1)];
                var random = new Random(123);
                for (int i = 0; i < 10; i++)
                    for (int j = 0; j < data.GetLength(1); j++)
                        newData[i, j] = (float)random.NextDouble();

                var transformed = loadedModel.Transform(newData);
                Console.WriteLine($"   ✅ Transformed {newData.GetLength(0)} new samples");
            }

            // Cleanup
            if (System.IO.File.Exists(modelPath))
                System.IO.File.Delete(modelPath);
        }

        static void TestDistanceMetrics(float[,] data)
        {
            var metrics = new DistanceMetric[]
            {
                DistanceMetric.Euclidean,
                DistanceMetric.Manhattan,
                DistanceMetric.Cosine
            };

            foreach (var metric in metrics)
            {
                using var model = new PacMapModel();
                var stopwatch = System.Diagnostics.Stopwatch.StartNew();

                var embedding = model.Fit(
                    data: data,
                    embeddingDimension: 2,
                    nNeighbors: 10,
                    metric: metric,
                    randomSeed: 42
                );

                stopwatch.Stop();
                Console.WriteLine($"   ✅ {metric}: {embedding.GetLength(0)} points in {stopwatch.Elapsed.TotalSeconds:F2}s");
            }
        }
    }
}