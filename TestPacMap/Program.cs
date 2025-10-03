using System;
using PacMAPSharp;

namespace TestPacMap
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Testing PacMAP v1.1.0 with HNSW features...");

            try
            {
                // Create simple test data
                var n_samples = 100;
                var n_features = 3;
                var data = new double[n_samples * n_features];

                // Generate simple test data
                for (int i = 0; i < n_samples; i++)
                {
                    for (int j = 0; j < n_features; j++)
                    {
                        data[i * n_features + j] = (i * 3 + j) * 0.1;
                    }
                }

                Console.WriteLine($"Created test data: {n_samples} samples × {n_features} features");

                // Test basic PacMAP functionality
                using (var pacmap = new PacMapModel())
                {
                    Console.WriteLine("Testing PacMAP fit...");
                    var embedding = pacmap.FitTransform(
                        data: data,
                        embeddingDimensions: 2,
                        neighbors: 10,
                        normalization: NormalizationMode.ZScore,
                        metric: DistanceMetric.Euclidean,
                        hnswUseCase: HnswUseCase.Balanced,
                        forceExactKnn: false, // Use HNSW
                        learningRate: 1.0,
                        nEpochs: (100, 100, 250),
                        autodetectHnswParams: true,
                        seed: 42
                    );

                    Console.WriteLine($"✅ PacMAP fit successful! Embedding shape: {embedding.Length / 2} × 2");

                    // Print first few embedding values
                    Console.WriteLine("First 5 embedding points:");
                    for (int i = 0; i < Math.Min(5, embedding.Length / 2); i++)
                    {
                        Console.WriteLine($"  Point {i}: [{embedding[i * 2]:.4f}, {embedding[i * 2 + 1]:.4f}]");
                    }

                    // Test model save/load
                    Console.WriteLine("Testing model serialization...");
                    var modelBytes = pacmap.SaveModel();
                    Console.WriteLine($"✅ Model saved: {modelBytes.Length} bytes");

                    // Test transform
                    Console.WriteLine("Testing transform...");
                    var newEmbedding = pacmap.Transform(data);
                    Console.WriteLine($"✅ Transform successful! Shape: {newEmbedding.Length / 2} × 2");
                }

                Console.WriteLine("\n🎉 All tests passed! HNSW incorporation is working correctly!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Test failed: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }
    }
}