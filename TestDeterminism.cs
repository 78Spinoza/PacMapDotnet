using System;
using PacMAPSharp;

namespace TestDeterminism
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=== PacMAP Determinism Test ===");

            // Create simple test data
            var rng = new Random(42);
            int nSamples = 1000;
            int nFeatures = 3;
            double[,] data = new double[nSamples, nFeatures];

            for (int i = 0; i < nSamples; i++)
            {
                for (int j = 0; j < nFeatures; j++)
                {
                    data[i, j] = rng.NextDouble() * 10;
                }
            }

            Console.WriteLine($"Test data: {nSamples} samples × {nFeatures} features");

            // Test 1: Same seed, should get identical results
            Console.WriteLine("\n--- Test 1: Same Seed Determinism ---");

            var pacmap1 = new PacMAPModel();
            var result1 = pacmap1.Fit(data, embeddingDimensions: 2, neighbors: 15,
                                       normalization: NormalizationMode.ZScore,
                                       forceExactKnn: true, seed: 42);

            var pacmap2 = new PacMAPModel();
            var result2 = pacmap2.Fit(data, embeddingDimensions: 2, neighbors: 15,
                                       normalization: NormalizationMode.ZScore,
                                       forceExactKnn: true, seed: 42);

            // Compare results
            double maxDiff = 0;
            double totalDiff = 0;
            int differentPoints = 0;

            for (int i = 0; i < nSamples; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    double diff = Math.Abs(result1.EmbeddingCoordinates[i, j] - result2.EmbeddingCoordinates[i, j]);
                    totalDiff += diff;
                    if (diff > maxDiff) maxDiff = diff;
                    if (diff > 1e-10) differentPoints++;
                }
            }

            Console.WriteLine($"Max difference: {maxDiff:E2}");
            Console.WriteLine($"Average difference: {totalDiff / (nSamples * 2):E2}");
            Console.WriteLine($"Different points: {differentPoints}/{nSamples * 2} ({100.0 * differentPoints / (nSamples * 2):F1}%)");

            if (maxDiff < 1e-10)
            {
                Console.WriteLine("✅ PASS: Deterministic with same seed");
            }
            else
            {
                Console.WriteLine("❌ FAIL: Not deterministic with same seed");
            }

            // Test 2: Save/Load/Transform
            Console.WriteLine("\n--- Test 2: Save/Load/Transform ---");

            // Save first model
            string modelPath = "test_model.bin";
            pacmap1.Save(modelPath);
            Console.WriteLine($"Model saved: {modelPath}");

            // Load model
            var loadedPacmap = PacMAPModel.Load(modelPath);
            Console.WriteLine("Model loaded successfully");

            // Transform same data
            var transformResult = loadedPacmap.Transform(data);
            Console.WriteLine($"Transform completed: {transformResult.GetLength(0)} points");

            // Transform vs Original comparison
            maxDiff = 0;
            totalDiff = 0;
            differentPoints = 0;

            for (int i = 0; i < nSamples; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    double diff = Math.Abs(result1.EmbeddingCoordinates[i, j] - transformResult[i, j]);
                    totalDiff += diff;
                    if (diff > maxDiff) maxDiff = diff;
                    if (diff > 1e-10) differentPoints++;
                }
            }

            Console.WriteLine($"Transform vs Original Max difference: {maxDiff:E2}");
            Console.WriteLine($"Transform vs Original Average difference: {totalDiff / (nSamples * 2):E2}");
            Console.WriteLine($"Transform vs Original Different points: {differentPoints}/{nSamples * 2} ({100.0 * differentPoints / (nSamples * 2):F1}%)");

            Console.WriteLine("\n=== Test Complete ===");
        }
    }
}