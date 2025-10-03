using System;
using System.IO;
using PacMAPSharp;

namespace TestTransform
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=== PacMAP Transform Consistency Test ===");
            Console.WriteLine("Mode: Direct KNN, Quantization OFF");
            Console.WriteLine();

            // Use a smaller dataset for easier debugging
            int nSamples = 1000;
            int nFeatures = 3;
            double[,] data = new double[nSamples, nFeatures];

            // Create deterministic test data
            Random rng = new Random(42);
            for (int i = 0; i < nSamples; i++)
            {
                for (int j = 0; j < nFeatures; j++)
                {
                    data[i, j] = rng.NextDouble() * 10;
                }
            }

            Console.WriteLine($"Test data: {nSamples} samples × {nFeatures} features");
            Console.WriteLine($"Random seed: 42");
            Console.WriteLine();

            // === TEST 1: Fit + Transform (should be the same as Transform) ===
            Console.WriteLine("=== TEST 1: Fit + Transform Consistency ===");

            var pacmap1 = new PacMAPModel();
            var fitResult = pacmap1.Fit(
                data: data,
                embeddingDimensions: 2,
                neighbors: 15,
                normalization: NormalizationMode.ZScore,
                forceExactKnn: true,
                seed: 42,
                useQuantization: false
            );

            Console.WriteLine($"✅ Fit completed: {fitResult.EmbeddingCoordinates.GetLength(0)} points");

            // Transform the same data with the same model
            var transformResult = pacmap1.Transform(data);
            Console.WriteLine($"✅ Transform completed: {transformResult.GetLength(0)} points");

            // Compare Fit vs Transform results
            double maxDiff1 = 0;
            double totalDiff1 = 0;
            int diffCount1 = 0;

            for (int i = 0; i < nSamples; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    double diff = Math.Abs(fitResult.EmbeddingCoordinates[i, j] - transformResult[i, j]);
                    totalDiff1 += diff;
                    if (diff > maxDiff1) maxDiff1 = diff;
                    if (diff > 1e-12) diffCount1++;
                }
            }

            Console.WriteLine($"Fit vs Transform Results:");
            Console.WriteLine($"  Max difference: {maxDiff1:E2}");
            Console.WriteLine($"  Average difference: {totalDiff1 / (nSamples * 2):E2}");
            Console.WriteLine($"  Different points: {diffCount1}/{nSamples * 2} ({100.0 * diffCount1 / (nSamples * 2):F1}%)");

            if (maxDiff1 < 1e-10)
            {
                Console.WriteLine("✅ PASS: Fit and Transform produce identical results");
            }
            else
            {
                Console.WriteLine("❌ FAIL: Fit and Transform produce different results");
            }

            Console.WriteLine();

            // === TEST 2: Save/Load/Transform Consistency ===
            Console.WriteLine("=== TEST 2: Save/Load/Transform Consistency ===");

            // Save the model
            string modelPath = "test_model.bin";
            pacmap1.Save(modelPath);
            Console.WriteLine($"✅ Model saved: {modelPath}");

            // Load the model
            var loadedPacmap = PacMAPModel.Load(modelPath);
            Console.WriteLine("✅ Model loaded");

            // Transform with loaded model
            var loadedTransformResult = loadedPacmap.Transform(data);
            Console.WriteLine($"✅ Loaded Transform completed: {loadedTransformResult.GetLength(0)} points");

            // Compare original Transform vs Loaded Transform
            double maxDiff2 = 0;
            double totalDiff2 = 0;
            int diffCount2 = 0;

            for (int i = 0; i < nSamples; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    double diff = Math.Abs(transformResult[i, j] - loadedTransformResult[i, j]);
                    totalDiff2 += diff;
                    if (diff > maxDiff2) maxDiff2 = diff;
                    if (diff > 1e-12) diffCount2++;
                }
            }

            Console.WriteLine($"Original Transform vs Loaded Transform Results:");
            Console.WriteLine($"  Max difference: {maxDiff2:E2}");
            Console.WriteLine($"  Average difference: {totalDiff2 / (nSamples * 2):E2}");
            Console.WriteLine($"  Different points: {diffCount2}/{nSamples * 2} ({100.0 * diffCount2 / (nSamples * 2):F1}%)");

            if (maxDiff2 < 1e-10)
            {
                Console.WriteLine("✅ PASS: Original and Loaded transforms are identical");
            }
            else
            {
                Console.WriteLine("❌ FAIL: Original and Loaded transforms are different");
            }

            Console.WriteLine();

            // === TEST 3: Fit vs Fit (same seed) ===
            Console.WriteLine("=== TEST 3: Fit Determinism (same seed) ===");

            var pacmap2 = new PacMAPModel();
            var fitResult2 = pacmap2.Fit(
                data: data,
                embeddingDimensions: 2,
                neighbors: 15,
                normalization: NormalizationMode.ZScore,
                forceExactKnn: true,
                seed: 42,
                useQuantization: false
            );

            // Compare two fits with same seed
            double maxDiff3 = 0;
            double totalDiff3 = 0;
            int diffCount3 = 0;

            for (int i = 0; i < nSamples; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    double diff = Math.Abs(fitResult.EmbeddingCoordinates[i, j] - fitResult2.EmbeddingCoordinates[i, j]);
                    totalDiff3 += diff;
                    if (diff > maxDiff3) maxDiff3 = diff;
                    if (diff > 1e-12) diffCount3++;
                }
            }

            Console.WriteLine($"Fit1 vs Fit2 Results (same seed):");
            Console.WriteLine($"  Max difference: {maxDiff3:E2}");
            Console.WriteLine($"  Average difference: {totalDiff3 / (nSamples * 2):E2}");
            Console.WriteLine($"  Different points: {diffCount3}/{nSamples * 2} ({100.0 * diffCount3 / (nSamples * 2):F1}%)");

            if (maxDiff3 < 1e-10)
            {
                Console.WriteLine("✅ PASS: Two fits with same seed are identical");
            }
            else
            {
                Console.WriteLine("❌ FAIL: Two fits with same seed are different");
            }

            // Cleanup
            if (File.Exists(modelPath))
            {
                File.Delete(modelPath);
                Console.WriteLine($"🗑️  Cleaned up: {modelPath}");
            }

            Console.WriteLine();
            Console.WriteLine("=== Test Complete ===");
        }
    }
}