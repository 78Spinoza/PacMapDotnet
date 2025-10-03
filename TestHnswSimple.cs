using System;
using PacMAPSharp;

class HnswTest
{
    static void Main(string[] args)
    {
        Console.WriteLine("Testing PacMAPSharp v1.1.0 with HNSW...");

        try
        {
            // Test library loading
            Console.WriteLine("[1/4] Testing library loading...");
            if (!PacMAPModel.IsLibraryLoaded())
            {
                Console.WriteLine("❌ FAIL: Native library not loaded");
                return;
            }
            Console.WriteLine("✅ PASS: Native library loaded successfully");

            // Test version verification
            Console.WriteLine("[2/4] Testing version verification...");
            PacMAPModel.VerifyVersion();
            Console.WriteLine("✅ PASS: Version verification passed");

            // Create test data (simple 2D dataset)
            Console.WriteLine("[3/4] Creating test data...");
            int nSamples = 1000;
            int nDimensions = 10;
            var data = new double[nSamples, nDimensions];
            var random = new Random(42);

            for (int i = 0; i < nSamples; i++)
            {
                for (int j = 0; j < nDimensions; j++)
                {
                    data[i, j] = random.NextDouble();
                }
            }
            Console.WriteLine($"✅ Created test data: {nSamples} samples x {nDimensions} dimensions");

            // Test PacMAP fitting with HNSW
            Console.WriteLine("[4/4] Testing PacMAP fitting with HNSW...");
            var model = new PacMAPModel();

            // Progress callback
            void ProgressCallback(PacMAPProgress progress)
            {
                Console.WriteLine($"Progress: {progress.Stage} - {progress.PercentComplete:F1}% - {progress.Message}");
            }

            var result = model.Fit(
                data: data,
                nDimensions: 2,
                nNeighbors: 10,
                normalizationMode: NormalizationMode.Auto,
                metric: DistanceMetric.Euclidean,
                hnswUseCase: HnswUseCase.Balanced,
                forceExactKnn: false,
                seed: 42,
                progressCallback: ProgressCallback
            );

            Console.WriteLine($"✅ PASS: PacMAP fitting completed");
            Console.WriteLine($"   Output dimensions: {result.Embedding.GetLength(1)}");
            Console.WriteLine($"   HNSW recall: {result.Quality.HnswRecallPercentage:F1}%");
            Console.WriteLine($"   Transform time: {result.Quality.TransformTimeMs}ms");

            // Display first few embedding coordinates
            Console.WriteLine("\nFirst 5 sample embeddings:");
            for (int i = 0; i < Math.Min(5, result.Embedding.GetLength(0)); i++)
            {
                Console.WriteLine($"  Sample {i}: ({result.Embedding[i, 0]:F4}, {result.Embedding[i, 1]:F4})");
            }

            Console.WriteLine("\n🎉 ALL TESTS PASSED!");
            Console.WriteLine($"HNSW Status: {(result.Quality.UsedHnsw ? "ENABLED" : "DISABLED")}");
            Console.WriteLine($"Model Info: {model.Info}");

        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ FAIL: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
        }
    }
}