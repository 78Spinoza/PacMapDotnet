using System;
using PacMAPSharp;

namespace PacMapDemo
{
    public static class HnswQuickTest
    {
        public static void RunMinimalHnswTest()
        {
            Console.WriteLine("🔧 HNSW Quick Debug Test");
            Console.WriteLine("========================");

            // Create tiny synthetic dataset (just enough to trigger HNSW: >1000 samples)
            int samples = 1200;  // Above HNSW threshold
            int features = 3;
            var data = new double[samples, features];

            // Simple clustered data
            var random = new Random(42);
            for (int i = 0; i < samples; i++)
            {
                // Create 3 simple clusters
                int cluster = i % 3;
                double baseX = cluster * 10.0;
                data[i, 0] = baseX + random.NextDouble() * 2.0;
                data[i, 1] = random.NextDouble() * 2.0;
                data[i, 2] = random.NextDouble() * 2.0;
            }

            Console.WriteLine($"📊 Test data: {samples}x{features} (triggers HNSW)");

            // Test 1: Exact KNN (should work)
            Console.WriteLine("\n🔍 Test 1: Exact KNN (baseline)");
            try
            {
                using var modelKNN = new PacMAPModel();
                var resultKNN = modelKNN.Fit(data, embeddingDimensions: 2, neighbors: 5,
                    forceExactKnn: true, seed: 42);
                Console.WriteLine($"✅ Exact KNN: SUCCESS ({resultKNN.ConfidenceScore:F3})");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Exact KNN: FAILED - {ex.Message}");
                return; // If exact KNN fails, something is fundamentally wrong
            }

            // Test 2: HNSW (the problem case)
            Console.WriteLine("\n⚡ Test 2: HNSW (debug target)");
            try
            {
                using var modelHNSW = new PacMAPModel();
                var resultHNSW = modelHNSW.Fit(data, embeddingDimensions: 2, neighbors: 5,
                    hnswUseCase: HnswUseCase.Balanced, forceExactKnn: false, seed: 42);
                Console.WriteLine($"✅ HNSW: SUCCESS ({resultHNSW.ConfidenceScore:F3})");
                Console.WriteLine("🎉 HNSW WORKS! Issue may be mammoth-dataset specific.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ HNSW: FAILED - {ex.Message}");
                Console.WriteLine("🔍 Confirmed: HNSW issue is systematic, not data-specific");

                // Quick parameter tests
                Console.WriteLine("\n🧪 Testing different HNSW parameters:");

                // Test with minimal HNSW settings
                try
                {
                    using var modelMinimal = new PacMAPModel();
                    var resultMinimal = modelMinimal.Fit(data, embeddingDimensions: 2, neighbors: 3,
                        hnswUseCase: HnswUseCase.FastConstruction, forceExactKnn: false, seed: 42);
                    Console.WriteLine("✅ HNSW Fast: SUCCESS");
                }
                catch
                {
                    Console.WriteLine("❌ HNSW Fast: FAILED");
                }

                // Test with different embedding dimensions
                try
                {
                    using var model3D = new PacMAPModel();
                    var result3D = model3D.Fit(data, embeddingDimensions: 3, neighbors: 5,
                        hnswUseCase: HnswUseCase.Balanced, forceExactKnn: false, seed: 42);
                    Console.WriteLine("✅ HNSW 3D: SUCCESS");
                }
                catch
                {
                    Console.WriteLine("❌ HNSW 3D: FAILED");
                }
            }

            Console.WriteLine("\n🔧 Quick test completed.");
        }
    }
}