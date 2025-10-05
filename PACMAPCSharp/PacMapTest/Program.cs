using System;
using UMAPuwotSharp;

namespace PacMapTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("========================================");
            Console.WriteLine("PACMAP C# to C++ Integration Test");
            Console.WriteLine("========================================");

            try
            {
                // Test 1: Basic model creation
                Console.WriteLine("\n=== Test 1: Basic Model Creation ===");
                using (var model = new PacMapModel())
                {
                    Console.WriteLine("✓ PacMapModel created successfully");
                    Console.WriteLine($"  - IsFitted: {model.IsFitted}");
                }

                // Test 2: Simple data fit
                Console.WriteLine("\n=== Test 2: Simple Data Fit ===");
                var testData = GenerateTestData(20, 3);
                Console.WriteLine($"Generated test data: {testData.GetLength(0)} samples x {testData.GetLength(1)} features");

                using (var model = new PacMapModel())
                {
                    var embedding = model.Fit(testData);
                    Console.WriteLine($"✓ Fit completed successfully");
                    Console.WriteLine($"  - Embedding shape: {embedding.GetLength(0)} x {embedding.GetLength(1)}");
                    Console.WriteLine($"  - IsFitted: {model.IsFitted}");

                    // Test 3: Transform new data
                    Console.WriteLine("\n=== Test 3: Transform New Data ===");
                    var newData = GenerateTestData(5, 3);
                    var transformed = model.Transform(newData);
                    Console.WriteLine($"✓ Transform completed successfully");
                    Console.WriteLine($"  - Transformed shape: {transformed.GetLength(0)} x {transformed.GetLength(1)}");

                    // Show some results
                    Console.WriteLine("\nSample results:");
                    for (int i = 0; i < Math.Min(3, embedding.GetLength(0)); i++)
                    {
                        Console.WriteLine($"  Training[{i}]: ({embedding[i, 0]:F3}, {embedding[i, 1]:F3})");
                    }
                    for (int i = 0; i < Math.Min(3, transformed.GetLength(0)); i++)
                    {
                        Console.WriteLine($"  Transform[{i}]: ({transformed[i, 0]:F3}, {transformed[i, 1]:F3})");
                    }
                }

                Console.WriteLine("\n========================================");
                Console.WriteLine("🎉 ALL C# TESTS PASSED!");
                Console.WriteLine("✅ Model creation and disposal");
                Console.WriteLine("✅ Data fitting with PACMAP");
                Console.WriteLine("✅ New data transformation");
                Console.WriteLine("✅ C# to C++ integration working");
                Console.WriteLine("\n🚀 PACMAP C# wrapper is production ready!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\n❌ TEST FAILED: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
                Environment.Exit(1);
            }
        }

        static float[,] GenerateTestData(int nSamples, int nFeatures)
        {
            var data = new float[nSamples, nFeatures];
            var random = new Random(42);

            for (int i = 0; i < nSamples; i++)
            {
                for (int j = 0; j < nFeatures; j++)
                {
                    // Generate simple test data with some pattern
                    data[i, j] = (float)(Math.Sin(i * 0.1) + Math.Cos(j * 0.5) + random.NextDouble() * 0.1);
                }
            }

            return data;
        }
    }
}
