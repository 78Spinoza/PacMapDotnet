using System;
using System.Diagnostics;
using PacMapSharp;

namespace PacMapBenchmarks
{
    class QuickBenchmark
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=== PACMAP Quick Performance Benchmark ===");
            Console.WriteLine("Data Size\tFeatures\tBuild Time (ms)\tTransform Time (ms)\tMemory (MB)");
            Console.WriteLine(new string('-', 80));

            // Test different data sizes
            int[] dataSizes = { 1000, 5000, 10000 };
            int[] dimensions = { 50, 100, 300 };

            foreach (int nSamples in dataSizes)
            {
                foreach (int nFeatures in dimensions)
                {
                    RunQuickBenchmark(nSamples, nFeatures);
                }
            }

            Console.WriteLine("\n=== Performance Analysis Complete ===");
        }

        static void RunQuickBenchmark(int nSamples, int nFeatures)
        {
            // Generate test data
            double[,] data = GenerateTestData(nSamples, nFeatures);

            // Memory before
            long memoryBefore = GC.GetTotalMemory(true);

            // Create model
            var model = new PacMapModel(mnRatio: 0.5f, fpRatio: 2.0f);

            // Benchmark fitting
            var sw = Stopwatch.StartNew();
            double[,] embedding = model.Fit(data, embeddingDimension: 2, nNeighbors: 10);
            sw.Stop();
            long fitTime = sw.ElapsedMilliseconds;

            // Generate transform test data
            double[,] newData = GenerateTestData(100, nFeatures);

            // Benchmark transformation
            sw.Restart();
            double[,] newEmbedding = model.Transform(newData);
            sw.Stop();
            long transformTime = sw.ElapsedMilliseconds;

            // Memory after
            long memoryAfter = GC.GetTotalMemory(false);
            double memoryUsedMB = (memoryAfter - memoryBefore) / (1024.0 * 1024.0);

            // Output results
            Console.WriteLine($"{nSamples}\t\t{nFeatures}\t\t{fitTime}\t\t{transformTime}\t\t{memoryUsedMB:F1}");

            // Cleanup
            model.Dispose();
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        static double[,] GenerateTestData(int nSamples, int nFeatures)
        {
            double[,] data = new double[nSamples, nFeatures];
            Random rand = new Random(42);

            for (int i = 0; i < nSamples; i++)
            {
                for (int j = 0; j < nFeatures; j++)
                {
                    data[i, j] = rand.NextDouble() * 10.0 - 5.0;
                }
            }

            return data;
        }
    }
}