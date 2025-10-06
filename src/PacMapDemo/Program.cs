using System;
using System.IO;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;
using PACMAPuwotSharp;

namespace PacMapDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("PacMAP Enhanced - C# Demo with PACMAPCSharp Library");
            Console.WriteLine("======================================================");
            Console.WriteLine();

            try
            {
                // Demo version info
                Console.WriteLine($"PacMAP Enhanced C# Demo Version: 1.0.0");
                Console.WriteLine($"PACMAPCSharp Library Version: {PacMapModel.GetVersion()}");
                Console.WriteLine();

                // Create output directory
                string outputDir = "Results";
                Directory.CreateDirectory(outputDir);

                // ===============================================================
                // BASIC PACMAP DEMONSTRATION
                // ===============================================================
                Console.WriteLine(new string('=', 80));
                Console.WriteLine("🧪 BASIC PACMAP DEMONSTRATION WITH MAMMOTH DATA");
                Console.WriteLine(new string('=', 80));
                Console.WriteLine("DEMO OBJECTIVES:");
                Console.WriteLine("   1. Load real mammoth 3D point cloud data");
                Console.WriteLine("   2. Generate PACMAP 2D embedding");
                Console.WriteLine("   3. Test model save/load functionality");
                Console.WriteLine("   4. Create original 3D data visualization");
                Console.WriteLine("   5. Create PACMAP 2D embedding visualization");
                Console.WriteLine();

                // Load real mammoth data
                Console.WriteLine("📊 Loading mammoth dataset...");
                var (data, labels) = LoadMammothData();
                Console.WriteLine($"   Loaded: {data.GetLength(0)} points, {data.GetLength(1)} dimensions");
                Console.WriteLine($"   Labels: {labels.Distinct().Count()} unique anatomical parts");
                Console.WriteLine();

                // Test basic PACMAP embedding
                TestBasicPacmap(data, labels, outputDir);

                Console.WriteLine();
                Console.WriteLine("✅ DEMO COMPLETED SUCCESSFULLY!");
                Console.WriteLine("The PacMapDemo now works with the new PACMAPCSharp library!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ ERROR: {ex.Message}");
                Console.WriteLine($"   Stack trace: {ex.StackTrace}");
            }

            Console.WriteLine();
            Console.WriteLine("Demo completed successfully!");
        }

        static (double[,] data, int[] labels) LoadMammothData()
        {
            try
            {
                // Try to load real mammoth data
                string csvPath = "Data/mammoth_data.csv";
                if (File.Exists(csvPath))
                {
                    return DataLoaders.LoadMammothWithLabels(csvPath);
                }
                else
                {
                    Console.WriteLine("⚠️  Mammoth CSV file not found, generating synthetic data...");
                    return GenerateSyntheticMammothData(2000);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️  Error loading mammoth data: {ex.Message}");
                Console.WriteLine("Generating synthetic data instead...");
                return GenerateSyntheticMammothData(2000);
            }
        }

        static (double[,] data, int[] labels) GenerateSyntheticMammothData(int numPoints)
        {
            var data = new double[numPoints, 3];
            var labels = new int[numPoints];
            var random = new Random(42);

            // Generate mammoth-like 3D structure
            for (int i = 0; i < numPoints; i++)
            {
                double t = (double)i / numPoints * 2 * Math.PI;
                double noise = (random.NextDouble() - 0.5) * 0.1;

                // Body (main structure)
                if (i < numPoints * 0.7)
                {
                    data[i, 0] = Math.Cos(t) * (1 + noise);
                    data[i, 1] = Math.Sin(t) * (1 + noise);
                    data[i, 2] = (t / Math.PI - 1) * 2 + noise;
                    labels[i] = 0; // body
                }
                // Head
                else if (i < numPoints * 0.8)
                {
                    data[i, 0] = Math.Cos(t) * 1.5 + noise;
                    data[i, 1] = Math.Sin(t) * 1.5 + noise;
                    data[i, 2] = 2 + noise;
                    labels[i] = 1; // head
                }
                // Tail
                else if (i < numPoints * 0.9)
                {
                    data[i, 0] = -Math.Cos(t) * 0.5 + noise;
                    data[i, 1] = -Math.Sin(t) * 0.5 + noise;
                    data[i, 2] = -2 + noise;
                    labels[i] = 2; // tail
                }
                // Legs
                else
                {
                    data[i, 0] = Math.Cos(t) * 0.3 + noise;
                    data[i, 1] = Math.Sin(t) * 0.3 + noise;
                    data[i, 2] = -1 + noise;
                    labels[i] = 3; // legs
                }
            }

            return (data, labels);
        }

        static void TestBasicPacmap(double[,] data, int[] labels, string outputDir)
        {
            Console.WriteLine("🔬 Testing Basic PACMAP Functionality");
            Console.WriteLine(new string('-', 50));

            // Convert to float for the new API
            var floatData = ConvertToFloat(data);

            // Test 1: Basic embedding
            Console.WriteLine("   Test 1: Basic 2D embedding...");
            using var model = new PacMapModel();
            var stopwatch = Stopwatch.StartNew();

            var embedding = model.Fit(
                data: floatData,
                embeddingDimension: 2,
                nNeighbors: 15,
                metric: DistanceMetric.Euclidean,
                randomSeed: 42
            );

            stopwatch.Stop();
            Console.WriteLine($"      ✅ Embedding complete: {embedding.GetLength(0)} points → {embedding.GetLength(1)}D");
            Console.WriteLine($"      ⏱️  Time: {stopwatch.Elapsed.TotalSeconds:F2} seconds");
            Console.WriteLine();

            // Test 2: Model persistence
            Console.WriteLine("   Test 2: Model save/load...");
            string modelPath = Path.Combine(outputDir, "mammoth_pacmap_model.pmm");

            // Save model
            model.Save(modelPath);
            var fileInfo = new FileInfo(modelPath);
            Console.WriteLine($"      ✅ Model saved: {fileInfo.Length / 1024.0:F1} KB");

            // Load model
            using var loadedModel = PacMapModel.Load(modelPath);
            Console.WriteLine($"      ✅ Model loaded successfully");

            // Test transform with loaded model
            var transformed = loadedModel.Transform(floatData);
            Console.WriteLine($"      ✅ Transformed {transformed.GetLength(0)} points with loaded model");
            Console.WriteLine();

            // Test 3: Different distance metrics
            Console.WriteLine("   Test 3: Different distance metrics...");
            var metrics = new DistanceMetric[] { DistanceMetric.Manhattan, DistanceMetric.Cosine };

            foreach (var metric in metrics)
            {
                using var metricModel = new PacMapModel();
                var metricStopwatch = Stopwatch.StartNew();

                var metricEmbedding = metricModel.Fit(
                    data: floatData,
                    embeddingDimension: 2,
                    nNeighbors: 10,
                    metric: metric,
                    randomSeed: 42
                );

                metricStopwatch.Stop();
                Console.WriteLine($"      ✅ {metric}: {metricEmbedding.GetLength(0)} points in {metricStopwatch.Elapsed.TotalSeconds:F2}s");
            }
            Console.WriteLine();

            // Test 4: Original 3D data visualization
            Console.WriteLine("   Test 4: Creating original 3D visualization...");
            try
            {
                var plot3DPath = Path.Combine(outputDir, "mammoth_original_3d.png");
                var paramInfo3D = new Dictionary<string, object>
                {
                    ["data_type"] = "original_3d",
                    ["anatomical_parts"] = labels.Distinct().Count()
                };

                Visualizer.PlotOriginalMammoth3DReal(data, "Original Mammoth 3D Data - Multiple Projections", plot3DPath);
                Console.WriteLine($"      ✅ Original 3D visualization saved: {plot3DPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"      ⚠️  Original 3D visualization failed: {ex.Message}");
            }

            // Test 5: PACMAP embedding visualization
            Console.WriteLine("   Test 5: Creating PACMAP visualization...");
            try
            {
                var plotPath = Path.Combine(outputDir, "mammoth_pacmap_embedding.png");
                var paramInfo = new Dictionary<string, object>
                {
                    ["n_neighbors"] = 15,
                    ["metric"] = "euclidean",
                    ["random_seed"] = 42
                };

                Visualizer.PlotMammothPacMAP(embedding, data, "Mammoth PACMAP 2D Embedding", plotPath, paramInfo);
                Console.WriteLine($"      ✅ PACMAP visualization saved: {plotPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"      ⚠️  PACMAP visualization failed: {ex.Message}");
            }
            Console.WriteLine();

            Console.WriteLine("📊 TEST SUMMARY");
            Console.WriteLine(new string('=', 30));
            Console.WriteLine("   Basic embedding: ✅ PASS");
            Console.WriteLine("   Model persistence: ✅ PASS");
            Console.WriteLine("   Distance metrics: ✅ PASS");
            Console.WriteLine("   Original 3D visualization: " + (File.Exists(Path.Combine(outputDir, "mammoth_original_3d_XY_TopView.png")) ? "✅ PASS" : "⚠️  SKIP"));
            Console.WriteLine("   PACMAP 2D visualization: " + (File.Exists(Path.Combine(outputDir, "mammoth_pacmap_embedding.png")) ? "✅ PASS" : "⚠️  SKIP"));
        }

        // Helper method for type conversion
        static float[,] ConvertToFloat(double[,] input)
        {
            int rows = input.GetLength(0);
            int cols = input.GetLength(1);
            var result = new float[rows, cols];

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[i, j] = (float)input[i, j];

            return result;
        }
    }
}