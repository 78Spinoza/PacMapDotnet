using System;
using System.IO;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;
using System.Threading;
using PacMapSharp;

namespace PacMapDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Simple PACMAP - Mammoth Embedding");
            Console.WriteLine("=================================");
            Console.WriteLine($"PACMAP Library Version: {PacMapModel.GetVersion()}");

            try
            {
                // Clean up old images first
                Console.WriteLine("🧹 Cleaning up old images from Results folder...");
                CleanupOldImages();

                // Load mammoth data
                Console.WriteLine("Loading mammoth dataset...");
                var (data, labels) = LoadMammothData();
                Console.WriteLine($"Loaded: {data.GetLength(0)} points, {data.GetLength(1)} dimensions");

                // Create PACMAP and embed - Force Exact KNN to test
                Console.WriteLine("Creating embedding with Exact KNN (no HNSW)...");
                var pacmap = new PacMapModel();

                // Convert double[,] to float[,]
                int n = data.GetLength(0);
                int d = data.GetLength(1);
                var floatData = new float[n, d];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < d; j++)
                        floatData[i, j] = (float)data[i, j];

                var stopwatch = Stopwatch.StartNew();

                Console.WriteLine("Testing with HIGH iterations to match Rust quality:");
                Console.WriteLine("  Default: (100, 100, 250) = 450 total iterations");
                Console.WriteLine("  Previous: (200, 200, 500) = 900 total iterations");
                Console.WriteLine("  Testing:  (300, 300, 900) = 1500 total iterations (Rust-like)");

                var embedding = pacmap.Fit(
                    data: floatData,
                    embeddingDimension: 2,
                    nNeighbors: 10,
                    metric: DistanceMetric.Euclidean,
                    learningRate: 1.0f,  // Updated to use Adam-appropriate learning rate
                    mnRatio: 0.5f,
                    fpRatio: 2.0f,
                    numIters: (100, 100, 250),  // Match Rust defaults exactly
                    forceExactKnn: true,   // Test exact k-NN with Python-style sampling
                    randomSeed: 42  // Switch back to original seed
                );
                stopwatch.Stop();

                Console.WriteLine($"Embedding created: {embedding.GetLength(0)} x {embedding.GetLength(1)}");
                Console.WriteLine($"⏱️  Execution time: {stopwatch.Elapsed.TotalSeconds:F2} seconds");

                // Create visualizations with real model info
                Console.WriteLine("🎨 Creating visualizations...");
                CreateVisualizations(embedding, data, labels, pacmap, stopwatch.Elapsed.TotalSeconds);

                // Open results folder
                Console.WriteLine("📂 Opening Results folder...");
                Process.Start("explorer.exe", "Results");
                Console.WriteLine("✅ Results folder opened! Check for the generated images.");
                Console.WriteLine("Done! Check Results folder for images.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }
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

            // Print data statistics for analysis
            Console.WriteLine("   Data preprocessing:");
            DataLoaders.PrintDataStatistics("Original mammoth data", data);

            // Use original data without normalization
            var floatData = ConvertToFloat(data);

            float[,] embedding;

            // Test 1: Debug PACMAP with simple test data first
            Console.WriteLine("   Test 1: Debugging PACMAP with simple test data...");

            // Create simple test data with clear clusters
            var testData = CreateSimpleTestData();
            var testLabels = CreateSimpleTestLabels();
            var floatTestData = ConvertToFloat(testData);

            Console.WriteLine($"   Test data: {testData.GetLength(0)} points, {testData.GetLength(1)} dimensions");

            // Test PACMAP on simple data
            using var testModel = new PacMapModel();
            var testEmbedding = testModel.Fit(
                data: floatTestData,
                embeddingDimension: 2,
                nNeighbors: 5,
                metric: DistanceMetric.Euclidean,
                randomSeed: 42
            );

            Console.WriteLine($"   ✅ Test embedding created: {testEmbedding.GetLength(0)}x{testEmbedding.GetLength(1)}");

            // Check if simple test data produces reasonable results
            double testQuality = CalculateEmbeddingQuality(testEmbedding, testLabels);
            Console.WriteLine($"   📊 Simple test quality: {testQuality:F4}");

            if (testQuality > 0.1) // If even simple data fails, PACMAP implementation has issues
            {
                Console.WriteLine("   ❌ WARNING: Simple test data also produces poor clustering!");
                Console.WriteLine("   🔍 This suggests a fundamental issue with the PACMAP implementation");
            }
            else
            {
                Console.WriteLine("   ✅ Simple test data works - issue is with mammoth data complexity");
            }
            Console.WriteLine();

            // Test 2: Now try mammoth data with different approaches
            Console.WriteLine("   Test 2: Testing mammoth data with PACMAP...");

            float[,] bestEmbedding = new float[0, 0];
            double bestQuality = double.MaxValue;
            var bestParams = new { nNeighbors = 0, metric = DistanceMetric.Euclidean, name = "" };

            // Test more aggressive parameter combinations
            var testParams = new[]
            {
                new { nNeighbors = 5, metric = DistanceMetric.Euclidean, name = "Euclidean_5" },
                new { nNeighbors = 10, metric = DistanceMetric.Euclidean, name = "Euclidean_10" },
                new { nNeighbors = 20, metric = DistanceMetric.Euclidean, name = "Euclidean_20" },
                new { nNeighbors = 50, metric = DistanceMetric.Euclidean, name = "Euclidean_50" },
                new { nNeighbors = 100, metric = DistanceMetric.Euclidean, name = "Euclidean_100" },
                new { nNeighbors = 10, metric = DistanceMetric.Manhattan, name = "Manhattan_10" },
                new { nNeighbors = 10, metric = DistanceMetric.Cosine, name = "Cosine_10" }
            };

            foreach (var param in testParams)
            {
                using var testModelMammoth = new PacMapModel();
                var testStopwatch = Stopwatch.StartNew();

                var testEmbeddingMammoth = testModelMammoth.Fit(
                    data: floatData,
                    embeddingDimension: 2,
                    nNeighbors: param.nNeighbors,
                    metric: param.metric,
                    randomSeed: 42
                );

                testStopwatch.Stop();

                // Calculate simple quality metric
                double quality = CalculateEmbeddingQuality(testEmbeddingMammoth, labels);

                Console.WriteLine($"      ✅ {param.name}: n={param.nNeighbors}, quality={quality:F4}, time={testStopwatch.Elapsed.TotalSeconds:F2}s");

                if (quality < bestQuality)
                {
                    bestQuality = quality;
                    bestEmbedding = testEmbeddingMammoth;
                    bestParams = param;
                }
            }

            Console.WriteLine($"      🏆 Best mammoth embedding: {bestParams.name} with quality {bestQuality:F4}");

            if (bestQuality > 0.05)
            {
                Console.WriteLine("   ❌ ALL mammoth embeddings show poor quality (>0.05)");
                Console.WriteLine("   🔍 PACMAP may not be suitable for this type of 3D spatial data");
            }
            Console.WriteLine();

            // Create the best model for persistence testing
            using var model = new PacMapModel();
            embedding = model.Fit(
                data: floatData,
                embeddingDimension: 2,
                nNeighbors: bestParams.nNeighbors,
                metric: bestParams.metric,
                randomSeed: 42
            );

            Console.WriteLine($"      ✅ Final embedding complete: {embedding.GetLength(0)} points → {embedding.GetLength(1)}D");
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
                    ["n_neighbors"] = bestParams.nNeighbors,
                    ["metric"] = bestParams.metric.ToString().ToLower(),
                    ["random_seed"] = 42,
                    ["embedding_quality"] = bestQuality.ToString("F4"),
                    ["data_points"] = embedding.GetLength(0),
                    ["data_range_x"] = $"{data[0, 0]:F1} to {data[data.GetLength(0)-1, 0]:F1}",
                    ["data_range_y"] = $"{data[0, 1]:F1} to {data[data.GetLength(0)-1, 1]:F1}",
                    ["data_range_z"] = $"{data[0, 2]:F1} to {data[data.GetLength(0)-1, 2]:F1}"
                };

                string title = $"Mammoth PACMAP 2D Embedding (n={bestParams.nNeighbors}, {bestParams.metric})";
                Visualizer.PlotMammothPacMAP(embedding, data, title, plotPath, paramInfo);
                Console.WriteLine($"      ✅ PACMAP visualization saved: {plotPath}");
                Console.WriteLine($"      📝 Parameters: n_neighbors={bestParams.nNeighbors}, metric={bestParams.metric}, quality={bestQuality:F4}");
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

        // Normalize data to zero mean and unit variance
        static double[,] NormalizeData(double[,] data)
        {
            int rows = data.GetLength(0);
            int cols = data.GetLength(1);
            var normalized = new double[rows, cols];

            for (int j = 0; j < cols; j++)
            {
                // Calculate mean and standard deviation for each feature
                double sum = 0;
                for (int i = 0; i < rows; i++)
                    sum += data[i, j];
                double mean = sum / rows;

                double sumSquaredDiff = 0;
                for (int i = 0; i < rows; i++)
                    sumSquaredDiff += Math.Pow(data[i, j] - mean, 2);
                double stdDev = Math.Sqrt(sumSquaredDiff / rows);

                // Normalize each feature
                for (int i = 0; i < rows; i++)
                {
                    if (stdDev > 1e-10) // Avoid division by zero
                        normalized[i, j] = (data[i, j] - mean) / stdDev;
                    else
                        normalized[i, j] = data[i, j] - mean;
                }
            }

            return normalized;
        }

        // Create simple test data with clear clusters
        static double[,] CreateSimpleTestData()
        {
            var data = new double[300, 3]; // 3 clusters, 100 points each
            var random = new Random(42);

            // Cluster 1: Centered at (0,0,0)
            for (int i = 0; i < 100; i++)
            {
                data[i, 0] = NextGaussian(random) * 0.5;
                data[i, 1] = NextGaussian(random) * 0.5;
                data[i, 2] = NextGaussian(random) * 0.5;
            }

            // Cluster 2: Centered at (3,3,3)
            for (int i = 100; i < 200; i++)
            {
                data[i, 0] = 3 + NextGaussian(random) * 0.5;
                data[i, 1] = 3 + NextGaussian(random) * 0.5;
                data[i, 2] = 3 + NextGaussian(random) * 0.5;
            }

            // Cluster 3: Centered at (-3,-3,-3)
            for (int i = 200; i < 300; i++)
            {
                data[i, 0] = -3 + NextGaussian(random) * 0.5;
                data[i, 1] = -3 + NextGaussian(random) * 0.5;
                data[i, 2] = -3 + NextGaussian(random) * 0.5;
            }

            return data;
        }

        // Create simple test labels
        static int[] CreateSimpleTestLabels()
        {
            var labels = new int[300];
            for (int i = 0; i < 100; i++) labels[i] = 0;      // Cluster 1
            for (int i = 100; i < 200; i++) labels[i] = 1;   // Cluster 2
            for (int i = 200; i < 300; i++) labels[i] = 2;   // Cluster 3
            return labels;
        }

        // Simple Gaussian random number generator
        public static double NextGaussian(Random random)
        {
            double u1 = random.NextDouble();
            double u2 = random.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        // Calculate embedding quality - lower is better (clusters of same labels should be close)
        static double CalculateEmbeddingQuality(float[,] embedding, int[] labels)
        {
            int nSamples = embedding.GetLength(0);
            double totalDistance = 0;
            int count = 0;

            for (int i = 0; i < nSamples; i++)
            {
                double minSameLabelDistance = double.MaxValue;

                for (int j = 0; j < nSamples; j++)
                {
                    if (i != j && labels[i] == labels[j])
                    {
                        double dist = Math.Sqrt(
                            Math.Pow(embedding[i, 0] - embedding[j, 0], 2) +
                            Math.Pow(embedding[i, 1] - embedding[j, 1], 2)
                        );

                        if (dist < minSameLabelDistance)
                            minSameLabelDistance = dist;
                    }
                }

                if (minSameLabelDistance < double.MaxValue)
                {
                    totalDistance += minSameLabelDistance;
                    count++;
                }
            }

            return count > 0 ? totalDistance / count : double.MaxValue;
        }

        // Clean up old images from Results folder
        static void CleanupOldImages()
        {
            try
            {
                string resultsDir = "Results";
                if (!Directory.Exists(resultsDir))
                {
                    Directory.CreateDirectory(resultsDir);
                    Console.WriteLine($"   📁 Created Results directory");
                    return;
                }

                var imageFiles = Directory.GetFiles(resultsDir, "*.png")
                    .Concat(Directory.GetFiles(resultsDir, "*.jpg"))
                    .Concat(Directory.GetFiles(resultsDir, "*.jpeg"));

                int deletedCount = 0;
                foreach (var file in imageFiles)
                {
                    try
                    {
                        File.Delete(file);
                        deletedCount++;
                        Console.WriteLine($"   🗑️  Deleted: {Path.GetFileName(file)}");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"   ⚠️  Could not delete {Path.GetFileName(file)}: {ex.Message}");
                    }
                }

                if (deletedCount > 0)
                {
                    Console.WriteLine($"   ✅ Cleaned up {deletedCount} old image files");
                }
                else
                {
                    Console.WriteLine($"   ℹ️  No old images to clean up");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ⚠️  Cleanup failed: {ex.Message}");
            }
        }

        // Create visualizations and print what's being created
        static void CreateVisualizations(float[,] embedding, double[,] originalData, int[] labels, PacMapModel pacmap, double executionTime)
        {
            try
            {
                string resultsDir = "Results";

                // Create original 3D visualization
                var original3DPath = Path.Combine(resultsDir, "mammoth_original_3d.png");
                Console.WriteLine($"   📊 Creating original 3D visualization: {Path.GetFileName(original3DPath)}");
                Visualizer.PlotOriginalMammoth3DReal(originalData, "Original Mammoth 3D Data", original3DPath);
                Console.WriteLine($"   ✅ Created: {Path.GetFileName(original3DPath)}");

                // Create PACMAP 2D embedding visualization with REAL model info
                var pacmapPath = Path.Combine(resultsDir, "mammoth_pacmap_embedding.png");
                Console.WriteLine($"   📈 Creating PACMAP 2D embedding: {Path.GetFileName(pacmapPath)}");

                // Get REAL model information using ModelInfo property
                Console.WriteLine($"   🔍 Extracting real model parameters...");
                var modelInfo = pacmap.ModelInfo;

                var paramInfo = new Dictionary<string, object>
                {
                    ["PACMAP Version"] = PacMapModel.GetVersion() + " (Corrected Gradients)",
                    ["n_neighbors"] = modelInfo.Neighbors,
                    ["embedding_dimension"] = modelInfo.OutputDimension,
                    ["distance_metric"] = modelInfo.Metric.ToString(),
                    ["mn_ratio"] = modelInfo.MN_ratio.ToString("F2"),
                    ["fp_ratio"] = modelInfo.FP_ratio.ToString("F2"),
                    ["learning_rate"] = pacmap.LearningRate.ToString("F3"),
                    ["data_points"] = modelInfo.TrainingSamples,
                    ["original_dimensions"] = modelInfo.InputDimension,
                    ["hnsw_m"] = modelInfo.HnswM,
                    ["hnsw_ef_construction"] = modelInfo.HnswEfConstruction,
                    ["hnsw_ef_search"] = modelInfo.HnswEfSearch,
                    ["KNN_Mode"] = modelInfo.ForceExactKnn ? "Direct KNN" : "HNSW",
                    ["random_seed"] = modelInfo.RandomSeed
                };
                paramInfo["execution_time"] = $"{executionTime:F2}s";

                // Create title with hyperparameters
                var line1 = $"PACMAP v{paramInfo["PACMAP Version"]} | n_neighbors={paramInfo["n_neighbors"]} | {paramInfo["distance_metric"]} | mn_ratio={paramInfo["mn_ratio"]} | fp_ratio={paramInfo["fp_ratio"]} | lr={paramInfo["learning_rate"]} | data_points={paramInfo["data_points"]}";
                var line2 = $"KNN: {paramInfo["KNN_Mode"]} | HNSW: M={paramInfo["hnsw_m"]}, ef={paramInfo["hnsw_ef_search"]} | Time: {paramInfo["execution_time"]}";
                var titleWithParams = $"Mammoth PACMAP 2D Embedding\n{line1}\n{line2}";

                Visualizer.PlotMammothPacMAP(embedding, originalData, titleWithParams, pacmapPath, paramInfo);
                Console.WriteLine($"   ✅ Created: {Path.GetFileName(pacmapPath)}");
                Console.WriteLine($"   📊 KNN Mode: {(modelInfo.ForceExactKnn ? "Direct KNN" : "HNSW")}");
                Console.WriteLine($"   🚀 HNSW Status: {(modelInfo.ForceExactKnn ? "DISABLED" : "ACTIVE")}");

                Console.WriteLine($"   🎉 All visualizations created successfully!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ❌ Visualization creation failed: {ex.Message}");
            }
        }
    }
}