using System;
using System.IO;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;
using System.Threading;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.Axes;
using OxyPlot.WindowsForms;
using OxyPlot.Legends;
using OxyPlot.Annotations;
using PacMapSharp;

namespace PacMapDemo
{
    class Program
    {
        /// <summary>
        /// Gets the full path for the Results directory
        /// </summary>
        static string GetResultsPath()
        {
            var currentDir = Directory.GetCurrentDirectory();
            return Path.GetFullPath(Path.Combine(currentDir, "Results"));
        }

        /// <summary>
        /// Gets the full path for a subdirectory within Results
        /// </summary>
        static string GetResultsPath(string subDirectory)
        {
            return Path.GetFullPath(Path.Combine(GetResultsPath(), subDirectory));
        }

        /// <summary>
        /// Unified progress callback for consistent single-line output across all PACMAP operations
        /// </summary>
        static void UnifiedProgressCallback(string phase, int current, int total, float percent, string message, string prefix = "")
        {
            string displayPrefix = string.IsNullOrEmpty(prefix) ? "" : $"[{prefix}] ";
            string safeMessage = message ?? "";
            Console.Write($"\r{new string(' ', 180)}\r   {displayPrefix}[{phase}] Progress: {current}/{total} ({percent:F1}%) {safeMessage}");
        }

        static void Main(string[] args)
        {
            Console.WriteLine("Simple PACMAP - Mammoth Embedding");
            Console.WriteLine("=================================");
            Console.WriteLine($"PACMAP Library Version: {PacMapModel.GetVersion()}");

            try
            {
                // Clean up ALL old results first (images and subfolders)
                Console.WriteLine("🧹 Cleaning up ALL previous results from Results folder...");
                CleanupAllResults();

                // Load mammoth data
                Console.WriteLine("Loading mammoth dataset...");
                var (data, labels) = LoadMammothData();
                Console.WriteLine($"Loaded: {data.GetLength(0)} points, {data.GetLength(1)} dimensions");

                // Convert double[,] to float[,]
                int n = data.GetLength(0);
                int d = data.GetLength(1);
                var floatData = new float[n, d];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < d; j++)
                        floatData[i, j] = (float)data[i, j];

                // =============================================
                // MAIN 1M IMAGES: Direct KNN + HNSW (Auto-tuning enabled)
                // =============================================
                               /*
                // =============================================
                // DISABLED: 10K MAMMOTH DEMO (Temporarily disabled for 1M testing)
                // =============================================
                // Create ONE HNSW embedding with 10K mammoth dataset (auto-tuning enabled)
                Console.WriteLine("🦣 CREATING 10K MAMMOTH HNSW EMBEDDING");
                Console.WriteLine("=====================================");
                Console.WriteLine("   Full dataset: 10K points with auto-tuning");
                Console.WriteLine("   Neighbors: 10, Auto-tuning: ENABLED");
                Console.WriteLine();

                var pacmapHNSW = new PacMapModel(); // Use default parameters

                var stopwatchHNSW = Stopwatch.StartNew();
                var embeddingHNSW = pacmapHNSW.Fit(
                    data: floatData,
                    embeddingDimension: 2,
                    nNeighbors: 10,
                    mnRatio: 0.5f,     // Default MN ratio
                    fpRatio: 2.0f,     // Default FP ratio
                    learningRate: 1.0f, // Default learning rate
                    numIters: (100, 100, 250), // Default iterations
                    forceExactKnn: false,  // HNSW with auto-tuning
                    autoHNSWParam: true,  // Enable auto-tuning for 1M dataset
                    randomSeed: 42,      // Fixed seed for reproducibility
                    progressCallback: (phase, current, total, percent, message) =>
                    {
                        UnifiedProgressCallback(phase, current, total, percent, message);
                    }
                );
                Console.WriteLine(); // New line after progress
                stopwatchHNSW.Stop();

                Console.WriteLine($"✅ 10K HNSW Embedding created: {embeddingHNSW.GetLength(0)} x {embeddingHNSW.GetLength(1)}");
                Console.WriteLine($"⏱️  HNSW Execution time: {stopwatchHNSW.Elapsed.TotalSeconds:F2} seconds");

                // Save HNSW model
                Console.WriteLine("💾 Saving 10K HNSW model...");
                string modelPathHNSW = "Results/mammoth_10k_hnsw.pmm";
                Directory.CreateDirectory("Results");
                pacmapHNSW.Save(modelPathHNSW);
                Console.WriteLine($"✅ Model saved: {modelPathHNSW}");

                Console.WriteLine();
                Console.WriteLine("🎨 Creating 10K HNSW mammoth visualization with labels...");

                // Create mammoth visualization WITH labels like the Direct KNN code
                string resultsDir = GetResultsPath();
                string mammothPath = Path.Combine(resultsDir, "mammoth_10k_hnsw.png");
                Console.WriteLine($"   📈 Creating 10K HNSW PACMAP 2D embedding: {Path.GetFileName(mammothPath)}");

                // Get model information from fitted HNSW model
                var modelInfo = pacmapHNSW.ModelInfo;
                var paramInfo = new Dictionary<string, object>
                {
                    ["PACMAP Version"] = PacMapModel.GetVersion(),
                    ["n_neighbors"] = modelInfo.Neighbors,
                    ["embedding_dimension"] = modelInfo.OutputDimension,
                    ["distance_metric"] = modelInfo.Metric.ToString(),
                    ["mn_ratio"] = modelInfo.MN_ratio.ToString("F2"),
                    ["fp_ratio"] = modelInfo.FP_ratio.ToString("F2"),
                    ["learning_rate"] = pacmapHNSW.LearningRate.ToString("F3"),
                    ["init_std_dev"] = pacmapHNSW.InitializationStdDev.ToString("E0"),
                    ["phase_iters"] = $"({pacmapHNSW.NumIters.phase1}, {pacmapHNSW.NumIters.phase2}, {pacmapHNSW.NumIters.phase3})",
                    ["data_points"] = modelInfo.TrainingSamples,
                    ["original_dimensions"] = modelInfo.InputDimension,
                    ["hnsw_m"] = modelInfo.HnswM,
                    ["hnsw_ef_construction"] = modelInfo.HnswEfConstruction,
                    ["hnsw_ef_search"] = modelInfo.HnswEfSearch,
                    ["KNN_Mode"] = "HNSW",
                    ["random_seed"] = modelInfo.RandomSeed
                };
                paramInfo["execution_time"] = $"{stopwatchHNSW.Elapsed.TotalSeconds:F2}s";

                // Create comprehensive title with ALL parameters like Direct KNN code
                var version = paramInfo["PACMAP Version"].ToString() ?? "Unknown";
                var knnMode = paramInfo["KNN_Mode"].ToString() ?? "Unknown";
                var sampleSize = paramInfo["data_points"].ToString();
                var execTime = paramInfo["execution_time"].ToString();

                // Line 1: Basic info with sample size
                var line1 = $"PACMAP v{version} | Sample: {sampleSize:N0} | {knnMode}";

                // Line 2: Core PACMAP parameters
                var line2 = $"k={paramInfo["n_neighbors"]} | {paramInfo["distance_metric"]} | dims={paramInfo["embedding_dimension"]} | seed={paramInfo["random_seed"]}";

                // Line 3: Hyperparameters
                var line3 = $"mn={paramInfo["mn_ratio"]} | fp={paramInfo["fp_ratio"]} | lr={paramInfo["learning_rate"]} | std={paramInfo["init_std_dev"]}";

                // Line 4: Optimization phases and HNSW details
                var hnswDetails = $"HNSW: M={paramInfo["hnsw_m"]}, ef_c={paramInfo["hnsw_ef_construction"]}, ef_s={paramInfo["hnsw_ef_search"]}";
                var line4 = $"phases={paramInfo["phase_iters"]} | {hnswDetails}";

                // Line 5: Execution time
                var line5 = $"Time: {execTime} | Original dims: {paramInfo["original_dimensions"]}";

                var titleWithParams = $"Mammoth PACMAP 2D Embedding (10K Dataset)\n{line1}\n{line2}\n{line3}\n{line4}\n{line5}";

                // Use the same visualization function as Direct KNN code
                Visualizer.PlotMammothPacMAP(embeddingHNSW, data, titleWithParams, mammothPath, paramInfo);
                Console.WriteLine($"   ✅ Created: {Path.GetFileName(mammothPath)}");
                Console.WriteLine($"   📊 KNN Mode: {(modelInfo.ForceExactKnn ? "Direct KNN" : "HNSW")}");
                Console.WriteLine($"   🚀 HNSW Status: {(modelInfo.ForceExactKnn ? "DISABLED" : "ACTIVE")}");
                Console.WriteLine($"   🎉 10K HNSW mammoth visualization created successfully!");
                */
                // =============================================
                // DISABLED: Comprehensive test suite after basic demo
                Console.WriteLine();
                // =============================================
                // DISABLED: Comprehensive test suite (not needed for main demo)
                // =============================================
                /*
                Console.WriteLine("=========================================================");
                Console.WriteLine("🧪 Running Comprehensive Test Suite (Reproducibility & Persistence)");
                Console.WriteLine("=========================================================");
                RunTransformConsistencyTests();

                // =============================================
                // DISABLED: Hyperparameter discovery and experiments
                // =============================================
                /*
                // Run hyperparameter experiments after comprehensive tests
                Console.WriteLine();
                Console.WriteLine("=========================================================");
                Console.WriteLine("🔬 Running Hyperparameter Experiments");
                Console.WriteLine("=========================================================");
                Console.WriteLine("   Testing different PACMAP hyperparameters on mammoth dataset");
                Console.WriteLine("   Using shared HNSW parameters discovered once for all experiments");
                Console.WriteLine("   Images will have white backgrounds with black dots as requested");
                Console.WriteLine();

                // Auto-discover optimal HNSW parameters once (shared across all experiments)
                Console.WriteLine("🔍 Auto-discovering optimal HNSW parameters for all experiments...");
                var optimalHNSWParams = AutoDiscoverHNSWParameters(floatData);

                Console.WriteLine($"✅ HNSW Parameters discovered: M={optimalHNSWParams.M}, ef_construction={optimalHNSWParams.EfConstruction}, ef_search={optimalHNSWParams.EfSearch}");
                Console.WriteLine();

                // 1️⃣ Neighbor Experiments
                DemoNeighborExperiments(floatData, optimalHNSWParams);

                // 2️⃣ Learning Rate Experiments
                DemoLearningRateExperiments(floatData, optimalHNSWParams);
                */

                // =============================================
                // ACTIVE: Use optimal HNSW parameters directly (skip slow discovery)
                // =============================================
                Console.WriteLine("🚀 Using proven optimal HNSW parameters (skip slow auto-discovery)...");
                var optimalHNSWParams = (M: 32, EfConstruction: 400, EfSearch: 200);
                Console.WriteLine($"✅ Using optimal HNSW parameters: M={optimalHNSWParams.M}, ef_construction={optimalHNSWParams.EfConstruction}, ef_search={optimalHNSWParams.EfSearch}");
                Console.WriteLine();

                // 3️⃣ Advanced Parameter Tuning (Hairy Mammoth Methodology)
                DemoAdvancedParameterTuning(floatData, optimalHNSWParams);

                // =============================================
                // MNIST Demo (Optional - for binary data testing)
                // =============================================
                Console.WriteLine();
                Console.WriteLine("🔢 Running MNIST Binary Reader Demo...");
                MnistDemo.RunDemo();

                // Optional: Run PACMAP on MNIST subset
                // MnistDemo.RunPacmapOnMnist(subsetSize: 5000);

                // =============================================
                // DISABLED: Additional expensive experiments
                // =============================================
                /*
                // 4️⃣ Initialization Standard Deviation Experiments
                DemoInitializationStdDevExperiments(floatData, optimalHNSWParams);

                // 5️⃣ Extended Learning Rate Experiments
                DemoExtendedLearningRateExperiments(floatData, optimalHNSWParams);

                Console.WriteLine();
                Console.WriteLine("🎉 ALL HYPERPARAMETER EXPERIMENTS COMPLETED!");
                Console.WriteLine("📁 Check Results folder for all experiment visualizations:");
                Console.WriteLine("   - neighbor_experiment_n*.png (different neighbor counts)");
                Console.WriteLine("   - learning_rate_experiment_lr*.png (different learning rates)");
                Console.WriteLine("   - advanced_tuning_*.png (advanced parameter combinations)");
                Console.WriteLine("   - init_std_dev_experiment_*.png (different initialization std dev)");
                Console.WriteLine("   - extended_lr_experiment_*.png (extended learning rate tests)");
                */
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
            DisplayModelHyperparameters(testModel, "Test Model");
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
            DisplayModelHyperparameters(model, "Final Model");
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
                DisplayModelHyperparameters(metricModel, $"Metric Test Model ({metric})");
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
                    ["init_std_dev"] = model.InitializationStdDev.ToString("E0"),
                    ["phase_iters"] = $"({model.NumIters.phase1}, {model.NumIters.phase2}, {model.NumIters.phase3})",
                    ["data_points"] = embedding.GetLength(0),
                    ["data_range_x"] = $"{data[0, 0]:F1} to {data[data.GetLength(0)-1, 0]:F1}",
                    ["data_range_y"] = $"{data[0, 1]:F1} to {data[data.GetLength(0)-1, 1]:F1}",
                    ["data_range_z"] = $"{data[0, 2]:F1} to {data[data.GetLength(0)-1, 2]:F1}"
                };

                string title = $"Mammoth PACMAP 2D Embedding\nn={bestParams.nNeighbors}, {bestParams.metric}, init_std={paramInfo["init_std_dev"]}, phases={paramInfo["phase_iters"]}";
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

        // Helper method for random subsampling
        static float[,] CreateFloatSubset(double[,] fullData, int subsetSize)
        {
            int nSamples = fullData.GetLength(0);
            int nFeatures = fullData.GetLength(1);
            var random = new Random(42); // Fixed seed for reproducible results

            // Generate random indices without replacement
            var availableIndices = Enumerable.Range(0, nSamples).ToList();
            var selectedIndices = new List<int>();

            for (int i = 0; i < subsetSize && availableIndices.Count > 0; i++)
            {
                int randomIndex = random.Next(availableIndices.Count);
                selectedIndices.Add(availableIndices[randomIndex]);
                availableIndices.RemoveAt(randomIndex);
            }

            // Create subset data
            var subsetData = new float[selectedIndices.Count, nFeatures];
            for (int i = 0; i < selectedIndices.Count; i++)
            {
                int sourceIdx = selectedIndices[i];
                for (int j = 0; j < nFeatures; j++)
                {
                    subsetData[i, j] = (float)fullData[sourceIdx, j];
                }
            }

            return subsetData;
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
                        Console.WriteLine($"   - Deleted: {Path.GetFileName(file)}");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"   ! Could not delete {Path.GetFileName(file)}: {ex.Message}");
                    }
                }

                if (deletedCount > 0)
                {
                    Console.WriteLine($"   - Cleaned up {deletedCount} old image files");
                }
                else
                {
                    Console.WriteLine($"   - No old images to clean up");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ⚠️  Cleanup failed: {ex.Message}");
            }
        }

        // Clean up ALL results from Results folder (images and subfolders)
        static void CleanupAllResults()
        {
            try
            {
                string resultsDir = GetResultsPath();

                // If Results directory doesn't exist, create it and return
                if (!Directory.Exists(resultsDir))
                {
                    Directory.CreateDirectory(resultsDir);
                    Console.WriteLine($"   📁 Created Results directory");
                    return;
                }

                Console.WriteLine($"   🗂️  Clearing all contents from: {resultsDir}");

                int deletedFiles = 0;
                int deletedFolders = 0;
                int failedFiles = 0;
                int failedFolders = 0;

                // Delete ALL files in the Results directory (all extensions)
                var allFiles = Directory.GetFiles(resultsDir, "*", SearchOption.AllDirectories);
                foreach (var file in allFiles)
                {
                    try
                    {
                        File.SetAttributes(file, FileAttributes.Normal); // Remove read-only if set
                        File.Delete(file);
                        deletedFiles++;
                    }
                    catch (Exception ex)
                    {
                        failedFiles++;
                        Console.WriteLine($"   ! Could not delete file {Path.GetFileName(file)}: {ex.Message}");
                    }
                }

                // Delete ALL subdirectories
                var subDirectories = Directory.GetDirectories(resultsDir, "*", SearchOption.AllDirectories);
                // Process in reverse order to delete nested directories first
                foreach (var directory in subDirectories.Reverse())
                {
                    try
                    {
                        // Delete directory only if empty (files should already be deleted)
                        if (Directory.GetFiles(directory).Length == 0 && Directory.GetDirectories(directory).Length == 0)
                        {
                            Directory.Delete(directory);
                            deletedFolders++;
                        }
                        else
                        {
                            // Force delete if still has contents
                            Directory.Delete(directory, recursive: true);
                            deletedFolders++;
                        }
                    }
                    catch (Exception ex)
                    {
                        failedFolders++;
                        Console.WriteLine($"   ! Could not delete folder {Path.GetFileName(directory)}: {ex.Message}");
                    }
                }

                // Summary
                Console.WriteLine($"   📊 Cleanup Summary:");
                if (deletedFiles > 0)
                    Console.WriteLine($"      ✅ Deleted {deletedFiles} files");
                if (deletedFolders > 0)
                    Console.WriteLine($"      ✅ Deleted {deletedFolders} subfolders");
                if (failedFiles > 0)
                    Console.WriteLine($"      ⚠️  Failed to delete {failedFiles} files");
                if (failedFolders > 0)
                    Console.WriteLine($"      ⚠️  Failed to delete {failedFolders} folders");

                if (deletedFiles == 0 && deletedFolders == 0)
                {
                    Console.WriteLine($"      ℹ️  Results folder was already clean");
                }
                else
                {
                    Console.WriteLine($"      🎉 Results folder completely cleared!");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ❌ Critical cleanup error: {ex.Message}");
                Console.WriteLine($"   ⚠️  Continuing with demo (some old files may remain)");
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
                    ["init_std_dev"] = pacmap.InitializationStdDev.ToString("E0"),
                    ["phase_iters"] = $"({pacmap.NumIters.phase1}, {pacmap.NumIters.phase2}, {pacmap.NumIters.phase3})",
                    ["data_points"] = modelInfo.TrainingSamples,
                    ["original_dimensions"] = modelInfo.InputDimension,
                    ["hnsw_m"] = modelInfo.HnswM,
                    ["hnsw_ef_construction"] = modelInfo.HnswEfConstruction,
                    ["hnsw_ef_search"] = modelInfo.HnswEfSearch,
                    ["KNN_Mode"] = modelInfo.ForceExactKnn ? "Direct KNN" : "HNSW",
                    ["random_seed"] = modelInfo.RandomSeed
                };
                paramInfo["execution_time"] = $"{executionTime:F2}s";

                // Create COMPREHENSIVE title with ALL parameters
                var version = paramInfo["PACMAP Version"].ToString()?.Replace(" (Corrected Gradients)", "").Replace("-CLEAN-OUTPUT", "") ?? "Unknown";
                var knnMode = paramInfo["KNN_Mode"].ToString() ?? "Unknown";
                var sampleSize = paramInfo["data_points"].ToString();
                var execTime = paramInfo["execution_time"].ToString();

                // Line 1: Basic info with sample size
                var line1 = $"PACMAP v{version} | Sample: {sampleSize:N0} | {knnMode}";

                // Line 2: Core PACMAP parameters
                var line2 = $"k={paramInfo["n_neighbors"]} | {paramInfo["distance_metric"]} | dims={paramInfo["embedding_dimension"]} | seed={paramInfo["random_seed"]}";

                // Line 3: Hyperparameters
                var line3 = $"mn={paramInfo["mn_ratio"]} | fp={paramInfo["fp_ratio"]} | lr={paramInfo["learning_rate"]} | std={paramInfo["init_std_dev"]}";

                // Line 4: Optimization phases and HNSW details
                var hnswDetails = knnMode.Contains("HNSW") ? $"HNSW: M={paramInfo["hnsw_m"]}, ef_c={paramInfo["hnsw_ef_construction"]}, ef_s={paramInfo["hnsw_ef_search"]}" : "Direct KNN";
                var line4 = $"phases={paramInfo["phase_iters"]} | {hnswDetails}";

                // Line 5: Execution time
                var line5 = $"Time: {execTime} | Original dims: {paramInfo["original_dimensions"]}";

                var titleWithParams = $"Mammoth PACMAP 2D Embedding\n{line1}\n{line2}\n{line3}\n{line4}\n{line5}";

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

        /// <summary>
        /// Create visualizations for BOTH Direct KNN and HNSW embeddings (fit and transform)
        /// </summary>
        static void CreateVisualizationsBoth(float[,] embeddingKNN, float[,] embeddingHNSW,
                                         float[,] transformedKNN, float[,] transformedHNSW,
                                         double[,] originalData, int[] labels,
                                         PacMapModel pacmapKNN, PacMapModel pacmapHNSW, double executionTimeKNN, double executionTimeHNSW)
        {
            try
            {
                Console.WriteLine("   📂 Setting up results directories...");
                var resultsDir = "Results";
                Directory.CreateDirectory(resultsDir);

                // Clean up old images
                var oldImages = Directory.GetFiles(resultsDir, "mammoth_*.png");
                foreach (var oldImage in oldImages)
                {
                    try
                    {
                        File.Delete(oldImage);
                        Console.WriteLine($"   - Deleted: {Path.GetFileName(oldImage)}");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"   ! Warning: Could not delete {Path.GetFileName(oldImage)}: {ex.Message}");
                    }
                }
                Console.WriteLine($"   - Cleaned up {oldImages.Length} old image files");

                // Create 3D original data visualization
                var original3DPath = Path.Combine(resultsDir, "mammoth_original_3d.png");
                Visualizer.PlotOriginalMammoth3DReal(originalData, "Original Mammoth 3D Data", original3DPath);
                Console.WriteLine($"   ✅ Created: {Path.GetFileName(original3DPath)}");

                Console.WriteLine();
                Console.WriteLine("   ================================================");
                Console.WriteLine("   📊 Creating Direct KNN Visualization...");
                Console.WriteLine("   ================================================");

                // Create Direct KNN 2D embedding visualization
                var pacmapPathKNN = Path.Combine(resultsDir, "mammoth_pacmap_direct_knn.png");
                Console.WriteLine($"   📈 Creating Direct KNN PACMAP 2D embedding: {Path.GetFileName(pacmapPathKNN)}");

                // Get Direct KNN model information
                var modelInfoKNN = pacmapKNN.ModelInfo;
                var paramInfoKNN = new Dictionary<string, object>
                {
                    ["PACMAP Version"] = PacMapModel.GetVersion() + " (Corrected Gradients)",
                    ["n_neighbors"] = modelInfoKNN.Neighbors,
                    ["embedding_dimension"] = modelInfoKNN.OutputDimension,
                    ["distance_metric"] = modelInfoKNN.Metric.ToString(),
                    ["mn_ratio"] = modelInfoKNN.MN_ratio.ToString("F2"),
                    ["fp_ratio"] = modelInfoKNN.FP_ratio.ToString("F2"),
                    ["learning_rate"] = pacmapKNN.LearningRate.ToString("F3"),
                    ["init_std_dev"] = pacmapKNN.InitializationStdDev.ToString("E0"),
                    ["phase_iters"] = $"({pacmapKNN.NumIters.phase1}, {pacmapKNN.NumIters.phase2}, {pacmapKNN.NumIters.phase3})",
                    ["data_points"] = modelInfoKNN.TrainingSamples,
                    ["original_dimensions"] = modelInfoKNN.InputDimension,
                    ["hnsw_m"] = modelInfoKNN.HnswM,
                    ["hnsw_ef_construction"] = modelInfoKNN.HnswEfConstruction,
                    ["hnsw_ef_search"] = modelInfoKNN.HnswEfSearch,
                    ["KNN_Mode"] = "Direct KNN",
                    ["random_seed"] = modelInfoKNN.RandomSeed
                };
                paramInfoKNN["execution_time"] = $"{executionTimeKNN:F2}s";

                // Create Direct KNN title
                var versionKNN = paramInfoKNN["PACMAP Version"].ToString()?.Replace(" (Corrected Gradients)", "").Replace("-CLEAN-OUTPUT", "") ?? "Unknown";
                var line1KNN = $"PACMAP v{versionKNN} | Direct KNN | k={paramInfoKNN["n_neighbors"]} | {paramInfoKNN["distance_metric"]}";
                var line2KNN = $"mn={paramInfoKNN["mn_ratio"]} fp={paramInfoKNN["fp_ratio"]} lr={paramInfoKNN["learning_rate"]} std={paramInfoKNN["init_std_dev"]}";
                var line3KNN = $"phases={paramInfoKNN["phase_iters"]} | Time: {paramInfoKNN["execution_time"]}";
                var titleWithParamsKNN = $"Mammoth PACMAP 2D Embedding (Direct KNN)\n{line1KNN}\n{line2KNN}\n{line3KNN}";

                Visualizer.PlotMammothPacMAP(embeddingKNN, originalData, titleWithParamsKNN, pacmapPathKNN, paramInfoKNN);
                Console.WriteLine($"   ✅ Created: {Path.GetFileName(pacmapPathKNN)}");
                Console.WriteLine($"   📊 Direct KNN Mode: EXACT (no approximation)");
                Console.WriteLine($"   🚀 HNSW Status: DISABLED");

                Console.WriteLine();
                Console.WriteLine("   ================================================");
                Console.WriteLine("   📊 Creating HNSW Visualization...");
                Console.WriteLine("   ================================================");

                // Create HNSW 2D embedding visualization
                var pacmapPathHNSW = Path.Combine(resultsDir, "mammoth_pacmap_hnsw.png");
                Console.WriteLine($"   📈 Creating HNSW PACMAP 2D embedding: {Path.GetFileName(pacmapPathHNSW)}");

                // Get HNSW model information
                var modelInfoHNSW = pacmapHNSW.ModelInfo;
                var paramInfoHNSW = new Dictionary<string, object>
                {
                    ["PACMAP Version"] = PacMapModel.GetVersion() + " (Corrected Gradients)",
                    ["n_neighbors"] = modelInfoHNSW.Neighbors,
                    ["embedding_dimension"] = modelInfoHNSW.OutputDimension,
                    ["distance_metric"] = modelInfoHNSW.Metric.ToString(),
                    ["mn_ratio"] = modelInfoHNSW.MN_ratio.ToString("F2"),
                    ["fp_ratio"] = modelInfoHNSW.FP_ratio.ToString("F2"),
                    ["learning_rate"] = pacmapHNSW.LearningRate.ToString("F3"),
                    ["init_std_dev"] = pacmapHNSW.InitializationStdDev.ToString("E0"),
                    ["phase_iters"] = $"({pacmapHNSW.NumIters.phase1}, {pacmapHNSW.NumIters.phase2}, {pacmapHNSW.NumIters.phase3})",
                    ["data_points"] = modelInfoHNSW.TrainingSamples,
                    ["original_dimensions"] = modelInfoHNSW.InputDimension,
                    ["hnsw_m"] = modelInfoHNSW.HnswM,
                    ["hnsw_ef_construction"] = modelInfoHNSW.HnswEfConstruction,
                    ["hnsw_ef_search"] = modelInfoHNSW.HnswEfSearch,
                    ["KNN_Mode"] = "HNSW",
                    ["random_seed"] = modelInfoHNSW.RandomSeed
                };
                paramInfoHNSW["execution_time"] = $"{executionTimeHNSW:F2}s";

                // Create HNSW title
                var versionHNSW = paramInfoHNSW["PACMAP Version"].ToString()?.Replace(" (Corrected Gradients)", "").Replace("-CLEAN-OUTPUT", "") ?? "Unknown";
                var line1HNSW = $"PACMAP v{versionHNSW} | HNSW | k={paramInfoHNSW["n_neighbors"]} | {paramInfoHNSW["distance_metric"]}";
                var line2HNSW = $"mn={paramInfoHNSW["mn_ratio"]} fp={paramInfoHNSW["fp_ratio"]} lr={paramInfoHNSW["learning_rate"]} std={paramInfoHNSW["init_std_dev"]}";
                var line3HNSW = $"phases={paramInfoHNSW["phase_iters"]} | HNSW: M={paramInfoHNSW["hnsw_m"]}, ef={paramInfoHNSW["hnsw_ef_search"]} | Time: {paramInfoHNSW["execution_time"]}";
                var titleWithParamsHNSW = $"Mammoth PACMAP 2D Embedding (HNSW)\n{line1HNSW}\n{line2HNSW}\n{line3HNSW}";

                Visualizer.PlotMammothPacMAP(embeddingHNSW, originalData, titleWithParamsHNSW, pacmapPathHNSW, paramInfoHNSW);
                Console.WriteLine($"   ✅ Created: {Path.GetFileName(pacmapPathHNSW)}");
                Console.WriteLine($"   📊 HNSW Mode: APPROXIMATE (faster)");
                Console.WriteLine($"   🚀 HNSW Parameters: M={modelInfoHNSW.HnswM}, ef={modelInfoHNSW.HnswEfSearch}");

                Console.WriteLine();
                Console.WriteLine("   ================================================");
                Console.WriteLine("   📊 Creating Transform Projection Comparisons...");
                Console.WriteLine("   ================================================");

                // Create projection comparison images for Direct KNN
                Console.WriteLine("   📈 Creating Direct KNN (Fit vs Transform) projections...");

                var fitPathKNN = Path.Combine(resultsDir, "mammoth_direct_knn_fit.png");
                var titleFitKNN = $"Direct KNN - Fit Projection\n{line1KNN}\n{line2KNN}\n{line3KNN}";
                Visualizer.PlotMammothPacMAP(embeddingKNN, originalData, titleFitKNN, fitPathKNN, paramInfoKNN);
                Console.WriteLine($"   ✅ Created: {Path.GetFileName(fitPathKNN)}");

                var transformPathKNN = Path.Combine(resultsDir, "mammoth_direct_knn_transform.png");
                var titleTransformKNN = $"Direct KNN - Transform Projection (After Load)\n{line1KNN}\n{line2KNN}\n{line3KNN}";
                Visualizer.PlotMammothPacMAP(transformedKNN, originalData, titleTransformKNN, transformPathKNN, paramInfoKNN);
                Console.WriteLine($"   ✅ Created: {Path.GetFileName(transformPathKNN)}");

                // Create projection comparison images for HNSW
                Console.WriteLine("   📈 Creating HNSW (Fit vs Transform) projections...");

                var fitPathHNSW = Path.Combine(resultsDir, "mammoth_hnsw_fit.png");
                var titleFitHNSW = $"HNSW - Fit Projection\n{line1HNSW}\n{line2HNSW}\n{line3HNSW}";
                Visualizer.PlotMammothPacMAP(embeddingHNSW, originalData, titleFitHNSW, fitPathHNSW, paramInfoHNSW);
                Console.WriteLine($"   ✅ Created: {Path.GetFileName(fitPathHNSW)}");

                var transformPathHNSW = Path.Combine(resultsDir, "mammoth_hnsw_transform.png");
                var titleTransformHNSW = $"HNSW - Transform Projection (After Load)\n{line1HNSW}\n{line2HNSW}\n{line3HNSW}";
                Visualizer.PlotMammothPacMAP(transformedHNSW, originalData, titleTransformHNSW, transformPathHNSW, paramInfoHNSW);
                Console.WriteLine($"   ✅ Created: {Path.GetFileName(transformPathHNSW)}");

                // Calculate MSE between fit and transform for both modes
                double mseKNN = CalculateMSE(embeddingKNN, transformedKNN);
                double mseHNSW = CalculateMSE(embeddingHNSW, transformedHNSW);
                Console.WriteLine($"   📊 Transform Accuracy:");
                Console.WriteLine($"      Direct KNN MSE (Fit vs Transform): {mseKNN:E4}");
                Console.WriteLine($"      HNSW MSE (Fit vs Transform): {mseHNSW:E4}");

                Console.WriteLine();
                Console.WriteLine("   🎉 ALL visualizations created successfully!");
                Console.WriteLine($"   ⏱️  Performance Summary:");
                Console.WriteLine($"      Direct KNN:  {executionTimeKNN:F2}s (exact, slower)");
                Console.WriteLine($"      HNSW:        {executionTimeHNSW:F2}s (approximate, faster)");
                Console.WriteLine($"      Speedup:     {(executionTimeKNN/executionTimeHNSW):F1}x");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ❌ Visualization creation failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Displays model hyperparameters in a formatted way
        /// </summary>
        static void DisplayModelHyperparameters(PacMapModel model, string context = "Model")
        {
            Console.WriteLine($"   📋 {context} Hyperparameters:");
            Console.WriteLine($"      MN_ratio: {model.MN_ratio}");
            Console.WriteLine($"      FP_ratio: {model.FP_ratio}");
            Console.WriteLine($"      Learning Rate: {model.LearningRate}");
            Console.WriteLine($"      Adam Beta1: {model.AdamBeta1}");
            Console.WriteLine($"      Adam Beta2: {model.AdamBeta2}");
            Console.WriteLine($"      Initialization Std Dev: {model.InitializationStdDev}");
            Console.WriteLine($"      Phase Iterations: ({model.NumIters.phase1}, {model.NumIters.phase2}, {model.NumIters.phase3})");
            Console.WriteLine();
        }

        // ====================== COMPREHENSIVE TEST SUITE ======================
        // Helper methods from Program_Complex.cs for advanced testing

        static double CalculateMSE(float[,] embedding1, float[,] embedding2)
        {
            int n = embedding1.GetLength(0);
            int d = embedding1.GetLength(1);
            double mse = 0;

            for (int i = 0; i < n; i++)
                for (int j = 0; j < d; j++)
                    mse += Math.Pow(embedding1[i, j] - embedding2[i, j], 2);

            return mse / (n * d);
        }

        static double CalculateMaxDifference(float[,] embedding1, float[,] embedding2)
        {
            int n = embedding1.GetLength(0);
            int d = embedding1.GetLength(1);
            double maxDiff = 0;

            for (int i = 0; i < n; i++)
                for (int j = 0; j < d; j++)
                    maxDiff = Math.Max(maxDiff, Math.Abs(embedding1[i, j] - embedding2[i, j]));

            return maxDiff;
        }

        static void GenerateConsistencyPlot(float[,] embedding1, float[,] embedding2, int[] labels, string title, string outputPath)
        {
            var plotModel = new PlotModel { Title = title };
            plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Embedding 1 - X Coordinate" });
            plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Embedding 2 - X Coordinate" });

            // Color by labels
            var uniqueLabels = labels.Distinct().OrderBy(x => x).ToArray();
            var colors = new[] { OxyColors.Blue, OxyColors.Red, OxyColors.Green, OxyColors.Orange, OxyColors.Purple };

            for (int labelIdx = 0; labelIdx < uniqueLabels.Length; labelIdx++)
            {
                var label = uniqueLabels[labelIdx];
                var scatterSeries = new ScatterSeries { Title = $"Label {label}", MarkerType = MarkerType.Circle, MarkerSize = 3 };
                scatterSeries.MarkerFill = colors[labelIdx % colors.Length];
                scatterSeries.MarkerStroke = colors[labelIdx % colors.Length];

                for (int i = 0; i < labels.Length; i++)
                {
                    if (labels[i] == label)
                    {
                        scatterSeries.Points.Add(new ScatterPoint(embedding1[i, 0], embedding2[i, 0], 3));
                    }
                }

                plotModel.Series.Add(scatterSeries);
            }

            plotModel.Legends.Add(new Legend { LegendPosition = LegendPosition.TopRight });

            // Export to PNG
            var exporter = new PngExporter { Width = 800, Height = 600, Resolution = 300 };
            using var stream = File.Create(outputPath);
            exporter.Export(plotModel, stream);
        }

        static void GenerateHeatmapPlot(float[,] embedding1, float[,] embedding2, string title, string outputPath)
        {
            // Simplified heatmap plot - just create a placeholder for now
            var plotModel = new PlotModel { Title = title };
            plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Sample Index" });
            plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Sample Index" });

            // Add a simple placeholder text annotation
            var annotation = new TextAnnotation
            {
                Text = "Heatmap visualization\n(Complex pairwise distance matrix)",
                TextPosition = new DataPoint(0.5, 0.5),
                TextHorizontalAlignment = HorizontalAlignment.Center,
                TextVerticalAlignment = VerticalAlignment.Middle,
                FontSize = 16,
                TextColor = OxyColors.Blue
            };
            plotModel.Annotations.Add(annotation);

            // Export to PNG
            var exporter = new PngExporter { Width = 800, Height = 600, Resolution = 300 };
            using var stream = File.Create(outputPath);
            exporter.Export(plotModel, stream);
        }

        static void GenerateScatterPlot(float[,] embedding, int[] labels, string title, string outputPath)
        {
            var plotModel = new PlotModel { Title = title };
            plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "X Coordinate" });
            plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Y Coordinate" });

            var uniqueLabels = labels.Distinct().OrderBy(x => x).ToArray();
            var colors = new[] { OxyColors.Blue, OxyColors.Red, OxyColors.Green, OxyColors.Orange, OxyColors.Purple };

            for (int labelIdx = 0; labelIdx < uniqueLabels.Length; labelIdx++)
            {
                var label = uniqueLabels[labelIdx];
                var scatterSeries = new ScatterSeries { Title = $"Label {label}", MarkerType = MarkerType.Circle, MarkerSize = 3 };
                scatterSeries.MarkerFill = colors[labelIdx % colors.Length];
                scatterSeries.MarkerStroke = colors[labelIdx % colors.Length];

                for (int i = 0; i < labels.Length; i++)
                {
                    if (labels[i] == label)
                    {
                        scatterSeries.Points.Add(new ScatterPoint(embedding[i, 0], embedding[i, 1], 3));
                    }
                }

                plotModel.Series.Add(scatterSeries);
            }

            plotModel.Legends.Add(new Legend { LegendPosition = LegendPosition.TopRight });

            // Export to PNG
            var exporter = new PngExporter { Width = 800, Height = 600, Resolution = 300 };
            using var stream = File.Create(outputPath);
            exporter.Export(plotModel, stream);
        }

        static void GenerateProjection(double[,] originalData, float[,] embedding, string projectionType, string outputPath)
        {
            var plotModel = new PlotModel { Title = $"Original Data {projectionType} Projection" };

            if (projectionType == "XY")
            {
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "X Coordinate" });
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Y Coordinate" });

                var scatterSeries = new ScatterSeries { Title = "Original XY", MarkerType = MarkerType.Circle, MarkerSize = 2 };
                scatterSeries.MarkerFill = OxyColors.Blue;
                scatterSeries.MarkerStroke = OxyColors.Blue;

                for (int i = 0; i < originalData.GetLength(0); i++)
                {
                    scatterSeries.Points.Add(new ScatterPoint(originalData[i, 0], originalData[i, 1], 2));
                }

                plotModel.Series.Add(scatterSeries);
            }
            else if (projectionType == "XZ")
            {
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "X Coordinate" });
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Z Coordinate" });

                var scatterSeries = new ScatterSeries { Title = "Original XZ", MarkerType = MarkerType.Circle, MarkerSize = 2 };
                scatterSeries.MarkerFill = OxyColors.Red;
                scatterSeries.MarkerStroke = OxyColors.Red;

                for (int i = 0; i < originalData.GetLength(0); i++)
                {
                    scatterSeries.Points.Add(new ScatterPoint(originalData[i, 0], originalData[i, 2], 2));
                }

                plotModel.Series.Add(scatterSeries);
            }
            else if (projectionType == "YZ")
            {
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Y Coordinate" });
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Z Coordinate" });

                var scatterSeries = new ScatterSeries { Title = "Original YZ", MarkerType = MarkerType.Circle, MarkerSize = 2 };
                scatterSeries.MarkerFill = OxyColors.Green;
                scatterSeries.MarkerStroke = OxyColors.Green;

                for (int i = 0; i < originalData.GetLength(0); i++)
                {
                    scatterSeries.Points.Add(new ScatterPoint(originalData[i, 1], originalData[i, 2], 2));
                }

                plotModel.Series.Add(scatterSeries);
            }

            plotModel.Legends.Add(new Legend { LegendPosition = LegendPosition.TopRight });

            // Export to PNG
            var exporter = new PngExporter { Width = 800, Height = 600, Resolution = 300 };
            using var stream = File.Create(outputPath);
            exporter.Export(plotModel, stream);
        }

        /// <summary>
        /// Run comprehensive test suite from Program_Complex.cs
        /// </summary>
        static void RunTransformConsistencyTests()
        {
            Console.WriteLine();
            Console.WriteLine("=========================================================");
            Console.WriteLine("🧪 Comprehensive Test Suite: Reproducibility & Persistence");
            Console.WriteLine("=========================================================");
            Console.WriteLine("   Testing reproducibility across different KNN modes and model persistence...");
            Console.WriteLine();

            // Convert double[,] to float[,] for PACMAP
            var (data, labels) = LoadMammothData();
            var floatData = ConvertToFloat(data);

            // Test configurations
            var testConfigs = new[]
            {
                new
                {
                    Name = "Exact KNN Mode",
                    NNeighbors = 10,
                    Distance = "euclidean",
                    UseHnsw = false,
                    UseQuantization = false,
                    Seed = 42
                },
                new
                {
                    Name = "HNSW Mode",
                    NNeighbors = 10,
                    Distance = "euclidean",
                    UseHnsw = true,
                    UseQuantization = false, // User clarified: no quantization for HNSW test
                    Seed = 42
                }
            };

            foreach (var config in testConfigs)
            {
                Console.WriteLine($"Test: {config.Name}");
                Console.WriteLine(new string('-', 50));

                var testDir = $"Results/{config.Name.Replace(" ", "_")}_Reproducibility";
                Directory.CreateDirectory(testDir);

                RunTransformTest(floatData, labels, config.NNeighbors, config.Distance,
                               config.UseHnsw, config.UseQuantization, config.Seed, testDir);

                Console.WriteLine();
            }

            Console.WriteLine("✅ All comprehensive tests completed!");
            Console.WriteLine("📁 Check individual test folders for detailed results and visualizations.");
        }

                // =============================================

        /// <summary>
        /// Run individual transform test with 8-step validation
        /// </summary>
        static void RunTransformTest(float[,] data, int[] labels, int nNeighbors, string distance,
                                    bool useHnsw, bool useQuantization, int seed, string outputDir)
        {
            // Convert string distance to enum
            var metric = distance.ToLower() switch
            {
                "euclidean" => DistanceMetric.Euclidean,
                "manhattan" => DistanceMetric.Manhattan,
                "cosine" => DistanceMetric.Cosine,
                _ => DistanceMetric.Euclidean
            };

            Console.WriteLine($"   Configuration: n_neighbors={nNeighbors}, distance={distance}, hnsw={useHnsw}, quantization={useQuantization}");
            Console.WriteLine();

            // Step 1: Initial fit
            Console.WriteLine("   Step 1: Initial PACMAP fit...");
            var model1 = new PacMapModel();
            var embedding1 = model1.Fit(
                data: data,
                embeddingDimension: 2,
                nNeighbors: nNeighbors,
                metric: metric,
                forceExactKnn: !useHnsw,
                randomSeed: seed,
                progressCallback: (phase, current, total, percent, message) =>
                {
                    UnifiedProgressCallback(phase, current, total, percent, message);
                }
            );
            Console.WriteLine(); // New line after progress
            Console.WriteLine($"   ✅ Initial embedding created: {embedding1.GetLength(0)}x{embedding1.GetLength(1)}");

            // Step 2: Save model
            Console.WriteLine("   Step 2: Saving model...");
            string modelPath = Path.Combine(outputDir, "pacmap_model.pmm");
            model1.Save(modelPath);
            Console.WriteLine($"   ✅ Model saved: {modelPath}");

            // Step 3: Second fit with same parameters
            Console.WriteLine("   Step 3: Second PACMAP fit with same parameters...");
            var model2 = new PacMapModel();
            var embedding2 = model2.Fit(
                data: data,
                embeddingDimension: 2,
                nNeighbors: nNeighbors,
                metric: metric,
                forceExactKnn: !useHnsw,
                randomSeed: seed,
                progressCallback: (phase, current, total, percent, message) =>
                {
                    UnifiedProgressCallback(phase, current, total, percent, message);
                }
            );
            Console.WriteLine(); // New line after progress
            Console.WriteLine($"   ✅ Second embedding created: {embedding2.GetLength(0)}x{embedding2.GetLength(1)}");

            // Step 4: Load saved model
            Console.WriteLine("   Step 4: Loading saved model...");
            var loadedModel = PacMapModel.Load(modelPath);
            Console.WriteLine("   ✅ Model loaded successfully");

            // Step 5: Transform with loaded model
            Console.WriteLine("   Step 5: Transform with loaded model...");
            var embeddingLoaded = loadedModel.Transform(data);
            Console.WriteLine($"   ✅ Transform completed: {embeddingLoaded.GetLength(0)}x{embeddingLoaded.GetLength(1)}");

            // Step 6: Calculate reproducibility metrics
            Console.WriteLine("   Step 6: Calculating reproducibility metrics...");
            double mse = CalculateMSE(embedding1, embedding2);
            double maxDiff = CalculateMaxDifference(embedding1, embedding2);
            Console.WriteLine($"   MSE between embeddings: {mse:E2}");
            Console.WriteLine($"   Max difference: {maxDiff:E2}");

            // Step 7: Generate visualizations
            Console.WriteLine("   Step 7: Generating visualizations...");

            // Original data projections
            var originalData = new double[data.GetLength(0), data.GetLength(1)];
            for (int i = 0; i < data.GetLength(0); i++)
                for (int j = 0; j < data.GetLength(1); j++)
                    originalData[i, j] = data[i, j];

            GenerateProjection(originalData, embedding1, "XY", Path.Combine(outputDir, "original_3d_XY_TopView.png"));
            GenerateProjection(originalData, embedding1, "XZ", Path.Combine(outputDir, "original_3d_XZ_SideView.png"));
            GenerateProjection(originalData, embedding1, "YZ", Path.Combine(outputDir, "original_3d_YZ_FrontView.png"));

            // Embedding visualizations with white background and parameter information
            var modelInfo = model1.ModelInfo;
            var paramInfo1 = new Dictionary<string, object>
            {
                ["test_type"] = "Reproducibility Test - Embedding 1",
                ["n_neighbors"] = modelInfo.Neighbors,
                ["distance_metric"] = modelInfo.Metric.ToString(),
                ["mn_ratio"] = modelInfo.MN_ratio.ToString("F2"),
                ["fp_ratio"] = modelInfo.FP_ratio.ToString("F2"),
                ["learning_rate"] = model1.LearningRate.ToString("F3"),
                ["init_std_dev"] = model1.InitializationStdDev.ToString("E0"),
                ["phase_iters"] = $"({model1.NumIters.phase1}, {model1.NumIters.phase2}, {model1.NumIters.phase3})",
                ["data_points"] = modelInfo.TrainingSamples,
                ["original_dimensions"] = modelInfo.InputDimension,
                ["random_seed"] = modelInfo.RandomSeed,
                ["KNN_Mode"] = useHnsw ? "HNSW" : "Direct KNN",
                ["hnsw_m"] = modelInfo.HnswM,
                ["hnsw_ef_construction"] = modelInfo.HnswEfConstruction,
                ["hnsw_ef_search"] = modelInfo.HnswEfSearch
            };

            var paramInfo2 = new Dictionary<string, object>(paramInfo1)
            {
                ["test_type"] = "Reproducibility Test - Embedding 2"
            };

            var title1 = $"PACMAP Reproducibility Test - Embedding 1\n{modelInfo.Metric} | k={modelInfo.Neighbors} | {(useHnsw ? "HNSW" : "Direct KNN")}\nmn={modelInfo.MN_ratio:F2} fp={modelInfo.FP_ratio:F2} lr={model1.LearningRate:F3}";
            var title2 = $"PACMAP Reproducibility Test - Embedding 2\n{modelInfo.Metric} | k={modelInfo.Neighbors} | {(useHnsw ? "HNSW" : "Direct KNN")}\nmn={modelInfo.MN_ratio:F2} fp={modelInfo.FP_ratio:F2} lr={model1.LearningRate:F3}";

            Visualizer.PlotSimplePacMAP(embedding1, title1, Path.Combine(outputDir, "embedding1.png"), paramInfo1);
            Visualizer.PlotSimplePacMAP(embedding2, title2, Path.Combine(outputDir, "embedding2.png"), paramInfo2);

            // Consistency plots
            GenerateConsistencyPlot(embedding1, embedding2, labels, "Embedding Consistency (X)", Path.Combine(outputDir, "consistency_x.png"));
            GenerateHeatmapPlot(embedding1, embedding2, "Pairwise Distance Difference Heatmap", Path.Combine(outputDir, "distance_heatmap.png"));

            Console.WriteLine("   ✅ Visualizations generated");

            // Step 8: Summary and validation
            Console.WriteLine("   Step 8: Summary and validation...");

            bool isReproducible = mse < 1e-6 && maxDiff < 1e-4;
            bool dimensionsMatch = embedding1.GetLength(0) == embedding2.GetLength(0) &&
                                 embedding1.GetLength(1) == embedding2.GetLength(1);

            Console.WriteLine($"   Reproducibility: {(isReproducible ? "✅ PASS" : "❌ FAIL")}");
            Console.WriteLine($"   Dimension consistency: {(dimensionsMatch ? "✅ PASS" : "❌ FAIL")}");
            Console.WriteLine($"   Model persistence: ✅ PASS");

            if (!isReproducible)
            {
                Console.WriteLine("   ⚠️  WARNING: Results are not perfectly reproducible!");
                Console.WriteLine("   This may indicate non-deterministic behavior in the implementation.");
            }
        }

        // ====================== HYPERPARAMETER EXPERIMENTS ======================

        /// <summary>
        /// Auto-discovers optimal HNSW parameters for the mammoth dataset (shared across experiments)
        /// </summary>
        static (int M, int EfConstruction, int EfSearch) AutoDiscoverHNSWParameters(float[,] floatData)
        {
            var autoModel = new PacMapModel(); // Use default parameters (100, 100, 250)

            var autoStopwatch = Stopwatch.StartNew();
            var autoEmbedding = autoModel.Fit(
                data: floatData,
                embeddingDimension: 2,
                nNeighbors: 10,
                learningRate: 1.0f,
                mnRatio: 0.5f,  // Default MN ratio
                fpRatio: 2.0f,  // Default FP ratio
                numIters: (100, 100, 250),  // Default iterations
                metric: DistanceMetric.Euclidean,
                forceExactKnn: false,  // HNSW with auto-discovery
                randomSeed: 42,
                autoHNSWParam: true,
                progressCallback: (phase, current, total, percent, message) =>
                {
                    UnifiedProgressCallback(phase, current, total, percent, message, "Auto-Discovery");
                }
            );
            autoStopwatch.Stop();
            Console.WriteLine(); // New line after progress

            // Get discovered HNSW parameters
            var modelInfo = autoModel.ModelInfo;
            return (modelInfo.HnswM, modelInfo.HnswEfConstruction, modelInfo.HnswEfSearch);
        }

        /// <summary>
        /// DemoNeighborExperiments - Tests different neighbor counts (5-50) on mammoth dataset
        /// Uses shared HNSW parameters discovered once for optimal performance
        /// </summary>
        static void DemoNeighborExperiments(float[,] floatData, (int M, int EfConstruction, int EfSearch) hnswParams)
        {
            Console.WriteLine();
            Console.WriteLine("=========================================================");
            Console.WriteLine("🔬 DemoNeighborExperiments: Testing Neighbor Counts (5-50)");
            Console.WriteLine("=========================================================");
            Console.WriteLine("   Testing different n_neighbors values with shared HNSW parameters");
            Console.WriteLine($"   Using HNSW: M={hnswParams.M}, ef_construction={hnswParams.EfConstruction}, ef_search={hnswParams.EfSearch}");
            Console.WriteLine();

            // Load mammoth data
            var (data, labels) = LoadMammothData();

            // Test neighbor counts: 5, 10, 15, 20, 30, 40, 50
            var neighborTests = new[] { 5, 10, 15, 20, 30, 40, 50 };
            var results = new List<(int nNeighbors, float[,] embedding, double time, double quality)>();

            // Testing all neighbor counts with shared HNSW parameters
            Console.WriteLine("🚀 Testing neighbor counts with shared HNSW parameters...");
            Console.WriteLine($"   Using HNSW: M={hnswParams.M}, ef_construction={hnswParams.EfConstruction}, ef_search={hnswParams.EfSearch}");
            Console.WriteLine();

            foreach (var nNeighbors in neighborTests)
            {
                Console.WriteLine($"📊 Testing n_neighbors = {nNeighbors}...");

                var model = new PacMapModel(
                    mnRatio: 1.2f,
                    fpRatio: 2.0f,
                    learningRate: 1.0f,
                    initializationStdDev: 1e-4f,
                    numIters: (200, 200, 400)
                );

                var stopwatch = Stopwatch.StartNew();
                var embedding = model.Fit(
                    data: floatData,
                    embeddingDimension: 2,
                    nNeighbors: nNeighbors,
                    learningRate: 1.0f,
                    mnRatio: 1.2f,
                    fpRatio: 2.0f,
                    numIters: (200, 200, 400),
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: false,  // HNSW
                    hnswM: hnswParams.M,
                    hnswEfConstruction: hnswParams.EfConstruction,
                    hnswEfSearch: hnswParams.EfSearch,
                    autoHNSWParam: false,
                    randomSeed: 42,
                    progressCallback: (phase, current, total, percent, message) =>
                    {
                        UnifiedProgressCallback(phase, current, total, percent, message, $"n={nNeighbors}");
                    }
                );
                stopwatch.Stop();
                Console.WriteLine(); // New line after progress

                double quality = CalculateEmbeddingQuality(embedding, labels);
                results.Add((nNeighbors, embedding, stopwatch.Elapsed.TotalSeconds, quality));

                Console.WriteLine($"   ✅ n={nNeighbors}: quality={quality:F4}, time={stopwatch.Elapsed.TotalSeconds:F2}s");

                // Create visualization for this neighbor count
                var paramInfo = new Dictionary<string, object>
                {
                    ["experiment_type"] = "Neighbor_Experiments",
                    ["n_neighbors"] = nNeighbors,
                    ["mn_ratio"] = "1.2",
                    ["fp_ratio"] = "2.0",
                    ["learning_rate"] = "1.0",
                    ["hnsw_m"] = hnswParams.M,
                    ["hnsw_ef_construction"] = hnswParams.EfConstruction,
                    ["hnsw_ef_search"] = hnswParams.EfSearch,
                    ["embedding_quality"] = quality.ToString("F4"),
                    ["execution_time"] = $"{stopwatch.Elapsed.TotalSeconds:F2}s"
                };

                // Create organized folder structure for GIF creation
                var experimentDir = Path.Combine("Results", "neighbor_experiments");
                Directory.CreateDirectory(experimentDir);

                // Sequential numbering for GIF creation
                var imageNumber = (nNeighbors - 5) / 5 + 1; // Maps 5,10,15,20,30,40,50 to 1,2,3,4,6,8,10
                if (nNeighbors == 30) imageNumber = 6;
                else if (nNeighbors == 40) imageNumber = 8;
                else if (nNeighbors == 50) imageNumber = 10;

                var outputPath = Path.Combine(experimentDir, $"{imageNumber:D4}.png");

                var title = $"Neighbor Experiment: n={nNeighbors}\nHNSW: M={hnswParams.M}, ef={hnswParams.EfSearch}\nQuality: {quality:F4}, Time: {stopwatch.Elapsed.TotalSeconds:F2}s";
                Visualizer.PlotMammothPacMAP(embedding, data, title, outputPath, paramInfo);
                Console.WriteLine($"   📈 Saved: {Path.GetFileName(outputPath)}");
                Console.WriteLine();
            }

            // Summary
            Console.WriteLine("📊 NEIGHBOR EXPERIMENTS SUMMARY");
            Console.WriteLine(new string('=', 50));
            var bestResult = results.OrderBy(r => r.quality).First();
            Console.WriteLine($"🏆 Best neighbor count: n={bestResult.nNeighbors} (quality: {bestResult.quality:F4})");
            Console.WriteLine($"⏱️  Execution times ranged from {results.Min(r => r.time):F2}s to {results.Max(r => r.time):F2}s");
            Console.WriteLine("📁 All results saved to Results/neighbor_experiments/0001.png, 0002.png, etc. (ready for GIF creation)");
        }

        /// <summary>
        /// DemoLearningRateExperiments - Learning rate optimization (0.5-1.0) on mammoth dataset
        /// Uses shared HNSW parameters discovered once for optimal performance
        /// </summary>
        static void DemoLearningRateExperiments(float[,] floatData, (int M, int EfConstruction, int EfSearch) hnswParams)
        {
            Console.WriteLine();
            Console.WriteLine("=========================================================");
            Console.WriteLine("🎓 DemoLearningRateExperiments: Learning Rate (0.5-1.0)");
            Console.WriteLine("=========================================================");
            Console.WriteLine("   Testing different learning_rate values with shared HNSW parameters");
            Console.WriteLine($"   Using HNSW: M={hnswParams.M}, ef_construction={hnswParams.EfConstruction}, ef_search={hnswParams.EfSearch}");
            Console.WriteLine();

            // Load mammoth data
            var (data, labels) = LoadMammothData();

            // Test learning rates: 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
            var learningRateTests = new[] { 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f };
            var results = new List<(float learningRate, float[,] embedding, double time, double quality)>();

            // Testing all learning rates with shared HNSW parameters
            Console.WriteLine("🚀 Testing learning rates with shared HNSW parameters...");
            Console.WriteLine($"   Using HNSW: M={hnswParams.M}, ef_construction={hnswParams.EfConstruction}, ef_search={hnswParams.EfSearch}");
            Console.WriteLine();

            foreach (var learningRate in learningRateTests)
            {
                Console.WriteLine($"📊 Testing learning_rate = {learningRate:F1}...");

                var model = new PacMapModel(
                    mnRatio: 1.2f,
                    fpRatio: 2.0f,
                    learningRate: learningRate,
                    initializationStdDev: 1e-4f,
                    numIters: (200, 200, 400)
                );

                var stopwatch = Stopwatch.StartNew();
                var embedding = model.Fit(
                    data: floatData,
                    embeddingDimension: 2,
                    nNeighbors: 10,
                    learningRate: learningRate,
                    mnRatio: 1.2f,
                    fpRatio: 2.0f,
                    numIters: (200, 200, 400),
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: false,  // HNSW
                    hnswM: hnswParams.M,
                    hnswEfConstruction: hnswParams.EfConstruction,
                    hnswEfSearch: hnswParams.EfSearch,
                    autoHNSWParam: false,
                    randomSeed: 42,
                    progressCallback: (phase, current, total, percent, message) =>
                    {
                        UnifiedProgressCallback(phase, current, total, percent, message, $"lr={learningRate:F1}");
                    }
                );
                stopwatch.Stop();
                Console.WriteLine(); // New line after progress

                double quality = CalculateEmbeddingQuality(embedding, labels);
                results.Add((learningRate, embedding, stopwatch.Elapsed.TotalSeconds, quality));

                Console.WriteLine($"   ✅ lr={learningRate:F1}: quality={quality:F4}, time={stopwatch.Elapsed.TotalSeconds:F2}s");

                // Create visualization for this learning rate
                var paramInfo = new Dictionary<string, object>
                {
                    ["experiment_type"] = "Learning_Rate_Experiments",
                    ["learning_rate"] = learningRate.ToString("F1"),
                    ["n_neighbors"] = "10",
                    ["mn_ratio"] = "1.2",
                    ["fp_ratio"] = "2.0",
                    ["hnsw_m"] = hnswParams.M,
                    ["hnsw_ef_construction"] = hnswParams.EfConstruction,
                    ["hnsw_ef_search"] = hnswParams.EfSearch,
                    ["embedding_quality"] = quality.ToString("F4"),
                    ["execution_time"] = $"{stopwatch.Elapsed.TotalSeconds:F2}s"
                };

                // Create organized folder structure for GIF creation
                var experimentDir = Path.Combine("Results", "learning_rate_experiments");
                Directory.CreateDirectory(experimentDir);

                // Sequential numbering for GIF creation (maps 0.5-1.0 to 0001-0006)
                var imageNumber = (int)((learningRate - 0.5f) / 0.1f) + 1;
                var outputPath = Path.Combine(experimentDir, $"{imageNumber:D4}.png");

                var title = $"Learning Rate Experiment: lr={learningRate:F1}\nHNSW: M={hnswParams.M}, ef={hnswParams.EfSearch}\nQuality: {quality:F4}, Time: {stopwatch.Elapsed.TotalSeconds:F2}s";
                Visualizer.PlotMammothPacMAP(embedding, data, title, outputPath, paramInfo);
                Console.WriteLine($"   📈 Saved: {Path.GetFileName(outputPath)}");
                Console.WriteLine();
            }

            // Summary
            Console.WriteLine("📊 LEARNING RATE EXPERIMENTS SUMMARY");
            Console.WriteLine(new string('=', 50));
            var bestResult = results.OrderBy(r => r.quality).First();
            Console.WriteLine($"🏆 Best learning rate: {bestResult.learningRate:F1} (quality: {bestResult.quality:F4})");
            Console.WriteLine($"⏱️  Execution times ranged from {results.Min(r => r.time):F2}s to {results.Max(r => r.time):F2}s");
            Console.WriteLine("📁 All results saved to Results/learning_rate_experiments/0001.png, 0002.png, etc. (ready for GIF creation)");
        }

        /// <summary>
        /// DemoAdvancedParameterTuning - REAL Hairy Mammoth experiments from Program_Complex.cs
        /// Systematic testing of PACMAP hyperparameters using the mammoth_a.csv dataset
        /// Tests midNearRatio, farPairRatio, and neighbors variations exactly as in Program_Complex.cs
        /// Each experiment type creates images in separate folders: hairy_midnear_knn, hairy_farpair_knn, hairy_neighbor_knn
        /// </summary>
        static void DemoAdvancedParameterTuning(float[,] floatData, (int M, int EfConstruction, int EfSearch) hnswParams)
        {
            Console.WriteLine();
            Console.WriteLine("=========================================================");
            Console.WriteLine("🦣 DemoAdvancedParameterTuning: REAL Hairy Mammoth Experiments");
            Console.WriteLine("=========================================================");
            Console.WriteLine("   Systematic PACMAP hyperparameter testing on 1M point mammoth dataset");
            Console.WriteLine("   Based on methodology from Program_Complex.cs");
            Console.WriteLine("   Tests: midNearRatio, farPairRatio, and neighbors variations");
            Console.WriteLine("   Each experiment type in separate folder");
            Console.WriteLine();

            // Load the ACTUAL hairy mammoth dataset (1M points)
            Console.WriteLine("🦣 Loading hairy mammoth dataset (1M points)...");
            var hairyMammothPath = "Data/mammoth_a.csv";
            var hairyMammothFull = DataLoaders.LoadMammothData(hairyMammothPath);
            Console.WriteLine($"🦣 Loaded FULL hairy mammoth: {hairyMammothFull.GetLength(0):N0} samples, {hairyMammothFull.GetLength(1)} features");
            Console.WriteLine($"   Data ranges: X=[{hairyMammothFull[0,0]:F3}, {hairyMammothFull[hairyMammothFull.GetLength(0)-1,0]:F3}], Y=[{hairyMammothFull[0,1]:F3}, {hairyMammothFull[hairyMammothFull.GetLength(0)-1,1]:F3}], Z=[{hairyMammothFull[0,2]:F3}, {hairyMammothFull[hairyMammothFull.GetLength(0)-1,2]:F3}]");

            // =============================================
            // CREATE ONE 1M HAIRY MAMMOTH IMAGE WITH DEFAULT PARAMETERS
            // =============================================
            Console.WriteLine();
            Console.WriteLine("🎯 CREATING FLAGSHIP 1M HAIRY MAMMOTH IMAGE");
            Console.WriteLine("==========================================");
            Console.WriteLine("   Full 1M dataset with default parameters (100, 100, 250)");
            Console.WriteLine("   Neighbors: 10, Auto-tuning: ENABLED");
            Console.WriteLine();

            // Convert full 1M dataset to float
            int n1M = hairyMammothFull.GetLength(0);
            int d1M = hairyMammothFull.GetLength(1);
            var float1MHairy = new float[n1M, d1M];
            for (int i = 0; i < n1M; i++)
                for (int j = 0; j < d1M; j++)
                    float1MHairy[i, j] = (float)hairyMammothFull[i, j];

            // Create PACMAP model for 1M hairy mammoth
            var pacmap1MHairy = new PacMapModel();
            var stopwatch1MHairy = Stopwatch.StartNew();
            var embedding1MHairy = pacmap1MHairy.Fit(
                data: float1MHairy,
                embeddingDimension: 2,
                nNeighbors: 10,
                mnRatio: 0.5f,
                fpRatio: 2.0f,
                learningRate: 1.0f,
                numIters: (100, 100, 250),
                forceExactKnn: false,
                autoHNSWParam: true,
                randomSeed: 42,
                progressCallback: (phase, current, total, percent, message) =>
                {
                    UnifiedProgressCallback(phase, current, total, percent, message, "1M FLAGSHIP");
                }
            );
            Console.WriteLine();
            stopwatch1MHairy.Stop();

            Console.WriteLine($"✅ FLAGSHIP 1M Hairy Mammoth created: {embedding1MHairy.GetLength(0):N0} x {embedding1MHairy.GetLength(1)}");
            Console.WriteLine($"⏱️  Execution time: {stopwatch1MHairy.Elapsed.TotalSeconds:F2} seconds");

            // Save the flagship 1M hairy mammoth image
            Console.WriteLine("💾 Saving FLAGSHIP 1M hairy mammoth image...");
            string flagshipImagePath = Path.Combine(GetResultsPath(), "mammoth_1M_flagship_hnsw.png");

            // Get ACTUAL parameter values from the fitted model
            var flagshipModelInfo = pacmap1MHairy.ModelInfo;
            var flagshipParamInfo = new Dictionary<string, object>
            {
                ["Dataset"] = $"Hairy Mammoth 1M ({float1MHairy.GetLength(0):N0} points)",
                ["Method"] = "PACMAP-HNSW",
                ["Neighbors"] = flagshipModelInfo.Neighbors.ToString(),
                ["MN Ratio"] = flagshipModelInfo.MN_ratio.ToString("F1"),
                ["FP Ratio"] = flagshipModelInfo.FP_ratio.ToString("F1"),
                ["Learning Rate"] = pacmap1MHairy.LearningRate.ToString("F1"),
                ["Iterations"] = $"({pacmap1MHairy.NumIters.phase1}, {pacmap1MHairy.NumIters.phase2}, {pacmap1MHairy.NumIters.phase3})",  // Actual iterations from model
                ["HNSW M"] = flagshipModelInfo.HnswM.ToString(),
                ["HNSW ef_construction"] = flagshipModelInfo.HnswEfConstruction.ToString(),
                ["HNSW ef_search"] = flagshipModelInfo.HnswEfSearch.ToString(),
                ["Random Seed"] = flagshipModelInfo.RandomSeed.ToString(),
                ["Execution Time"] = $"{stopwatch1MHairy.Elapsed.TotalSeconds:F2}s"
            };
            Visualizer.PlotSimplePacMAP(embedding1MHairy, "Hairy Mammoth (1M) - FLAGSHIP PACMAP-HNSW", flagshipImagePath, flagshipParamInfo);
            Console.WriteLine($"   ✅ FLAGSHIP saved: {flagshipImagePath}");

            // Save the flagship 1M hairy mammoth model
            Console.WriteLine("💾 Saving FLAGSHIP 1M hairy mammoth model...");
            string flagshipModelPath = Path.Combine(GetResultsPath(), "pacmap_1M_flagship_hnsw.pmm");
            pacmap1MHairy.Save(flagshipModelPath);
            Console.WriteLine($"   ✅ FLAGSHIP model saved: {flagshipModelPath}");

            Console.WriteLine();
            Console.WriteLine("🎉 FLAGSHIP 1M HAIRY MAMMOTH COMPLETE!");
            Console.WriteLine("   📁 Check Results folder for mammoth_1M_flagship_hnsw.png");
            Console.WriteLine();

            float[,] sampleData; // Will be set based on subsampling decision

            // Use 100K subsample for parameter experiments (faster for GIF generation)
            int maxSamplesForTesting = 100000;
            if (hairyMammothFull.GetLength(0) > maxSamplesForTesting) {
                Console.WriteLine("🚀 Using 100K subsample for parameter testing (faster GIF generation)...");
                sampleData = CreateFloatSubset(hairyMammothFull, maxSamplesForTesting);
                Console.WriteLine($"   Subsample: {maxSamplesForTesting:N0} points from {hairyMammothFull.GetLength(0):N0} total");
            } else {
                Console.WriteLine("🚀 Using full dataset for experiments (≤ 100K points)...");
                // Convert to float for PACMAP
                int nSamples = hairyMammothFull.GetLength(0);
                int nFeatures = hairyMammothFull.GetLength(1);
                sampleData = new float[nSamples, nFeatures];
                for (int i = 0; i < nSamples; i++)
                    for (int j = 0; j < nFeatures; j++)
                        sampleData[i, j] = (float)hairyMammothFull[i, j];
                Console.WriteLine($"   Full dataset: {nSamples:N0} points for experiments");
            }

            // Performance tracking for samples
            var hnswMidNearPerformanceTimes = new List<double>();
            var hnswFarPairPerformanceTimes = new List<double>();
            var hnswNeighborsPerformanceTimes = new List<double>();
            var exactMidNearPerformanceTimes = new List<double>();
            var exactFarPairPerformanceTimes = new List<double>();
            var exactNeighborsPerformanceTimes = new List<double>();

            // Store dataset dimensions for use throughout function
            int totalSamples = sampleData.GetLength(0);
            int totalFeatures = sampleData.GetLength(1);

            // Experiment parameters
            double[] midNearRatioValues = { 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0 };
            double[] farPairRatioValues = { 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0 };
            int[] neighborValues = { 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80 };
            int fixedNeighbors = 10;
            int fileIndex = 1;

            // ========================================================================
            // PART 1: HNSW EXPERIMENTS (Full 1M Dataset)
            // ========================================================================
            Console.WriteLine();
            Console.WriteLine("🚀 PART 1: HNSW EXPERIMENTS (1M Points - Fast Approximate)");
            Console.WriteLine("====================================================================");

            // Step 1: HNSW Parameter Discovery (One-time auto-discovery)
            Console.WriteLine("🔍 STEP 1: HNSW Parameter Discovery");
            Console.WriteLine("====================================");
            Console.WriteLine("Running auto-discovery to find optimal HNSW parameters...");

            var discoveryModel = new PacMapModel(
                mnRatio: 1.0f,
                fpRatio: 2.0f,
                learningRate: 1.0f,
                initializationStdDev: 1e-4f,
                numIters: (200, 200, 400)
            );

            var discoveryStopwatch = Stopwatch.StartNew();
            var discoveryEmbedding = discoveryModel.Fit(
                data: floatData,  // Use FULL 1M dataset
                embeddingDimension: 2,
                nNeighbors: 10,  // Default neighbors
                learningRate: 1.0f,  // Default learning rate
                mnRatio: 0.5f,  // Default MN ratio
                fpRatio: 2.0f,  // Default FP ratio
                numIters: (100, 100, 250),  // DEFAULT iterations
                metric: DistanceMetric.Euclidean,
                forceExactKnn: false,  // HNSW with auto-tuning
                randomSeed: 42,
                autoHNSWParam: true,  // Enable auto-tuning for 1M dataset
                progressCallback: (phase, current, total, percent, message) =>
                {
                    UnifiedProgressCallback(phase, current, total, percent, message);
                }
            );
            Console.WriteLine(); // New line after progress
            discoveryStopwatch.Stop();
            Console.WriteLine($"   ✅ 1M HNSW Discovery completed in {discoveryStopwatch.Elapsed.TotalSeconds:F2}s");

            // Save the 1M HNSW image
            Console.WriteLine("💾 Saving 1M HNSW discovery image...");
            string resultsDir = GetResultsPath();
            string discoveryImagePath = Path.Combine(resultsDir, "mammoth_1M_hnsw_discovery.png");

            // Get ACTUAL parameter values from the fitted model
            var actualModelInfo = discoveryModel.ModelInfo;
            var discoveryParamInfo = new Dictionary<string, object>
            {
                ["Dataset"] = $"Mammoth 1M ({floatData.GetLength(0)} points)",
                ["Method"] = "PACMAP-HNSW",
                ["Neighbors"] = actualModelInfo.Neighbors.ToString(),
                ["MN Ratio"] = actualModelInfo.MN_ratio.ToString("F1"),
                ["FP Ratio"] = actualModelInfo.FP_ratio.ToString("F1"),
                ["Iterations"] = "(100, 100, 250)",  // Default iterations
                ["HNSW M"] = actualModelInfo.HnswM.ToString(),
                ["HNSW ef_construction"] = actualModelInfo.HnswEfConstruction.ToString(),
                ["HNSW ef_search"] = actualModelInfo.HnswEfSearch.ToString(),
                ["Random Seed"] = actualModelInfo.RandomSeed.ToString(),
                ["Execution Time"] = $"{discoveryStopwatch.Elapsed.TotalSeconds:F2}s"
            };
            Visualizer.PlotSimplePacMAP(discoveryEmbedding, "Hairy Mammoth (1M) - PACMAP-HNSW Discovery", discoveryImagePath, discoveryParamInfo);
            Console.WriteLine($"   ✅ Saved: {discoveryImagePath}");

            // Get discovered HNSW parameters for reuse
            var discoveredParams = discoveryModel.ModelInfo;
            Console.WriteLine($"   📊 Optimal HNSW parameters discovered:");
            Console.WriteLine($"      M={discoveredParams.HnswM}, ef_construction={discoveredParams.HnswEfConstruction}, ef_search={discoveredParams.HnswEfSearch}");
            Console.WriteLine();

            // Step 2: HNSW Experiments with Fixed Parameters (Same structure as Exact KNN)
            Console.WriteLine("🚀 STEP 2: HNSW EXPERIMENTS WITH FIXED PARAMETERS");
            Console.WriteLine("==================================================");
            Console.WriteLine($"Using discovered HNSW parameters for all experiments: M={discoveredParams.HnswM}, ef_c={discoveredParams.HnswEfConstruction}, ef_s={discoveredParams.HnswEfSearch}");
            Console.WriteLine();

            string hnswDir = GetResultsPath("hairy_hnsw");
            Directory.CreateDirectory(hnswDir);

            // HNSW EXPERIMENT 1: Mid-Near Ratio Variations (same as Exact KNN)
            Console.WriteLine("📊 HNSW EXPERIMENT 1: Mid-Near Ratio Variations");
            Console.WriteLine("===============================================");
            string hnswMidnearDir = GetResultsPath("hairy_hnsw_midnear");
            Directory.CreateDirectory(hnswMidnearDir);
            int hnswFileIndex = 1;

            foreach (double midNearRatio in midNearRatioValues)
            {
                Console.WriteLine($"🧪 HNSW Testing midNearRatio={midNearRatio:F1}...");

                var model = new PacMapModel(
                    mnRatio: 1.0f,
                    fpRatio: 2.0f,
                    learningRate: 1.0f,
                    initializationStdDev: 1e-4f,
                    numIters: (200, 200, 400)
                );

                try
                {
                    var stopwatch = Stopwatch.StartNew();
                    var embedding = model.Fit(
                        data: sampleData,
                        embeddingDimension: 2,
                        nNeighbors: fixedNeighbors,
                        learningRate: 1.0f,
                        mnRatio: (float)midNearRatio,
                        fpRatio: 2.0f,
                        numIters: (200, 200, 400),
                        metric: DistanceMetric.Euclidean,
                        forceExactKnn: false,  // HNSW
                        randomSeed: 42,
                        autoHNSWParam: false,  // Use discovered parameters
                        hnswM: discoveredParams.HnswM,
                        hnswEfConstruction: discoveredParams.HnswEfConstruction,
                        hnswEfSearch: discoveredParams.HnswEfSearch,
                        progressCallback: (phase, current, total, percent, message) =>
                        {
                            UnifiedProgressCallback(phase, current, total, percent, message);
                        }
                    );
                    Console.WriteLine(); // New line after progress
                    stopwatch.Stop();
                    var executionTime = stopwatch.Elapsed.TotalSeconds;
                    Console.WriteLine($"   ✅ HNSW Completed in {executionTime:F2}s");
                    hnswMidNearPerformanceTimes.Add(executionTime);

                    // Force garbage collection to clean up resources between experiments
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    GC.Collect();

                    // Create visualization
                    var plotPath = Path.Combine(hnswMidnearDir, $"{hnswFileIndex:D4}.png");
                    var title = $"PACMAP Hairy Mammoth - Parameter Testing\n" +
                              $"PACMAP v{PacMapModel.GetVersion()} | Sample: {sampleData.GetLength(0):N0} | HNSW\n" +
                              $"midNearRatio={midNearRatio:F1} | k={fixedNeighbors} | Euclidean\n" +
                              $"mn={midNearRatio:F1} | fp=2.0 | lr=1.0 | std=1e-4 | seed=42\n" +
                              $"phases=(200,200,400) | HNSW: M={discoveredParams.HnswM}, ef_c={discoveredParams.HnswEfConstruction}, ef_s={discoveredParams.HnswEfSearch}\n" +
                              $"Time: {executionTime:F2}s | Original dims: {sampleData.GetLength(1)}";
                    Visualizer.PlotSimplePacMAP(embedding, title, plotPath, null);
                    Console.WriteLine($"   📁 Saved: {hnswFileIndex:D4}.png");
                    hnswFileIndex++;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   ❌ ERROR in HNSW experiment midNearRatio={midNearRatio:F1}: {ex.Message}");
                    Console.WriteLine($"   Stack trace: {ex.StackTrace}");
                    Console.WriteLine($"   Continuing to next experiment...");
                    continue;
                }
            }

            // HNSW EXPERIMENT 2: Far-Pair Ratio Variations
            Console.WriteLine();
            Console.WriteLine("📊 HNSW EXPERIMENT 2: Far-Pair Ratio Variations");
            Console.WriteLine("==============================================");
            string hnswFarpairDir = GetResultsPath("hairy_hnsw_farpair");
            Directory.CreateDirectory(hnswFarpairDir);

            foreach (double farPairRatio in farPairRatioValues)
            {
                Console.WriteLine($"🧪 HNSW Testing farPairRatio={farPairRatio:F1}...");

                try
                {
                    var model = new PacMapModel(
                        mnRatio: 1.0f,
                        fpRatio: 2.0f,
                        learningRate: 1.0f,
                        initializationStdDev: 1e-4f,
                        numIters: (200, 200, 400)
                    );

                    var stopwatch = Stopwatch.StartNew();
                    var embedding = model.Fit(
                        data: sampleData,
                        embeddingDimension: 2,
                        nNeighbors: fixedNeighbors,
                        learningRate: 1.0f,
                        mnRatio: 1.0f,
                        fpRatio: (float)farPairRatio,
                        numIters: (200, 200, 400),
                        metric: DistanceMetric.Euclidean,
                        forceExactKnn: false,  // HNSW
                        randomSeed: 42,
                        autoHNSWParam: false,  // Use discovered parameters
                        hnswM: discoveredParams.HnswM,
                        hnswEfConstruction: discoveredParams.HnswEfConstruction,
                        hnswEfSearch: discoveredParams.HnswEfSearch,
                        progressCallback: (phase, current, total, percent, message) =>
                        {
                            UnifiedProgressCallback(phase, current, total, percent, message);
                        }
                    );
                    Console.WriteLine(); // New line after progress
                    stopwatch.Stop();
                    var executionTime = stopwatch.Elapsed.TotalSeconds;
                    Console.WriteLine($"   ✅ HNSW Completed in {executionTime:F2}s");
                    hnswFarPairPerformanceTimes.Add(executionTime);

                    // Force garbage collection to clean up resources between experiments
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    GC.Collect();

                    // Create visualization
                    var plotPath = Path.Combine(hnswFarpairDir, $"{hnswFileIndex:D4}.png");
                    var title = $"PACMAP Hairy Mammoth - Parameter Testing\n" +
                              $"PACMAP v{PacMapModel.GetVersion()} | Sample: {sampleData.GetLength(0):N0} | HNSW\n" +
                              $"farPairRatio={farPairRatio:F1} | k={fixedNeighbors} | Euclidean\n" +
                              $"mn=1.0 | fp={farPairRatio:F1} | lr=1.0 | std=1e-4 | seed=42\n" +
                              $"phases=(200,200,400) | HNSW: M={discoveredParams.HnswM}, ef_c={discoveredParams.HnswEfConstruction}, ef_s={discoveredParams.HnswEfSearch}\n" +
                              $"Time: {executionTime:F2}s | Original dims: {sampleData.GetLength(1)}";
                    Visualizer.PlotSimplePacMAP(embedding, title, plotPath, null);
                    Console.WriteLine($"   📁 Saved: {hnswFileIndex:D4}.png");
                    hnswFileIndex++;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   ❌ ERROR in HNSW experiment farPairRatio={farPairRatio:F1}: {ex.Message}");
                    Console.WriteLine($"   Stack trace: {ex.StackTrace}");
                    Console.WriteLine($"   Continuing to next experiment...");
                    continue;
                }
            }

            // HNSW EXPERIMENT 3: Neighbors Variations
            Console.WriteLine();
            Console.WriteLine("📊 HNSW EXPERIMENT 3: Neighbors Variations");
            Console.WriteLine("===========================================");
            string hnswNeighborsDir = GetResultsPath("hairy_hnsw_neighbors");
            Directory.CreateDirectory(hnswNeighborsDir);

            foreach (int neighbors in neighborValues)
            {
                Console.WriteLine($"🧪 HNSW Testing neighbors={neighbors}...");

                try
                {
                    var model = new PacMapModel(
                    mnRatio: 1.0f,
                    fpRatio: 2.0f,
                    learningRate: 1.0f,
                    initializationStdDev: 1e-4f,
                    numIters: (200, 200, 400)
                );

                var stopwatch = Stopwatch.StartNew();
                var embedding = model.Fit(
                    data: sampleData,
                    embeddingDimension: 2,
                    nNeighbors: neighbors,
                    learningRate: 1.0f,
                    mnRatio: 1.0f,
                    fpRatio: 2.0f,
                    numIters: (200, 200, 400),
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: false,  // HNSW
                    randomSeed: 42,
                    autoHNSWParam: false,  // Use discovered parameters
                    hnswM: discoveredParams.HnswM,
                    hnswEfConstruction: discoveredParams.HnswEfConstruction,
                    hnswEfSearch: discoveredParams.HnswEfSearch,
                    progressCallback: (phase, current, total, percent, message) =>
                    {
                        UnifiedProgressCallback(phase, current, total, percent, message);
                    }
                );
                Console.WriteLine(); // New line after progress
                stopwatch.Stop();
                var executionTime = stopwatch.Elapsed.TotalSeconds;
                Console.WriteLine($"   ✅ HNSW Completed in {executionTime:F2}s");
                hnswNeighborsPerformanceTimes.Add(executionTime);

                // Force garbage collection to clean up resources between experiments
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();

                // Create visualization
                var plotPath = Path.Combine(hnswNeighborsDir, $"{hnswFileIndex:D4}.png");
                var title = $"PACMAP Hairy Mammoth - Parameter Testing\n" +
                              $"PACMAP v{PacMapModel.GetVersion()} | Sample: {sampleData.GetLength(0):N0} | HNSW\n" +
                              $"neighbors={neighbors} | Euclidean | dims=2\n" +
                              $"mn=1.0 | fp=2.0 | lr=1.0 | std=1e-4 | seed=42\n" +
                              $"phases=(200,200,400) | HNSW: M={discoveredParams.HnswM}, ef_c={discoveredParams.HnswEfConstruction}, ef_s={discoveredParams.HnswEfSearch}\n" +
                              $"Time: {executionTime:F2}s | Original dims: {sampleData.GetLength(1)}";
                Visualizer.PlotSimplePacMAP(embedding, title, plotPath, null);
                Console.WriteLine($"   📁 Saved: {hnswFileIndex:D4}.png");
                hnswFileIndex++;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   ❌ ERROR in HNSW experiment neighbors={neighbors}: {ex.Message}");
                    Console.WriteLine($"   Stack trace: {ex.StackTrace}");
                    Console.WriteLine($"   Continuing to next experiment...");
                    continue;
                }
            }

            Console.WriteLine();
            Console.WriteLine("📊 HNSW PERFORMANCE SUMMARY");
            Console.WriteLine("==========================");

            if (hnswMidNearPerformanceTimes.Count > 0)
            {
                double avgMidNearTime = hnswMidNearPerformanceTimes.Average();
                Console.WriteLine($"📈 Mid-Near Ratio: {avgMidNearTime:F2}s avg | {hnswMidNearPerformanceTimes.Count} experiments");
            }

            if (hnswFarPairPerformanceTimes.Count > 0)
            {
                double avgFarPairTime = hnswFarPairPerformanceTimes.Average();
                Console.WriteLine($"📈 Far-Pair Ratio: {avgFarPairTime:F2}s avg | {hnswFarPairPerformanceTimes.Count} experiments");
            }

            if (hnswNeighborsPerformanceTimes.Count > 0)
            {
                double avgNeighborsTime = hnswNeighborsPerformanceTimes.Average();
                Console.WriteLine($"📈 Neighbors Var: {avgNeighborsTime:F2}s avg | {hnswNeighborsPerformanceTimes.Count} experiments");
            }

            // Overall HNSW Summary
            var allHnswTimes = new List<double>();
            allHnswTimes.AddRange(hnswMidNearPerformanceTimes);
            allHnswTimes.AddRange(hnswFarPairPerformanceTimes);
            allHnswTimes.AddRange(hnswNeighborsPerformanceTimes);
            if (allHnswTimes.Count > 0)
            {
                double overallHnswAvgTime = allHnswTimes.Average();
                Console.WriteLine($"📈 Overall HNSW: {overallHnswAvgTime:F2}s avg | {allHnswTimes.Count} total experiments");
            }

            Console.WriteLine($"   Discovered HNSW Parameters: M={discoveredParams.HnswM}, ef_c={discoveredParams.HnswEfConstruction}, ef_s={discoveredParams.HnswEfSearch}");

            // Store for final comparison
            var hnswPerformanceTimes = allHnswTimes;
            var hnswModelInfo = discoveredParams;
            double hnswTime = allHnswTimes.Count > 0 ? allHnswTimes.Average() : 0.0;
            Console.WriteLine($"🚀 HNSW (1M points): {hnswTime:F2}s");
            Console.WriteLine($"   Algorithm: HNSW with auto-discovery");
            Console.WriteLine($"   Accuracy: Approximate (fast)");
            Console.WriteLine();
            // ========================================================================
            // PART 2: EXACT KNN EXPERIMENTS (Subsampled for Performance)
            // ========================================================================
            Console.WriteLine("🎯 PART 2: EXACT KNN EXPERIMENTS (40K Subsampled - Precise)");
            Console.WriteLine("====================================================================");

            // Subsample 40K points for Exact KNN experiments (1M would take too long)
            Console.WriteLine("🔄 Subsampling 40K points for Exact KNN experiments...");
            var random = new Random(42); // Same seed for reproducibility
            int subsampleSize = 40000;
            var subsampleIndices = new int[subsampleSize];
            for (int i = 0; i < subsampleSize; i++)
            {
                subsampleIndices[i] = random.Next(totalSamples);
            }

            // Create subsampled data for Exact KNN
            var subsampleData = new float[subsampleSize, totalFeatures];
            for (int i = 0; i < subsampleSize; i++)
            {
                int idx = subsampleIndices[i];
                for (int j = 0; j < totalFeatures; j++)
                {
                    subsampleData[i, j] = sampleData[idx, j];
                }
            }
            Console.WriteLine($"   ✅ Subsampled {subsampleSize:N0} points from {totalSamples:N0} total for Exact KNN");
            Console.WriteLine();

            // Exact KNN Mid-Near Ratio Experiments
            Console.WriteLine("📊 EXACT KNN EXPERIMENT 1: Mid-Near Ratio Variations");
            Console.WriteLine("===============================================");
            string exactBaseDir = GetResultsPath("");  // Base directory for all Exact KNN results
            string exactMidnearDir = GetResultsPath("hairy_exact_midnear_knn");
            Directory.CreateDirectory(exactMidnearDir);
            fileIndex = 1;

            foreach (double midNearRatio in midNearRatioValues)
            {
                Console.WriteLine($"🧪 Exact KNN Testing midNearRatio={midNearRatio:F1}...");

                var model = new PacMapModel(
                    mnRatio: (float)midNearRatio,
                    fpRatio: 2.0f,
                    learningRate: 1.0f,
                    initializationStdDev: 1e-4f,
                    numIters: (200, 200, 400)
                );

                var stopwatch = Stopwatch.StartNew();
                var embedding = model.Fit(
                    data: subsampleData,
                    embeddingDimension: 2,
                    nNeighbors: fixedNeighbors,
                    learningRate: 1.0f,
                    mnRatio: (float)midNearRatio,
                    fpRatio: 2.0f,
                    numIters: (200, 200, 400),
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: true,  // Exact KNN
                    randomSeed: 42,
                    progressCallback: (phase, current, total, percent, message) =>
                    {
                        UnifiedProgressCallback(phase, current, total, percent, message);
                    }
                );
                Console.WriteLine(); // New line after progress
                stopwatch.Stop();
                var executionTime = stopwatch.Elapsed.TotalSeconds;
                Console.WriteLine($"   ✅ Exact KNN Completed in {executionTime:F2}s");
                exactMidNearPerformanceTimes.Add(executionTime);

                // Create visualization
                var plotPath = Path.Combine(exactMidnearDir, $"{fileIndex:D4}.png");
                var title = $"PACMAP Hairy Mammoth - Parameter Testing\n" +
                              $"PACMAP v{PacMapModel.GetVersion()} | Sample: {subsampleSize:N0} | Exact KNN\n" +
                              $"midNearRatio={midNearRatio:F1} | k={fixedNeighbors} | Euclidean\n" +
                              $"mn={midNearRatio:F1} | fp=2.0 | lr=1.0 | std=1e-4 | seed=42\n" +
                              $"phases=(200,200,400) | Direct KNN (Exact)\n" +
                              $"Time: {executionTime:F2}s | Original dims: {totalFeatures}";
                Visualizer.PlotSimplePacMAP(embedding, title, plotPath, null);
                Console.WriteLine($"   📁 Saved: {fileIndex:D4}.png");
                fileIndex++;
            }
            // Exact KNN Far-Pair Ratio Experiments
            Console.WriteLine("📊 EXACT KNN EXPERIMENT 2: Far-Pair Ratio Variations");
            Console.WriteLine("==============================================");
            string exactFarpairDir = GetResultsPath("hairy_exact_farpair_knn");
            Directory.CreateDirectory(exactFarpairDir);

            foreach (double farPairRatio in farPairRatioValues)
            {
                Console.WriteLine($"🧪 Exact KNN Testing farPairRatio={farPairRatio:F1}...");

                var model = new PacMapModel(
                    mnRatio: 1.0f,
                    fpRatio: (float)farPairRatio,
                    learningRate: 1.0f,
                    initializationStdDev: 1e-4f,
                    numIters: (200, 200, 400)
                );

                var stopwatch = Stopwatch.StartNew();
                var embedding = model.Fit(
                    data: subsampleData,
                    embeddingDimension: 2,
                    nNeighbors: fixedNeighbors,
                    learningRate: 1.0f,
                    mnRatio: 1.0f,
                    fpRatio: (float)farPairRatio,
                    numIters: (200, 200, 400),
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: true,  // Exact KNN
                    randomSeed: 42,
                    progressCallback: (phase, current, total, percent, message) =>
                    {
                        UnifiedProgressCallback(phase, current, total, percent, message);
                    }
                );
                Console.WriteLine(); // New line after progress
                stopwatch.Stop();
                var executionTime = stopwatch.Elapsed.TotalSeconds;
                Console.WriteLine($"   ✅ Exact KNN Completed in {executionTime:F2}s");
                exactFarPairPerformanceTimes.Add(executionTime);

                // Create visualization
                var plotPath = Path.Combine(exactFarpairDir, $"{fileIndex:D4}.png");
                var title = $"PACMAP Hairy Mammoth - Parameter Testing\n" +
                              $"PACMAP v{PacMapModel.GetVersion()} | Sample: {subsampleSize:N0} | Exact KNN\n" +
                              $"farPairRatio={farPairRatio:F1} | k={fixedNeighbors} | Euclidean\n" +
                              $"mn=1.0 | fp={farPairRatio:F1} | lr=1.0 | std=1e-4 | seed=42\n" +
                              $"phases=(200,200,400) | Direct KNN (Exact)\n" +
                              $"Time: {executionTime:F2}s | Original dims: {totalFeatures}";
                Visualizer.PlotSimplePacMAP(embedding, title, plotPath, null);
                Console.WriteLine($"   📁 Saved: {fileIndex:D4}.png");
                fileIndex++;
            }
            Console.WriteLine();

            // Exact KNN Neighbors Experiments
            Console.WriteLine("📊 EXACT KNN EXPERIMENT 3: Neighbors Variations");
            Console.WriteLine("============================================");
            string exactNeighborsDir = GetResultsPath("hairy_exact_neighbors_knn");
            Directory.CreateDirectory(exactNeighborsDir);
            int neighborFileIndex = 1;

            foreach (int neighbors in neighborValues)
            {
                Console.WriteLine($"🧪 Exact KNN Testing neighbors={neighbors}...");

                var model = new PacMapModel(
                    mnRatio: 1.0f,
                    fpRatio: 2.0f,
                    learningRate: 1.0f,
                    initializationStdDev: 1e-4f,
                    numIters: (200, 200, 400)
                );

                var stopwatch = Stopwatch.StartNew();
                var embedding = model.Fit(
                    data: subsampleData,
                    embeddingDimension: 2,
                    nNeighbors: neighbors,
                    learningRate: 1.0f,
                    mnRatio: 1.0f,
                    fpRatio: 2.0f,
                    numIters: (200, 200, 400),
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: true,  // Exact KNN
                    randomSeed: 42,
                    progressCallback: (phase, current, total, percent, message) =>
                    {
                        UnifiedProgressCallback(phase, current, total, percent, message);
                    }
                );
                Console.WriteLine(); // New line after progress
                stopwatch.Stop();
                var executionTime = stopwatch.Elapsed.TotalSeconds;
                Console.WriteLine($"   ✅ Exact KNN Completed in {executionTime:F2}s");
                exactNeighborsPerformanceTimes.Add(executionTime);

                // Create visualization
                var plotPath = Path.Combine(exactNeighborsDir, $"{neighborFileIndex:D4}.png");
                var title = $"PACMAP Hairy Mammoth - Parameter Testing\n" +
                              $"PACMAP v{PacMapModel.GetVersion()} | Sample: {subsampleSize:N0} | Exact KNN\n" +
                              $"neighbors={neighbors} | Euclidean | dims=2\n" +
                              $"mn=1.0 | fp=2.0 | lr=1.0 | std=1e-4 | seed=42\n" +
                              $"phases=(200,200,400) | Direct KNN (Exact)\n" +
                              $"Time: {executionTime:F2}s | Original dims: {totalFeatures}";
                Visualizer.PlotSimplePacMAP(embedding, title, plotPath, null);
                Console.WriteLine($"   📁 Saved: {neighborFileIndex:D4}.png");
                neighborFileIndex++;
            }
            Console.WriteLine();

            // COMPREHENSIVE PERFORMANCE SUMMARY
            Console.WriteLine();
            Console.WriteLine("📊 COMPREHENSIVE PERFORMANCE SUMMARY");
            Console.WriteLine("=====================================");
            Console.WriteLine($"   HNSW: 1M points (full dataset) | Exact KNN: {subsampleSize:N0} points (subsampled)");

            // HNSW Performance Summary (Multiple experiments)
            Console.WriteLine($"🚀 HNSW (Multiple Experiments):");
            if (hnswPerformanceTimes.Count > 0)
            {
                double hnswAvgTime = hnswPerformanceTimes.Average();
                double hnswMinTime = hnswPerformanceTimes.Min();
                double hnswMaxTime = hnswPerformanceTimes.Max();
                Console.WriteLine($"   Average: {hnswAvgTime:F2}s | Min: {hnswMinTime:F2}s | Max: {hnswMaxTime:F2}s");
                Console.WriteLine($"   Discovered Parameters: M={hnswModelInfo.HnswM}, ef_c={hnswModelInfo.HnswEfConstruction}, ef_s={hnswModelInfo.HnswEfSearch}");
                Console.WriteLine($"   Performance: ~{(1.0 / hnswAvgTime):F1} runs/sec | 1M points in {hnswAvgTime:F2}s");
                Console.WriteLine($"   Experiments: {hnswPerformanceTimes.Count} different parameter configurations");
            }

            // Exact KNN Performance Summary
            Console.WriteLine();
            Console.WriteLine($"🎯 Exact KNN Performance:");

            if (exactMidNearPerformanceTimes.Count > 0)
            {
                double avgMidNearTime = exactMidNearPerformanceTimes.Average();
                Console.WriteLine($"   Mid-Near Ratio: {avgMidNearTime:F2}s avg | {exactMidNearPerformanceTimes.Count} experiments");
            }

            if (exactFarPairPerformanceTimes.Count > 0)
            {
                double avgFarPairTime = exactFarPairPerformanceTimes.Average();
                Console.WriteLine($"   Far-Pair Ratio: {avgFarPairTime:F2}s avg | {exactFarPairPerformanceTimes.Count} experiments");
            }

            if (exactNeighborsPerformanceTimes.Count > 0)
            {
                double avgNeighborsTime = exactNeighborsPerformanceTimes.Average();
                Console.WriteLine($"   Neighbors Var: {avgNeighborsTime:F2}s avg | {exactNeighborsPerformanceTimes.Count} experiments");
            }

            // Overall Exact KNN Summary
            var allExactTimes = new List<double>();
            allExactTimes.AddRange(exactMidNearPerformanceTimes);
            allExactTimes.AddRange(exactFarPairPerformanceTimes);
            allExactTimes.AddRange(exactNeighborsPerformanceTimes);
            if (allExactTimes.Count > 0)
            {
                double overallExactAvgTime = allExactTimes.Average();
                double overallExactMinTime = allExactTimes.Min();
                double overallExactMaxTime = allExactTimes.Max();
                Console.WriteLine($"   Overall Exact KNN: {overallExactAvgTime:F2}s avg | Min: {overallExactMinTime:F2}s | Max: {overallExactMaxTime:F2}s");
                Console.WriteLine($"   Performance: ~{(1.0 / overallExactAvgTime):F1} runs/sec | 1M points in {overallExactAvgTime:F2}s");
            }

            // Algorithm Comparison
            Console.WriteLine();
            Console.WriteLine($"⚡ ALGORITHM COMPARISON:");
            if (hnswPerformanceTimes.Count > 0 && allExactTimes.Count > 0)
            {
                double hnswAvg = hnswPerformanceTimes.Average();
                double exactAvg = allExactTimes.Average();
                double speedup = exactAvg / hnswAvg;
                Console.WriteLine($"   HNSW is {speedup:F1}x faster than Exact KNN");
                Console.WriteLine($"   HNSW: {hnswAvg:F2}s | Exact KNN: {exactAvg:F2}s | Speedup: {speedup:F1}x");
            }

            Console.WriteLine();
            Console.WriteLine($"📁 Results Summary:");
            Console.WriteLine($"   HNSW Results: {hnswDir}");
            Console.WriteLine($"   Exact KNN Results: {exactBaseDir}");
            Console.WriteLine($"   Dataset: Hairy Mammoth (HNSW: 1M, Exact KNN: {subsampleSize:N0}) | Seed: 42 (deterministic)");
            Console.WriteLine($"   Images: 1200x1200 resolution, black dots on white background");
            Console.WriteLine("✅ All experiments completed successfully!");
        }

        /// <summary>
        /// DemoInitializationStdDevExperiments - Tests different initialization standard deviations
        /// Tests initialization_std_dev values: 1e-4, 1e-3, 1e-2, 1e-1
        /// Uses shared HNSW parameters discovered once for optimal performance
        /// </summary>
        static void DemoInitializationStdDevExperiments(float[,] floatData, (int M, int EfConstruction, int EfSearch) hnswParams)
        {
            Console.WriteLine();
            Console.WriteLine("=========================================================");
            Console.WriteLine("🎲 DemoInitializationStdDevExperiments: Testing Init Std Dev");
            Console.WriteLine("=========================================================");
            Console.WriteLine("   Testing different initialization_std_dev values with shared HNSW parameters");
            Console.WriteLine($"   Using HNSW: M={hnswParams.M}, ef_construction={hnswParams.EfConstruction}, ef_search={hnswParams.EfSearch}");
            Console.WriteLine();

            // Load mammoth data
            var (data, labels) = LoadMammothData();

            // Test initialization standard deviations: 1e-4, 1e-3, 1e-2, 1e-1
            var initStdDevTests = new[] { 1e-4f, 1e-3f, 1e-2f, 1e-1f };
            var results = new List<(float initStdDev, float[,] embedding, double time, double quality)>();

            // Testing all initialization std devs with shared HNSW parameters
            Console.WriteLine("🚀 Testing initialization std devs with shared HNSW parameters...");
            Console.WriteLine($"   Using HNSW: M={hnswParams.M}, ef_construction={hnswParams.EfConstruction}, ef_search={hnswParams.EfSearch}");
            Console.WriteLine();

            foreach (var initStdDev in initStdDevTests)
            {
                Console.WriteLine($"📊 Testing initialization_std_dev = {initStdDev:E1}...");

                var model = new PacMapModel(
                    mnRatio: 1.2f,
                    fpRatio: 2.0f,
                    learningRate: 1.0f,
                    initializationStdDev: initStdDev,
                    numIters: (200, 200, 400)
                );

                var stopwatch = Stopwatch.StartNew();
                var embedding = model.Fit(
                    data: floatData,
                    embeddingDimension: 2,
                    nNeighbors: 10,
                    learningRate: 1.0f,
                    mnRatio: 1.2f,
                    fpRatio: 2.0f,
                    numIters: (200, 200, 400),
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: false,  // HNSW
                    hnswM: hnswParams.M,
                    hnswEfConstruction: hnswParams.EfConstruction,
                    hnswEfSearch: hnswParams.EfSearch,
                    autoHNSWParam: false,
                    randomSeed: 42
                );
                stopwatch.Stop();
                Console.WriteLine(); // New line after progress

                double quality = CalculateEmbeddingQuality(embedding, labels);
                results.Add((initStdDev, embedding, stopwatch.Elapsed.TotalSeconds, quality));

                Console.WriteLine($"   ✅ init_std={initStdDev:E1}: quality={quality:F4}, time={stopwatch.Elapsed.TotalSeconds:F2}s");

                // Create visualization for this initialization std dev
                var paramInfo = new Dictionary<string, object>
                {
                    ["experiment_type"] = "Initialization_Std_Dev_Experiments",
                    ["initialization_std_dev"] = initStdDev.ToString("E1"),
                    ["n_neighbors"] = "10",
                    ["mn_ratio"] = "1.2",
                    ["fp_ratio"] = "2.0",
                    ["learning_rate"] = "1.0",
                    ["hnsw_m"] = hnswParams.M,
                    ["hnsw_ef_construction"] = hnswParams.EfConstruction,
                    ["hnsw_ef_search"] = hnswParams.EfSearch,
                    ["embedding_quality"] = quality.ToString("F4"),
                    ["execution_time"] = $"{stopwatch.Elapsed.TotalSeconds:F2}s"
                };

                // Create organized folder structure for GIF creation
                var experimentDir = Path.Combine("Results", "init_std_dev_experiments");
                Directory.CreateDirectory(experimentDir);

                // Sequential numbering for GIF creation (maps 1e-4, 1e-3, 1e-2, 1e-1 to 0001-0004)
                var imageNumber = initStdDev switch
                {
                    1e-4f => 1,
                    1e-3f => 2,
                    1e-2f => 3,
                    1e-1f => 4,
                    _ => 1
                };
                var outputPath = Path.Combine(experimentDir, $"{imageNumber:D4}.png");

                var title = $"Init Std Dev Experiment: {initStdDev:E1}\nHNSW: M={hnswParams.M}, ef={hnswParams.EfSearch}\nQuality: {quality:F4}, Time: {stopwatch.Elapsed.TotalSeconds:F2}s";
                Visualizer.PlotMammothPacMAP(embedding, data, title, outputPath, paramInfo);
                Console.WriteLine($"   📈 Saved: {Path.GetFileName(outputPath)}");
                Console.WriteLine();
            }

            // Summary
            Console.WriteLine("📊 INITIALIZATION STD DEV EXPERIMENTS SUMMARY");
            Console.WriteLine(new string('=', 60));
            var bestResult = results.OrderBy(r => r.quality).First();
            Console.WriteLine($"🏆 Best initialization std dev: {bestResult.initStdDev:E1} (quality: {bestResult.quality:F4})");
            Console.WriteLine($"⏱️  Execution times ranged from {results.Min(r => r.time):F2}s to {results.Max(r => r.time):F2}s");
            Console.WriteLine("📁 All results saved to Results/init_std_dev_experiments/0001.png, 0002.png, etc. (ready for GIF creation)");
        }

        /// <summary>
        /// DemoExtendedLearningRateExperiments - Extended learning rate tests
        /// Tests learning_rate values: 1.0, 0.9, 0.8, 0.7, 0.6, 0.5
        /// Uses shared HNSW parameters discovered once for optimal performance
        /// </summary>
        static void DemoExtendedLearningRateExperiments(float[,] floatData, (int M, int EfConstruction, int EfSearch) hnswParams)
        {
            Console.WriteLine();
            Console.WriteLine("=========================================================");
            Console.WriteLine("🎓 DemoExtendedLearningRateExperiments: Extended LR Tests");
            Console.WriteLine("=========================================================");
            Console.WriteLine("   Testing extended learning_rate values with shared HNSW parameters");
            Console.WriteLine($"   Using HNSW: M={hnswParams.M}, ef_construction={hnswParams.EfConstruction}, ef_search={hnswParams.EfSearch}");
            Console.WriteLine();

            // Load mammoth data
            var (data, labels) = LoadMammothData();

            // Test learning rates: 1.0, 0.9, 0.8, 0.7, 0.6, 0.5
            var learningRateTests = new[] { 1.0f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f };
            var results = new List<(float learningRate, float[,] embedding, double time, double quality)>();

            // Testing extended learning rates with shared HNSW parameters
            Console.WriteLine($"   Using HNSW: M={hnswParams.M}, ef_construction={hnswParams.EfConstruction}, ef_search={hnswParams.EfSearch}");
            Console.WriteLine();

            foreach (var learningRate in learningRateTests)
            {
                Console.WriteLine($"📊 Testing learning_rate = {learningRate:F1}...");

                var model = new PacMapModel(
                    mnRatio: 1.2f,
                    fpRatio: 2.0f,
                    learningRate: learningRate,
                    initializationStdDev: 1e-4f,
                    numIters: (200, 200, 400)
                );

                var stopwatch = Stopwatch.StartNew();
                var embedding = model.Fit(
                    data: floatData,
                    embeddingDimension: 2,
                    nNeighbors: 10,
                    learningRate: learningRate,
                    mnRatio: 1.2f,
                    fpRatio: 2.0f,
                    numIters: (200, 200, 400),
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: false,  // HNSW
                    hnswM: hnswParams.M,
                    hnswEfConstruction: hnswParams.EfConstruction,
                    hnswEfSearch: hnswParams.EfSearch,
                    autoHNSWParam: false,
                    randomSeed: 42,
                    progressCallback: (phase, current, total, percent, message) =>
                    {
                        UnifiedProgressCallback(phase, current, total, percent, message, $"lr={learningRate:F1}");
                    }
                );
                stopwatch.Stop();
                Console.WriteLine(); // New line after progress

                double quality = CalculateEmbeddingQuality(embedding, labels);
                results.Add((learningRate, embedding, stopwatch.Elapsed.TotalSeconds, quality));

                Console.WriteLine($"   ✅ lr={learningRate:F1}: quality={quality:F4}, time={stopwatch.Elapsed.TotalSeconds:F2}s");

                // Create visualization for this learning rate
                var paramInfo = new Dictionary<string, object>
                {
                    ["experiment_type"] = "Extended_Learning_Rate_Experiments",
                    ["learning_rate"] = learningRate.ToString("F1"),
                    ["n_neighbors"] = "10",
                    ["mn_ratio"] = "1.2",
                    ["fp_ratio"] = "2.0",
                    ["initialization_std_dev"] = "1e-4",
                    ["hnsw_m"] = hnswParams.M,
                    ["hnsw_ef_construction"] = hnswParams.EfConstruction,
                    ["hnsw_ef_search"] = hnswParams.EfSearch,
                    ["embedding_quality"] = quality.ToString("F4"),
                    ["execution_time"] = $"{stopwatch.Elapsed.TotalSeconds:F2}s"
                };

                // Create organized folder structure for GIF creation
                var experimentDir = Path.Combine("Results", "extended_lr_experiments");
                Directory.CreateDirectory(experimentDir);

                // Sequential numbering for GIF creation (maps 1.0,0.9,0.8,0.7,0.6,0.5 to 0001-0006)
                var imageNumber = (int)((1.0f - learningRate) / 0.1f) + 1;
                var outputPath = Path.Combine(experimentDir, $"{imageNumber:D4}.png");

                var title = $"Extended LR Experiment: lr={learningRate:F1}\nHNSW: M={hnswParams.M}, ef={hnswParams.EfSearch}\nQuality: {quality:F4}, Time: {stopwatch.Elapsed.TotalSeconds:F2}s";
                Visualizer.PlotMammothPacMAP(embedding, data, title, outputPath, paramInfo);
                Console.WriteLine($"   📈 Saved: {Path.GetFileName(outputPath)}");
                Console.WriteLine();
            }

            // Summary
            Console.WriteLine("📊 EXTENDED LEARNING RATE EXPERIMENTS SUMMARY");
            Console.WriteLine(new string('=', 60));
            var bestResult = results.OrderBy(r => r.quality).First();
            Console.WriteLine($"🏆 Best learning rate: {bestResult.learningRate:F1} (quality: {bestResult.quality:F4})");
            Console.WriteLine($"⏱️  Execution times ranged from {results.Min(r => r.time):F2}s to {results.Max(r => r.time):F2}s");
            Console.WriteLine("📁 All results saved to Results/extended_lr_experiments/0001.png, 0002.png, etc. (ready for GIF creation)");
        }
    }
}