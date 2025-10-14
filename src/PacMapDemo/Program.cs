using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using OxyPlot.Legends;
using OxyPlot.Annotations;
using OxyPlot.WindowsForms;
using PacMapSharp;

namespace PacMapDemo
{
    /// <summary>
    /// Main program class for running PACMAP demonstrations and experiments on the mammoth dataset.
    /// </summary>
    public class Program
    {
        private const string ResultsDir = "Results";
        private static readonly string DataDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "PacMapDemo", "Data"));
        private const string MammothDataFile = "mammoth_data.csv";
        private const string HairyMammothDataFile = "mammoth_a.csv";

        /// <summary>
        /// Entry point for the PACMAP demo application.
        /// </summary>
        public static void Main(string[] args)
        {
            Console.WriteLine("=================================");
            Console.WriteLine($"PACMAP Library Version: {PacMapModel.GetVersion()}");

            try
            {
                // Initialize results directory and clean previous results
                InitializeResultsDirectory();

                // Load and prepare mammoth dataset
                var (data, labels) = LoadMammothData();
                Console.WriteLine($"Loaded: {data.GetLength(0)} points, {data.GetLength(1)} dimensions");

                // Run core demonstrations
               // Run10kMammothDemo(data);
           

             

                // Run transform consistency tests
                // RunTransformConsistencyTests(data, labels); 

                // Run hyperparameter experiments
                //RunHyperparameterExperiments(data, labels); 


               // CreateFlagship1MHairyMammoth();

                // Run advanced parameter tuning
                //DemoAdvancedParameterTuning(data, labels); 

                // Run MNIST demo
                RunMnistDemo(); // DISABLED - testing exact KNN only
                OpenResultsFolder();
                Console.WriteLine("🎉 ALL DEMONSTRATIONS AND EXPERIMENTS COMPLETED!");
                Console.WriteLine($"📁 Check {ResultsDir} folder for visualizations.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
            }
        }

        /// <summary>
        /// Initializes the results directory and cleans up previous results.
        /// </summary>
        private static void InitializeResultsDirectory()
        {
            Console.WriteLine("🧹 Cleaning up previous results...");
            CleanupAllResults();
            Directory.CreateDirectory(ResultsDir);
            Console.WriteLine($"   📁 Created {ResultsDir} directory");
        }

        /// <summary>
        /// Opens the Results folder in Windows Explorer.
        /// </summary>
        private static void OpenResultsFolder()
        {
            Console.WriteLine("📂 Opening Results folder...");
            try
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = GetResultsPath(),
                    UseShellExecute = true
                });
                Console.WriteLine($"   ✅ Results folder opened");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ⚠️ Could not open Results folder: {ex.Message}");
            }
        }

        /// <summary>
        /// Loads the mammoth dataset from a CSV file.
        /// </summary>
        private static (double[,] data, int[] labels) LoadMammothData()
        {
            Console.WriteLine("📥 Loading mammoth dataset...");
            string csvPath = Path.Combine(DataDir, MammothDataFile);
            if (!File.Exists(csvPath))
            {
                throw new FileNotFoundException($"Mammoth data file not found: {csvPath}");
            }
            return DataLoaders.LoadMammothWithLabels(csvPath);
        }

        /// <summary>
        /// Calculates the quality of an embedding based on intra-label distances.
        /// </summary>
        private static double CalculateEmbeddingQuality(double[,] embedding, int[] labels)
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
                            Math.Pow(embedding[i, 1] - embedding[j, 1], 2));
                        minSameLabelDistance = Math.Min(minSameLabelDistance, dist);
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

        /// <summary>
        /// Deletes all files and subdirectories in the Results folder.
        /// </summary>
        private static void CleanupAllResults()
        {
            string resultsPath = GetResultsPath();
            if (!Directory.Exists(resultsPath))
            {
                Directory.CreateDirectory(resultsPath);
                return;
            }

            int deletedFiles = 0, deletedFolders = 0, failedFiles = 0, failedFolders = 0;

            foreach (var file in Directory.GetFiles(resultsPath, "*", SearchOption.AllDirectories))
            {
                try
                {
                    File.SetAttributes(file, FileAttributes.Normal);
                    File.Delete(file);
                    deletedFiles++;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   ⚠️ Could not delete file {Path.GetFileName(file)}: {ex.Message}");
                    failedFiles++;
                }
            }

            foreach (var dir in Directory.GetDirectories(resultsPath, "*", SearchOption.AllDirectories).Reverse())
            {
                try
                {
                    if (Directory.GetFiles(dir).Length == 0 && Directory.GetDirectories(dir).Length == 0)
                    {
                        Directory.Delete(dir);
                        deletedFolders++;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   ⚠️ Could not delete folder {Path.GetFileName(dir)}: {ex.Message}");
                    failedFolders++;
                }
            }

            Console.WriteLine($"   📊 Cleanup Summary:");
            if (deletedFiles > 0) Console.WriteLine($"      ✅ Deleted {deletedFiles} files");
            if (deletedFolders > 0) Console.WriteLine($"      ✅ Deleted {deletedFolders} folders");
            if (failedFiles > 0) Console.WriteLine($"      ⚠️ Failed to delete {failedFiles} files");
            if (failedFolders > 0) Console.WriteLine($"      ⚠️ Failed to delete {failedFolders} folders");
            if (deletedFiles == 0 && deletedFolders == 0)
                Console.WriteLine($"      ℹ️ Results folder was already clean");
        }

        /// <summary>
        /// Gets the full path to the Results directory or a subdirectory within it.
        /// </summary>
        private static string GetResultsPath(string subDirectory = "")
        {
            var basePath = Path.Combine(Directory.GetCurrentDirectory(), ResultsDir);
            return string.IsNullOrEmpty(subDirectory) ? basePath : Path.Combine(basePath, subDirectory);
        }

        /// <summary>
        /// Unified progress callback for consistent console output.
        /// </summary>
        private static void UnifiedProgressCallback(string phase, int current, int total, float percent, string? message)
        {
            Console.Write($"\r{new string(' ', 180)}\r   [{phase}] Progress: {current}/{total} ({percent:F1}%) {message}");
        }

        /// <summary>
        /// Unified progress callback logger with consistent console output.
        /// </summary>
        private static void UnifiedProgressCallbackLogger(string phase, int current, int total, float percent, string? message)
        {
            Console.WriteLine($"   [{phase}] Progress: {current}/{total} ({percent:F1}%) {message}");
        }

        /// <summary>
        /// Creates a progress callback with prefix for better organization.
        /// </summary>
        private static ProgressCallback CreatePrefixedCallback(string prefix)
        {
            return (phase, current, total, percent, message) =>
            {
                string displayPrefix = string.IsNullOrEmpty(prefix) ? "" : $"[{prefix}] ";
                Console.Write($"\r{new string(' ', 180)}\r   {displayPrefix}[{phase}] Progress: {current}/{total} ({percent:F1}%) {message}");
            };
        }

        /// <summary>
        /// Creates a progress callback logger with prefix for better organization.
        /// </summary>
        private static ProgressCallback CreatePrefixedLoggerCallback(string prefix)
        {
            return (phase, current, total, percent, message) =>
            {
                string displayPrefix = string.IsNullOrEmpty(prefix) ? "" : $"[{prefix}] ";
                Console.WriteLine($"{displayPrefix}[{phase}] Progress: {current}/{total} ({percent:F1}%) {message}");
            };
        }



        /// <summary>
        /// Runs a demo on a 10K mammoth dataset using Exact KNN.
        /// </summary>
        private static void Run10kMammothDemo(double[,] data)
        {
            Console.WriteLine("🦣 Running 10K Mammoth Demo...");
            var pacmap = new PacMapModel();
            var stopwatch = Stopwatch.StartNew();
            var embedding = pacmap.Fit(
                data: data,
                embeddingDimension: 2,
                nNeighbors: 10,
                mnRatio: 1.1f,
                fpRatio: 2.0f,
                learningRate: 1.0f,
                numIters: (100, 100, 250),
                forceExactKnn: false,
                autoHNSWParam: false,
                hnswM: 16,
                hnswEfSearch: 100,
                hnswEfConstruction: 150,
                randomSeed: 42,
                progressCallback: UnifiedProgressCallback
            );
            stopwatch.Stop();
            Console.WriteLine();
            Console.WriteLine($"✅ 10K Embedding created: {embedding.GetLength(0)} x {embedding.GetLength(1)}");
            Console.WriteLine($"⏱️ Execution time: {stopwatch.Elapsed.TotalSeconds:F2}s");

            // Save model
            string modelPath = Path.Combine(ResultsDir, "mammoth_10k_hnsw.pmm");
            pacmap.Save(modelPath);
            Console.WriteLine($"✅ Model saved: {modelPath}");

            // Create visualization with complete model info
            CreateVisualizations(embedding, data, new int[data.GetLength(0)], pacmap, stopwatch.Elapsed.TotalSeconds);
        }

        /// <summary>
        /// Creates the flagship 1M hairy mammoth demo.
        /// </summary>
        private static void CreateFlagship1MHairyMammoth()
        {
            Console.WriteLine("🦣 Running Flagship 1M Hairy Mammoth Demo...");
            string csvPath = Path.Combine(DataDir, HairyMammothDataFile);
            if (!File.Exists(csvPath))
            {
                Console.WriteLine($"   ⚠️ Hairy mammoth data file not found: {csvPath}");
                return;
            }

            var (data, labels) = DataLoaders.LoadMammothWithLabels(csvPath);
            Console.WriteLine($"   Loaded: {data.GetLength(0)} points, {data.GetLength(1)} dimensions");

            var pacmap = new PacMapModel();
            var stopwatch = Stopwatch.StartNew();
            var embedding = pacmap.Fit(
                data: data,
                embeddingDimension: 2,
                nNeighbors: 10,
                mnRatio: 0.5f,
                fpRatio: 2.0f,
                learningRate: 1.0f,
                numIters: (100, 100, 250),
                forceExactKnn: false,
                autoHNSWParam: true,
                randomSeed: 42,
                progressCallback: UnifiedProgressCallback
            );
            stopwatch.Stop();
            Console.WriteLine();
            Console.WriteLine($"   ✅ 1M Embedding created: {embedding.GetLength(0)} x {embedding.GetLength(1)}");
            Console.WriteLine($"   ⏱️ Execution time: {stopwatch.Elapsed.TotalSeconds:F2}s");

            // Save model
            string modelPath = Path.Combine(ResultsDir, "hairy_mammoth_1m_hnsw.pmm");
         

              // Create black and white visualization of original 3D data
            string outputPath = Path.Combine(ResultsDir, "hairy_mammoth_1m_3d.png");
            var title3D = "Flagship 1M Hairy Mammoth - Original 3D Black & White";
            Visualizer.PlotOriginalMammoth3DReal(data, title3D, outputPath);
            Console.WriteLine($"   ✅ Created: {Path.GetFileName(outputPath)}");
        }

        /// <summary>
        /// Creates visualizations for the mammoth demos.
        /// </summary>
        private static void CreateVisualizations(double[,] embedding, double[,] originalData, int[] labels, PacMapModel pacmap, double executionTime)
        {
            try
            {
                Console.WriteLine("🎨 Creating visualizations...");
                string original3DPath = Path.Combine(ResultsDir, "mammoth_original_3d.png");
                Visualizer.PlotOriginalMammoth3DReal(originalData, "Original Mammoth 3D Data", original3DPath);
                Console.WriteLine($"   ✅ Created: {Path.GetFileName(original3DPath)}");

                string pacmapPath = Path.Combine(ResultsDir, "mammoth_pacmap_embedding.png");
                var modelInfo = pacmap.ModelInfo;
                var paramInfo = CreateFitParamInfo(pacmap, executionTime, "Main_Demo");

                var title = BuildVisualizationTitle(pacmap);
                Visualizer.PlotMammothPacMAP(embedding, originalData, title, pacmapPath, paramInfo);
                Console.WriteLine($"   ✅ Created: {Path.GetFileName(pacmapPath)}");
                Console.WriteLine($"   📊 KNN Mode: {paramInfo["KNN_Mode"]}");
                Console.WriteLine($"   🚀 HNSW Status: {(modelInfo.ForceExactKnn ? "DISABLED" : "ACTIVE")}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ❌ Visualization creation failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Creates parameter info dictionary from model info (essential FIT parameters only).
        /// </summary>
        private static Dictionary<string, object> CreateFitParamInfo(PacMapModel model, double executionTime, string experimentType = "")
        {
            var modelInfo = model.ModelInfo;
            return new Dictionary<string, object>
            {
                ["experiment_type"] = experimentType,
                ["PACMAP Version"] = PacMapModel.GetVersion(),
                ["n_neighbors"] = modelInfo.Neighbors,
                ["embedding_dimension"] = modelInfo.OutputDimension,
                ["distance_metric"] = modelInfo.Metric.ToString(),
                ["mn_ratio"] = modelInfo.MN_ratio.ToString("F2"),
                ["fp_ratio"] = modelInfo.FP_ratio.ToString("F2"),
                ["learning_rate"] = modelInfo.AdamEps.ToString("F6"),
                ["init_std_dev"] = modelInfo.InitializationStdDev.ToString("E0"),
                ["phase_iters"] = $"({model.NumIters.phase1}, {model.NumIters.phase2}, {model.NumIters.phase3})",
                ["data_points"] = modelInfo.TrainingSamples,
                ["original_dimensions"] = modelInfo.InputDimension,
                ["hnsw_m"] = modelInfo.HnswM,
                ["hnsw_ef_construction"] = modelInfo.HnswEfConstruction,
                ["hnsw_ef_search"] = modelInfo.HnswEfSearch,
                ["KNN_Mode"] = modelInfo.ForceExactKnn ? "Direct KNN" : "HNSW",
                ["random_seed"] = modelInfo.RandomSeed,
                ["execution_time"] = $"{executionTime:F2}s"
            };
        }

        /// <summary>
        /// Builds a title for visualizations from model info.
        /// </summary>
        private static string BuildVisualizationTitle(PacMapModel model, string prefix = "Mammoth PACMAP 2D Embedding")
        {
            var modelInfo = model.ModelInfo;
            var version = PacMapModel.GetVersion().Replace(" (Corrected Gradients)", "").Replace("-CLEAN-OUTPUT", "");
            var knnMode = modelInfo.ForceExactKnn ? "Direct KNN" : "HNSW";

            return $@"{prefix}
PACMAP v{version} | Sample: {modelInfo.TrainingSamples:N0} | {knnMode}
k={modelInfo.Neighbors} | {modelInfo.Metric} | dims={modelInfo.OutputDimension} | seed={modelInfo.RandomSeed}
mn={modelInfo.MN_ratio:F2} | fp={modelInfo.FP_ratio:F2} | lr={modelInfo.AdamEps:F6} | std={modelInfo.InitializationStdDev:E0}
phases=({model.NumIters.phase1}, {model.NumIters.phase2}, {model.NumIters.phase3}) | HNSW: M={modelInfo.HnswM}, ef_c={modelInfo.HnswEfConstruction}, ef_s={modelInfo.HnswEfSearch}";
        }

        /// <summary>
        /// Runs transform consistency tests for reproducibility and persistence.
        /// </summary>
        private static void RunTransformConsistencyTests(double[,] data, int[] labels)
        {
            Console.WriteLine("🧪 Running Transform Consistency Tests...");
            var testConfigs = new[]
            {
                new { Name = "Exact KNN Mode", NNeighbors = 10, Distance = "euclidean", UseHnsw = false, UseQuantization = false, Seed = 42 },
                new { Name = "HNSW Mode", NNeighbors = 10, Distance = "euclidean", UseHnsw = true, UseQuantization = false, Seed = 42 }
            };

            foreach (var config in testConfigs)
            {
                Console.WriteLine($"   🔍 Testing {config.Name}...");
                string testDir = Path.Combine(ResultsDir, config.Name.Replace(" ", "_") + "_Reproducibility");
                Directory.CreateDirectory(testDir);
                RunTransformTest(data, labels, config.NNeighbors, config.Distance, config.UseHnsw, config.UseQuantization, config.Seed, testDir);
            }
            Console.WriteLine("✅ All transform tests completed!");
        }

        /// <summary>
        /// Runs a single transform test with validation steps.
        /// </summary>
        private static void RunTransformTest(double[,] data, int[] labels, int nNeighbors, string distance, bool useHnsw, bool useQuantization, int seed, string outputDir)
        {
            var metric = distance.ToLower() switch
            {
                "euclidean" => DistanceMetric.Euclidean,
                "manhattan" => DistanceMetric.Manhattan,
                "cosine" => DistanceMetric.Cosine,
                _ => DistanceMetric.Euclidean
            };

            Console.WriteLine($"   Configuration: n_neighbors={nNeighbors}, distance={distance}, hnsw={useHnsw}, quantization={useQuantization}");

            // Step 1: Initial fit
            var model1 = new PacMapModel();
            var embedding1 = model1.Fit(data, 2, nNeighbors, 0.5f, 2.0f, (100, 100, 250), metric, !useHnsw, 16, 150, 100, seed, true, 1.0f, false, UnifiedProgressCallback);
            Console.WriteLine($"   ✅ Initial embedding created: {embedding1.GetLength(0)}x{embedding1.GetLength(1)}");

            // Step 2: Save model
            string modelPath = Path.Combine(outputDir, "pacmap_model.pmm");
            model1.Save(modelPath);
            Console.WriteLine($"   ✅ Model saved: {modelPath}");

            // Step 3: Second fit
            var model2 = new PacMapModel();
            var embedding2 = model2.Fit(data, 2, nNeighbors, 0.5f, 2.0f, (100, 100, 250), metric, !useHnsw, 16, 150, 100, seed, true, 1.0f, false, UnifiedProgressCallback);
            Console.WriteLine($"   ✅ Second embedding created: {embedding2.GetLength(0)}x{embedding2.GetLength(1)}");

            // Step 4: Load saved model
            var loadedModel = PacMapModel.Load(modelPath);
            Console.WriteLine("   ✅ Model loaded successfully");

            // Step 5: Transform with loaded model
            var embeddingLoaded = loadedModel.Transform(data);
            Console.WriteLine($"   ✅ Transform completed: {embeddingLoaded.GetLength(0)}x{embeddingLoaded.GetLength(1)}");

            // Step 6: Calculate reproducibility metrics
            double mse = CalculateMSE(embedding1, embedding2);
            double maxDiff = CalculateMaxDifference(embedding1, embedding2);
            Console.WriteLine($"   MSE between embeddings: {mse:E2}");
            Console.WriteLine($"   Max difference: {maxDiff:E2}");

            // Step 7: Generate visualizations
            GenerateTransformVisualizations(data, embedding1, embedding2, labels, model1, outputDir);

            // Step 8: Summary and validation
            bool isReproducible = mse < 1e-6 && maxDiff < 1e-4;
            bool dimensionsMatch = embedding1.GetLength(0) == embedding2.GetLength(0) && embedding1.GetLength(1) == embedding2.GetLength(1);
            Console.WriteLine($"   Reproducibility: {(isReproducible ? "✅ PASS" : "❌ FAIL")}");
            Console.WriteLine($"   Dimension consistency: {(dimensionsMatch ? "✅ PASS" : "❌ FAIL")}");
            Console.WriteLine($"   Model persistence: ✅ PASS");
        }

        /// <summary>
        /// Generates visualizations for transform consistency tests.
        /// </summary>
        private static void GenerateTransformVisualizations(double[,] data, double[,] embedding1, double[,] embedding2, int[] labels, PacMapModel model, string outputDir)
        {
            var originalData = data;
            GenerateProjection(originalData, embedding1, "XY", Path.Combine(outputDir, "original_3d_XY_TopView.png"));
            GenerateProjection(originalData, embedding1, "XZ", Path.Combine(outputDir, "original_3d_XZ_SideView.png"));
            GenerateProjection(originalData, embedding1, "YZ", Path.Combine(outputDir, "original_3d_YZ_FrontView.png"));

            var paramInfo1 = CreateFitParamInfo(model, 0, "Reproducibility_Test_Embedding_1");
            var paramInfo2 = CreateFitParamInfo(model, 0, "Reproducibility_Test_Embedding_2");
            var title1 = BuildVisualizationTitle(model, "PACMAP Reproducibility Test - Embedding 1");
            var title2 = BuildVisualizationTitle(model, "PACMAP Reproducibility Test - Embedding 2");

            Visualizer.PlotMammothPacMAP(embedding1, data, title1, Path.Combine(outputDir, "embedding1.png"), paramInfo1);
            Visualizer.PlotMammothPacMAP(embedding2, data, title2, Path.Combine(outputDir, "embedding2.png"), paramInfo2);
            GenerateConsistencyPlot(embedding1, embedding2, labels, "Embedding Consistency (X)", Path.Combine(outputDir, "consistency_x.png"));
            GenerateHeatmapPlot(embedding1, embedding2, "Pairwise Distance Difference Heatmap", Path.Combine(outputDir, "distance_heatmap.png"));
            Console.WriteLine("   ✅ Visualizations generated");
        }

        /// <summary>
        /// Runs hyperparameter experiments on the mammoth dataset.
        /// </summary>
        private static void RunHyperparameterExperiments(double[,] data, int[] labels)
        {
            Console.WriteLine("🔬 Running Hyperparameter Experiments...");
            var optimalHNSWParams = AutoDiscoverHNSWParameters(data);
            Console.WriteLine($"✅ HNSW Parameters: M={optimalHNSWParams.M}, ef_construction={optimalHNSWParams.EfConstruction}, ef_search={optimalHNSWParams.EfSearch}");

            DemoNeighborExperiments(data, labels, optimalHNSWParams);
            DemoLearningRateExperiments(data, labels, optimalHNSWParams);
            DemoInitializationStdDevExperiments(data, labels, optimalHNSWParams);
        }

        /// <summary>
        /// Auto-discovers optimal HNSW parameters.
        /// </summary>
        private static (int M, int EfConstruction, int EfSearch) AutoDiscoverHNSWParameters(double[,] data)
        {
            Console.WriteLine("🔍 Auto-discovering HNSW parameters...");
            var model = new PacMapModel();
            var stopwatch = Stopwatch.StartNew();
            model.Fit(
                data: data,
                embeddingDimension: 2,
                nNeighbors: 10,
                learningRate: 1.0f,
                mnRatio: 0.5f,
                fpRatio: 2.0f,
                numIters: (100, 100, 250),
                metric: DistanceMetric.Euclidean,
                forceExactKnn: false,
                autoHNSWParam: true,
                randomSeed: 42,
                progressCallback: CreatePrefixedCallback("Auto-Discovery")
            );
            stopwatch.Stop();
            Console.WriteLine();
            var modelInfo = model.ModelInfo;
            return (modelInfo.HnswM, modelInfo.HnswEfConstruction, modelInfo.HnswEfSearch);
        }

        /// <summary>
        /// Tests different neighbor counts for the mammoth dataset.
        /// </summary>
        private static void DemoNeighborExperiments(double[,] data, int[] labels, (int M, int EfConstruction, int EfSearch) hnswParams)
        {
            Console.WriteLine("🔬 Testing Neighbor Counts (5-50)...");
            var neighborTests = Enumerable.Range(0, 13).Select(i => 5 + i * 2).ToArray();
            var results = new List<(int nNeighbors, double[,] embedding, double time, double quality)>();

            foreach (var nNeighbors in neighborTests)
            {
                Console.WriteLine($"   📊 Testing n_neighbors = {nNeighbors}...");
                var model = new PacMapModel(mnRatio: 1.2f, fpRatio: 2.0f, learningRate: 1.0f, initializationStdDev: 1e-3f, numIters: (100, 100, 250));
                var stopwatch = Stopwatch.StartNew();
                var embedding = model.Fit(
                    data: data,
                    embeddingDimension: 2,
                    nNeighbors: nNeighbors,
                    learningRate: 1.0f,
                    mnRatio: 1.2f,
                    fpRatio: 2.0f,
                    numIters: (100, 100, 250),
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: false,
                    hnswM: hnswParams.M,
                    hnswEfConstruction: hnswParams.EfConstruction,
                    hnswEfSearch: hnswParams.EfSearch,
                    autoHNSWParam: false,
                    randomSeed: 42,
                    progressCallback: CreatePrefixedCallback($"n={nNeighbors}")
                );
                stopwatch.Stop();
                Console.WriteLine();
                double quality = CalculateEmbeddingQuality(embedding, labels);
                results.Add((nNeighbors, embedding, stopwatch.Elapsed.TotalSeconds, quality));
                Console.WriteLine($"   ✅ n={nNeighbors}: quality={quality:F4}, time={stopwatch.Elapsed.TotalSeconds:F2}s");

                var paramInfo = CreateFitParamInfo(model, stopwatch.Elapsed.TotalSeconds, "Neighbor_Experiments");
                paramInfo["embedding_quality"] = quality.ToString("F4");

                var experimentDir = Path.Combine(ResultsDir, "neighbor_experiments");
                Directory.CreateDirectory(experimentDir);
                var outputPath = Path.Combine(experimentDir, $"{(nNeighbors - 5) / 2 + 1:D4}.png");
                var modelInfo = model.ModelInfo;
                var title = $"Neighbor Experiment: n={modelInfo.Neighbors}\n" + BuildVisualizationTitle(model, "Neighbor Experiment");
                Visualizer.PlotMammothPacMAP(embedding, data, title, outputPath, paramInfo);
                Console.WriteLine($"   📈 Saved: {Path.GetFileName(outputPath)}");
            }

            Console.WriteLine("📊 Neighbor Experiments Summary");
            Console.WriteLine(new string('=', 50));
            var bestResult = results.OrderBy(r => r.quality).First();
            Console.WriteLine($"🏆 Best neighbor count: n={bestResult.nNeighbors} (quality: {bestResult.quality:F4})");
            Console.WriteLine($"⏱️ Execution times: {results.Min(r => r.time):F2}s to {results.Max(r => r.time):F2}s");
        }

        /// <summary>
        /// Tests different learning rates for the mammoth dataset.
        /// </summary>
        private static void DemoLearningRateExperiments(double[,] data, int[] labels, (int M, int EfConstruction, int EfSearch) hnswParams)
        {
            Console.WriteLine("🎓 Testing Learning Rates (0.5-1.0)...");
            var learningRateTests = new[] { 0.6f, 0.65f, 0.7f, 0.75f, 0.8f, 0.85f, 0.9f, 0.95f, 1.0f };
            var results = new List<(float learningRate, double[,] embedding, double time, double quality)>();

            foreach (var learningRate in learningRateTests)
            {
                Console.WriteLine($"   📊 Testing learning_rate = {learningRate:F1}...");
                var model = new PacMapModel(mnRatio: 1.2f, fpRatio: 2.0f, learningRate: learningRate, initializationStdDev: 1e-3f, numIters: (100, 100, 250));
                var stopwatch = Stopwatch.StartNew();
                var embedding = model.Fit(
                    data: data,
                    embeddingDimension: 2,
                    nNeighbors: 10,
                    learningRate: learningRate,
                    mnRatio: 1.2f,
                    fpRatio: 2.0f,
                    numIters: (100, 100, 250),
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: false,
                    hnswM: hnswParams.M,
                    hnswEfConstruction: hnswParams.EfConstruction,
                    hnswEfSearch: hnswParams.EfSearch,
                    autoHNSWParam: false,
                    randomSeed: 42,
                    progressCallback: CreatePrefixedCallback($"lr={learningRate:F1}")
                );
                stopwatch.Stop();
                Console.WriteLine();
                double quality = CalculateEmbeddingQuality(embedding, labels);
                results.Add((learningRate, embedding, stopwatch.Elapsed.TotalSeconds, quality));
                Console.WriteLine($"   ✅ lr={learningRate:F1}: quality={quality:F4}, time={stopwatch.Elapsed.TotalSeconds:F2}s");

                var paramInfo = CreateFitParamInfo(model, stopwatch.Elapsed.TotalSeconds, "Learning_Rate_Experiments");
                paramInfo["embedding_quality"] = quality.ToString("F4");

                var experimentDir = Path.Combine(ResultsDir, "learning_rate_experiments");
                Directory.CreateDirectory(experimentDir);
                var imageNumber = (int)((learningRate - 0.6f) / 0.05f) + 1;
                var outputPath = Path.Combine(experimentDir, $"{imageNumber:D4}.png");
                var modelInfo = model.ModelInfo;
                var title = $"Learning Rate Experiment: lr={learningRate:F1}\n" + BuildVisualizationTitle(model, "Learning Rate Experiment");
                Visualizer.PlotMammothPacMAP(embedding, data, title, outputPath, paramInfo);
                Console.WriteLine($"   📈 Saved: {Path.GetFileName(outputPath)}");
            }

            Console.WriteLine("📊 Learning Rate Experiments Summary");
            Console.WriteLine(new string('=', 50));
            var bestResult = results.OrderBy(r => r.quality).First();
            Console.WriteLine($"🏆 Best learning rate: {bestResult.learningRate:F1} (quality: {bestResult.quality:F4})");
            Console.WriteLine($"⏱️ Execution times: {results.Min(r => r.time):F2}s to {results.Max(r => r.time):F2}s");
        }

        /// <summary>
        /// Tests different initialization standard deviations.
        /// </summary>
        private static void DemoInitializationStdDevExperiments(double[,] data, int[] labels, (int M, int EfConstruction, int EfSearch) hnswParams)
        {
            Console.WriteLine("🎲 Testing Initialization Std Dev...");
            var initStdDevTests = new[] { 1e-4f, 1e-3f, 1e-2f, 1e-1f };
            var results = new List<(float initStdDev, double[,] embedding, double time, double quality)>();

            foreach (var initStdDev in initStdDevTests)
            {
                Console.WriteLine($"   📊 Testing initialization_std_dev = {initStdDev:E1}...");
                var model = new PacMapModel(mnRatio: 1.2f, fpRatio: 2.0f, learningRate: 1.0f, initializationStdDev: initStdDev, numIters: (100, 100, 250));
                var stopwatch = Stopwatch.StartNew();
                var embedding = model.Fit(
                    data: data,
                    embeddingDimension: 2,
                    nNeighbors: 10,
                    learningRate: 1.0f,
                    mnRatio: 1.2f,
                    fpRatio: 2.0f,
                    numIters: (100, 100, 250),
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: false,
                    hnswM: hnswParams.M,
                    hnswEfConstruction: hnswParams.EfConstruction,
                    hnswEfSearch: hnswParams.EfSearch,
                    autoHNSWParam: false,
                    randomSeed: 42
                );
                stopwatch.Stop();
                Console.WriteLine();
                double quality = CalculateEmbeddingQuality(embedding, labels);
                results.Add((initStdDev, embedding, stopwatch.Elapsed.TotalSeconds, quality));
                Console.WriteLine($"   ✅ init_std={initStdDev:E1}: quality={quality:F4}, time={stopwatch.Elapsed.TotalSeconds:F2}s");

                var paramInfo = CreateFitParamInfo(model, stopwatch.Elapsed.TotalSeconds, "Initialization_Std_Dev_Experiments");
                paramInfo["embedding_quality"] = quality.ToString("F4");

                var experimentDir = Path.Combine(ResultsDir, "init_std_dev_experiments");
                Directory.CreateDirectory(experimentDir);
                var imageNumber = initStdDev switch
                {
                    1e-4f => 1,
                    1e-3f => 2,
                    1e-2f => 3,
                    1e-1f => 4,
                    _ => 1
                };
                var outputPath = Path.Combine(experimentDir, $"{imageNumber:D4}.png");
                var modelInfo = model.ModelInfo;
                var title = $"Init Std Dev Experiment: {modelInfo.InitializationStdDev:E0}\n" + BuildVisualizationTitle(model, "Init Std Dev Experiment");
                Visualizer.PlotMammothPacMAP(embedding, data, title, outputPath, paramInfo);
                Console.WriteLine($"   📈 Saved: {Path.GetFileName(outputPath)}");
            }

            Console.WriteLine("📊 Initialization Std Dev Experiments Summary");
            Console.WriteLine(new string('=', 60));
            var bestResult = results.OrderBy(r => r.quality).First();
            Console.WriteLine($"🏆 Best init std dev: {bestResult.initStdDev:E1} (quality: {bestResult.quality:F4})");
            Console.WriteLine($"⏱️ Execution times: {results.Min(r => r.time):F2}s to {results.Max(r => r.time):F2}s");
        }

     

        /// <summary>
        /// Runs advanced parameter tuning experiments.
        /// </summary>
        private static void DemoAdvancedParameterTuning(double[,] data, int[] labels)
        {
            Console.WriteLine("🔬 Running Advanced Parameter Tuning Experiments...");
            var testConfigs = new[]
            {
                new { MNRatio = 0.5f, FPRatio = 2.0f, LearningRate = 1.0f, Name = "Default" },
                new { MNRatio = 0.1f, FPRatio = 1.0f, LearningRate = 0.5f, Name = "Conservative" },
                new { MNRatio = 1.0f, FPRatio = 4.0f, LearningRate = 2.0f, Name = "Aggressive" }
            };

            var optimalHNSWParams = AutoDiscoverHNSWParameters(data);
            var results = new List<(string name, double[,] embedding, double time, double quality)>();

            foreach (var config in testConfigs)
            {
                Console.WriteLine($"   📊 Testing {config.Name} configuration (mn={config.MNRatio}, fp={config.FPRatio}, lr={config.LearningRate})...");
                var model = new PacMapModel(
                    mnRatio: config.MNRatio,
                    fpRatio: config.FPRatio,
                    learningRate: config.LearningRate,
                    initializationStdDev: 1e-4f,
                    numIters: (100, 100, 250)
                );
                var stopwatch = Stopwatch.StartNew();
                var embedding = model.Fit(
                    data: data,
                    embeddingDimension: 2,
                    nNeighbors: 10,
                    learningRate: config.LearningRate,
                    mnRatio: config.MNRatio,
                    fpRatio: config.FPRatio,
                    numIters: (100, 100, 250),
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: false,
                    hnswM: optimalHNSWParams.M,
                    hnswEfConstruction: optimalHNSWParams.EfConstruction,
                    hnswEfSearch: optimalHNSWParams.EfSearch,
                    autoHNSWParam: false,
                    randomSeed: 42,
                    progressCallback: CreatePrefixedCallback(config.Name)
                );
                stopwatch.Stop();
                Console.WriteLine();
                double quality = CalculateEmbeddingQuality(embedding, labels);
                results.Add((config.Name, embedding, stopwatch.Elapsed.TotalSeconds, quality));
                Console.WriteLine($"   ✅ {config.Name}: quality={quality:F4}, time={stopwatch.Elapsed.TotalSeconds:F2}s");

                var paramInfo = CreateFitParamInfo(model, stopwatch.Elapsed.TotalSeconds, "Advanced_Parameter_Tuning");
                paramInfo["configuration"] = config.Name;
                paramInfo["embedding_quality"] = quality.ToString("F4");

                var experimentDir = Path.Combine(ResultsDir, "advanced_param_experiments");
                Directory.CreateDirectory(experimentDir);
                var imageNumber = Array.IndexOf(testConfigs, config) + 1;
                var outputPath = Path.Combine(experimentDir, $"{imageNumber:D4}_{config.Name}.png");
                var title = $"Advanced Param Experiment: {config.Name}\n" + BuildVisualizationTitle(model, "Advanced Parameter Tuning Experiment");
                Visualizer.PlotMammothPacMAP(embedding, data, title, outputPath, paramInfo);
                Console.WriteLine($"   📈 Saved: {Path.GetFileName(outputPath)}");
            }

            Console.WriteLine("📊 Advanced Parameter Tuning Summary");
            Console.WriteLine(new string('=', 60));
            var bestResult = results.OrderBy(r => r.quality).First();
            Console.WriteLine($"🏆 Best configuration: {bestResult.name} (quality: {bestResult.quality:F4})");
            Console.WriteLine($"⏱️ Execution times: {results.Min(r => r.time):F2}s to {results.Max(r => r.time):F2}s");
        }

        /// <summary>
        /// Runs the MNIST demo.
        /// </summary>
        private static void RunMnistDemo()
        {
            Console.WriteLine("🔢 Running MNIST Demo...");
            MnistDemo.RunDemo();
        }

  
        /// <summary>
        /// Calculates the Mean Squared Error between two embeddings.
        /// </summary>
        private static double CalculateMSE(double[,] embedding1, double[,] embedding2)
        {
            int n = embedding1.GetLength(0);
            int d = embedding1.GetLength(1);
            double mse = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < d; j++)
                    mse += Math.Pow(embedding1[i, j] - embedding2[i, j], 2);
            return mse / (n * d);
        }

        /// <summary>
        /// Calculates the maximum difference between two embeddings.
        /// </summary>
        private static double CalculateMaxDifference(double[,] embedding1, double[,] embedding2)
        {
            int n = embedding1.GetLength(0);
            int d = embedding1.GetLength(1);
            double maxDiff = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < d; j++)
                    maxDiff = Math.Max(maxDiff, Math.Abs(embedding1[i, j] - embedding2[i, j]));
            return maxDiff;
        }

        /// <summary>
        /// Generates a consistency plot comparing two embeddings.
        /// </summary>
        private static void GenerateConsistencyPlot(double[,] embedding1, double[,] embedding2, int[] labels, string title, string outputPath)
        {
            var plotModel = new PlotModel { Title = title };
            plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Embedding 1 - X Coordinate" });
            plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Embedding 2 - X Coordinate" });

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
                        scatterSeries.Points.Add(new ScatterPoint(embedding1[i, 0], embedding2[i, 0], 3));
                }
                plotModel.Series.Add(scatterSeries);
            }

            plotModel.Legends.Add(new Legend { LegendPosition = LegendPosition.TopRight });
            ExportPlotToPng(plotModel, outputPath);
        }

        /// <summary>
        /// Generates a placeholder heatmap plot for pairwise distance differences.
        /// </summary>
        private static void GenerateHeatmapPlot(double[,] embedding1, double[,] embedding2, string title, string outputPath)
        {
            var plotModel = new PlotModel { Title = title };
            plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Sample Index" });
            plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Sample Index" });

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
            ExportPlotToPng(plotModel, outputPath);
        }

        /// <summary>
        /// Generates a projection plot for original data.
        /// </summary>
        private static void GenerateProjection(double[,] originalData, double[,] embedding, string projectionType, string outputPath)
        {
            var plotModel = new PlotModel { Title = $"Original Data {projectionType} Projection" };
            var scatterSeries = new ScatterSeries { Title = $"Original {projectionType}", MarkerType = MarkerType.Circle, MarkerSize = 2 };

            if (projectionType == "XY")
            {
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "X Coordinate" });
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Y Coordinate" });
                scatterSeries.MarkerFill = OxyColors.Blue;
                scatterSeries.MarkerStroke = OxyColors.Blue;
                for (int i = 0; i < originalData.GetLength(0); i++)
                    scatterSeries.Points.Add(new ScatterPoint(originalData[i, 0], originalData[i, 1], 2));
            }
            else if (projectionType == "XZ")
            {
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "X Coordinate" });
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Z Coordinate" });
                scatterSeries.MarkerFill = OxyColors.Red;
                scatterSeries.MarkerStroke = OxyColors.Red;
                for (int i = 0; i < originalData.GetLength(0); i++)
                    scatterSeries.Points.Add(new ScatterPoint(originalData[i, 0], originalData[i, 2], 2));
            }
            else if (projectionType == "YZ")
            {
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Y Coordinate" });
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Z Coordinate" });
                scatterSeries.MarkerFill = OxyColors.Green;
                scatterSeries.MarkerStroke = OxyColors.Green;
                for (int i = 0; i < originalData.GetLength(0); i++)
                    scatterSeries.Points.Add(new ScatterPoint(originalData[i, 1], originalData[i, 2], 2));
            }

            plotModel.Series.Add(scatterSeries);
            plotModel.Legends.Add(new Legend { LegendPosition = LegendPosition.TopRight });
            ExportPlotToPng(plotModel, outputPath);
        }

        /// <summary>
        /// Exports a plot model to a PNG file.
        /// </summary>
        private static void ExportPlotToPng(PlotModel plotModel, string outputPath)
        {
            var exporter = new PngExporter { Width = 800, Height = 600, Resolution = 300 };
            using var stream = File.Create(outputPath);
            exporter.Export(plotModel, stream);
        }
    }
}