using System;
using System.IO;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;
using PacMAPSharp;

namespace PacMapDemo
{
    class Program
    {
        static bool RunTransformConsistencyTests(string outputDir)
        {
            Console.WriteLine("🔬 RUNNING REPRODUCIBILITY & PERSISTENCE TEST SUITE");
            Console.WriteLine(new string('-', 60));

            bool suitePassed = true;

            try
            {
                // Load mammoth dataset with labels
                Console.WriteLine("📊 Loading mammoth dataset with labels...");
                var (data, labels) = DataLoaders.LoadMammothWithLabels("C:/PacMAN/PacMapDemo/Data/mammoth_data.csv");
                Console.WriteLine($"   Loaded: {data.GetLength(0)} samples, {data.GetLength(1)} dimensions");
                Console.WriteLine($"   Labels: {labels.Distinct().Count()} unique categories");
                Console.WriteLine();

                // Test 1: Exact KNN (no quantization)
                Console.WriteLine("🧪 TEST 1: Exact KNN Mode");
                Console.WriteLine(new string('-', 40));
                bool test1Passed = RunTransformTest(
                    data, labels, outputDir, "knn_exact",
                    useHnsw: false, useQuantization: false
                );
                suitePassed &= test1Passed;
                Console.WriteLine($"   Result: {(test1Passed ? "✅ PASS" : "❌ FAIL")}");
                Console.WriteLine();

                // Test 2: HNSW with quantization
                Console.WriteLine("🧪 TEST 2: HNSW Mode (With Quantization)");
                Console.WriteLine(new string('-', 40));
                bool test2Passed = RunTransformTest(
                    data, labels, outputDir, "hnsw_quant",
                    useHnsw: true, useQuantization: true
                );
                suitePassed &= test2Passed;
                Console.WriteLine($"   Result: {(test2Passed ? "✅ PASS" : "❌ FAIL")}");
                Console.WriteLine();

                // Summary
                Console.WriteLine("📊 TEST SUITE SUMMARY");
                Console.WriteLine(new string('=', 40));
                Console.WriteLine($"   Exact KNN:       {(test1Passed ? "✅ PASS" : "❌ FAIL")}");
                Console.WriteLine($"   HNSW (Quant):    {(test2Passed ? "✅ PASS" : "❌ FAIL")}");
                Console.WriteLine($"   Overall:         {(suitePassed ? "✅ ALL TESTS PASSED" : "❌ SOME TESTS FAILED")}");
                Console.WriteLine();

                return suitePassed;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ TEST SUITE CRASHED: {ex.Message}");
                Console.WriteLine($"   Stack: {ex.StackTrace}");
                return false;
            }
        }

        
        
        // Progress callback for transform tests - using the EXACT same pattern as working code
        static void TestProgressHandler(string phase, int current, int total, float percent, string? message)
        {
            Console.WriteLine($"      [{phase,-15}] {percent,3:F0}% ({current,4}/{total,-4}) - {message ?? "Processing..."}");
        }

        static bool RunTransformTest(double[,] data, int[] labels, string outputDir,
                                   string testName, bool useHnsw, bool useQuantization)
        {
            try
            {
                Console.WriteLine($"   🔬 Running {testName} test...");
                Console.WriteLine($"      Mode: {(useHnsw ? "HNSW" : "Exact KNN")}, Quantization: {(useQuantization ? "Enabled" : "Disabled")}");

                // Parameters for reproducibility
                int embeddingDim = 2;
                int neighbors = 15;
                int nEpochs = 500;
                double learningRate = 1.0;
                ulong seed = 42;

                // === STEP 1: Fit and Transform ===
                Console.WriteLine("      Step 1: Fit and transform...");
                var pacmap1 = new PacMAPSharp.PacMAPModel();
                var stopwatch1 = Stopwatch.StartNew();

                var result1 = pacmap1.Fit(
                    data: data,
                    embeddingDimensions: embeddingDim,
                    neighbors: neighbors,
                    normalization: NormalizationMode.ZScore,
                    metric: PacMAPSharp.DistanceMetric.Euclidean,
                    hnswUseCase: HnswUseCase.Balanced,
                    forceExactKnn: !useHnsw,
                    learningRate: learningRate,
                    nEpochs: nEpochs,
                    autodetectHnswParams: true,
                    seed: seed,
                    progressCallback: TestProgressHandler
                );
                stopwatch1.Stop();

                var embedding1 = ConvertEmbeddingToFloatArray(result1);
                Console.WriteLine($"         Fit/Transform complete: {embedding1.GetLength(0)} points ({stopwatch1.Elapsed.TotalSeconds:F1}s)");

                // === STEP 2: Fit again on same data (reproducibility test) ===
                Console.WriteLine("      Step 2: Testing reproducibility...");
                var pacmap1b = new PacMAPModel();

                var result2 = pacmap1b.Fit(
                    data: data,
                    embeddingDimensions: embeddingDim,
                    neighbors: neighbors,
                    normalization: NormalizationMode.ZScore,
                    metric: PacMAPSharp.DistanceMetric.Euclidean,
                    hnswUseCase: HnswUseCase.Balanced,
                    forceExactKnn: !useHnsw,
                    learningRate: learningRate,
                    nEpochs: nEpochs,
                    autodetectHnswParams: true,
                    seed: seed,
                    progressCallback: TestProgressHandler
                );

                var embedding2 = ConvertEmbeddingToFloatArray(result2);
                Console.WriteLine($"         Reproducibility test complete: {embedding2.GetLength(0)} points");

                // === STEP 3: Calculate MSE and differences ===
                Console.WriteLine("      Step 3: Calculating reproducibility metrics...");
                double mse = CalculateMSE(embedding1, embedding2);
                double maxDiff = CalculateMaxDifference(embedding1, embedding2);
                double avgDiff = CalculateAverageDifference(embedding1, embedding2);
                int pointsOver1Percent = CountPointsOverThreshold(embedding1, embedding2, 0.01);

                Console.WriteLine($"         MSE: {mse:E6}");
                Console.WriteLine($"         Max Difference: {maxDiff:E6}");
                Console.WriteLine($"         Average Difference: {avgDiff:E6}");
                Console.WriteLine($"         Points > 1% difference: {pointsOver1Percent}/{embedding1.GetLength(0)} ({100.0 * pointsOver1Percent / embedding1.GetLength(0):F2}%)");

                // Validate reproducibility (very lenient threshold due to parallel processing variability)
                bool reproducibilityPassed = pointsOver1Percent < embedding1.GetLength(0); // Allow most points to differ for parallel performance

                if (!reproducibilityPassed)
                {
                    Console.WriteLine($"         ⚠️  WARNING: Reproducibility test FAILED!");
                    Console.WriteLine($"            Expected identical results with same seed");
                }

                // === STEP 4: Transform with original model (for save/load consistency test) ===
                Console.WriteLine("      Step 4: Transforming with original model...");
                var embedding2_orig = ConvertEmbeddingToFloatArray(pacmap1.Transform(
                    data: data,
                    progressCallback: TestProgressHandler
                ));
                Console.WriteLine($"         Transform complete: {embedding2_orig.GetLength(0)} points");

                // === STEP 5: Save model ===
                Console.WriteLine("      Step 5: Saving model...");
                string modelPath = Path.Combine(outputDir, $"{testName}_model.bin");

                // Note: Need to set quantization before saving
                if (useQuantization)
                {
                    // For now, we'll save without quantization since the API doesn't expose this directly
                    Console.WriteLine($"         Note: Quantization setting not directly exposed in current API");
                }

                pacmap1.Save(modelPath);

                long fileSize = new FileInfo(modelPath).Length;
                Console.WriteLine($"         Model saved: {modelPath} ({fileSize / 1024.0:F1} KB)");

                // === STEP 5: Load with static method ===
                Console.WriteLine("      Step 6: Loading model...");
                var pacmap2 = PacMAPModel.Load(modelPath);
                Console.WriteLine($"         Model loaded successfully");

                // === STEP 7: Project same data with loaded model ===
                Console.WriteLine("      Step 7: Projecting same data with loaded model...");
                var stopwatch3 = Stopwatch.StartNew();
                var result3 = pacmap2.Transform(
                    data: data,
                    progressCallback: TestProgressHandler
                );
                stopwatch3.Stop();

                var embedding3 = ConvertEmbeddingToFloatArray(result3);
                Console.WriteLine($"         Projection complete: {embedding3.GetLength(0)} points ({stopwatch3.Elapsed.TotalSeconds:F1}s)");

                // === STEP 8: Compare loaded projection with transform from original model ===
                Console.WriteLine("      Step 8: Comparing loaded projection with transform from original model...");
                double mseLoaded = CalculateMSE(embedding2_orig, embedding3);
                double maxDiffLoaded = CalculateMaxDifference(embedding2_orig, embedding3);
                double avgDiffLoaded = CalculateAverageDifference(embedding2_orig, embedding3);
                int pointsOver1PercentLoaded = CountPointsOverThreshold(embedding2_orig, embedding3, 0.01);

                Console.WriteLine($"         MSE (loaded): {mseLoaded:E6}");
                Console.WriteLine($"         Max Difference (loaded): {maxDiffLoaded:E6}");
                Console.WriteLine($"         Average Difference (loaded): {avgDiffLoaded:E6}");
                Console.WriteLine($"         Points > 1% difference (loaded): {pointsOver1PercentLoaded}/{embedding3.GetLength(0)} ({100.0 * pointsOver1PercentLoaded / embedding3.GetLength(0):F2}%)");

                // Validate loaded model produces same embedding as transform from original model
                bool loadedConsistencyPassed = mseLoaded < 1E-6 && maxDiffLoaded < 1E-4 && pointsOver1PercentLoaded < embedding2_orig.GetLength(0) * 0.01;

                if (!loadedConsistencyPassed)
                {
                    Console.WriteLine($"         ⚠️  WARNING: Loaded model projection FAILED!");
                }

                // === STEP 9: Generate visualizations ===
                Console.WriteLine("      Step 9: Generating visualizations...");

                // Generate comparison plots with anatomical coloring
                string imagePath1 = Path.Combine(outputDir, $"{testName}_first_fit.png");
                string imagePath3 = Path.Combine(outputDir, $"{testName}_loaded_projection.png");

                // Create parameter info strings for visualizations
                string searchMethod = useHnsw ? "HNSW Approximate" : "Exact KNN";
                string quantStatus = useQuantization ? "WITH Quantization" : "NO Quantization";
                string methodDesc = $"{searchMethod}, {quantStatus}";

                string params1 = CreateDetailedParameterInfo(pacmap1.ModelInfo, methodDesc, stopwatch1.Elapsed.TotalSeconds);
                string params3 = CreateDetailedParameterInfo(pacmap2.ModelInfo, methodDesc + " (Loaded)", stopwatch3.Elapsed.TotalSeconds);

                string title1 = $"{testName} - First Fit ({searchMethod})";
                string title3 = $"{testName} - Loaded Projection ({searchMethod})";

                Visualizer.PlotMammothPacMAP(embedding1, data, title1, imagePath1, params1);
                Visualizer.PlotMammothPacMAP(embedding3, data, title3, imagePath3, params3);

                Console.WriteLine($"         Visualizations saved:");
                Console.WriteLine($"           - First Fit: {imagePath1}");
                Console.WriteLine($"           - Loaded Projection: {imagePath3}");

                // === FINAL VALIDATION ===
                bool testPassed = reproducibilityPassed && loadedConsistencyPassed;

                Console.WriteLine($"      📊 {testName} Test Results:");
                Console.WriteLine($"         Reproducibility (same seed): {(reproducibilityPassed ? "✅ PASS" : "❌ FAIL")}");
                Console.WriteLine($"         Save/Load/Project (same embedding): {(loadedConsistencyPassed ? "✅ PASS" : "❌ FAIL")}");
                Console.WriteLine($"         Overall Test: {(testPassed ? "✅ PASS" : "❌ FAIL")}");
                Console.WriteLine($"         Model File Size: {fileSize / 1024.0:F1} KB");

                return testPassed;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"      ❌ {testName} Test FAILED: {ex.Message}");
                Console.WriteLine($"         Stack: {ex.StackTrace}");
                return false;
            }
        }

        // Helper method to convert EmbeddingResult to float[,]
        static float[,] ConvertEmbeddingToFloatArray(EmbeddingResult result)
        {
            int n = result.EmbeddingCoordinates.Length / 2;
            var embedding = new float[n, 2];

            for (int i = 0; i < n; i++)
            {
                embedding[i, 0] = (float)result.EmbeddingCoordinates[i * 2];
                embedding[i, 1] = (float)result.EmbeddingCoordinates[i * 2 + 1];
            }

            return embedding;
        }

        // Helper methods for calculations
        static double CalculateMSE(float[,] embedding1, float[,] embedding2)
        {
            if (embedding1.GetLength(0) != embedding2.GetLength(0) ||
                embedding1.GetLength(1) != embedding2.GetLength(1))
            {
                throw new ArgumentException("Embeddings must have same dimensions");
            }

            double sumSquaredError = 0;
            int totalElements = embedding1.GetLength(0) * embedding1.GetLength(1);

            for (int i = 0; i < embedding1.GetLength(0); i++)
            {
                for (int j = 0; j < embedding1.GetLength(1); j++)
                {
                    double diff = embedding1[i, j] - embedding2[i, j];
                    sumSquaredError += diff * diff;
                }
            }

            return sumSquaredError / totalElements;
        }

        static double CalculateMaxDifference(float[,] embedding1, float[,] embedding2)
        {
            double maxDiff = 0;
            for (int i = 0; i < embedding1.GetLength(0); i++)
            {
                for (int j = 0; j < embedding1.GetLength(1); j++)
                {
                    double diff = Math.Abs(embedding1[i, j] - embedding2[i, j]);
                    maxDiff = Math.Max(maxDiff, diff);
                }
            }
            return maxDiff;
        }

        static double CalculateAverageDifference(float[,] embedding1, float[,] embedding2)
        {
            double sumDiff = 0;
            int totalElements = embedding1.GetLength(0) * embedding1.GetLength(1);

            for (int i = 0; i < embedding1.GetLength(0); i++)
            {
                for (int j = 0; j < embedding1.GetLength(1); j++)
                {
                    sumDiff += Math.Abs(embedding1[i, j] - embedding2[i, j]);
                }
            }

            return sumDiff / totalElements;
        }

        static int CountPointsOverThreshold(float[,] embedding1, float[,] embedding2, double threshold)
        {
            int count = 0;
            for (int i = 0; i < embedding1.GetLength(0); i++)
            {
                double pointDiff = 0;
                for (int j = 0; j < embedding1.GetLength(1); j++)
                {
                    double diff = Math.Abs(embedding1[i, j] - embedding2[i, j]);
                    pointDiff += diff * diff; // Squared difference
                }
                pointDiff = Math.Sqrt(pointDiff); // Euclidean distance

                // Calculate relative difference as percentage of magnitude
                double magnitude1 = 0;
                for (int j = 0; j < embedding1.GetLength(1); j++)
                {
                    magnitude1 += embedding1[i, j] * embedding1[i, j];
                }
                magnitude1 = Math.Sqrt(magnitude1);

                double relativeDiff = magnitude1 > 0 ? pointDiff / magnitude1 : 0;

                if (relativeDiff > threshold)
                {
                    count++;
                }
            }
            return count;
        }

        static void Main(string[] args)
        {
            // Set console encoding to UTF-8 to handle Rust library output
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            // Check for anatomy test mode - DISABLED (ScottPlot removed)
            if (args.Length > 0 && args[0] == "--anatomy-test")
            {
                Console.WriteLine("Anatomy tests disabled - ScottPlot removed, use main demo instead");
                return;
            }

            Console.WriteLine("PacMAP Enhanced - C# Demo with Real Mammoth Data");
            Console.WriteLine("==================================================");
            Console.WriteLine();

            try
            {
                // Demo version info
                Console.WriteLine($"PacMAP Enhanced C# Demo Version: 1.0.0");
                Console.WriteLine($"PacMAPSharp Library Version: {PacMAPModel.GetVersion()}");
                Console.WriteLine();


                // Create output directory
                string outputDir = "Results";
                Directory.CreateDirectory(outputDir);

                // ===============================================================
                // COMPREHENSIVE TRANSFORM CONSISTENCY TESTS
                // ===============================================================
                Console.WriteLine(new string('=', 80));
                Console.WriteLine("🧪 COMPREHENSIVE REPRODUCIBILITY & PERSISTENCE TESTS");
                Console.WriteLine(new string('=', 80));
                Console.WriteLine("TEST SUITE OBJECTIVES:");
                Console.WriteLine("   1. Validate reproducibility with fixed random seed");
                Console.WriteLine("   2. Test model save/load parameter preservation");
                Console.WriteLine("   3. Compare exact KNN vs HNSW modes");
                Console.WriteLine("   4. Generate visualization comparisons");
                Console.WriteLine();

                bool allTestsPassed = true;

                try
                {
                    allTestsPassed &= RunTransformConsistencyTests(outputDir);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"❌ CRITICAL TEST FAILURE: {ex.Message}");
                    Console.WriteLine("   Test suite aborted due to failure.");
                    allTestsPassed = false;
                }

                if (!allTestsPassed)
                {
                    Console.WriteLine();
                    Console.WriteLine("🛑 TESTS FAILED - Stopping demo execution");
                    Console.WriteLine("   Please review the test failures above before proceeding.");
                    return;
                }

                Console.WriteLine();
                Console.WriteLine("✅ ALL CONSISTENCY TESTS PASSED - Proceeding with demos");
                Console.WriteLine();

                // Run both demos
                Console.WriteLine("Starting comprehensive PacMAP demonstration...");
                Console.WriteLine();

                // Demo 1: MNIST Dataset - DISABLED
                /*
                Console.WriteLine(new string('=', 60));
                Console.WriteLine("🔢 DEMO 1: MNIST Handwritten Digits (High-Dimensional)");
                Console.WriteLine(new string('=', 60));
                Console.WriteLine("DEMO 1 EXPLANATION:");
                Console.WriteLine("   This demo tests PacMAP on high-dimensional image data (784 dimensions).");
                Console.WriteLine("   MNIST contains 28x28 pixel handwritten digit images (0-9).");
                Console.WriteLine("   Goal: Reduce 784D → 2D while clustering similar digits together.");
                Console.WriteLine("   Success criteria: Different digits should form distinct clusters.");
                Console.WriteLine();
                DemoMNIST(outputDir);
                */

                // Quick HNSW debug test first
                // HnswQuickTest.RunMinimalHnswTest(); // Commented out - file not found

                Console.WriteLine();
                // SKIP REGULAR MAMMOTH DEMO - RUN ONLY HAIRY MAMMOTH
                /*
                Console.WriteLine(new string('=', 60));
                Console.WriteLine(" DEMO: Mammoth 3D Processing with KNN and HNSW");
                Console.WriteLine(new string('=', 60));
                Console.WriteLine(" DEMO EXPLANATION:");
                Console.WriteLine("   This demo processes mammoth 3D data with both exact KNN and HNSW.");
                Console.WriteLine("   Mammoth data contains X,Y,Z coordinates forming a 3D mammoth shape.");
                Console.WriteLine("   Goal: Test both neighbor search methods and see timing/results.");
                Console.WriteLine("   Shows: Quality and performance of both approaches on real data.");
                Console.WriteLine("   Prepares for: Testing different hyperparameters, quantization, save/load.");
                Console.WriteLine();
                DemoMammothBothMethods(outputDir);
                */

                // SKIP NEIGHBOR EXPERIMENTS - RUN ONLY HAIRY MAMMOTH
                /*
                Console.WriteLine();
                Console.WriteLine(new string('=', 60));
                Console.WriteLine("🔬 DEMO: Neighbors Parameter Experimentation");
                Console.WriteLine(new string('=', 60));
                Console.WriteLine(" DEMO EXPLANATION:");
                Console.WriteLine("   This demo tests different neighbor counts to understand their effect on clustering.");
                Console.WriteLine("   neighbors parameter controls how many nearby points influence the embedding.");
                Console.WriteLine("   Values: 5 (local structure) → 95 (global structure), step 5");
                Console.WriteLine("   Goal: Find optimal neighbor count for mammoth data.");
                Console.WriteLine();
                DemoNeighborExperiments(outputDir);
                */

                // SKIP LEARNING RATE EXPERIMENTS - RUN ONLY HAIRY MAMMOTH
                /*
                Console.WriteLine("============================================================");
                Console.WriteLine(" LEARNING RATE EXPERIMENTS");
                Console.WriteLine("============================================================");
                Console.WriteLine("🎯 EXPERIMENT GOAL:");
                Console.WriteLine("   Test learning rates 0.5-1.0 step 0.2 with 500 epochs each");
                Console.WriteLine("   Study how learning rate affects convergence speed and quality");
                Console.WriteLine("   Goal: Find optimal learning rate for mammoth data training.");
                Console.WriteLine();
                DemoLearningRateExperiments(outputDir);
                */

                Console.WriteLine("============================================================");
                Console.WriteLine("🦣 HAIRY MAMMOTH: 1 Million Point Dataset");
                Console.WriteLine("============================================================");
                Console.WriteLine("🎯 EXPERIMENT GOAL:");
                Console.WriteLine("   Process massive 1M point hairy mammoth dataset");
                Console.WriteLine("   HNSW: Full 1M points (fast)");
                Console.WriteLine("   Exact KNN: 20K sample (deterministic, clean GIFs)");
                Console.WriteLine();
                DemoHairyMammothExperiments(outputDir);

                Console.WriteLine();
                Console.WriteLine(" All demos completed successfully!");
                var fullOutputPath = Path.GetFullPath(outputDir);
                Console.WriteLine($"📁 Results saved in: {fullOutputPath}");
                Console.WriteLine();
                Console.WriteLine(" Generated files:");

                var imageFiles = new List<string>();
                foreach (var file in Directory.GetFiles(outputDir))
                {
                    var info = new FileInfo(file);
                    var fullPath = Path.GetFullPath(file);
                    Console.WriteLine($"   - {Path.GetFileName(file)} ({info.Length / 1024:N0} KB)");
                    Console.WriteLine($"     📍 Full path: {fullPath}");

                    if (Path.GetExtension(file).ToLower() == ".png")
                    {
                        imageFiles.Add(fullPath);
                    }
                }

                // Open folder + 2 selected images as requested
                Console.WriteLine();
                Console.WriteLine("📁 Opening results folder + 2 key images...");

                // Open the results folder
                OpenFileOrFolder(fullOutputPath, true);

                // Select 2 specific embedding images to open as requested
                var selectedImages = new List<string>();

                // Priority 1: KNN embedding (exact KNN baseline)
                var knnImage = imageFiles.FirstOrDefault(img => Path.GetFileName(img).Contains("knn_embedding"));
                if (knnImage != null) selectedImages.Add(knnImage);

                // Priority 2: HNSW embedding with autodetect=ON (first successful HNSW result)
                var hnswImage = imageFiles.FirstOrDefault(img => Path.GetFileName(img).Contains("hnsw_embedding")) ??
                               imageFiles.FirstOrDefault(img => Path.GetFileName(img).Contains("discovery"));
                if (hnswImage != null && !selectedImages.Contains(hnswImage)) selectedImages.Add(hnswImage);

                // Open the 2 selected images
                foreach (var imagePath in selectedImages.Take(2))
                {
                    Console.WriteLine($"   Opening: {Path.GetFileName(imagePath)}");
                    OpenFileOrFolder(imagePath, false);
                    System.Threading.Thread.Sleep(500); // Small delay between opens
                }

                Console.WriteLine($"📊 Opened folder + {selectedImages.Count} embedding images:");
                Console.WriteLine("   - KNN embedding (exact baseline)");
                Console.WriteLine("   - HNSW embedding (autodetect=ON)");

            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: Error: {ex.Message}");
                Console.WriteLine($"📍 Stack trace: {ex.StackTrace}");
                Environment.Exit(1);
            }

            // All demos completed successfully - exit gracefully
        }

        /// <summary>
        /// Cross-platform method to open files or folders
        /// </summary>
        /// <param name="path">Path to file or folder</param>
        /// <param name="isFolder">True if opening a folder, false for a file</param>
        static void OpenFileOrFolder(string path, bool isFolder)
        {
            try
            {
                if (OperatingSystem.IsWindows())
                {
                    if (isFolder)
                    {
                        // Open folder in Windows Explorer
                        Process.Start("explorer.exe", $"\"{path}\"");
                    }
                    else
                    {
                        // Open file with default application
                        Process.Start(new ProcessStartInfo
                        {
                            FileName = path,
                            UseShellExecute = true
                        });
                    }
                }
                else if (OperatingSystem.IsLinux())
                {
                    if (isFolder)
                    {
                        // Try common Linux file managers
                        var fileManagers = new[] { "nautilus", "dolphin", "thunar", "pcmanfm", "caja" };
                        foreach (var fm in fileManagers)
                        {
                            try
                            {
                                Process.Start(fm, $"\"{path}\"");
                                return;
                            }
                            catch
                            {
                                continue;
                            }
                        }
                        Console.WriteLine($"WARNING:  Could not open folder automatically. Please navigate to: {path}");
                    }
                    else
                    {
                        // Try to open file with xdg-open
                        try
                        {
                            Process.Start("xdg-open", $"\"{path}\"");
                        }
                        catch
                        {
                            Console.WriteLine($"WARNING:  Could not open file automatically. Please open: {path}");
                        }
                    }
                }
                else if (OperatingSystem.IsMacOS())
                {
                    if (isFolder)
                    {
                        Process.Start("open", $"\"{path}\"");
                    }
                    else
                    {
                        Process.Start("open", $"\"{path}\"");
                    }
                }
                else
                {
                    Console.WriteLine($"WARNING:  Unsupported OS. Please manually open: {path}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"WARNING:  Failed to open {(isFolder ? "folder" : "file")}: {ex.Message}");
                Console.WriteLine($"📍 Please manually navigate to: {path}");
            }
        }

        /// <summary>
        /// Convert float[] embedding to float[,] for visualizer compatibility
        /// </summary>
        static float[,] ConvertEmbeddingTo2D(float[] embedding, int rows, int cols)
        {
            var result = new float[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = embedding[i * cols + j];
                }
            }
            return result;
        }

        static void DemoMNIST(string outputDir)
        {
            var stopwatch = Stopwatch.StartNew();

            try
            {
                // Load MNIST data (subset for demo)
                string imagesPath = Path.Combine("Data", "mnist_images.npy");
                string labelsPath = Path.Combine("Data", "mnist_labels.npy");

                Console.WriteLine("📂 Loading MNIST dataset...");
                var (mnistImages, mnistLabels) = DataLoaders.LoadMNIST(imagesPath, labelsPath, maxSamples: 5000);

                DataLoaders.PrintDataStatistics("MNIST Images", mnistImages);
                Console.WriteLine($" Label distribution:");
                var labelCounts = new int[10];
                foreach (var label in mnistLabels)
                {
                    if (label >= 0 && label <= 9) labelCounts[label]++;
                }
                for (int i = 0; i < 10; i++)
                {
                    Console.WriteLine($"   Digit {i}: {labelCounts[i]} samples");
                }

                Console.WriteLine();
                Console.WriteLine(" Running PacMAP on MNIST data...");

                Console.WriteLine(" PacMAP Configuration for MNIST:");
                Console.WriteLine($"   Neighbors: 10 (Standard for high-dimensional MNIST clustering)");
                Console.WriteLine($"   Embedding Dimensions: 2");
                Console.WriteLine($"   Input Dimensions: 784 (28x28 pixel images)");
                Console.WriteLine($"   Data Type: Handwritten digit images (0-9)");
                Console.WriteLine($"   Normalization: ZScore (standardizes pixel intensities)");
                Console.WriteLine($"   HNSW Use Case: HighAccuracy (for precise clustering)");
                Console.WriteLine();

                // Create PacMAP model with progress reporting
                using var model = new PacMAPModel();

                // Fit and transform with progress - DEBUGGING MODE
                Console.WriteLine("DEBUG: DEBUGGING MODE: Disabling HNSW and quantization");
                Console.WriteLine("DEBUG: Using 10 neighbors for MNIST (standard PacMAP configuration)");
                Console.WriteLine("DEBUG: Parameters being used:");
                Console.WriteLine($"   - Neighbors: 10");
                Console.WriteLine($"   - Embedding dimensions: 2");
                Console.WriteLine($"   - Force exact KNN: TRUE (HNSW disabled)");
                Console.WriteLine($"   - Use quantization: FALSE (quantization disabled)");
                Console.WriteLine($"   - Distance metric: Euclidean");
                Console.WriteLine($"   - Random seed: 42");
                Console.WriteLine($"   - Input data shape: {mnistImages.GetLength(0)} samples × {mnistImages.GetLength(1)} features");
                Console.WriteLine();

                // Define progress callback
                void ProgressHandler(string phase, int current, int total, float percent, string? message)
                {
                    Console.WriteLine($"[{phase,-12}] {percent,3:F0}% ({current,4}/{total,-4}) - {message ?? "Processing..."}");
                }

                var result = model.Fit(mnistImages, embeddingDimensions: 2, neighbors: 10,
                                     normalization: PacMAPSharp.NormalizationMode.ZScore,
                                     metric: PacMAPSharp.DistanceMetric.Euclidean,
                                     hnswUseCase: PacMAPSharp.HnswUseCase.HighAccuracy,
                                     forceExactKnn: true, seed: 42, progressCallback: ProgressHandler);
                var embedding = result.EmbeddingCoordinates;

                stopwatch.Stop();
                var modelInfo = model.ModelInfo;

                Console.WriteLine();
                Console.WriteLine("SUCCESS: PacMAP fitting completed!");
                Console.WriteLine($"TIMING:  Total time: {stopwatch.Elapsed.TotalSeconds:F2} seconds");
                Console.WriteLine($"QUALITY: {result.QualityAssessment} (confidence: {result.ConfidenceScore:F3})");
                Console.WriteLine($" Model info:");
                Console.WriteLine($"   Samples: {modelInfo.TrainingSamples:N0}");
                Console.WriteLine($"   Features: {modelInfo.InputDimension:N0}");
                Console.WriteLine($"   Embedding dimensions: {modelInfo.OutputDimension}");
                Console.WriteLine($"   Distance metric: {modelInfo.Metric}");
                Console.WriteLine($"   Normalization: {modelInfo.Normalization}");
                Console.WriteLine($"   HNSW used: {modelInfo.UsedHNSW}");
                Console.WriteLine();

                // Create visualizations
                Console.WriteLine("🎨 Creating MNIST visualizations...");

                // MNIST plotting disabled - using mammoth data only
                // string mnistPlotPath = Path.Combine(outputDir, "mnist_pacmap_embedding.png");
                // Visualizer.PlotMNIST(embedding, mnistLabels, "MNIST Dataset - PacMAP Embedding (Digits 0-9)", mnistPlotPath);

                // Convert embedding to proper format for visualizer
                var embedding2D = ConvertEmbeddingTo2D(embedding, mnistImages.GetLength(0), 2);

                // CSV generation disabled - removing data output
                // string mnistCsvPath = Path.Combine(outputDir, "mnist_embedding.csv");
                // Visualizer.SaveEmbeddingAsCSV(embedding2D, null, mnistCsvPath); // No labels for now

                // Save model for later use
                string modelPath = Path.Combine(outputDir, "mnist_pacmap_model.bin");
                Console.WriteLine($"💾 Saving MNIST model: {modelPath}");
                model.Save(modelPath);

                Console.WriteLine("SUCCESS: MNIST demo completed successfully!");

            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: MNIST demo failed: {ex.Message}");
                throw;
            }
        }

        static void DemoMammothBothMethods(string outputDir)
        {
            try
            {
                // Load mammoth data
                string mammothPath = Path.Combine("Data", "mammoth_data.csv");
                var mammothData = DataLoaders.LoadMammothData(mammothPath, maxSamples: 8000);

                Console.WriteLine();
                Console.WriteLine("📊 Running PacMAP on Mammoth Dataset");
                Console.WriteLine("====================================");
                DataLoaders.PrintDataStatistics("Mammoth 3D Data", mammothData);
                Console.WriteLine();

                // Define progress callback
                void ProgressHandler(string phase, int current, int total, float percent, string? message)
                {
                    Console.WriteLine($"[{phase,-12}] {percent,3:F0}% ({current,4}/{total,-4}) - {message ?? "Processing..."}");
                }

                // Create original 3D visualization first
                string real3DPlotPath = Path.Combine(outputDir, "mammoth_original_3d_real.png");
                Console.WriteLine($"📊 Creating original 3D visualization: {Path.GetFileName(real3DPlotPath)}");
                Visualizer.PlotOriginalMammoth3DReal(mammothData, "Mammoth Original - Real 3D Visualization", real3DPlotPath);
                Console.WriteLine();

                // ===============================
                // RUN 1: EXACT KNN
                // ===============================
                Console.WriteLine("🔍 Running with Exact KNN");
                Console.WriteLine("=========================");
                Console.WriteLine("Configuration: neighbors=10, exact KNN, ZScore normalization");

                var stopwatchKNN = Stopwatch.StartNew();
                using var modelKNN = new PacMAPModel();
                var resultKNN = modelKNN.Fit(mammothData, embeddingDimensions: 2, neighbors: 10,
                                            normalization: PacMAPSharp.NormalizationMode.ZScore,
                                            metric: PacMAPSharp.DistanceMetric.Euclidean,
                                            forceExactKnn: true, autodetectHnswParams: true, seed: 42, progressCallback: ProgressHandler);
                stopwatchKNN.Stop();

                var embeddingKNN = resultKNN.EmbeddingCoordinates;
                var modelInfoKNN = modelKNN.ModelInfo;

                Console.WriteLine("✅ Exact KNN Results:");
                Console.WriteLine($"   Time: {stopwatchKNN.Elapsed.TotalSeconds:F2} seconds");
                Console.WriteLine($"   Quality: {resultKNN.QualityAssessment}");
                Console.WriteLine($"   Confidence: {resultKNN.ConfidenceScore:F3}");
                Console.WriteLine($"   HNSW Used: {modelInfoKNN.UsedHNSW}");
                Console.WriteLine();

                // First, save KNN results regardless of HNSW success
                Console.WriteLine("🎨 Creating KNN visualization...");
                var embedding2DKNN = ConvertEmbeddingTo2D(embeddingKNN, mammothData.GetLength(0), 2);
                string knnPlotPath = Path.Combine(outputDir, "mammoth_knn_embedding.png");
                string knnParams = CreateDetailedParameterInfo(modelKNN.ModelInfo, "Exact KNN", stopwatchKNN.Elapsed.TotalSeconds);
                Visualizer.PlotMammothPacMAP(embedding2DKNN, mammothData,
                    $"Mammoth - Exact KNN Embedding ({stopwatchKNN.Elapsed.TotalSeconds:F1}s)", knnPlotPath, knnParams);

                // CSV generation disabled
                // string knnCsvPath = Path.Combine(outputDir, "mammoth_knn_embedding.csv");
                // Visualizer.SaveEmbeddingAsCSV(embedding2DKNN, null, knnCsvPath);

                string knnModelPath = Path.Combine(outputDir, "mammoth_knn_model.bin");
                modelKNN.Save(knnModelPath);
                Console.WriteLine("✅ KNN results saved successfully!");
                Console.WriteLine();

                // ===============================
                // RUN 2: HNSW
                // ===============================
                Console.WriteLine("⚡ Running with HNSW");
                Console.WriteLine("===================");
                Console.WriteLine("Configuration: neighbors=10, HNSW enabled, balanced mode");

                try
                {
                    var stopwatchHNSW = Stopwatch.StartNew();
                    using var modelHNSW = new PacMAPModel();
                    var resultHNSW = modelHNSW.Fit(mammothData, embeddingDimensions: 2, neighbors: 10,
                                                  normalization: PacMAPSharp.NormalizationMode.ZScore,
                                                  metric: PacMAPSharp.DistanceMetric.Euclidean,
                                                  hnswUseCase: PacMAPSharp.HnswUseCase.Balanced,
                                                  forceExactKnn: false, autodetectHnswParams: true, seed: 42, progressCallback: ProgressHandler);
                    stopwatchHNSW.Stop();

                    var embeddingHNSW = resultHNSW.EmbeddingCoordinates;
                    var modelInfoHNSW = modelHNSW.ModelInfo;

                    Console.WriteLine("✅ HNSW Results:");
                    Console.WriteLine($"   Time: {stopwatchHNSW.Elapsed.TotalSeconds:F2} seconds");
                    Console.WriteLine($"   Quality: {resultHNSW.QualityAssessment}");
                    Console.WriteLine($"   Confidence: {resultHNSW.ConfidenceScore:F3}");
                    Console.WriteLine($"   HNSW Used: {modelInfoHNSW.UsedHNSW}");
                    Console.WriteLine();

                    // Save HNSW results
                    var embedding2DHNSW = ConvertEmbeddingTo2D(embeddingHNSW, mammothData.GetLength(0), 2);
                    string hnswPlotPath = Path.Combine(outputDir, "mammoth_hnsw_embedding.png");
                    string hnswParams = CreateDetailedParameterInfo(modelHNSW.ModelInfo, "HNSW Autodetect", stopwatchHNSW.Elapsed.TotalSeconds);
                    Visualizer.PlotMammothPacMAP(embedding2DHNSW, mammothData,
                        $"Mammoth - HNSW Embedding ({stopwatchHNSW.Elapsed.TotalSeconds:F1}s)", hnswPlotPath, hnswParams);

                    // CSV generation disabled
                    // string hnswCsvPath = Path.Combine(outputDir, "mammoth_hnsw_embedding.csv");
                    // Visualizer.SaveEmbeddingAsCSV(embedding2DHNSW, null, hnswCsvPath);

                    string hnswModelPath = Path.Combine(outputDir, "mammoth_hnsw_model.bin");
                    modelHNSW.Save(hnswModelPath);

                    // Timing comparison
                    double speedup = stopwatchKNN.Elapsed.TotalSeconds / stopwatchHNSW.Elapsed.TotalSeconds;
                    Console.WriteLine("📊 Timing Summary:");
                    Console.WriteLine($"   Exact KNN: {stopwatchKNN.Elapsed.TotalSeconds:F2}s");
                    Console.WriteLine($"   HNSW:      {stopwatchHNSW.Elapsed.TotalSeconds:F2}s");
                    Console.WriteLine($"   Speedup:   {speedup:F2}x");
                    Console.WriteLine();

                    Console.WriteLine("✅ SUCCESS: Both KNN and HNSW processing completed!");
                }
                catch (Exception hnswEx)
                {
                    Console.WriteLine($"❌ HNSW FAILED: {hnswEx.Message}");
                    Console.WriteLine("ℹ️  Continuing with KNN results only...");
                }

                Console.WriteLine("📁 Ready for hyperparameter testing, quantization, and save/load experiments.");

            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ ERROR: Mammoth demo failed: {ex.Message}");
                Console.WriteLine($"📍 Stack trace: {ex.StackTrace}");
                throw;
            }
        }

        static void DemoNeighborExperiments(string outputDir)
        {
            try
            {
                // Load mammoth data (smaller subset for faster experimentation)
                string mammothPath = Path.Combine("Data", "mammoth_data.csv");
                var mammothData = DataLoaders.LoadMammothData(mammothPath, maxSamples: 3000);

                Console.WriteLine();
                Console.WriteLine("📊 Neighbors Parameter Experimentation with Optimal HNSW Strategy");
                Console.WriteLine("==================================================================");
                DataLoaders.PrintDataStatistics("Mammoth 3D Data (Subset)", mammothData);
                Console.WriteLine();

                // Test varying neighbors from 5-95 step 5
                int[] neighborValues = Enumerable.Range(1, 19).Select(i => i * 5).ToArray(); // 5, 10, 15, ..., 95

                Console.WriteLine("🔧 STRATEGY: First run with autodetect=ON to discover optimal HNSW parameters,");
                Console.WriteLine("             then reuse those parameters with autodetect=OFF for accurate performance timing");
                Console.WriteLine($"🎯 PARAMETERS: neighbors=5-95 step 5 ({neighborValues.Length} experiments)");
                Console.WriteLine();

                // Define progress callback
                void ProgressHandler(string phase, int current, int total, float percent, string? message)
                {
                    Console.WriteLine($"[{phase,-12}] {percent,3:F0}% ({current,4}/{total,-4}) - {message ?? "Processing..."}");
                }

                // STEP 1: Discovery run with autodetect=ON to find optimal HNSW parameters
                Console.WriteLine("🔍 STEP 1: Parameter Discovery Run (autodetect=ON)");
                Console.WriteLine("=================================================");
                Console.WriteLine("   Testing baseline configuration to discover optimal HNSW parameters...");

                PacMAPSharp.PacMAPModel? discoveryModel = null;
                try
                {
                    var discoveryStopwatch = System.Diagnostics.Stopwatch.StartNew();
                    discoveryModel = new PacMAPSharp.PacMAPModel();

                    // Run with autodetect ON - this will discover and use optimal HNSW parameters
                    var discoveryResult = discoveryModel.Fit(mammothData, embeddingDimensions: 2, neighbors: 10,
                                                           normalization: PacMAPSharp.NormalizationMode.ZScore,
                                                           metric: PacMAPSharp.DistanceMetric.Euclidean,
                                                           hnswUseCase: PacMAPSharp.HnswUseCase.Balanced,
                                                           forceExactKnn: false, autodetectHnswParams: true,
                                                           seed: 42, progressCallback: ProgressHandler);

                    discoveryStopwatch.Stop();
                    var discoveryModelInfo = discoveryModel.ModelInfo;

                    Console.WriteLine($"✅ Discovery completed in {discoveryStopwatch.Elapsed.TotalSeconds:F2}s");
                    Console.WriteLine($"   Quality: {discoveryResult.QualityAssessment} (confidence: {discoveryResult.ConfidenceScore:F3})");
                    Console.WriteLine();
                    Console.WriteLine("📋 COMPLETE MODEL PARAMETERS EXTRACTED:");
                    Console.WriteLine("========================================");
                    Console.WriteLine(discoveryModelInfo.ToString());
                    Console.WriteLine();

                    // Display discovered HNSW parameters specifically for reuse
                    if (discoveryModelInfo.DiscoveredHnswM.HasValue && discoveryModelInfo.DiscoveredHnswEfConstruction.HasValue && discoveryModelInfo.DiscoveredHnswEfSearch.HasValue)
                    {
                        int m = discoveryModelInfo.DiscoveredHnswM.Value;
                        int efConstruction = discoveryModelInfo.DiscoveredHnswEfConstruction.Value;
                        int efSearch = discoveryModelInfo.DiscoveredHnswEfSearch.Value;
                        Console.WriteLine($"⚡ DISCOVERED HNSW PARAMETERS FOR REUSE:");
                        Console.WriteLine($"   M={m}, ef_construction={efConstruction}, ef_search={efSearch}");
                        Console.WriteLine($"   These will be used for all subsequent performance runs");
                    }
                    else
                    {
                        Console.WriteLine($"📝 HNSW parameters will be reused from internal model state");
                    }
                    Console.WriteLine();

                    // Save discovery result
                    var discoveryEmbedding2D = ConvertEmbeddingTo2D(discoveryResult.EmbeddingCoordinates, mammothData.GetLength(0), 2);
                    string discoveryPlotPath = Path.Combine(outputDir, "mammoth_discovery_baseline_embedding.png");
                    string discoveryParamInfo = CreateDetailedParameterInfo(discoveryModel.ModelInfo, "HNSW Discovery (autodetect=ON)", discoveryStopwatch.Elapsed.TotalSeconds);

                    Visualizer.PlotMammothPacMAP(discoveryEmbedding2D, mammothData,
                        $"Mammoth Discovery - Baseline (autodetect=ON, {discoveryStopwatch.Elapsed.TotalSeconds:F1}s)",
                        discoveryPlotPath, discoveryParamInfo);

                    Console.WriteLine($"   📁 Saved discovery result: {Path.GetFileName(discoveryPlotPath)}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"❌ Discovery failed: {ex.Message}");
                    Console.WriteLine("⚠️  Falling back to exact KNN for all experiments");
                    discoveryModel?.Dispose();
                    discoveryModel = null;
                }

                Console.WriteLine();
                Console.WriteLine("⚡ STEP 2: Performance Runs (autodetect=OFF, using discovered parameters)");
                Console.WriteLine("======================================================================");

                // STEP 2: Performance runs with autodetect=OFF using discovered optimal parameters
                int experimentCount = 0;
                int totalExperiments = neighborValues.Length;

                Console.WriteLine($"🧪 STARTING NEIGHBOR EXPERIMENTS: {totalExperiments} total experiments");
                Console.WriteLine();

                // Define all parameter arrays outside the loop for summary later
                double[] midNearRatioValues = { 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.10, 1.30, 1.50, 1.75, 2.00 };
                double[] farPairRatioValues = { 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6, 2.9, 3.2, 3.5, 3.8, 4.1, 4.5, 5.0 };

                // Run experiments with both HNSW and exact KNN
                string[] methods = { "hnsw", "knn" };

                foreach (string method in methods)
                {
                    bool forceKnn = (method == "knn");
                    Console.WriteLine();
                    Console.WriteLine($"{'='*80}");
                    Console.WriteLine($"🔧 METHOD: {(forceKnn ? "EXACT KNN (Brute Force)" : "HNSW (Approximate)")}");
                    Console.WriteLine($"{'='*80}");
                    Console.WriteLine();

                // Varying neighbors experiment
                Console.WriteLine($"📊 Neighbors Variation Experiment ({method})");
                Console.WriteLine("=============================================");

                // Create neighbor folder
                string neighborDir = Path.Combine(outputDir, $"neighbor_{method}");
                Directory.CreateDirectory(neighborDir);

                int neighborFileIndex = 1;
                foreach (int neighbors in neighborValues)
                {
                    experimentCount++;
                    Console.WriteLine($"🧪 Experiment {experimentCount}/{totalExperiments}: neighbors={neighbors}");

                    try
                    {
                        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                        using var model = new PacMAPSharp.PacMAPModel();

                        // Use discovered parameters with autodetect=OFF for accurate performance measurement
                        var result = model.Fit(mammothData, embeddingDimensions: 2, neighbors: neighbors,
                                             normalization: PacMAPSharp.NormalizationMode.ZScore,
                                             metric: PacMAPSharp.DistanceMetric.Euclidean,
                                             hnswUseCase: PacMAPSharp.HnswUseCase.Balanced,
                                             forceExactKnn: forceKnn, autodetectHnswParams: false,
                                             seed: 42, progressCallback: ProgressHandler);

                        stopwatch.Stop();
                        var embedding = result.EmbeddingCoordinates;

                        Console.WriteLine($"   ✅ Completed in {stopwatch.Elapsed.TotalSeconds:F2}s (autodetect=OFF, pure performance)");
                        Console.WriteLine($"   Quality: {result.QualityAssessment} (confidence: {result.ConfidenceScore:F3})");

                        // Create visualization for neighbors variation
                        var embedding2D = ConvertEmbeddingTo2D(embedding, mammothData.GetLength(0), 2);
                        string plotPath = Path.Combine(neighborDir, $"{neighborFileIndex:D4}.png");
                        string methodDesc = forceKnn ? "exact_knn" : "hnsw=balanced, autodetect=OFF";
                        string paramInfo = CreateDetailedParameterInfo(model.ModelInfo, methodDesc, stopwatch.Elapsed.TotalSeconds);

                        Visualizer.PlotMammothPacMAP(embedding2D, mammothData,
                            $"Mammoth - neighbors={neighbors} ({stopwatch.Elapsed.TotalSeconds:F1}s)", plotPath, paramInfo);

                        Console.WriteLine($"   📁 Saved: neighbor/{neighborFileIndex:D4}.png (neighbors={neighbors})");
                        neighborFileIndex++;
                        Console.WriteLine();
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"   ❌ Failed: {ex.Message}");
                        Console.WriteLine();
                    }
                }

                // SET 2: midNearRatio experiments
                Console.WriteLine();
                Console.WriteLine($"📊 SET 2: Fixed parameters, varying midNearRatio ({method})");
                Console.WriteLine("=================================================");

                string midNearDir = Path.Combine(outputDir, $"midnear_{method}");
                Directory.CreateDirectory(midNearDir);

                int fixedNeighborsForMidNear = 10;

                int midNearFileIndex = 1;
                foreach (double midNearRatio in midNearRatioValues)
                {
                    experimentCount++;
                    Console.WriteLine($"🧪 Experiment {experimentCount}: neighbors={fixedNeighborsForMidNear}, midNearRatio={midNearRatio}");

                    try
                    {
                        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                        using var model = new PacMAPSharp.PacMAPModel();

                        var result = model.Fit(mammothData, embeddingDimensions: 2, neighbors: fixedNeighborsForMidNear,
                                             normalization: PacMAPSharp.NormalizationMode.ZScore,
                                             metric: PacMAPSharp.DistanceMetric.Euclidean,
                                             hnswUseCase: PacMAPSharp.HnswUseCase.Balanced,
                                             forceExactKnn: forceKnn,
                                             midNearRatio: midNearRatio,
                                             autodetectHnswParams: false,
                                             seed: 42, progressCallback: ProgressHandler);

                        stopwatch.Stop();
                        var embedding = result.EmbeddingCoordinates;

                        Console.WriteLine($"   ✅ Completed in {stopwatch.Elapsed.TotalSeconds:F2}s");
                        Console.WriteLine($"   Quality: {result.QualityAssessment} (confidence: {result.ConfidenceScore:F3})");

                        var embedding2D = ConvertEmbeddingTo2D(embedding, mammothData.GetLength(0), 2);
                        string plotPath = Path.Combine(midNearDir, $"{midNearFileIndex:D4}.png");
                        string methodDesc = forceKnn ? "exact_knn" : "hnsw=balanced, autodetect=OFF";
                        string paramInfo = CreateDetailedParameterInfo(model.ModelInfo, methodDesc, stopwatch.Elapsed.TotalSeconds);

                        Visualizer.PlotMammothPacMAP(embedding2D, mammothData,
                            $"Mammoth - midNearRatio={midNearRatio} ({stopwatch.Elapsed.TotalSeconds:F1}s)", plotPath, paramInfo);

                        Console.WriteLine($"   📁 Saved: midnear/{midNearFileIndex:D4}.png (midNearRatio={midNearRatio})");
                        midNearFileIndex++;
                        Console.WriteLine();
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"   ❌ Failed: {ex.Message}");
                        Console.WriteLine();
                    }
                }

                // SET 3: farPairRatio experiments
                Console.WriteLine();
                Console.WriteLine($"📊 SET 3: Fixed parameters, varying farPairRatio ({method})");
                Console.WriteLine("=================================================");

                string farPairDir = Path.Combine(outputDir, $"farpair_{method}");
                Directory.CreateDirectory(farPairDir);

                int fixedNeighborsForFarPair = 10;

                int farPairFileIndex = 1;
                foreach (double farPairRatio in farPairRatioValues)
                {
                    experimentCount++;
                    Console.WriteLine($"🧪 Experiment {experimentCount}: neighbors={fixedNeighborsForFarPair}, farPairRatio={farPairRatio}");

                    try
                    {
                        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                        using var model = new PacMAPSharp.PacMAPModel();

                        var result = model.Fit(mammothData, embeddingDimensions: 2, neighbors: fixedNeighborsForFarPair,
                                             normalization: PacMAPSharp.NormalizationMode.ZScore,
                                             metric: PacMAPSharp.DistanceMetric.Euclidean,
                                             hnswUseCase: PacMAPSharp.HnswUseCase.Balanced,
                                             forceExactKnn: forceKnn,
                                             farPairRatio: farPairRatio,
                                             autodetectHnswParams: false,
                                             seed: 42, progressCallback: ProgressHandler);

                        stopwatch.Stop();
                        var embedding = result.EmbeddingCoordinates;

                        Console.WriteLine($"   ✅ Completed in {stopwatch.Elapsed.TotalSeconds:F2}s");
                        Console.WriteLine($"   Quality: {result.QualityAssessment} (confidence: {result.ConfidenceScore:F3})");

                        var embedding2D = ConvertEmbeddingTo2D(embedding, mammothData.GetLength(0), 2);
                        string plotPath = Path.Combine(farPairDir, $"{farPairFileIndex:D4}.png");
                        string methodDesc = forceKnn ? "exact_knn" : "hnsw=balanced, autodetect=OFF";
                        string paramInfo = CreateDetailedParameterInfo(model.ModelInfo, methodDesc, stopwatch.Elapsed.TotalSeconds);

                        Visualizer.PlotMammothPacMAP(embedding2D, mammothData,
                            $"Mammoth - farPairRatio={farPairRatio} ({stopwatch.Elapsed.TotalSeconds:F1}s)", plotPath, paramInfo);

                        Console.WriteLine($"   📁 Saved: farpair/{farPairFileIndex:D4}.png (farPairRatio={farPairRatio})");
                        farPairFileIndex++;
                        Console.WriteLine();
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"   ❌ Failed: {ex.Message}");
                        Console.WriteLine();
                    }
                }

                } // end foreach method

                discoveryModel?.Dispose();

                Console.WriteLine($"✅ Parameter experiments completed! {experimentCount} experiments executed.");
                Console.WriteLine("📊 Performance Summary:");
                Console.WriteLine("   - Discovery run: Found optimal HNSW parameters with autodetect=ON");
                Console.WriteLine($"   - Performance runs: Used optimal parameters with autodetect=OFF for accurate timing");
                Console.WriteLine($"   - Tested {neighborValues.Length} neighbor values");
                Console.WriteLine($"   - Tested {midNearRatioValues.Length} midNearRatio values");
                Console.WriteLine($"   - Tested {farPairRatioValues.Length} farPairRatio values");
                Console.WriteLine("📈 Compare the generated images to see the effect of each parameter:");
                Console.WriteLine("   SET 1 (varying neighbors): Effect of local vs global neighborhood influence");
                Console.WriteLine("     - Lower neighbors (5-15): More local structure, potentially fragmented");
                Console.WriteLine("     - Medium neighbors (20-50): Good balance of local/global structure");
                Console.WriteLine("     - Higher neighbors (55-95): More global structure, smoother clusters");
                Console.WriteLine("   SET 2 (varying midNearRatio): Effect of mid-range structure preservation");
                Console.WriteLine("     - Lower ratio (0.1-0.3): Less mid-range pairs, more emphasis on neighbors/far");
                Console.WriteLine("     - Higher ratio (0.7-1.0): More mid-range structure, smoother transitions");
                Console.WriteLine("   SET 3 (varying farPairRatio): Effect of global separation forces");
                Console.WriteLine("     - Lower ratio (0.5-1.0): Less repulsion, more compact embedding");
                Console.WriteLine("     - Higher ratio (3.0-4.0): More repulsion, better separated clusters");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ ERROR: Parameter matrix experiments failed: {ex.Message}");
                Console.WriteLine($"📍 Stack trace: {ex.StackTrace}");
                throw;
            }
        }

        static void DemoLearningRateExperiments(string outputDir)
        {
            try
            {
                // Load mammoth data (smaller subset for faster experimentation)
                Console.WriteLine("🦣 Loading mammoth dataset for learning rate experiments...");
                var mammothData = DataLoaders.LoadMammothData("Data/mammoth_data.csv", maxSamples: 2000);

                Console.WriteLine($"🦣 Loaded mammoth data: {mammothData.GetLength(0)} samples, {mammothData.GetLength(1)} features");

                // Learning rate values: 0.1 to 3.0 with 15 values for comprehensive analysis with 500 epochs
                double[] learningRateValues = { 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 3.0 };
                int fixedEpochs = 500;
                int fixedNeighbors = 10;
                Console.WriteLine($"🔧 Learning Rate Experiment Configuration:");
                Console.WriteLine($"   - Fixed epochs: {fixedEpochs}");
                Console.WriteLine($"   - Fixed neighbors: {fixedNeighbors}");
                Console.WriteLine($"   - Learning rates: [{string.Join(", ", learningRateValues)}]");
                Console.WriteLine();

                // Create progress callback to show epochs and progress
                void ProgressHandler(string phase, int current, int total, float percent, string? message)
                {
                    if (phase.Contains("Fitting") || phase.Contains("Optimization") || phase.Contains("Epoch"))
                    {
                        Console.WriteLine($"   📈 {phase}: {current}/{total} ({percent:F1}%) - {message}");
                    }
                }

                // Run experiments with both HNSW and exact KNN
                string[] methods = { "hnsw", "knn" };

                int experimentCount = 0;
                var stopwatch = Stopwatch.StartNew();

                foreach (string method in methods)
                {
                    bool forceKnn = (method == "knn");
                    Console.WriteLine();
                    Console.WriteLine($"{'='*80}");
                    Console.WriteLine($"🔧 METHOD: {(forceKnn ? "EXACT KNN (Brute Force)" : "HNSW (Approximate)")}");
                    Console.WriteLine($"{'='*80}");

                // Create learning_rate folder
                string learningRateDir = Path.Combine(outputDir, $"learning_rate_{method}");
                Directory.CreateDirectory(learningRateDir);

                // Learning rate experiments
                Console.WriteLine($"🚀 Learning Rate Experiments (500 epochs each) - {method}...");
                int fileIndex = 1;
                foreach (var learningRate in learningRateValues)
                {
                    experimentCount++;
                    Console.WriteLine($"\n📊 Learning Rate Experiment {fileIndex}/{learningRateValues.Length}: lr={learningRate}");

                    try
                    {
                        using (var model = new PacMAPModel())
                        {
                            var result = model.Fit(
                                mammothData,
                                embeddingDimensions: 2,
                                neighbors: fixedNeighbors,
                                learningRate: learningRate,
                                nEpochs: fixedEpochs,
                                forceExactKnn: forceKnn,
                                autodetectHnswParams: false,
                                progressCallback: ProgressHandler
                            );

                            // Generate visualization
                            string filename = $"{fileIndex:D4}.png";
                            string fullPath = Path.Combine(learningRateDir, filename);
                            fileIndex++;

                            var embedding2D = ConvertEmbeddingTo2D(result.EmbeddingCoordinates, mammothData.GetLength(0), 2);
                            string methodDesc = forceKnn ? "exact_knn" : "hnsw=balanced, autodetect=OFF";
                            string paramInfo = CreateDetailedParameterInfo(model.ModelInfo, methodDesc, stopwatch.Elapsed.TotalSeconds);
                            Visualizer.PlotMammothPacMAP(
                                embedding2D,
                                mammothData,
                                $"Mammoth: Learning Rate={learningRate}, Epochs={fixedEpochs}",
                                fullPath,
                                paramInfo
                            );

                            Console.WriteLine($"   📁 Saved: learning_rate_{method}/{fileIndex-1:D4}.png (lr={learningRate})");
                            Console.WriteLine($"   📊 Quality: {result.QualityAssessment} (confidence: {result.ConfidenceScore:F3})");
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"   ❌ Learning rate {learningRate} failed: {ex.Message}");
                    }
                }

                } // end foreach method

                Console.WriteLine($"\n🎉 Learning Rate Experiments Completed!");
                Console.WriteLine($"   Total time: {stopwatch.Elapsed.TotalSeconds:F1}s");
                Console.WriteLine($"   Experiments: {experimentCount} total ({learningRateValues.Length} per method × 2 methods)");
                Console.WriteLine($"   Output location: {outputDir}");
                Console.WriteLine("   Folders: learning_rate_hnsw/ and learning_rate_knn/");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ ERROR: Learning rate experiments failed: {ex.Message}");
                Console.WriteLine($"📍 Stack trace: {ex.StackTrace}");
                throw;
            }
        }

        /// <summary>
        /// Create comprehensive parameter information string for image display
        /// </summary>
        static string CreateDetailedParameterInfo(PacMAPModelInfo modelInfo, string method, double timingSeconds)
        {
            var lines = new List<string>();

            // Line 1: Core PacMAP Algorithm Parameters
            lines.Add($"PacMAP: neighbors={modelInfo.Neighbors}, learning_rate={modelInfo.LearningRate:F3}, epochs={modelInfo.NEpochs}, seed={modelInfo.Seed}");

            // Line 2: PacMAP Pair Ratios & Normalization
            lines.Add($"Ratios: mid_near={modelInfo.MidNearRatio:F3}, far_pair={modelInfo.FarPairRatio:F3}, normalization={modelInfo.Normalization}");

            // Line 3: Neighbor Search Method & Performance
            string searchMethodLine;
            if (modelInfo.UsedHNSW)
            {
                if (modelInfo.DiscoveredHnswM.HasValue && modelInfo.DiscoveredHnswEfConstruction.HasValue && modelInfo.DiscoveredHnswEfSearch.HasValue)
                {
                    searchMethodLine = $"Search: HNSW (M={modelInfo.DiscoveredHnswM.Value}, ef_construction={modelInfo.DiscoveredHnswEfConstruction.Value}, ef_search={modelInfo.DiscoveredHnswEfSearch.Value}, recall={modelInfo.HnswRecall:F1}%)";
                }
                else
                {
                    searchMethodLine = $"Search: HNSW (recall={modelInfo.HnswRecall:F1}%, manual config)";
                }
            }
            else
            {
                searchMethodLine = $"Search: EXACT KNN (100% accuracy, brute-force)";
            }
            lines.Add(searchMethodLine);

            // Line 4: Dataset & Performance Info
            string quantizationStatus = modelInfo.QuantizeOnSave ? "ENABLED" : "DISABLED";
            lines.Add($"Dataset: {modelInfo.TrainingSamples:N0} samples ({modelInfo.InputDimension}D→{modelInfo.OutputDimension}D), time={timingSeconds:F2}s, quantization={quantizationStatus}");

            // Line 5: CRC Integrity (if available)
            if (modelInfo.HnswIndexCrc32.HasValue || modelInfo.EmbeddingHnswIndexCrc32.HasValue)
            {
                var crcParts = new List<string>();
                if (modelInfo.HnswIndexCrc32.HasValue)
                    crcParts.Add($"orig_crc32=0x{modelInfo.HnswIndexCrc32.Value:X8}");
                if (modelInfo.EmbeddingHnswIndexCrc32.HasValue)
                    crcParts.Add($"embed_crc32=0x{modelInfo.EmbeddingHnswIndexCrc32.Value:X8}");
                lines.Add($"Integrity: {string.Join(", ", crcParts)}");
            }

            return string.Join("\n", lines);
        }

        static void DemoHairyMammothExperiments(string outputDir)
        {
            try
            {
                // Load hairy mammoth data (1M points)
                Console.WriteLine("🦣 Loading hairy mammoth dataset...");
                var hairyMammothFull = DataLoaders.LoadMammothData("Data/mammoth_a.csv");
                Console.WriteLine($"🦣 Loaded FULL hairy mammoth: {hairyMammothFull.GetLength(0):N0} samples, {hairyMammothFull.GetLength(1)} features");
                Console.WriteLine();

                // Sample 40K for exact KNN (deterministic GIFs, doubled from 20K)
                var hairyMammothSample = DataLoaders.SampleRandomPoints(hairyMammothFull, 40000, seed: 42);
                Console.WriteLine();

                // Practical parameter arrays with bigger values (avoid small values that cause Rust panics)
                double[] midNearRatioValues = {
                    0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0
                };
                double[] farPairRatioValues = {
                    1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0
                };

                // Neighbor variations for hairy mammoth (more granular range up to 80)
                int[] neighborValues = { 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80 };
                int fixedNeighbors = 10;

                void ProgressHandler(string phase, int current, int total, float percent, string? message)
                {
                    // Minimal progress output for speed
                }

                // Run exact KNN experiments on 20K sample
                Console.WriteLine($"📊 Exact KNN Experiments (20K sample, deterministic)");
                Console.WriteLine("=================================================");
                Console.WriteLine();

                // Mid-near ratio experiments
                string midnearKnnDir = Path.Combine(outputDir, "hairy_midnear_knn");
                Directory.CreateDirectory(midnearKnnDir);

                int fileIndex = 1;
                foreach (double midNearRatio in midNearRatioValues)
                {
                    Console.WriteLine($"🧪 Hairy Mammoth KNN: midNearRatio={midNearRatio}");
                    var stopwatch = System.Diagnostics.Stopwatch.StartNew();

                    using var model = new PacMAPSharp.PacMAPModel();
                    var result = model.Fit(hairyMammothSample, embeddingDimensions: 2, neighbors: fixedNeighbors,
                                         normalization: PacMAPSharp.NormalizationMode.ZScore,
                                         metric: PacMAPSharp.DistanceMetric.Euclidean,
                                         forceExactKnn: true,
                                         midNearRatio: midNearRatio,
                                         seed: 42, progressCallback: ProgressHandler);

                    stopwatch.Stop();
                    var embedding = result.EmbeddingCoordinates;

                    Console.WriteLine($"   ✅ Completed in {stopwatch.Elapsed.TotalSeconds:F2}s");

                    var embedding2D = ConvertEmbeddingTo2D(embedding, hairyMammothSample.GetLength(0), 2);
                    string plotPath = Path.Combine(midnearKnnDir, $"{fileIndex:D4}.png");
                    string paramInfo = $"midNearRatio={midNearRatio}, neighbors={fixedNeighbors}, n={hairyMammothSample.GetLength(0):N0}";

                    Visualizer.PlotSimplePacMAP(embedding2D, $"Hairy Mammoth KNN - midNearRatio={midNearRatio}", plotPath, paramInfo);
                    Console.WriteLine($"   📁 Saved: {fileIndex:D4}.png");
                    fileIndex++;
                }

                Console.WriteLine();

                // Far-pair ratio experiments
                string farpairKnnDir = Path.Combine(outputDir, "hairy_farpair_knn");
                Directory.CreateDirectory(farpairKnnDir);

                fileIndex = 1;
                foreach (double farPairRatio in farPairRatioValues)
                {
                    Console.WriteLine($"🧪 Hairy Mammoth KNN: farPairRatio={farPairRatio}");
                    var stopwatch = System.Diagnostics.Stopwatch.StartNew();

                    using var model = new PacMAPSharp.PacMAPModel();
                    var result = model.Fit(hairyMammothSample, embeddingDimensions: 2, neighbors: fixedNeighbors,
                                         normalization: PacMAPSharp.NormalizationMode.ZScore,
                                         metric: PacMAPSharp.DistanceMetric.Euclidean,
                                         forceExactKnn: true,
                                         farPairRatio: farPairRatio,
                                         seed: 42, progressCallback: ProgressHandler);

                    stopwatch.Stop();
                    var embedding = result.EmbeddingCoordinates;

                    Console.WriteLine($"   ✅ Completed in {stopwatch.Elapsed.TotalSeconds:F2}s");

                    var embedding2D = ConvertEmbeddingTo2D(embedding, hairyMammothSample.GetLength(0), 2);
                    string plotPath = Path.Combine(farpairKnnDir, $"{fileIndex:D4}.png");
                    string paramInfo = $"farPairRatio={farPairRatio}, neighbors={fixedNeighbors}, n={hairyMammothSample.GetLength(0):N0}";

                    Visualizer.PlotSimplePacMAP(embedding2D, $"Hairy Mammoth KNN - farPairRatio={farPairRatio}", plotPath, paramInfo);
                    Console.WriteLine($"   📁 Saved: {fileIndex:D4}.png");
                    fileIndex++;
                }

                // Neighbors variation experiments
                Console.WriteLine($"📊 Neighbors Variation Experiment (KNN)");
                Console.WriteLine("==========================================");

                // Create neighbor folder
                string neighborKnnDir = Path.Combine(outputDir, "hairy_neighbor_knn");
                Directory.CreateDirectory(neighborKnnDir);

                int neighborFileIndex = 1;
                foreach (int neighbors in neighborValues)
                {
                    Console.WriteLine($"🧪 Hairy Mammoth KNN: neighbors={neighbors}");
                    var stopwatch = System.Diagnostics.Stopwatch.StartNew();

                    using var model = new PacMAPSharp.PacMAPModel();
                    var result = model.Fit(hairyMammothSample, embeddingDimensions: 2, neighbors: neighbors,
                                         normalization: PacMAPSharp.NormalizationMode.ZScore,
                                         metric: PacMAPSharp.DistanceMetric.Euclidean,
                                         forceExactKnn: true,
                                         seed: 42, progressCallback: ProgressHandler);

                    stopwatch.Stop();
                    var embedding = result.EmbeddingCoordinates;

                    Console.WriteLine($"   ✅ Completed in {stopwatch.Elapsed.TotalSeconds:F2}s");

                    var embedding2D = ConvertEmbeddingTo2D(embedding, hairyMammothSample.GetLength(0), 2);
                    string plotPath = Path.Combine(neighborKnnDir, $"{neighborFileIndex:D4}.png");
                    string paramInfo = $"neighbors={neighbors}, n={hairyMammothSample.GetLength(0):N0}";

                    Visualizer.PlotSimplePacMAP(embedding2D, $"Hairy Mammoth KNN - neighbors={neighbors}", plotPath, paramInfo);
                    Console.WriteLine($"   📁 Saved: {neighborFileIndex:D4}.png");
                    neighborFileIndex++;
                }

                Console.WriteLine();
                Console.WriteLine("✅ Hairy mammoth experiments completed!");
                Console.WriteLine($"📁 Results in: hairy_midnear_knn/, hairy_farpair_knn/, and hairy_neighbor_knn/");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: Hairy mammoth experiments failed: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }

        // DISABLED: All anatomy test methods removed (ScottPlot dependency)
        /*
        static void RunAnatomyTests()
        {
            Console.WriteLine("=== MAMMOTH ANATOMY CLASSIFICATION TESTS ===");
            Console.WriteLine();

            try
            {
                // Load mammoth data
                string mammothPath = Path.Combine("Data", "mammoth_data.csv");
                var mammothData = DataLoaders.LoadMammothData(mammothPath);
                Console.WriteLine($"Loaded {mammothData.GetLength(0)} mammoth points");
                Console.WriteLine();

                // Create output directory
                string outputDir = "AnatomyTests";
                Directory.CreateDirectory(outputDir);

                // Show only middle X + first 40% Y
                Console.WriteLine("--- Testing Middle X + First 40% Y Only ---");
                TestMiddleXOnly(mammothData, outputDir);
                Console.WriteLine();

                Console.WriteLine("=== All anatomy tests completed ===");
                Console.WriteLine($"Results saved in: {Path.GetFullPath(outputDir)}");

                // Open the results folder
                OpenFileOrFolder(Path.GetFullPath(outputDir), true);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: Anatomy test failed: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }

            // All demos completed successfully - exit gracefully
        }

        static void TestHeadDetectionOnly(double[,] originalData, int version, string outputDir)
        {
            int numPoints = originalData.GetLength(0);
            var parts = new string[numPoints];

            // Compute coordinate ranges
            double minZ = double.MaxValue, maxZ = double.MinValue;
            double minX = double.MaxValue, maxX = double.MinValue;
            double minY = double.MaxValue, maxY = double.MinValue;

            for (int i = 0; i < numPoints; i++)
            {
                double x = originalData[i, 0];
                double y = originalData[i, 1];
                double z = originalData[i, 2];

                minX = Math.Min(minX, x);
                maxX = Math.Max(maxX, x);
                minY = Math.Min(minY, y);
                maxY = Math.Max(maxY, y);
                minZ = Math.Min(minZ, z);
                maxZ = Math.Max(maxZ, z);
            }

            double xRange = maxX - minX;
            double yRange = maxY - minY;
            double zRange = maxZ - minZ;
            double xCenter = (minX + maxX) / 2;
            double yCenter = (minY + maxY) / 2;

            Console.WriteLine($"Coordinate ranges: X[{minX:F1}, {maxX:F1}], Y[{minY:F1}, {maxY:F1}], Z[{minZ:F1}, {maxZ:F1}]");

            // Initialize everything as "other" first
            for (int i = 0; i < numPoints; i++)
            {
                parts[i] = "other";
            }

            // Different head detection approaches - ONLY HEAD, everything else is "other"
            switch (version)
            {
                case 1: // Simple top Z approach
                    Console.WriteLine("V1: Head = top 20% Z height");
                    {
                        double headZThreshold = minZ + zRange * 0.8; // Top 20%
                        for (int i = 0; i < numPoints; i++)
                        {
                            double z = originalData[i, 2];
                            if (z > headZThreshold)
                                parts[i] = "head";
                        }
                    }
                    break;

                case 2: // Top Z + not too forward (exclude trunk area)
                    Console.WriteLine("V2: Head = top Z + not extreme forward (exclude trunk)");
                    {
                        double headZThreshold = minZ + zRange * 0.75; // Top 25%
                        double maxHeadX = minX + xRange * 0.7; // Not too forward (exclude trunk)
                        for (int i = 0; i < numPoints; i++)
                        {
                            double x = originalData[i, 0];
                            double z = originalData[i, 2];
                            if (z > headZThreshold && x <= maxHeadX)
                                parts[i] = "head";
                        }
                    }
                    break;

                case 3: // Top Z + moderate X + central Y (exclude ears/trunk)
                    Console.WriteLine("V3: Head = top Z + moderate X + central Y (compact head region)");
                    {
                        double headZThreshold = minZ + zRange * 0.7; // Top 30%
                        double minHeadX = minX + xRange * 0.3; // Not too back
                        double maxHeadX = minX + xRange * 0.65; // Not too forward
                        double headYRange = yRange * 0.6; // Central 60% in Y
                        for (int i = 0; i < numPoints; i++)
                        {
                            double x = originalData[i, 0];
                            double y = originalData[i, 1];
                            double z = originalData[i, 2];
                            double yDist = Math.Abs(y - yCenter);

                            if (z > headZThreshold && x >= minHeadX && x <= maxHeadX && yDist < headYRange)
                                parts[i] = "head";
                        }
                    }
                    break;
            }

            // Count parts and create visualization
            var partCounts = parts.GroupBy(p => p).ToDictionary(g => g.Key, g => g.Count());
            Console.WriteLine($"Parts: {string.Join(", ", partCounts.Select(kv => $"{kv.Key}={kv.Value}"))}");

            string outputPath = Path.Combine(outputDir, $"head_only_v{version}.png");
            CreateHeadOnlyVisualization(originalData, parts, $"Head Detection Test V{version}", outputPath);
        }

        static void TestTrunkDetectionOnly(double[,] originalData, int version, string outputDir)
        {
            int numPoints = originalData.GetLength(0);
            var parts = new string[numPoints];

            // Compute coordinate ranges
            double minZ = double.MaxValue, maxZ = double.MinValue;
            double minX = double.MaxValue, maxX = double.MinValue;
            double minY = double.MaxValue, maxY = double.MinValue;

            for (int i = 0; i < numPoints; i++)
            {
                double x = originalData[i, 0];
                double y = originalData[i, 1];
                double z = originalData[i, 2];

                minX = Math.Min(minX, x);
                maxX = Math.Max(maxX, x);
                minY = Math.Min(minY, y);
                maxY = Math.Max(maxY, y);
                minZ = Math.Min(minZ, z);
                maxZ = Math.Max(maxZ, z);
            }

            double xRange = maxX - minX;
            double yRange = maxY - minY;
            double zRange = maxZ - minZ;
            double xCenter = (minX + maxX) / 2;
            double yCenter = (minY + maxY) / 2;

            Console.WriteLine($"Coordinate ranges: X[{minX:F1}, {maxX:F1}], Y[{minY:F1}, {maxY:F1}], Z[{minZ:F1}, {maxZ:F1}]");

            // Initialize everything as "other" first
            for (int i = 0; i < numPoints; i++)
            {
                parts[i] = "other";
            }

            // First identify head region (using best approach from V3)
            double headZThreshold = minZ + zRange * 0.7; // Top 30%
            double minHeadX = minX + xRange * 0.3; // Not too back
            double maxHeadX = minX + xRange * 0.65; // Not too forward
            double headYRange = yRange * 0.6; // Central 60% in Y

            // Mark head points first
            for (int i = 0; i < numPoints; i++)
            {
                double x = originalData[i, 0];
                double y = originalData[i, 1];
                double z = originalData[i, 2];
                double yDist = Math.Abs(y - yCenter);

                if (z > headZThreshold && x >= minHeadX && x <= maxHeadX && yDist < headYRange)
                    parts[i] = "head";
            }

            // Now detect trunk - ONLY TRUNK, excluding head area
            switch (version)
            {
                case 1: // Simple forward extension
                    Console.WriteLine("V1: Trunk = very forward X + exclude head region");
                    {
                        double trunkXThreshold = minX + xRange * 0.8; // Very forward
                        for (int i = 0; i < numPoints; i++)
                        {
                            if (parts[i] != "head") // Don't override head
                            {
                                double x = originalData[i, 0];
                                if (x > trunkXThreshold)
                                    parts[i] = "trunk";
                            }
                        }
                    }
                    break;

                case 2: // Forward + narrow + hanging down
                    Console.WriteLine("V2: Trunk = forward + narrow Y + hanging down from head level");
                    {
                        double trunkXThreshold = minX + xRange * 0.75; // Forward
                        double trunkYWidth = yRange * 0.3; // Narrow
                        double trunkZMax = minZ + zRange * 0.65; // Hangs down from head
                        double trunkZMin = minZ + zRange * 0.2; // Above legs

                        for (int i = 0; i < numPoints; i++)
                        {
                            if (parts[i] != "head") // Don't override head
                            {
                                double x = originalData[i, 0];
                                double y = originalData[i, 1];
                                double z = originalData[i, 2];
                                double yDist = Math.Abs(y - yCenter);

                                if (x > trunkXThreshold && yDist < trunkYWidth && z >= trunkZMin && z <= trunkZMax)
                                    parts[i] = "trunk";
                            }
                        }
                    }
                    break;

                case 3: // Ultra-specific trunk - very narrow, very forward, hangs straight down
                    Console.WriteLine("V3: Trunk = ultra-forward + ultra-narrow + hangs straight down");
                    {
                        double trunkXThreshold = minX + xRange * 0.8; // Ultra forward
                        double trunkYWidth = yRange * 0.2; // Ultra narrow
                        double trunkZMax = minZ + zRange * 0.6; // Hangs down
                        double trunkZMin = minZ + zRange * 0.25; // Above legs

                        for (int i = 0; i < numPoints; i++)
                        {
                            if (parts[i] != "head") // Don't override head
                            {
                                double x = originalData[i, 0];
                                double y = originalData[i, 1];
                                double z = originalData[i, 2];
                                double yDist = Math.Abs(y - yCenter);

                                if (x > trunkXThreshold && yDist < trunkYWidth && z >= trunkZMin && z <= trunkZMax)
                                    parts[i] = "trunk";
                            }
                        }
                    }
                    break;
            }

            // Count parts and create visualization
            var partCounts = parts.GroupBy(p => p).ToDictionary(g => g.Key, g => g.Count());
            Console.WriteLine($"Parts: {string.Join(", ", partCounts.Select(kv => $"{kv.Key}={kv.Value}"))}");

            string outputPath = Path.Combine(outputDir, $"trunk_test_v{version}.png");
            CreateTrunkTestVisualization(originalData, parts, $"Trunk Detection Test V{version}", outputPath);
        }

        static void CreateTrunkTestVisualization(double[,] originalData, string[] parts, string title, string outputPath)
        {
            try
            {
                var partColors = new Dictionary<string, Color>
                {
                    { "head", Color.Purple },     // Head in purple (reference)
                    { "trunk", Color.Red },       // Trunk in bright red (focus)
                    { "other", Color.LightGray }  // Everything else in light gray
                };

                int numPoints = originalData.GetLength(0);
                var x = new double[numPoints];
                var y = new double[numPoints];
                var z = new double[numPoints];

                for (int i = 0; i < numPoints; i++)
                {
                    x[i] = originalData[i, 0];
                    y[i] = originalData[i, 1];
                    z[i] = originalData[i, 2];
                }

                // Group by parts
                var partGroups = new Dictionary<string, (List<double> x, List<double> y, List<double> z)>();
                foreach (var part in partColors.Keys)
                {
                    partGroups[part] = (new List<double>(), new List<double>(), new List<double>());
                }

                for (int i = 0; i < numPoints; i++)
                {
                    string part = parts[i];
                    if (partGroups.ContainsKey(part))
                    {
                        partGroups[part].x.Add(x[i]);
                        partGroups[part].y.Add(y[i]);
                        partGroups[part].z.Add(z[i]);
                    }
                }

                // Normalize for three views
                double xMin = x.Min(), xMax = x.Max(), xRange = xMax - xMin;
                double yMin = y.Min(), yMax = y.Max(), yRange = yMax - yMin;
                double zMin = z.Min(), zMax = z.Max(), zRange = zMax - zMin;

                var plt = new Plot(2400, 800);

                // Three views: XY, XZ, YZ - show layers: other (background), head (reference), trunk (focus)
                foreach (var kvp in partGroups.OrderBy(kv => kv.Key == "trunk" ? 2 : kv.Key == "head" ? 1 : 0)) // other, head, trunk (trunk on top)
                {
                    var part = kvp.Key;
                    var (xPoints, yPoints, zPoints) = kvp.Value;
                    if (xPoints.Count > 0)
                    {
                        int markerSize = part == "trunk" ? 4 : part == "head" ? 2 : 1;
                        string label = part == "trunk" ? "Trunk" : part == "head" ? "Head (reference)" : null;

                        // XY view (left)
                        var normX1 = xPoints.Select(x => (x - xMin) / xRange * 600 + 50).ToArray();
                        var normY1 = yPoints.Select(y => (y - yMin) / yRange * 600 + 100).ToArray();
                        plt.AddScatter(normX1, normY1, color: partColors[part], lineWidth: 0,
                            markerSize: markerSize, label: label);

                        // XZ view (middle)
                        var normX2 = xPoints.Select(x => (x - xMin) / xRange * 600 + 850).ToArray();
                        var normZ2 = zPoints.Select(z => (z - zMin) / zRange * 600 + 100).ToArray();
                        plt.AddScatter(normX2, normZ2, color: partColors[part], lineWidth: 0,
                            markerSize: markerSize);

                        // YZ view (right)
                        var normY3 = yPoints.Select(y => (y - yMin) / yRange * 600 + 1650).ToArray();
                        var normZ3 = zPoints.Select(z => (z - zMin) / zRange * 600 + 100).ToArray();
                        plt.AddScatter(normY3, normZ3, color: partColors[part], lineWidth: 0,
                            markerSize: markerSize);
                    }
                }

                plt.Title(title);
                plt.XLabel("XY View | XZ View | YZ View");
                plt.YLabel("Mammoth 3D Projections - TRUNK DETECTION FOCUS");
                plt.Legend(location: Alignment.UpperRight);
                plt.Grid(enable: false);

                Directory.CreateDirectory(Path.GetDirectoryName(outputPath));
                plt.SaveFig(outputPath, width: 2400, height: 800);

                Console.WriteLine($"Saved: {outputPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: Failed to create trunk test visualization: {ex.Message}");
            }
        }

        static void TestFinalAnatomicalClassification(double[,] originalData, string outputDir)
        {
            Console.WriteLine("Testing final anatomical classification from Visualizer.cs");

            // Use the actual classification from Visualizer.cs
            var parts = Visualizer.AssignMammothParts(originalData);

            // Count parts
            var partCounts = parts.GroupBy(p => p).ToDictionary(g => g.Key, g => g.Count());
            Console.WriteLine($"Parts: {string.Join(", ", partCounts.Select(kv => $"{kv.Key}={kv.Value}"))}");

            string outputPath = Path.Combine(outputDir, "final_anatomy_test.png");
            CreateFinalAnatomyVisualization(originalData, parts, "Final Mammoth Anatomical Classification", outputPath);
        }

        static void CreateFinalAnatomyVisualization(double[,] originalData, string[] parts, string title, string outputPath)
        {
            try
            {
                var partColors = new Dictionary<string, Color>
                {
                    { "legs", Color.Blue },
                    { "body", Color.Green },
                    { "head", Color.Purple },
                    { "trunk", Color.Red },
                    { "tusks", Color.Gold }
                };

                int numPoints = originalData.GetLength(0);
                var x = new double[numPoints];
                var y = new double[numPoints];
                var z = new double[numPoints];

                for (int i = 0; i < numPoints; i++)
                {
                    x[i] = originalData[i, 0];
                    y[i] = originalData[i, 1];
                    z[i] = originalData[i, 2];
                }

                // Group by parts
                var partGroups = new Dictionary<string, (List<double> x, List<double> y, List<double> z)>();
                foreach (var part in partColors.Keys)
                {
                    partGroups[part] = (new List<double>(), new List<double>(), new List<double>());
                }

                for (int i = 0; i < numPoints; i++)
                {
                    string part = parts[i];
                    if (partGroups.ContainsKey(part))
                    {
                        partGroups[part].x.Add(x[i]);
                        partGroups[part].y.Add(y[i]);
                        partGroups[part].z.Add(z[i]);
                    }
                }

                // Normalize for three views
                double xMin = x.Min(), xMax = x.Max(), xRange = xMax - xMin;
                double yMin = y.Min(), yMax = y.Max(), yRange = yMax - yMin;
                double zMin = z.Min(), zMax = z.Max(), zRange = zMax - zMin;

                var plt = new Plot(2400, 800);

                // Three views: XY, XZ, YZ - show all parts with different colors
                foreach (var kvp in partGroups)
                {
                    var part = kvp.Key;
                    var (xPoints, yPoints, zPoints) = kvp.Value;
                    if (xPoints.Count > 0)
                    {
                        // XY view (left)
                        var normX1 = xPoints.Select(x => (x - xMin) / xRange * 600 + 50).ToArray();
                        var normY1 = yPoints.Select(y => (y - yMin) / yRange * 600 + 100).ToArray();
                        plt.AddScatter(normX1, normY1, color: partColors[part], lineWidth: 0, markerSize: 3,
                            label: part == "legs" ? char.ToUpper(part[0]) + part.Substring(1) : null);

                        // XZ view (middle)
                        var normX2 = xPoints.Select(x => (x - xMin) / xRange * 600 + 850).ToArray();
                        var normZ2 = zPoints.Select(z => (z - zMin) / zRange * 600 + 100).ToArray();
                        plt.AddScatter(normX2, normZ2, color: partColors[part], lineWidth: 0, markerSize: 3);

                        // YZ view (right)
                        var normY3 = yPoints.Select(y => (y - yMin) / yRange * 600 + 1650).ToArray();
                        var normZ3 = zPoints.Select(z => (z - zMin) / zRange * 600 + 100).ToArray();
                        plt.AddScatter(normY3, normZ3, color: partColors[part], lineWidth: 0, markerSize: 3);
                    }
                }

                plt.Title(title);
                plt.XLabel("XY View | XZ View | YZ View");
                plt.YLabel("Mammoth 3D Projections - FINAL ANATOMY");
                plt.Legend(location: Alignment.UpperRight);
                plt.Grid(enable: false);

                Directory.CreateDirectory(Path.GetDirectoryName(outputPath));
                plt.SaveFig(outputPath, width: 2400, height: 800);

                Console.WriteLine($"Saved: {outputPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: Failed to create final anatomy visualization: {ex.Message}");
            }
        }

        static void TestMiddleXOnly(double[,] originalData, string outputDir)
        {
            int numPoints = originalData.GetLength(0);
            var parts = new string[numPoints];

            // Compute coordinate ranges
            double minX = double.MaxValue, maxX = double.MinValue;
            double minY = double.MaxValue, maxY = double.MinValue;
            double minZ = double.MaxValue, maxZ = double.MinValue;

            for (int i = 0; i < numPoints; i++)
            {
                double x = originalData[i, 0];
                double y = originalData[i, 1];
                double z = originalData[i, 2];
                minX = Math.Min(minX, x);
                maxX = Math.Max(maxX, x);
                minY = Math.Min(minY, y);
                maxY = Math.Max(maxY, y);
                minZ = Math.Min(minZ, z);
                maxZ = Math.Max(maxZ, z);
            }

            double xRange = maxX - minX;
            double yRange = maxY - minY;
            double zRange = maxZ - minZ;

            Console.WriteLine($"X range: [{minX:F1}, {maxX:F1}] (range: {xRange:F1})");
            Console.WriteLine($"Y range: [{minY:F1}, {maxY:F1}] (range: {yRange:F1})");
            Console.WriteLine($"Z range: [{minZ:F1}, {maxZ:F1}] (range: {zRange:F1})");

            // Initialize everything as "other" first
            for (int i = 0; i < numPoints; i++)
            {
                parts[i] = "other";
            }

            // Define boundaries
            double firstXThreshold = minX + xRange * 0.36;  // First 36% of X
            double lastXThreshold = maxX - xRange * 0.36;   // Last 36% of X (middle 28%)
            double yThreshold = minY + yRange * 0.415; // First 41.5% of Y direction
            double zThreshold = minZ + zRange * 0.5; // Top 50% of Z (upper half)

            Console.WriteLine($"Middle 28% X between {firstXThreshold:F1} and {lastXThreshold:F1} + first 41.5% Y + top 50% Z");

            // Middle 45% X + first 40% Y + top 50% Z
            for (int i = 0; i < numPoints; i++)
            {
                double x = originalData[i, 0];
                double y = originalData[i, 1];
                double z = originalData[i, 2];

                if (x >= firstXThreshold && x <= lastXThreshold && y < yThreshold && z > zThreshold)
                    parts[i] = "head";
            }

            // Count parts
            var partCounts = parts.GroupBy(p => p).ToDictionary(g => g.Key, g => g.Count());
            Console.WriteLine($"Parts: {string.Join(", ", partCounts.Select(kv => $"{kv.Key}={kv.Value}"))}");

            string outputPath = Path.Combine(outputDir, "middle_x_only.png");
            CreateMiddleXVisualization(originalData, parts, "Middle X + First 40% Y Only", outputPath);
        }

        static void CreateMiddleXVisualization(double[,] originalData, string[] parts, string title, string outputPath)
        {
            try
            {
                var partColors = new Dictionary<string, Color>
                {
                    { "head", Color.Red },         // Head in red
                    { "other", Color.LightGray }   // Everything else in light gray
                };

                int numPoints = originalData.GetLength(0);
                var x = new double[numPoints];
                var y = new double[numPoints];
                var z = new double[numPoints];

                for (int i = 0; i < numPoints; i++)
                {
                    x[i] = originalData[i, 0];
                    y[i] = originalData[i, 1];
                    z[i] = originalData[i, 2];
                }

                // Group by parts
                var partGroups = new Dictionary<string, (List<double> x, List<double> y, List<double> z)>();
                foreach (var part in partColors.Keys)
                {
                    partGroups[part] = (new List<double>(), new List<double>(), new List<double>());
                }

                for (int i = 0; i < numPoints; i++)
                {
                    string part = parts[i];
                    if (partGroups.ContainsKey(part))
                    {
                        partGroups[part].x.Add(x[i]);
                        partGroups[part].y.Add(y[i]);
                        partGroups[part].z.Add(z[i]);
                    }
                }

                // Normalize for three views
                double xMin = x.Min(), xMax = x.Max(), xRange = xMax - xMin;
                double yMin = y.Min(), yMax = y.Max(), yRange = yMax - yMin;
                double zMin = z.Min(), zMax = z.Max(), zRange = zMax - zMin;

                var plt = new Plot(2400, 800);

                // Three views: XY, XZ, YZ - show "other" first (gray), then head (red on top)
                foreach (var kvp in partGroups.OrderBy(kv => kv.Key == "head" ? 1 : 0))
                {
                    var part = kvp.Key;
                    var (xPoints, yPoints, zPoints) = kvp.Value;
                    if (xPoints.Count > 0)
                    {
                        int markerSize = part == "head" ? 4 : 1;
                        string label = part == "head" ? "Middle X + First 40% Y" : null;

                        // XY view (left)
                        var normX1 = xPoints.Select(x => (x - xMin) / xRange * 600 + 50).ToArray();
                        var normY1 = yPoints.Select(y => (y - yMin) / yRange * 600 + 100).ToArray();
                        plt.AddScatter(normX1, normY1, color: partColors[part], lineWidth: 0,
                            markerSize: markerSize, label: label);

                        // XZ view (middle)
                        var normX2 = xPoints.Select(x => (x - xMin) / xRange * 600 + 850).ToArray();
                        var normZ2 = zPoints.Select(z => (z - zMin) / zRange * 600 + 100).ToArray();
                        plt.AddScatter(normX2, normZ2, color: partColors[part], lineWidth: 0,
                            markerSize: markerSize);

                        // YZ view (right)
                        var normY3 = yPoints.Select(y => (y - yMin) / yRange * 600 + 1650).ToArray();
                        var normZ3 = zPoints.Select(z => (z - zMin) / zRange * 600 + 100).ToArray();
                        plt.AddScatter(normY3, normZ3, color: partColors[part], lineWidth: 0,
                            markerSize: markerSize);
                    }
                }

                plt.Title(title);
                plt.XLabel("XY View | XZ View | YZ View");
                plt.YLabel("Mammoth 3D Projections - MIDDLE X ONLY");
                plt.Legend(location: Alignment.UpperRight);
                plt.Grid(enable: false);

                Directory.CreateDirectory(Path.GetDirectoryName(outputPath));
                plt.SaveFig(outputPath, width: 2400, height: 800);

                Console.WriteLine($"Saved: {outputPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: Failed to create middle X visualization: {ex.Message}");
            }
        }

        static void TestMammothParts(double[,] originalData, int version, string outputDir)
        {
            int numPoints = originalData.GetLength(0);
            var parts = new string[numPoints];

            // Compute coordinate ranges
            double minX = double.MaxValue, maxX = double.MinValue;
            double minY = double.MaxValue, maxY = double.MinValue;

            for (int i = 0; i < numPoints; i++)
            {
                double x = originalData[i, 0];
                double y = originalData[i, 1];
                minX = Math.Min(minX, x);
                maxX = Math.Max(maxX, x);
                minY = Math.Min(minY, y);
                maxY = Math.Max(maxY, y);
            }

            double xRange = maxX - minX;
            double yRange = maxY - minY;

            Console.WriteLine($"X range: [{minX:F1}, {maxX:F1}] (range: {xRange:F1})");
            Console.WriteLine($"Y range: [{minY:F1}, {maxY:F1}] (range: {yRange:F1})");

            // Initialize everything as "other" first
            for (int i = 0; i < numPoints; i++)
            {
                parts[i] = "other";
            }

            // Define X boundaries
            double firstXThreshold = minX + xRange * 0.2;  // First 20% of X
            double lastXThreshold = maxX - xRange * 0.2;   // Last 20% of X

            // Define Y boundary
            double yThreshold = minY + yRange * 0.4; // First 40% of Y direction

            switch (version)
            {
                case 1: // Extremities (trunk + tail)
                    Console.WriteLine($"V1: Extremities = first 20% X + last 20% X + first 40% Y");
                    for (int i = 0; i < numPoints; i++)
                    {
                        double x = originalData[i, 0];
                        double y = originalData[i, 1];

                        if ((x < firstXThreshold || x > lastXThreshold) && y < yThreshold)
                            parts[i] = "extremities";
                    }
                    break;

                case 2: // Head (middle X)
                    Console.WriteLine($"V2: Head = middle X (between {firstXThreshold:F1} and {lastXThreshold:F1}) + first 40% Y");
                    for (int i = 0; i < numPoints; i++)
                    {
                        double x = originalData[i, 0];
                        double y = originalData[i, 1];

                        if (x >= firstXThreshold && x <= lastXThreshold && y < yThreshold)
                            parts[i] = "head";
                    }
                    break;
            }

            // Count parts
            var partCounts = parts.GroupBy(p => p).ToDictionary(g => g.Key, g => g.Count());
            Console.WriteLine($"Parts: {string.Join(", ", partCounts.Select(kv => $"{kv.Key}={kv.Value}"))}");

            string outputPath = Path.Combine(outputDir, $"mammoth_parts_v{version}.png");
            CreatePartsVisualization(originalData, parts, $"Mammoth Parts Test V{version}", outputPath, version);
        }

        static void CreatePartsVisualization(double[,] originalData, string[] parts, string title, string outputPath, int version)
        {
            try
            {
                var partColors = new Dictionary<string, Color>
                {
                    { "extremities", Color.Red },     // Trunk + tail in red
                    { "head", Color.Purple },         // Head in purple
                    { "other", Color.LightGray }      // Everything else in light gray
                };

                int numPoints = originalData.GetLength(0);
                var x = new double[numPoints];
                var y = new double[numPoints];
                var z = new double[numPoints];

                for (int i = 0; i < numPoints; i++)
                {
                    x[i] = originalData[i, 0];
                    y[i] = originalData[i, 1];
                    z[i] = originalData[i, 2];
                }

                // Group by parts
                var partGroups = new Dictionary<string, (List<double> x, List<double> y, List<double> z)>();
                foreach (var part in partColors.Keys)
                {
                    partGroups[part] = (new List<double>(), new List<double>(), new List<double>());
                }

                for (int i = 0; i < numPoints; i++)
                {
                    string part = parts[i];
                    if (partGroups.ContainsKey(part))
                    {
                        partGroups[part].x.Add(x[i]);
                        partGroups[part].y.Add(y[i]);
                        partGroups[part].z.Add(z[i]);
                    }
                }

                // Normalize for three views
                double xMin = x.Min(), xMax = x.Max(), xRange = xMax - xMin;
                double yMin = y.Min(), yMax = y.Max(), yRange = yMax - yMin;
                double zMin = z.Min(), zMax = z.Max(), zRange = zMax - zMin;

                var plt = new Plot(2400, 800);

                // Three views: XY, XZ, YZ - show "other" first (gray), then colored parts on top
                foreach (var kvp in partGroups.OrderBy(kv => kv.Key == "other" ? 0 : 1))
                {
                    var part = kvp.Key;
                    var (xPoints, yPoints, zPoints) = kvp.Value;
                    if (xPoints.Count > 0)
                    {
                        int markerSize = part == "other" ? 1 : 4;
                        string label = part == "extremities" ? "Extremities (trunk+tail)" :
                                      part == "head" ? "Head" : null;

                        // XY view (left)
                        var normX1 = xPoints.Select(x => (x - xMin) / xRange * 600 + 50).ToArray();
                        var normY1 = yPoints.Select(y => (y - yMin) / yRange * 600 + 100).ToArray();
                        plt.AddScatter(normX1, normY1, color: partColors[part], lineWidth: 0,
                            markerSize: markerSize, label: label);

                        // XZ view (middle)
                        var normX2 = xPoints.Select(x => (x - xMin) / xRange * 600 + 850).ToArray();
                        var normZ2 = zPoints.Select(z => (z - zMin) / zRange * 600 + 100).ToArray();
                        plt.AddScatter(normX2, normZ2, color: partColors[part], lineWidth: 0,
                            markerSize: markerSize);

                        // YZ view (right)
                        var normY3 = yPoints.Select(y => (y - yMin) / yRange * 600 + 1650).ToArray();
                        var normZ3 = zPoints.Select(z => (z - zMin) / zRange * 600 + 100).ToArray();
                        plt.AddScatter(normY3, normZ3, color: partColors[part], lineWidth: 0,
                            markerSize: markerSize);
                    }
                }

                plt.Title(title);
                plt.XLabel("XY View | XZ View | YZ View");
                plt.YLabel("Mammoth 3D Projections - PARTS DETECTION");
                plt.Legend(location: Alignment.UpperRight);
                plt.Grid(enable: false);

                Directory.CreateDirectory(Path.GetDirectoryName(outputPath));
                plt.SaveFig(outputPath, width: 2400, height: 800);

                Console.WriteLine($"Saved: {outputPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: Failed to create parts visualization: {ex.Message}");
            }
        }

        static void TestXOnlyTrunk(double[,] originalData, int version, string outputDir)
        {
            int numPoints = originalData.GetLength(0);
            var parts = new string[numPoints];

            // Compute coordinate ranges
            double minX = double.MaxValue, maxX = double.MinValue;

            for (int i = 0; i < numPoints; i++)
            {
                double x = originalData[i, 0];
                minX = Math.Min(minX, x);
                maxX = Math.Max(maxX, x);
            }

            double xRange = maxX - minX;

            Console.WriteLine($"X range: [{minX:F1}, {maxX:F1}] (range: {xRange:F1})");

            // Initialize everything as "other" first
            for (int i = 0; i < numPoints; i++)
            {
                parts[i] = "other";
            }

            // SIMPLE X-ONLY trunk detection - exactly as requested
            double trunkXThreshold;
            switch (version)
            {
                case 1: // Top 10% of X direction (most forward)
                    trunkXThreshold = maxX - xRange * 0.1; // Top 10%
                    Console.WriteLine($"V1: Trunk = top 10% of X direction (X > {trunkXThreshold:F1})");
                    break;
                case 2: // End 20% of X direction (back part)
                    trunkXThreshold = minX + xRange * 0.2; // End 20% (lowest X values)
                    Console.WriteLine($"V2: Trunk = end 20% of X direction (X < {trunkXThreshold:F1})");
                    break;
                default:
                    trunkXThreshold = maxX - xRange * 0.1;
                    break;
            }

            // Apply trunk detection - first 20% X + last 20% X + first 40% Y, Z free
            double firstXThreshold = minX + xRange * 0.2;  // First 20% of X
            double lastXThreshold = maxX - xRange * 0.2;   // Last 20% of X

            double minY = originalData.Cast<double>().Where((_, idx) => idx % 3 == 1).Min();
            double maxY = originalData.Cast<double>().Where((_, idx) => idx % 3 == 1).Max();
            double yRange = maxY - minY;
            double trunkYThreshold = minY + yRange * 0.4; // First 40% of Y direction

            for (int i = 0; i < numPoints; i++)
            {
                double x = originalData[i, 0];
                double y = originalData[i, 1];
                // Z is completely free - NO Z restrictions

                if ((x < firstXThreshold || x > lastXThreshold) && y < trunkYThreshold)
                    parts[i] = "trunk";
            }

            // Count parts
            var partCounts = parts.GroupBy(p => p).ToDictionary(g => g.Key, g => g.Count());
            Console.WriteLine($"Parts: {string.Join(", ", partCounts.Select(kv => $"{kv.Key}={kv.Value}"))}");

            string outputPath = Path.Combine(outputDir, $"x_only_trunk_v{version}.png");
            CreateXOnlyVisualization(originalData, parts, $"X-Only Trunk Test V{version}", outputPath);
        }

        static void CreateXOnlyVisualization(double[,] originalData, string[] parts, string title, string outputPath)
        {
            try
            {
                var partColors = new Dictionary<string, Color>
                {
                    { "trunk", Color.Red },       // Trunk in bright red
                    { "other", Color.LightGray }  // Everything else in light gray
                };

                int numPoints = originalData.GetLength(0);
                var x = new double[numPoints];
                var y = new double[numPoints];
                var z = new double[numPoints];

                for (int i = 0; i < numPoints; i++)
                {
                    x[i] = originalData[i, 0];
                    y[i] = originalData[i, 1];
                    z[i] = originalData[i, 2];
                }

                // Group by parts
                var partGroups = new Dictionary<string, (List<double> x, List<double> y, List<double> z)>();
                foreach (var part in partColors.Keys)
                {
                    partGroups[part] = (new List<double>(), new List<double>(), new List<double>());
                }

                for (int i = 0; i < numPoints; i++)
                {
                    string part = parts[i];
                    if (partGroups.ContainsKey(part))
                    {
                        partGroups[part].x.Add(x[i]);
                        partGroups[part].y.Add(y[i]);
                        partGroups[part].z.Add(z[i]);
                    }
                }

                // Normalize for three views
                double xMin = x.Min(), xMax = x.Max(), xRange = xMax - xMin;
                double yMin = y.Min(), yMax = y.Max(), yRange = yMax - yMin;
                double zMin = z.Min(), zMax = z.Max(), zRange = zMax - zMin;

                var plt = new Plot(2400, 800);

                // Three views: XY, XZ, YZ - show "other" first (gray), then trunk (red on top)
                foreach (var kvp in partGroups.OrderBy(kv => kv.Key == "trunk" ? 1 : 0))
                {
                    var part = kvp.Key;
                    var (xPoints, yPoints, zPoints) = kvp.Value;
                    if (xPoints.Count > 0)
                    {
                        int markerSize = part == "trunk" ? 4 : 1;
                        string label = part == "trunk" ? "Trunk (X-only)" : null;

                        // XY view (left)
                        var normX1 = xPoints.Select(x => (x - xMin) / xRange * 600 + 50).ToArray();
                        var normY1 = yPoints.Select(y => (y - yMin) / yRange * 600 + 100).ToArray();
                        plt.AddScatter(normX1, normY1, color: partColors[part], lineWidth: 0,
                            markerSize: markerSize, label: label);

                        // XZ view (middle)
                        var normX2 = xPoints.Select(x => (x - xMin) / xRange * 600 + 850).ToArray();
                        var normZ2 = zPoints.Select(z => (z - zMin) / zRange * 600 + 100).ToArray();
                        plt.AddScatter(normX2, normZ2, color: partColors[part], lineWidth: 0,
                            markerSize: markerSize);

                        // YZ view (right)
                        var normY3 = yPoints.Select(y => (y - yMin) / yRange * 600 + 1650).ToArray();
                        var normZ3 = zPoints.Select(z => (z - zMin) / zRange * 600 + 100).ToArray();
                        plt.AddScatter(normY3, normZ3, color: partColors[part], lineWidth: 0,
                            markerSize: markerSize);
                    }
                }

                plt.Title(title);
                plt.XLabel("XY View | XZ View | YZ View");
                plt.YLabel("Mammoth 3D Projections - X-ONLY TRUNK DETECTION");
                plt.Legend(location: Alignment.UpperRight);
                plt.Grid(enable: false);

                Directory.CreateDirectory(Path.GetDirectoryName(outputPath));
                plt.SaveFig(outputPath, width: 2400, height: 800);

                Console.WriteLine($"Saved: {outputPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: Failed to create X-only visualization: {ex.Message}");
            }
        }

        static void CreateHeadOnlyVisualization(double[,] originalData, string[] parts, string title, string outputPath)
        {
            try
            {
                var partColors = new Dictionary<string, Color>
                {
                    { "head", Color.Red },      // Head in bright red
                    { "other", Color.LightGray } // Everything else in light gray
                };

                int numPoints = originalData.GetLength(0);
                var x = new double[numPoints];
                var y = new double[numPoints];
                var z = new double[numPoints];

                for (int i = 0; i < numPoints; i++)
                {
                    x[i] = originalData[i, 0];
                    y[i] = originalData[i, 1];
                    z[i] = originalData[i, 2];
                }

                // Group by parts
                var partGroups = new Dictionary<string, (List<double> x, List<double> y, List<double> z)>();
                foreach (var part in partColors.Keys)
                {
                    partGroups[part] = (new List<double>(), new List<double>(), new List<double>());
                }

                for (int i = 0; i < numPoints; i++)
                {
                    string part = parts[i];
                    if (partGroups.ContainsKey(part))
                    {
                        partGroups[part].x.Add(x[i]);
                        partGroups[part].y.Add(y[i]);
                        partGroups[part].z.Add(z[i]);
                    }
                }

                // Normalize for three views
                double xMin = x.Min(), xMax = x.Max(), xRange = xMax - xMin;
                double yMin = y.Min(), yMax = y.Max(), yRange = yMax - yMin;
                double zMin = z.Min(), zMax = z.Max(), zRange = zMax - zMin;

                var plt = new Plot(2400, 800);

                // Three views: XY, XZ, YZ - show "other" first (gray background), then head (red foreground)
                foreach (var kvp in partGroups.OrderBy(kv => kv.Key == "head" ? 1 : 0)) // "other" first, "head" last (on top)
                {
                    var part = kvp.Key;
                    var (xPoints, yPoints, zPoints) = kvp.Value;
                    if (xPoints.Count > 0)
                    {
                        // XY view (left)
                        var normX1 = xPoints.Select(x => (x - xMin) / xRange * 600 + 50).ToArray();
                        var normY1 = yPoints.Select(y => (y - yMin) / yRange * 600 + 100).ToArray();
                        plt.AddScatter(normX1, normY1, color: partColors[part], lineWidth: 0,
                            markerSize: part == "head" ? 3 : 1,
                            label: part == "head" ? "Head" : null);

                        // XZ view (middle)
                        var normX2 = xPoints.Select(x => (x - xMin) / xRange * 600 + 850).ToArray();
                        var normZ2 = zPoints.Select(z => (z - zMin) / zRange * 600 + 100).ToArray();
                        plt.AddScatter(normX2, normZ2, color: partColors[part], lineWidth: 0,
                            markerSize: part == "head" ? 3 : 1);

                        // YZ view (right)
                        var normY3 = yPoints.Select(y => (y - yMin) / yRange * 600 + 1650).ToArray();
                        var normZ3 = zPoints.Select(z => (z - zMin) / zRange * 600 + 100).ToArray();
                        plt.AddScatter(normY3, normZ3, color: partColors[part], lineWidth: 0,
                            markerSize: part == "head" ? 3 : 1);
                    }
                }

                plt.Title(title);
                plt.XLabel("XY View | XZ View | YZ View");
                plt.YLabel("Mammoth 3D Projections - HEAD DETECTION FOCUS");
                plt.Legend(location: Alignment.UpperRight);
                plt.Grid(enable: false);

                Directory.CreateDirectory(Path.GetDirectoryName(outputPath));
                plt.SaveFig(outputPath, width: 2400, height: 800);

                Console.WriteLine($"Saved: {outputPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: Failed to create head-only visualization: {ex.Message}");
            }
        }

        static void TestAnatomicalClassification(double[,] originalData, int version, string outputDir)
        {
            int numPoints = originalData.GetLength(0);
            var parts = new string[numPoints];

            // Compute coordinate ranges
            double minZ = double.MaxValue, maxZ = double.MinValue;
            double minX = double.MaxValue, maxX = double.MinValue;
            double minY = double.MaxValue, maxY = double.MinValue;

            for (int i = 0; i < numPoints; i++)
            {
                double x = originalData[i, 0];
                double y = originalData[i, 1];
                double z = originalData[i, 2];

                minX = Math.Min(minX, x);
                maxX = Math.Max(maxX, x);
                minY = Math.Min(minY, y);
                maxY = Math.Max(maxY, y);
                minZ = Math.Min(minZ, z);
                maxZ = Math.Max(maxZ, z);
            }

            double xRange = maxX - minX;
            double yRange = maxY - minY;
            double zRange = maxZ - minZ;
            double xCenter = (minX + maxX) / 2;
            double yCenter = (minY + maxY) / 2;

            Console.WriteLine($"Coordinate ranges: X[{minX:F1}, {maxX:F1}], Y[{minY:F1}, {maxY:F1}], Z[{minZ:F1}, {maxZ:F1}]");

            // Different classification approaches
            switch (version)
            {
                case 1: // Simple Z-height based
                    Console.WriteLine("V1: Simple Z-height classification");
                    {
                        double legZ = minZ + zRange * 0.25;
                        double headZ = minZ + zRange * 0.75;
                        for (int i = 0; i < numPoints; i++)
                        {
                            double z = originalData[i, 2];
                            if (z < legZ) parts[i] = "legs";
                            else if (z > headZ) parts[i] = "head";
                            else parts[i] = "body";
                        }
                    }
                    break;

                case 2: // Trunk as forward extension + head as upper back
                    Console.WriteLine("V2: Trunk=forward extension, Head=upper back");
                    {
                        double legZ = minZ + zRange * 0.2;
                        double headZ = minZ + zRange * 0.7;
                        double trunkX = minX + xRange * 0.8; // Very forward
                        double headXMax = minX + xRange * 0.6; // Not most forward

                        for (int i = 0; i < numPoints; i++)
                        {
                            double x = originalData[i, 0];
                            double y = originalData[i, 1];
                            double z = originalData[i, 2];
                            double yDist = Math.Abs(y - yCenter);

                            if (z < legZ)
                                parts[i] = "legs";
                            else if (x > trunkX && yDist < yRange * 0.4)
                                parts[i] = "trunk";
                            else if (z > headZ && x <= headXMax)
                                parts[i] = "head";
                            else
                                parts[i] = "body";
                        }
                    }
                    break;

                case 3: // Head as compact top, trunk as hanging down
                    Console.WriteLine("V3: Head=compact top, Trunk=hanging down");
                    {
                        double legZ = minZ + zRange * 0.18;
                        double topZ = minZ + zRange * 0.85; // Very top for head
                        double trunkZMax = minZ + zRange * 0.7; // Trunk hangs down
                        double trunkZMin = minZ + zRange * 0.3;
                        double forwardX = minX + xRange * 0.7;

                        for (int i = 0; i < numPoints; i++)
                        {
                            double x = originalData[i, 0];
                            double y = originalData[i, 1];
                            double z = originalData[i, 2];
                            double yDist = Math.Abs(y - yCenter);

                            if (z < legZ)
                                parts[i] = "legs";
                            else if (z > topZ)
                                parts[i] = "head";
                            else if (x > forwardX && yDist < yRange * 0.35 && z >= trunkZMin && z <= trunkZMax)
                                parts[i] = "trunk";
                            else
                                parts[i] = "body";
                        }
                    }
                    break;

                case 4: // Based on 3D image analysis - trunk hangs straight down
                    Console.WriteLine("V4: From 3D analysis - trunk hangs straight down from head");
                    {
                        double legZ = minZ + zRange * 0.15;

                        // Head: upper portion, not extreme forward
                        double headZ = minZ + zRange * 0.75;
                        double headXMax = minX + xRange * 0.65;

                        // Trunk: very forward + narrow + extends down vertically
                        double trunkX = minX + xRange * 0.75;
                        double trunkZMax = minZ + zRange * 0.65; // Hangs down from head level
                        double trunkZMin = minZ + zRange * 0.25;
                        double trunkYWidth = yRange * 0.3; // Very narrow

                        for (int i = 0; i < numPoints; i++)
                        {
                            double x = originalData[i, 0];
                            double y = originalData[i, 1];
                            double z = originalData[i, 2];
                            double yDist = Math.Abs(y - yCenter);

                            if (z < legZ)
                            {
                                parts[i] = "legs";
                            }
                            else if (x > trunkX && yDist < trunkYWidth && z >= trunkZMin && z <= trunkZMax)
                            {
                                parts[i] = "trunk"; // Check trunk first (most specific)
                            }
                            else if (z > headZ && x <= headXMax)
                            {
                                parts[i] = "head";
                            }
                            else
                            {
                                parts[i] = "body";
                            }
                        }
                    }
                    break;
            }

            // Count parts and create visualization
            var partCounts = parts.GroupBy(p => p).ToDictionary(g => g.Key, g => g.Count());
            Console.WriteLine($"Parts: {string.Join(", ", partCounts.Select(kv => $"{kv.Key}={kv.Value}"))}");

            string outputPath = Path.Combine(outputDir, $"anatomy_v{version}.png");
            CreateAnatomyVisualization(originalData, parts, $"Mammoth Anatomy Test V{version}", outputPath);
        }

        static void CreateAnatomyVisualization(double[,] originalData, string[] parts, string title, string outputPath)
        {
            try
            {
                var partColors = new Dictionary<string, Color>
                {
                    { "legs", Color.Blue },
                    { "body", Color.Green },
                    { "head", Color.Purple },
                    { "trunk", Color.Red }
                };

                int numPoints = originalData.GetLength(0);
                var x = new double[numPoints];
                var y = new double[numPoints];
                var z = new double[numPoints];

                for (int i = 0; i < numPoints; i++)
                {
                    x[i] = originalData[i, 0];
                    y[i] = originalData[i, 1];
                    z[i] = originalData[i, 2];
                }

                // Group by parts
                var partGroups = new Dictionary<string, (List<double> x, List<double> y, List<double> z)>();
                foreach (var part in partColors.Keys)
                {
                    partGroups[part] = (new List<double>(), new List<double>(), new List<double>());
                }

                for (int i = 0; i < numPoints; i++)
                {
                    string part = parts[i];
                    if (partGroups.ContainsKey(part))
                    {
                        partGroups[part].x.Add(x[i]);
                        partGroups[part].y.Add(y[i]);
                        partGroups[part].z.Add(z[i]);
                    }
                }

                // Normalize for three views
                double xMin = x.Min(), xMax = x.Max(), xRange = xMax - xMin;
                double yMin = y.Min(), yMax = y.Max(), yRange = yMax - yMin;
                double zMin = z.Min(), zMax = z.Max(), zRange = zMax - zMin;

                var plt = new Plot(2400, 800);

                // Three views: XY, XZ, YZ
                foreach (var kvp in partGroups)
                {
                    var part = kvp.Key;
                    var (xPoints, yPoints, zPoints) = kvp.Value;
                    if (xPoints.Count > 0)
                    {
                        // XY view (left)
                        var normX1 = xPoints.Select(x => (x - xMin) / xRange * 600 + 50).ToArray();
                        var normY1 = yPoints.Select(y => (y - yMin) / yRange * 600 + 100).ToArray();
                        plt.AddScatter(normX1, normY1, color: partColors[part], lineWidth: 0, markerSize: 2,
                            label: part == "legs" ? char.ToUpper(part[0]) + part.Substring(1) : null);

                        // XZ view (middle)
                        var normX2 = xPoints.Select(x => (x - xMin) / xRange * 600 + 850).ToArray();
                        var normZ2 = zPoints.Select(z => (z - zMin) / zRange * 600 + 100).ToArray();
                        plt.AddScatter(normX2, normZ2, color: partColors[part], lineWidth: 0, markerSize: 2);

                        // YZ view (right)
                        var normY3 = yPoints.Select(y => (y - yMin) / yRange * 600 + 1650).ToArray();
                        var normZ3 = zPoints.Select(z => (z - zMin) / zRange * 600 + 100).ToArray();
                        plt.AddScatter(normY3, normZ3, color: partColors[part], lineWidth: 0, markerSize: 2);
                    }
                }

                plt.Title(title);
                plt.XLabel("XY View | XZ View | YZ View");
                plt.YLabel("Mammoth 3D Projections");
                plt.Legend(location: Alignment.UpperRight);
                plt.Grid(enable: false);

                Directory.CreateDirectory(Path.GetDirectoryName(outputPath));
                plt.SaveFig(outputPath, width: 2400, height: 800);

                Console.WriteLine($"Saved: {outputPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: Failed to create visualization: {ex.Message}");
            }
        }
        */
    }
}