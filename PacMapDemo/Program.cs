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
        static void Main(string[] args)
        {
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

                Console.WriteLine();
                Console.WriteLine(new string('=', 60));
                Console.WriteLine(" DEMO: Mammoth 3D Point Cloud (Topological Structure)");
                Console.WriteLine(new string('=', 60));
                Console.WriteLine(" DEMO 2 EXPLANATION:");
                Console.WriteLine("   This demo tests PacMAP on 3D spatial data (3 dimensions).");
                Console.WriteLine("   Mammoth data contains X,Y,Z coordinates forming a 3D mammoth shape.");
                Console.WriteLine("   Goal: Reduce 3D → 2D while preserving the mammoth's topology/shape.");
                Console.WriteLine("   Success criteria: 2D embedding should still look like a mammoth.");
                Console.WriteLine("   This tests PacMAP's ability to preserve global structure and shape.");
                Console.WriteLine();
                DemoMammoth(outputDir);

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

                Console.WriteLine();
                Console.WriteLine("  Opening visualization files...");

                // Open the results folder
                OpenFileOrFolder(fullOutputPath, true);

                // Open each image file
                foreach (var imagePath in imageFiles)
                {
                    Console.WriteLine($" Opening: {Path.GetFileName(imagePath)}");
                    OpenFileOrFolder(imagePath, false);
                    System.Threading.Thread.Sleep(500); // Small delay between opens
                }

            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: Error: {ex.Message}");
                Console.WriteLine($"📍 Stack trace: {ex.StackTrace}");
                Environment.Exit(1);
            }

            Console.WriteLine("\nPress any key to exit...");
            Console.ReadKey();
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

                string mnistCsvPath = Path.Combine(outputDir, "mnist_embedding.csv");
                Visualizer.SaveEmbeddingAsCSV(embedding2D, null, mnistCsvPath); // No labels for now

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

        static void DemoMammoth(string outputDir)
        {
            var stopwatch = Stopwatch.StartNew();

            try
            {
                // Load mammoth data
                string mammothPath = Path.Combine("Data", "mammoth_data.csv");

                var mammothData = DataLoaders.LoadMammothData(mammothPath, maxSamples: 8000);

                Console.WriteLine();
                Console.WriteLine(" Running PacMAP on mammoth data...");

                Console.WriteLine(" PacMAP Configuration for Mammoth 3D:");
                Console.WriteLine($"   Neighbors: 10 (Lower for preserving local topology)");
                Console.WriteLine($"   Embedding Dimensions: 2");
                Console.WriteLine($"   Input Dimensions: 3 (X, Y, Z coordinates)");
                Console.WriteLine($"   Data Type: 3D point cloud forming mammoth shape");
                Console.WriteLine($"   Goal: Preserve mammoth's topological structure in 2D");
                Console.WriteLine($"   Normalization: ZScore (standardizes coordinate scales)");
                Console.WriteLine($"   HNSW Use Case: Balanced (speed vs accuracy for 3D data)");
                Console.WriteLine();

                // Create PacMAP model with progress reporting - REAL MODEL
                using var model = new PacMAPModel();

                // Define progress callback
                void ProgressHandler(string phase, int current, int total, float percent, string? message)
                {
                    Console.WriteLine($"[{phase,-12}] {percent,3:F0}% ({current,4}/{total,-4}) - {message ?? "Processing..."}");
                }

                // Fit and transform with progress - REAL MODEL DEBUGGING
                Console.WriteLine("DEBUG: DEBUGGING MODE: Using REAL Rust model with timeout check");
                Console.WriteLine("DEBUG: Using 10 neighbors for mammoth (lower for preserving local 3D topology)");

                // Use full mammoth dataset to test embedding quality
                int testSize = mammothData.GetLength(0); // Use all 8000 points
                var testMammothData = new double[testSize, 3];
                for (int i = 0; i < testSize; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        testMammothData[i, j] = mammothData[i, j];
                    }
                }

                Console.WriteLine("DEBUG: Using FULL mammoth dataset to test embedding quality:");

                Console.WriteLine("DEBUG: Parameters being used:");
                Console.WriteLine($"   - Neighbors: 10");
                Console.WriteLine($"   - Embedding dimensions: 2");
                Console.WriteLine($"   - Force exact KNN: TRUE (HNSW disabled)");
                Console.WriteLine($"   - Use quantization: FALSE (quantization disabled)");
                Console.WriteLine($"   - Distance metric: Euclidean");
                Console.WriteLine($"   - Random seed: 42");
                Console.WriteLine($"   - Input data shape: {testSize} samples × {testMammothData.GetLength(1)} features (FULL DATASET)");
                Console.WriteLine();

                Console.WriteLine("TIMING:  Starting PacMAP with FULL mammoth dataset...");
                Console.WriteLine("TIMING:  This should now use HNSW for 8000 points and produce better mammoth shape");

                var result = model.Fit(testMammothData, embeddingDimensions: 2, neighbors: 10,
                                     normalization: PacMAPSharp.NormalizationMode.ZScore,
                                     metric: PacMAPSharp.DistanceMetric.Euclidean,
                                     hnswUseCase: PacMAPSharp.HnswUseCase.Balanced,
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
                Console.WriteLine($"   Features: {modelInfo.InputDimension}");
                Console.WriteLine($"   Embedding dimensions: {modelInfo.OutputDimension}");
                Console.WriteLine($"   Distance metric: {modelInfo.Metric}");
                Console.WriteLine($"   Normalization: {modelInfo.Normalization}");
                Console.WriteLine($"   HNSW used: {modelInfo.UsedHNSW}");
                Console.WriteLine();

                // Create visualizations
                Console.WriteLine("Creating mammoth visualizations...");

                // Plot real 3D data using OxyPlot only
                string real3DPlotPath = Path.Combine(outputDir, "mammoth_original_3d_real.png");
                Visualizer.PlotOriginalMammoth3DReal(mammothData, "Mammoth Original - Real 3D Visualization (OxyPlot)", real3DPlotPath);

                // Convert embedding to proper format for visualizer
                var embedding2D = ConvertEmbeddingTo2D(embedding, testSize, 2);

                // Plot PacMAP embedding using OxyPlot
                string mammothPlotPath = Path.Combine(outputDir, "mammoth_pacmap_embedding.png");
                Visualizer.PlotMammothPacMAP(embedding2D, mammothData, "Mammoth Dataset - PacMAP 2D Embedding with Anatomical Parts", mammothPlotPath);

                string mammothCsvPath = Path.Combine(outputDir, "mammoth_embedding.csv");
                Visualizer.SaveEmbeddingAsCSV(embedding2D, null, mammothCsvPath);

                // Save model
                string modelPath = Path.Combine(outputDir, "mammoth_pacmap_model.bin");
                Console.WriteLine($"💾 Saving mammoth model: {modelPath}");
                model.Save(modelPath);

                Console.WriteLine("SUCCESS: Mammoth demo completed successfully!");

            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: Mammoth demo failed: {ex.Message}");
                throw;
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

            Console.WriteLine("\nPress any key to exit...");
            Console.ReadKey();
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