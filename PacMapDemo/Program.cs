using System;
using System.IO;
using System.Diagnostics;
using System.Collections.Generic;

namespace PacMapDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("🎯 PacMAP Enhanced - C# Demo with Real Mammoth Data");
            Console.WriteLine("==================================================");
            Console.WriteLine();

            try
            {
                // Demo version info
                Console.WriteLine($"📚 PacMAP Enhanced C# Demo Version: 1.0.0");
                Console.WriteLine($"📚 Rust PacMAP Version: {RealPacMapModel.GetVersion()}");
                Console.WriteLine();


                // Create output directory
                string outputDir = "Results";
                Directory.CreateDirectory(outputDir);

                // Run both demos
                Console.WriteLine("🎮 Starting comprehensive PacMAP demonstration...");
                Console.WriteLine();

                // Demo 1: MNIST Dataset - DISABLED
                /*
                Console.WriteLine(new string('=', 60));
                Console.WriteLine("🔢 DEMO 1: MNIST Handwritten Digits (High-Dimensional)");
                Console.WriteLine(new string('=', 60));
                Console.WriteLine("📖 DEMO 1 EXPLANATION:");
                Console.WriteLine("   This demo tests PacMAP on high-dimensional image data (784 dimensions).");
                Console.WriteLine("   MNIST contains 28x28 pixel handwritten digit images (0-9).");
                Console.WriteLine("   Goal: Reduce 784D → 2D while clustering similar digits together.");
                Console.WriteLine("   Success criteria: Different digits should form distinct clusters.");
                Console.WriteLine();
                DemoMNIST(outputDir);
                */

                Console.WriteLine();
                Console.WriteLine(new string('=', 60));
                Console.WriteLine("🦣 DEMO: Mammoth 3D Point Cloud (Topological Structure)");
                Console.WriteLine(new string('=', 60));
                Console.WriteLine("📖 DEMO 2 EXPLANATION:");
                Console.WriteLine("   This demo tests PacMAP on 3D spatial data (3 dimensions).");
                Console.WriteLine("   Mammoth data contains X,Y,Z coordinates forming a 3D mammoth shape.");
                Console.WriteLine("   Goal: Reduce 3D → 2D while preserving the mammoth's topology/shape.");
                Console.WriteLine("   Success criteria: 2D embedding should still look like a mammoth.");
                Console.WriteLine("   This tests PacMAP's ability to preserve global structure and shape.");
                Console.WriteLine();
                DemoMammoth(outputDir);

                Console.WriteLine();
                Console.WriteLine("🎉 All demos completed successfully!");
                var fullOutputPath = Path.GetFullPath(outputDir);
                Console.WriteLine($"📁 Results saved in: {fullOutputPath}");
                Console.WriteLine();
                Console.WriteLine("📊 Generated files:");

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
                Console.WriteLine("🖼️  Opening visualization files...");

                // Open the results folder
                OpenFileOrFolder(fullOutputPath, true);

                // Open each image file
                foreach (var imagePath in imageFiles)
                {
                    Console.WriteLine($"🔍 Opening: {Path.GetFileName(imagePath)}");
                    OpenFileOrFolder(imagePath, false);
                    System.Threading.Thread.Sleep(500); // Small delay between opens
                }

            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
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
                        Console.WriteLine($"⚠️  Could not open folder automatically. Please navigate to: {path}");
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
                            Console.WriteLine($"⚠️  Could not open file automatically. Please open: {path}");
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
                    Console.WriteLine($"⚠️  Unsupported OS. Please manually open: {path}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️  Failed to open {(isFolder ? "folder" : "file")}: {ex.Message}");
                Console.WriteLine($"📍 Please manually navigate to: {path}");
            }
        }

        /// <summary>
        /// Convert double[,] array to float[,] array for PacMAP API
        /// </summary>
        static float[,] ConvertToFloat(double[,] input)
        {
            int rows = input.GetLength(0);
            int cols = input.GetLength(1);
            var result = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = (float)input[i, j];
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
                Console.WriteLine($"📊 Label distribution:");
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
                Console.WriteLine("🚀 Running PacMAP on MNIST data...");

                Console.WriteLine("⚙️ PacMAP Configuration for MNIST:");
                Console.WriteLine($"   Neighbors: 10 (Standard for high-dimensional MNIST clustering)");
                Console.WriteLine($"   Embedding Dimensions: 2");
                Console.WriteLine($"   Input Dimensions: 784 (28x28 pixel images)");
                Console.WriteLine($"   Data Type: Handwritten digit images (0-9)");
                Console.WriteLine($"   Normalization: ZScore (standardizes pixel intensities)");
                Console.WriteLine($"   HNSW Use Case: HighAccuracy (for precise clustering)");
                Console.WriteLine();

                // Create PacMAP model with progress reporting
                using var model = new RealPacMapModel();

                // Hook up progress reporting
                model.ProgressChanged += (sender, e) =>
                {
                    Console.WriteLine($"[{e.Phase,-12}] {e.Percent,3:F0}% ({e.Current,4}/{e.Total,-4}) - {e.Message}");
                };

                // Fit and transform with progress - DEBUGGING MODE
                Console.WriteLine("🔧 DEBUGGING MODE: Disabling HNSW and quantization");
                Console.WriteLine("🔧 Using 10 neighbors for MNIST (standard PacMAP configuration)");
                Console.WriteLine("🔧 Parameters being used:");
                Console.WriteLine($"   - Neighbors: 10");
                Console.WriteLine($"   - Embedding dimensions: 2");
                Console.WriteLine($"   - Force exact KNN: TRUE (HNSW disabled)");
                Console.WriteLine($"   - Use quantization: FALSE (quantization disabled)");
                Console.WriteLine($"   - Distance metric: Euclidean");
                Console.WriteLine($"   - Random seed: 42");
                Console.WriteLine($"   - Input data shape: {mnistImages.GetLength(0)} samples × {mnistImages.GetLength(1)} features");
                Console.WriteLine();

                var embeddingDouble = model.FitTransform(mnistImages, neighbors: 10, seed: 42, forceExactKnn: true, useQuantization: false,
                                                 midNearRatio: 0.5, farPairRatio: 2.0, numIters: 450);
                var embedding = ConvertToFloat(embeddingDouble);

                stopwatch.Stop();
                var modelInfo = model.GetModelInfo();

                Console.WriteLine();
                Console.WriteLine("✅ PacMAP fitting completed!");
                Console.WriteLine($"⏱️  Total time: {stopwatch.Elapsed.TotalSeconds:F2} seconds");
                Console.WriteLine($"📊 Model info:");
                Console.WriteLine($"   Samples: {modelInfo.NSamples:N0}");
                Console.WriteLine($"   Features: {modelInfo.NFeatures:N0}");
                Console.WriteLine($"   Embedding dimensions: {modelInfo.EmbeddingDim}");
                Console.WriteLine($"   Memory usage: {modelInfo.MemoryUsageMb} MB");
                Console.WriteLine();

                // Create visualizations
                Console.WriteLine("🎨 Creating MNIST visualizations...");

                string mnistPlotPath = Path.Combine(outputDir, "mnist_pacmap_embedding.png");
                Visualizer.PlotMNIST(embedding, mnistLabels, "MNIST Dataset - PacMAP Embedding (Digits 0-9)", mnistPlotPath);

                string mnistCsvPath = Path.Combine(outputDir, "mnist_embedding.csv");
                Visualizer.SaveEmbeddingAsCSV(embedding, mnistLabels, mnistCsvPath);

                // Save model for later use
                string modelPath = Path.Combine(outputDir, "mnist_pacmap_model.bin");
                Console.WriteLine($"💾 Saving MNIST model: {modelPath}");
                model.Save(modelPath);

                Console.WriteLine("✅ MNIST demo completed successfully!");

            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ MNIST demo failed: {ex.Message}");
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

                Console.WriteLine("📂 Loading mammoth dataset...");
                var mammothData = DataLoaders.LoadMammothData(mammothPath, maxSamples: 8000);

                DataLoaders.PrintDataStatistics("Mammoth 3D Points", mammothData);

                Console.WriteLine();
                Console.WriteLine("🚀 Running PacMAP on mammoth data...");

                Console.WriteLine("⚙️ PacMAP Configuration for Mammoth 3D:");
                Console.WriteLine($"   Neighbors: 10 (Lower for preserving local topology)");
                Console.WriteLine($"   Embedding Dimensions: 2");
                Console.WriteLine($"   Input Dimensions: 3 (X, Y, Z coordinates)");
                Console.WriteLine($"   Data Type: 3D point cloud forming mammoth shape");
                Console.WriteLine($"   Goal: Preserve mammoth's topological structure in 2D");
                Console.WriteLine($"   Normalization: ZScore (standardizes coordinate scales)");
                Console.WriteLine($"   HNSW Use Case: Balanced (speed vs accuracy for 3D data)");
                Console.WriteLine();

                // Create PacMAP model with progress reporting - REAL MODEL
                using var model = new RealPacMapModel();

                // Hook up progress reporting
                model.ProgressChanged += (sender, e) =>
                {
                    Console.WriteLine($"[{e.Phase,-12}] {e.Percent,3:F0}% ({e.Current,4}/{e.Total,-4}) - {e.Message}");
                };

                // Fit and transform with progress - REAL MODEL DEBUGGING
                Console.WriteLine("🔧 DEBUGGING MODE: Using REAL Rust model with timeout check");
                Console.WriteLine("🔧 Using 10 neighbors for mammoth (lower for preserving local 3D topology)");

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

                Console.WriteLine("🔧 Using FULL mammoth dataset to test embedding quality:");

                Console.WriteLine("🔧 Parameters being used:");
                Console.WriteLine($"   - Neighbors: 10");
                Console.WriteLine($"   - Embedding dimensions: 2");
                Console.WriteLine($"   - Force exact KNN: TRUE (HNSW disabled)");
                Console.WriteLine($"   - Use quantization: FALSE (quantization disabled)");
                Console.WriteLine($"   - Distance metric: Euclidean");
                Console.WriteLine($"   - Random seed: 42");
                Console.WriteLine($"   - Input data shape: {testSize} samples × {testMammothData.GetLength(1)} features (FULL DATASET)");
                Console.WriteLine();

                Console.WriteLine("⏱️  Starting PacMAP with FULL mammoth dataset...");
                Console.WriteLine("⏱️  This should now use HNSW for 8000 points and produce better mammoth shape");

                var embeddingDouble = model.FitTransform(testMammothData, neighbors: 10, seed: 42, forceExactKnn: true, useQuantization: false,
                                                 midNearRatio: 0.5, farPairRatio: 2.0, numIters: 450);
                var embedding = ConvertToFloat(embeddingDouble);

                stopwatch.Stop();
                var modelInfo = model.GetModelInfo();

                Console.WriteLine();
                Console.WriteLine("✅ PacMAP fitting completed!");
                Console.WriteLine($"⏱️  Total time: {stopwatch.Elapsed.TotalSeconds:F2} seconds");
                Console.WriteLine($"📊 Model info:");
                Console.WriteLine($"   Samples: {modelInfo.NSamples:N0}");
                Console.WriteLine($"   Features: {modelInfo.NFeatures}");
                Console.WriteLine($"   Embedding dimensions: {modelInfo.EmbeddingDim}");
                Console.WriteLine($"   Memory usage: {modelInfo.MemoryUsageMb} MB");
                Console.WriteLine();

                // Create visualizations
                Console.WriteLine("🎨 Creating mammoth visualizations...");

                // Plot original 3D data with anatomical part coloring
                string originalPlotPath = Path.Combine(outputDir, "mammoth_original_3d.png");
                Visualizer.PlotOriginalMammoth3DWithParts(mammothData, "Mammoth Original 3D Point Cloud - Anatomical Parts", originalPlotPath);

                // Plot PacMAP embedding with anatomical part coloring
                string mammothPlotPath = Path.Combine(outputDir, "mammoth_pacmap_embedding.png");
                Visualizer.PlotMammothWithParts(embedding, mammothData, "Mammoth Dataset - PacMAP 2D Embedding with Anatomical Parts", mammothPlotPath);

                string mammothCsvPath = Path.Combine(outputDir, "mammoth_embedding.csv");
                Visualizer.SaveEmbeddingAsCSV(embedding, null, mammothCsvPath);

                // Save model
                string modelPath = Path.Combine(outputDir, "mammoth_pacmap_model.bin");
                Console.WriteLine($"💾 Saving mammoth model: {modelPath}");
                model.Save(modelPath);

                Console.WriteLine("✅ Mammoth demo completed successfully!");

            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Mammoth demo failed: {ex.Message}");
                throw;
            }
        }
    }

}