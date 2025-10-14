using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
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
    /// MNIST Demo Program
    /// Demonstrates loading and using MNIST data with the binary reader
    /// </summary>
    public class MnistDemo
    {
        /// <summary>
        /// Run MNIST demonstration
        /// </summary>
        public static void RunDemo()
        {
            Console.WriteLine("🔢 MNIST Binary Reader Demo");
            Console.WriteLine("=========================");

            // Create and open Results folder before starting
            var resultsDir = "Results";
            Directory.CreateDirectory(resultsDir);

            try
            {
                // Open Results folder in Windows Explorer
                Process.Start(new ProcessStartInfo
                {
                    FileName = Path.GetFullPath(resultsDir),
                    UseShellExecute = true
                });
                Console.WriteLine($"📂 Opened Results folder: {Path.GetFullPath(resultsDir)}");
                Console.WriteLine();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️ Could not open Results folder: {ex.Message}");
                Console.WriteLine();
            }

            try
            {
                // Path to the binary MNIST file
                string dataPath = Path.Combine("Data", "mnist_binary.dat.zip");

                if (!File.Exists(dataPath))
                {
                    Console.WriteLine($"❌ MNIST binary file not found: {dataPath}");
                    Console.WriteLine("   Please run the Python converter first:");
                    Console.WriteLine("   cd Data && python mnist_converter.py");
                    return;
                }

                Console.WriteLine($"📁 Loading MNIST data from: {dataPath}");

                // Load MNIST data
                var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                var mnistData = MnistReader.Read(dataPath);
                stopwatch.Stop();

                Console.WriteLine($"✅ Loaded in {stopwatch.Elapsed.TotalMilliseconds:F1} ms");
                Console.WriteLine();

                // Print dataset information
                MnistReader.PrintInfo(mnistData);
                Console.WriteLine();

                // Demonstrate data access
                Console.WriteLine("🔍 Data Access Examples:");
                Console.WriteLine("=====================");

                // Show some sample images and labels
                var samples = MnistReader.GetRandomSamples(mnistData, samplesPerDigit: 3, seed: 42);
                Console.WriteLine($"Random samples (showing first {Math.Min(10, samples.Length)}):");

                for (int i = 0; i < Math.Min(10, samples.Length); i++)
                {
                    var index = samples[i];
                    var label = mnistData.Labels?[index] ?? 0;
                    Console.WriteLine($"   Sample {i + 1}: Index {index:D5}, Label: {label}");
                }

                Console.WriteLine();

                // Demonstrate conversion to float array for PACMAP
                Console.WriteLine("🔄 Data Conversion for PACMAP:");
                Console.WriteLine("===============================");

                // Use reasonable subset of MNIST dataset (10,000 images) - full dataset too large for PACMAP
                var subsetSize = Math.Min(10000, mnistData.NumImages);
                var doubleData = mnistData.GetDoubleArray(0, subsetSize);
                var labels = mnistData.Labels?.Take(subsetSize).ToArray() ?? Array.Empty<byte>();

                Console.WriteLine($"   Using MNIST subset: {doubleData.GetLength(0):N0} images of {mnistData.NumImages:N0} total");
                Console.WriteLine($"   Shape: [{doubleData.GetLength(0):N0}, {doubleData.GetLength(1)}]");
                Console.WriteLine($"   Memory: {doubleData.Length * 8.0 / 1024 / 1024:F1} MB");
                Console.WriteLine($"   ✅ Data loaded as double pixels [0-255] - no conversion needed!");
                Console.WriteLine();

                // Show actual label distribution from subset
                Console.WriteLine("📊 Subset Label Distribution:");
                Console.WriteLine("============================");
                var labelCounts = new int[10];
                for (int i = 0; i < labels.Length; i++)
                {
                    labelCounts[labels[i]]++;
                }
                for (int digit = 0; digit < 10; digit++)
                {
                    var percentage = (labelCounts[digit] * 100.0) / labels.Length;
                    Console.WriteLine($"   Digit {digit}: {labelCounts[digit]:D4} samples ({percentage:F1}%)");
                }
                Console.WriteLine();

                // Show some statistics about the double data
                Console.WriteLine("📊 Double Data Statistics:");
                Console.WriteLine("========================");

                double minVal = doubleData.Cast<double>().Min();
                double maxVal = doubleData.Cast<double>().Max();
                double meanVal = doubleData.Cast<double>().Average();

                Console.WriteLine($"   Value range: [{minVal:F3}, {maxVal:F3}]");
                Console.WriteLine($"   Mean value: {meanVal:F3}");
                Console.WriteLine($"   Expected range: [0, 255] (raw pixel values)");
                Console.WriteLine();

                // Create MNIST sample visualization first
                CreateMnistSampleVisualization(mnistData);

                // Print file paths that will be created
                Console.WriteLine();
                Console.WriteLine("📁 Files that will be created:");
                var samplePath = Path.Combine(Directory.GetCurrentDirectory(), "Results", "mnist_samples_visualization.png");
                Console.WriteLine($"   🎨 MNIST sample visualization: {samplePath}");
                Console.WriteLine($"   📊 2D HNSW embedding: {Path.Combine(Directory.GetCurrentDirectory(), "Results", "mnist_2d_embedding.png")}");
                Console.WriteLine($"   📊 2D DirectKNN embedding: {Path.Combine(Directory.GetCurrentDirectory(), "Results", "mnist_2d_KNN_embedding.png")}");
                Console.WriteLine($"   📊 2D HNSW transform: {Path.Combine(Directory.GetCurrentDirectory(), "Results", "mnist_2d_transform.png")}");
                Console.WriteLine($"   📊 2D DirectKNN transform: {Path.Combine(Directory.GetCurrentDirectory(), "Results", "mnist_2d_KNN_transform.png")}");
                Console.WriteLine();

                // Create 2D embedding using helper function (HNSW) and save model + timing for transform
                var (pacmapHNSW, hnswFitTime) = CreateMnistEmbeddingWithModel(doubleData, labels, nNeighbors: 30, mnRatio: 0.5f, fpRatio: 2.0f,
                    name: "mnist_2d_embedding", folderName: "", directKNN: false);

                // Create DirectKNN embedding for comparison and save model + timing for transform
                var (pacmapKNN, knnFitTime) = CreateMnistEmbeddingWithModel(doubleData, labels, nNeighbors: 30, mnRatio: 0.5f, fpRatio: 2.0f,
                    name: "mnist_2d_KNN_embedding", folderName: "", directKNN: true);

                // Create Transform API experiments using previously fitted models
                CreateTransformExperiments(doubleData, labels, pacmapHNSW, pacmapKNN, hnswFitTime, knnFitTime);

                // Create neighborMNSTI folder with k=5 to 60 experiments (HNSW only)
                CreateNeighborMNSTI_Experiments(doubleData, labels);

                // Create MNSTMnRatio experiments with mnRatio from 0.5 to 2.0 (increments of 0.2)
                CreateMNSTMnRatio_Experiments(doubleData, labels);

                // Create MNSTfpRatio experiments with fpRatio from 0.5 to 4.0 (increments of 0.5)
                CreateMNSTfpRatio_Experiments(doubleData, labels);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
                Console.WriteLine($"   Stack trace: {ex.StackTrace}");
            }
        }

        /// <summary>
        /// Run PACMAP on MNIST subset
        /// </summary>
        public static void RunPacmapOnMnist(int subsetSize = 5000, int embeddingDim = 2)
        {
            Console.WriteLine($"🎯 PACMAP on MNIST Demo (Subset: {subsetSize:N0})");
            Console.WriteLine("========================================");

            try
            {
                // Load MNIST data
                string dataPath = Path.Combine("Data", "mnist_binary.dat.zip");
                if (!File.Exists(dataPath))
                {
                    Console.WriteLine("❌ Please run mnist_converter.py first");
                    return;
                }

                var mnistData = MnistReader.Read(dataPath);
                var actualSubsetSize = Math.Min(subsetSize, mnistData.NumImages);

                Console.WriteLine($"📊 Using {actualSubsetSize:N0} MNIST samples for PACMAP");

                // Convert to float array
                var floatData = mnistData.GetDoubleArray(0, actualSubsetSize);
                var labels = mnistData.Labels?.Take(actualSubsetSize).ToArray() ?? Array.Empty<byte>();

                Console.WriteLine($"   Data shape: [{floatData.GetLength(0)}, {floatData.GetLength(1)}]");
                Console.WriteLine($"   Label range: {labels.Min()}-{labels.Max()}");

                // This is where you would integrate with PACMAP
                Console.WriteLine();
                Console.WriteLine("🔄 Ready for PACMAP integration!");
                Console.WriteLine("   The data is prepared and can be passed to PacMapModel.Fit()");
                Console.WriteLine("   Use the floatData variable as the 'data' parameter");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
            }
        }

        /// <summary>
        /// Creates a visualization of actual MNIST digit images (0-9) using OxyPlot
        /// </summary>
        private static void CreateMnistSampleVisualization(MnistReader.MnistData mnistData)
        {
            Console.WriteLine("🎨 Creating MNIST Sample Visualization...");

            try
            {
                // Get random samples - one for each digit 0-9
                var random = new Random(42);
                var selectedSamples = new Dictionary<int, int>();

                for (int digit = 0; digit < 10; digit++)
                {
                    var digitIndices = new List<int>();
                    for (int i = 0; i < (mnistData.Labels?.Length ?? 0); i++)
                    {
                        if (mnistData.Labels != null && mnistData.Labels[i] == digit)
                            digitIndices.Add(i);
                    }

                    if (digitIndices.Count > 0)
                    {
                        var randomIndex = digitIndices[random.Next(digitIndices.Count)];
                        selectedSamples[digit] = randomIndex;
                        Console.WriteLine($"   Digit {digit}: Sample index {randomIndex:D5}");
                    }
                }

                // Create plot model for displaying actual digit images
                var plotModel = new PlotModel
                {
                    Title = "MNIST Digit Samples (0-9) - Actual Images",
                    Background = OxyColors.White
                };

                // Create image annotations for each digit
                for (int row = 0; row < 2; row++)
                {
                    for (int col = 0; col < 5; col++)
                    {
                        int digit = row * 5 + col;
                        if (digit >= 10) break;

                        if (selectedSamples.ContainsKey(digit))
                        {
                            var sampleIndex = selectedSamples[digit];
                            var image = mnistData.GetImageAsByteArray(sampleIndex);

                            // Create simple scatter plot representation of the digit pixels
                            CreateDigitPixelScatter(plotModel, image, digit, col, 1 - row);

                            // Add label below the image
                            var labelAnnotation = new TextAnnotation
                            {
                                Text = $"Label: {digit}",
                                TextPosition = new DataPoint(col, 0.7 - row),
                                TextHorizontalAlignment = HorizontalAlignment.Center,
                                TextVerticalAlignment = VerticalAlignment.Middle,
                                FontSize = 12,
                                FontWeight = FontWeights.Bold,
                                TextColor = OxyColors.Black
                            };
                            plotModel.Annotations.Add(labelAnnotation);
                        }
                    }
                }

                // Configure axes
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Minimum = -0.5, Maximum = 4.5, Title = "" });
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Minimum = -0.5, Maximum = 1.5, Title = "" });

                // Save to file
                var outputPath = Path.Combine("Results", "mnist_samples_visualization.png");
                Directory.CreateDirectory("Results");

                var exporter = new PngExporter { Width = 1200, Height = 500, Resolution = 300 };
                using var stream = File.Create(outputPath);
                exporter.Export(plotModel, stream);

                Console.WriteLine($"✅ MNIST sample visualization saved: {outputPath}");
                Console.WriteLine($"   📊 Sample file: {Path.Combine(Directory.GetCurrentDirectory(), outputPath)}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error creating sample visualization: {ex.Message}");
            }
        }

        /// <summary>
        /// Creates a scatter plot representation of MNIST digit pixels
        /// </summary>
        private static void CreateDigitPixelScatter(PlotModel plotModel, byte[,] image, int digit, double centerX, double centerY)
        {
            var scatterSeries = new ScatterSeries
            {
                Title = $"Digit {digit}",
                MarkerType = MarkerType.Square,
                MarkerSize = 1.0,  // HUGE for clear digit visualization
                MarkerFill = OxyColors.Black,
                MarkerStroke = OxyColors.Black
            };

            int height = image.GetLength(0);
            int width = image.GetLength(1);

            // Scale and center the digit within the plot area
            double scale = 0.4; // Scale to fit within the 0.8x0.8 area
            double offsetX = centerX - scale / 2;
            double offsetY = centerY - scale / 2;

            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    // Use original pixel value (0 = white background, 255 = black ink)
                    var pixelValue = image[h, w];

                    // Only plot dark pixels (ink) - MNIST has black ink on white background
                    if (pixelValue > 128) // High values = black ink
                    {
                        double x = offsetX + (w / (double)width) * scale;
                        // Flip Y-axis to display digits upright
                        double y = offsetY + ((height - h) / (double)height) * scale;

                        scatterSeries.Points.Add(new ScatterPoint(x, y, 1.0));
                    }
                }
            }

            if (scatterSeries.Points.Count > 0)
                plotModel.Series.Add(scatterSeries);
        }

      
        /// <summary>
        /// Creates 2D embedding visualization with colored labels using PACMAP
        /// </summary>
        private static void Create2DEmbeddingVisualization(float[,] floatData, byte[] labels)
        {
            Console.WriteLine();
            Console.WriteLine("🚀 Starting 2D Embedding Transform...");
            Console.WriteLine("==================================");

            try
            {
                Console.WriteLine($"📊 Input data: [{floatData.GetLength(0):N0} samples, {floatData.GetLength(1)} features]");
                Console.WriteLine($"📊 Label range: {labels.Min()}-{labels.Max()}");

                // Use PACMAP for dimensionality reduction (similar to UMAP)
                var embeddingStopwatch = System.Diagnostics.Stopwatch.StartNew();

                // Convert float to double for PACMAP
                var doubleData = new double[floatData.GetLength(0), floatData.GetLength(1)];
                for (int i = 0; i < floatData.GetLength(0); i++)
                {
                    for (int j = 0; j < floatData.GetLength(1); j++)
                    {
                        doubleData[i, j] = floatData[i, j];
                    }
                }

                var pacmap = new PacMapModel();
                var embedding = pacmap.Fit(
                    data: doubleData,
                    embeddingDimension: 2,
                    nNeighbors: 15,
                    mnRatio: 0.5f,
                    fpRatio: 2.0f,
                    learningRate: 1.0f,
                    numIters: (100, 100, 250),
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: false,
                    randomSeed: 42
                );

                embeddingStopwatch.Stop();
                Console.WriteLine($"✅ Embedding completed in {embeddingStopwatch.Elapsed.TotalSeconds:F2}s");
                Console.WriteLine($"   Shape: [{embedding.GetLength(0)}, {embedding.GetLength(1)}]");

                // Create 2D visualization with colored labels
                var tempPacmap = new PacMapModel();
                Create2DScatterPlot(embedding, labels, tempPacmap, 0.0);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ 2D embedding error: {ex.Message}");
            }
        }

        /// <summary>
        /// Helper function to create MNIST embedding with specified parameters and return the fitted model + fit time
        /// </summary>
        private static (PacMapModel model, double fitTime) CreateMnistEmbeddingWithModel(double[,] data, byte[] labels, int nNeighbors, float mnRatio, float fpRatio, string name, string folderName = "", bool directKNN = false)
        {
            string knnType = directKNN ? "Direct KNN" : "HNSW";
            Console.WriteLine($"🚀 Creating {name} embedding (k={nNeighbors}, mn={mnRatio:F2}, fp={fpRatio:F2}, KNN={knnType})...");

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            var pacmap = new PacMapModel();
            var embedding = pacmap.Fit(
                data: data,
                embeddingDimension: 2,
                nNeighbors: nNeighbors,
                mnRatio: mnRatio,
                fpRatio: fpRatio,
                learningRate: 1.0f,
                numIters: (100, 100, 250),
                metric: DistanceMetric.Euclidean,
                forceExactKnn: directKNN,
                randomSeed: 42,
                autoHNSWParam: false,  // Use default HNSW values
                hnswM: 16,             // Default HNSW M
                hnswEfConstruction: 150,  // Default HNSW ef_construction
                hnswEfSearch: 300,    // Default HNSW ef_search
                progressCallback: (phase, current, total, percent, message) =>
                {
                    Console.Write($"\r[{phase}] {current}/{total} ({percent:F1}%) {message}".PadRight(180));
                }
            );

            stopwatch.Stop();
            Console.WriteLine($"✅ {name} embedding completed in {stopwatch.Elapsed.TotalSeconds:F2}s");
            Console.WriteLine($"   Shape: [{embedding.GetLength(0)}, {embedding.GetLength(1)}]");

            // Create 2D visualization with colored labels and model info
            Create2DScatterPlot(embedding, labels, pacmap, stopwatch.Elapsed.TotalSeconds, name, folderName);

            return (pacmap, stopwatch.Elapsed.TotalSeconds); // Return fitted model + fit time for reuse in Transform
        }

        /// <summary>
        /// Helper function to create MNIST embedding with specified parameters (legacy method for experiments)
        /// </summary>
        private static void CreateMnistEmbedding(double[,] data, byte[] labels, int nNeighbors, float mnRatio, float fpRatio, string name, string folderName = "", bool directKNN = false)
        {
            string knnType = directKNN ? "Direct KNN" : "HNSW";
            Console.WriteLine($"🚀 Creating {name} embedding (k={nNeighbors}, mn={mnRatio:F2}, fp={fpRatio:F2}, KNN={knnType})...");

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            var pacmap = new PacMapModel();
            var embedding = pacmap.Fit(
                data: data,
                embeddingDimension: 2,
                nNeighbors: nNeighbors,
                mnRatio: mnRatio,
                fpRatio: fpRatio,
                learningRate: 1.0f,
                numIters: (100, 100, 250),
                metric: DistanceMetric.Euclidean,
                forceExactKnn: directKNN,
                randomSeed: 42,
                autoHNSWParam: false,  // Use default HNSW values
                hnswM: 16,             // Default HNSW M
                hnswEfConstruction: 150,  // Default HNSW ef_construction
                hnswEfSearch: 300,    // Default HNSW ef_search
                progressCallback: (phase, current, total, percent, message) =>
                {
                    Console.Write($"\r[{phase}] {current}/{total} ({percent:F1}%) {message}".PadRight(180));
                }
            );

            stopwatch.Stop();
            Console.WriteLine($"✅ {name} embedding completed in {stopwatch.Elapsed.TotalSeconds:F2}s");
            Console.WriteLine($"   Shape: [{embedding.GetLength(0)}, {embedding.GetLength(1)}]");

            // Create 2D visualization with colored labels and model info
            Create2DScatterPlot(embedding, labels, pacmap, stopwatch.Elapsed.TotalSeconds, name, folderName);
        }

        /// <summary>
        /// Creates 2D scatter plot with colored labels, counts, and hyperparameters
        /// </summary>
        private static void Create2DScatterPlot(double[,] embedding, byte[] labels, PacMapModel pacmap, double executionTime, string name = "mnist_2d_embedding", string folderName = "Results")
        {
            Console.WriteLine("🎨 Creating 2D Embedding Visualization...");

            try
            {
                // Count labels for title
                var labelCounts = new int[10];
                for (int i = 0; i < labels.Length; i++)
                {
                    labelCounts[labels[i]]++;
                }

                // Build title with hyperparameters like mammoth visualizations
                var modelInfo = pacmap.ModelInfo;
                var version = PacMapModel.GetVersion().Replace(" (Corrected Gradients)", "").Replace("-CLEAN-OUTPUT", "");
                var knnMode = modelInfo.ForceExactKnn ? "Direct KNN" : "HNSW";

                // Get actual learning rate from the PACMAP parameters
                var actualLearningRate = 1.0f; // This is what we passed to Fit()

                var timeUnit = executionTime >= 60 ? $"{executionTime / 60.0:F1}m" : $"{executionTime:F1}s";
                var title = $@"MNIST 2D Embedding (PACMAP)
PACMAP v{version} | Sample: {embedding.GetLength(0):N0} | {knnMode} | Time: {timeUnit}
k={modelInfo.Neighbors} | {modelInfo.Metric} | dims={modelInfo.OutputDimension} | seed=42
mn={modelInfo.MN_ratio:F2} | fp={modelInfo.FP_ratio:F2} | lr={actualLearningRate:F2} | std={modelInfo.InitializationStdDev:E0}
phases=({pacmap.NumIters.phase1}, {pacmap.NumIters.phase2}, {pacmap.NumIters.phase3}) | HNSW: M={modelInfo.HnswM}, ef_c={modelInfo.HnswEfConstruction}, ef_s={modelInfo.HnswEfSearch}";

                var plotModel = new PlotModel
                {
                    Title = title,
                    Background = OxyColors.White
                };

                // Define digit groups with different markers and better colors
                // Group 1: 8, 9, 3 (same type - Square)
                // Group 2: 7, 1 (same type - Diamond)
                // Group 3: 0, 2, 4 (all separate - Triangle, Circle, Plus)
                // Group 4: 5 (separate - Star)
                // Group 5: 6 (separate - Cross)

                var digitConfigs = new[]
                {
                    new { Digit = 0, Color = OxyColors.Red, Marker = MarkerType.Triangle, Name = "0-Triangle" },
                    new { Digit = 1, Color = OxyColors.Blue, Marker = MarkerType.Diamond, Name = "1-Diamond" },
                    new { Digit = 2, Color = OxyColors.Green, Marker = MarkerType.Circle, Name = "2-Circle" },
                    new { Digit = 3, Color = OxyColors.Orange, Marker = MarkerType.Square, Name = "3-Square" },
                    new { Digit = 4, Color = OxyColors.Purple, Marker = MarkerType.Plus, Name = "4-Plus" },
                    new { Digit = 5, Color = OxyColors.Cyan, Marker = MarkerType.Star, Name = "5-Star" },
                    new { Digit = 6, Color = OxyColors.Magenta, Marker = MarkerType.Cross, Name = "6-Cross" },
                    new { Digit = 7, Color = OxyColors.Brown, Marker = MarkerType.Diamond, Name = "7-Diamond" },
                    new { Digit = 8, Color = OxyColors.Pink, Marker = MarkerType.Square, Name = "8-Square" },
                    new { Digit = 9, Color = OxyColors.Gray, Marker = MarkerType.Square, Name = "9-Square" }
                };

                // Create scatter series for each digit
                foreach (var config in digitConfigs)
                {
                    var scatterSeries = new ScatterSeries
                    {
                        Title = $"Digit {config.Digit} ({labelCounts[config.Digit]:D4}) - {config.Name}",
                        MarkerType = config.Marker,
                        MarkerSize = 4,
                        MarkerFill = config.Color,
                        MarkerStroke = config.Color,
                        MarkerStrokeThickness = 0.5
                    };

                    // Add points for this digit
                    for (int i = 0; i < embedding.GetLength(0); i++)
                    {
                        if (labels[i] == config.Digit)
                        {
                            scatterSeries.Points.Add(new ScatterPoint(embedding[i, 0], embedding[i, 1], 4));
                        }
                    }

                    if (scatterSeries.Points.Count > 0)
                        plotModel.Series.Add(scatterSeries);
                }

                // Calculate min/max for proper axis scaling
                double minX = embedding[0, 0], maxX = embedding[0, 0];
                double minY = embedding[0, 1], maxY = embedding[0, 1];

                for (int i = 1; i < embedding.GetLength(0); i++)
                {
                    if (embedding[i, 0] < minX) minX = embedding[i, 0];
                    if (embedding[i, 0] > maxX) maxX = embedding[i, 0];
                    if (embedding[i, 1] < minY) minY = embedding[i, 1];
                    if (embedding[i, 1] > maxY) maxY = embedding[i, 1];
                }

                // Add 20% padding to right side of X axis to fit labels
                double xPadding = (maxX - minX) * 0.2;

                // Configure axes with proper min/max - only add padding to right side
                plotModel.Axes.Add(new LinearAxis
                {
                    Position = AxisPosition.Bottom,
                    Title = "X Coordinate",
                    Minimum = minX,
                    Maximum = maxX + xPadding
                });
                plotModel.Axes.Add(new LinearAxis
                {
                    Position = AxisPosition.Left,
                    Title = "Y Coordinate",
                    Minimum = minY,
                    Maximum = maxY
                });

                // Add legend
                plotModel.Legends.Add(new Legend { LegendPosition = LegendPosition.TopRight });

                // Determine output path based on folder parameter
                string outputDir = string.IsNullOrEmpty(folderName) ? "Results" : Path.Combine("Results", folderName);
                Directory.CreateDirectory(outputDir);
                var outputPath = Path.Combine(outputDir, $"{name}.png");
                var exporter = new OxyPlot.WindowsForms.PngExporter { Width = 1200, Height = 900, Resolution = 300 };
                using var stream = File.Create(outputPath);
                exporter.Export(plotModel, stream);

                Console.WriteLine($"✅ 2D embedding visualization saved: {outputPath}");
                Console.WriteLine($"   Resolution: 1200x900, Points: {embedding.GetLength(0):N0}");
                Console.WriteLine($"   📊 Full path: {Path.GetFullPath(outputPath)}");

                // Auto-open only the HNSW transform image, not regular embeddings
                if (name == "mnist_2d_transform")
                {
                    try
                    {
                        Process.Start(new ProcessStartInfo
                        {
                            FileName = Path.GetFullPath(outputPath),
                            UseShellExecute = true
                        });
                        Console.WriteLine($"   📂 Opened {name} visualization");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"   ⚠️ Could not open {name}: {ex.Message}");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error creating 2D plot: {ex.Message}");
            }
        }

        /// <summary>
        /// Creates transform API experiments using previously fitted models
        /// </summary>
        private static void CreateTransformExperiments(double[,] data, byte[] labels, PacMapModel pacmapHNSW, PacMapModel pacmapKNN, double hnswFitTime, double knnFitTime)
        {
            Console.WriteLine();
            Console.WriteLine("🔄 Creating Transform API Experiments...");
            Console.WriteLine("======================================");

            try
            {
                var totalStopwatch = System.Diagnostics.Stopwatch.StartNew();

                // Transform using previously fitted HNSW model
                Console.WriteLine($"\n🎯 Transform Experiment 1/2: HNSW Transform (reusing fitted model)");
                CreateMnistTransformEmbedding(
                    data: data,
                    labels: labels,
                    fittedModel: pacmapHNSW,
                    originalFitTime: hnswFitTime,
                    name: "mnist_2d_transform",
                    folderName: ""
                );

                // Transform using previously fitted DirectKNN model
                Console.WriteLine($"\n🎯 Transform Experiment 2/2: DirectKNN Transform (reusing fitted model)");
                CreateMnistTransformEmbedding(
                    data: data,
                    labels: labels,
                    fittedModel: pacmapKNN,
                    originalFitTime: knnFitTime,
                    name: "mnist_2d_KNN_transform",
                    folderName: ""
                );

                totalStopwatch.Stop();

                Console.WriteLine();
                Console.WriteLine($"✅ All Transform API experiments completed!");
                Console.WriteLine($"   Total experiments: 2 (HNSW + DirectKNN - using fitted models)");
                Console.WriteLine($"   Total time: {totalStopwatch.Elapsed.TotalSeconds:F1}s");
                Console.WriteLine($"   Average time per experiment: {totalStopwatch.Elapsed.TotalSeconds / 2:F1}s");
                Console.WriteLine($"   📚 Note: Transform reuses previously fitted models - no retraining needed!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error in Transform API experiments: {ex.Message}");
            }
        }

        /// <summary>
        /// Helper function to create MNIST embedding using Transform API with pre-fitted model
        /// </summary>
        private static void CreateMnistTransformEmbedding(double[,] data, byte[] labels, PacMapModel fittedModel, double originalFitTime, string name, string folderName = "")
        {
            Console.WriteLine($"🚀 Creating {name} transform (using pre-fitted model)...");

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            Console.WriteLine($"   🔄 Transforming data using pre-fitted model...");
            // Transform data using the already fitted model - no retraining!
            var embedding = fittedModel.Transform(data);

            stopwatch.Stop();
            var transformTime = stopwatch.Elapsed.TotalSeconds;

            Console.WriteLine($"✅ {name} transform completed in {transformTime:F2}s");
            Console.WriteLine($"   Shape: [{embedding.GetLength(0)}, {embedding.GetLength(1)}]");
            Console.WriteLine($"   ⚡ Fast transform - no model training needed!");
            Console.WriteLine($"   📊 Original fit time: {originalFitTime:F2}s | Transform time: {transformTime:F2}s");

            // Get the original embedding from the fitted model to compare with transform result
                var originalEmbedding = fittedModel.GetEmbedding();

                // Compare transform vs original embeddings
                bool embeddingsMatch = CompareEmbeddings(embedding, originalEmbedding, name);

                // Create 2D visualization with colored labels and model info (shows both fit and transform times)
                CreateTransformScatterPlot(embedding, labels, fittedModel, originalFitTime, transformTime, name, folderName, embeddingsMatch);
        }

        /// <summary>
        /// Compares two embeddings to check if they are identical (within tolerance)
        /// </summary>
        private static bool CompareEmbeddings(double[,] embedding1, double[,] embedding2, string name)
        {
            Console.WriteLine($"   🔍 Comparing {name} embeddings...");

            if (embedding1.GetLength(0) != embedding2.GetLength(0) || embedding1.GetLength(1) != embedding2.GetLength(1))
            {
                Console.WriteLine($"   ❌ Embedding dimensions differ: [{embedding1.GetLength(0)},{embedding1.GetLength(1)}] vs [{embedding2.GetLength(0)},{embedding2.GetLength(1)}]");
                return false;
            }

            double maxDiff = 0.0;
            double totalDiff = 0.0;
            int diffCount = 0;

            for (int i = 0; i < embedding1.GetLength(0); i++)
            {
                for (int j = 0; j < embedding1.GetLength(1); j++)
                {
                    double diff = Math.Abs(embedding1[i, j] - embedding2[i, j]);
                    if (diff > 1e-10) // Small tolerance for floating point comparison
                    {
                        diffCount++;
                        totalDiff += diff;
                        maxDiff = Math.Max(maxDiff, diff);
                    }
                }
            }

            double avgDiff = diffCount > 0 ? totalDiff / diffCount : 0.0;

            if (diffCount == 0)
            {
                Console.WriteLine($"   ✅ Embeddings are IDENTICAL - Transform API working correctly!");
                return true;
            }
            else
            {
                Console.WriteLine($"   ❌ Embeddings DIFFER - Transform API has issues:");
                Console.WriteLine($"      Different points: {diffCount:N0} out of {embedding1.GetLength(0) * embedding1.GetLength(1):N0}");
                Console.WriteLine($"      Average difference: {avgDiff:E2}");
                Console.WriteLine($"      Maximum difference: {maxDiff:E2}");
                Console.WriteLine($"      This suggests the Transform API is not returning the same result as Fit!");
                return false;
            }
        }

        /// <summary>
        /// Creates 2D scatter plot for Transform API with both fit and transform times
        /// </summary>
        private static void CreateTransformScatterPlot(double[,] embedding, byte[] labels, PacMapModel pacmap, double originalFitTime, double transformTime, string name = "mnist_2d_transform", string folderName = "Results", bool embeddingsMatch = true)
        {
            Console.WriteLine("🎨 Creating 2D Transform Visualization...");

            try
            {
                // Count labels for title
                var labelCounts = new int[10];
                for (int i = 0; i < labels.Length; i++)
                {
                    labelCounts[labels[i]]++;
                }

                // Build title with both fit and transform times like mammoth visualizations
                var modelInfo = pacmap.ModelInfo;
                var version = PacMapModel.GetVersion().Replace(" (Corrected Gradients)", "").Replace("-CLEAN-OUTPUT", "");
                var knnMode = modelInfo.ForceExactKnn ? "Direct KNN" : "HNSW";
                var actualLearningRate = 1.0f;

                var matchStatus = embeddingsMatch ? "✅ MATCH" : "❌ DIFFER";
                var title = $@"MNIST 2D Transform (PACMAP)
PACMAP v{version} | Sample: {embedding.GetLength(0):N0} | {knnMode} | TRANSFORM | {matchStatus}
k={modelInfo.Neighbors} | {modelInfo.Metric} | dims={modelInfo.OutputDimension} | seed=42
mn={modelInfo.MN_ratio:F2} | fp={modelInfo.FP_ratio:F2} | lr={actualLearningRate:F2} | std={modelInfo.InitializationStdDev:E0}
Fit: {originalFitTime:F2}s | Transform: {transformTime:F2}s | Speedup: {(originalFitTime/transformTime):F1}x
phases=({pacmap.NumIters.phase1}, {pacmap.NumIters.phase2}, {pacmap.NumIters.phase3}) | HNSW: M={modelInfo.HnswM}, ef_c={modelInfo.HnswEfConstruction}, ef_s={modelInfo.HnswEfSearch}";

                var plotModel = new PlotModel
                {
                    Title = title,
                    Background = OxyColors.White
                };

                // Use the same digit configurations as regular plots
                var digitConfigs = new[]
                {
                    new { Digit = 0, Color = OxyColors.Red, Marker = MarkerType.Triangle, Name = "0-Triangle" },
                    new { Digit = 1, Color = OxyColors.Blue, Marker = MarkerType.Diamond, Name = "1-Diamond" },
                    new { Digit = 2, Color = OxyColors.Green, Marker = MarkerType.Circle, Name = "2-Circle" },
                    new { Digit = 3, Color = OxyColors.Orange, Marker = MarkerType.Square, Name = "3-Square" },
                    new { Digit = 4, Color = OxyColors.Purple, Marker = MarkerType.Plus, Name = "4-Plus" },
                    new { Digit = 5, Color = OxyColors.Cyan, Marker = MarkerType.Star, Name = "5-Star" },
                    new { Digit = 6, Color = OxyColors.Magenta, Marker = MarkerType.Cross, Name = "6-Cross" },
                    new { Digit = 7, Color = OxyColors.Brown, Marker = MarkerType.Diamond, Name = "7-Diamond" },
                    new { Digit = 8, Color = OxyColors.Pink, Marker = MarkerType.Square, Name = "8-Square" },
                    new { Digit = 9, Color = OxyColors.Gray, Marker = MarkerType.Square, Name = "9-Square" }
                };

                // Create scatter series for each digit
                foreach (var config in digitConfigs)
                {
                    var scatterSeries = new ScatterSeries
                    {
                        Title = $"Digit {config.Digit} ({labelCounts[config.Digit]:D4}) - {config.Name}",
                        MarkerType = config.Marker,
                        MarkerSize = 4,
                        MarkerFill = config.Color,
                        MarkerStroke = config.Color,
                        MarkerStrokeThickness = 0.5
                    };

                    // Add points for this digit
                    for (int i = 0; i < embedding.GetLength(0); i++)
                    {
                        if (labels[i] == config.Digit)
                        {
                            scatterSeries.Points.Add(new ScatterPoint(embedding[i, 0], embedding[i, 1], 4));
                        }
                    }

                    if (scatterSeries.Points.Count > 0)
                        plotModel.Series.Add(scatterSeries);
                }

                // Calculate min/max for proper axis scaling
                double minX = embedding[0, 0], maxX = embedding[0, 0];
                double minY = embedding[0, 1], maxY = embedding[0, 1];

                for (int i = 1; i < embedding.GetLength(0); i++)
                {
                    if (embedding[i, 0] < minX) minX = embedding[i, 0];
                    if (embedding[i, 0] > maxX) maxX = embedding[i, 0];
                    if (embedding[i, 1] < minY) minY = embedding[i, 1];
                    if (embedding[i, 1] > maxY) maxY = embedding[i, 1];
                }

                // Add 20% padding to right side of X axis to fit labels
                double xPadding = (maxX - minX) * 0.2;

                // Configure axes with proper min/max - only add padding to right side
                plotModel.Axes.Add(new LinearAxis
                {
                    Position = AxisPosition.Bottom,
                    Title = "X Coordinate",
                    Minimum = minX,
                    Maximum = maxX + xPadding
                });
                plotModel.Axes.Add(new LinearAxis
                {
                    Position = AxisPosition.Left,
                    Title = "Y Coordinate",
                    Minimum = minY,
                    Maximum = maxY
                });

                // Add legend
                plotModel.Legends.Add(new Legend { LegendPosition = LegendPosition.TopRight });

                // Determine output path based on folder parameter
                string outputDir = string.IsNullOrEmpty(folderName) ? "Results" : Path.Combine("Results", folderName);
                Directory.CreateDirectory(outputDir);
                var outputPath = Path.Combine(outputDir, $"{name}.png");
                var exporter = new OxyPlot.WindowsForms.PngExporter { Width = 1200, Height = 900, Resolution = 300 };
                using var stream = File.Create(outputPath);
                exporter.Export(plotModel, stream);

                Console.WriteLine($"✅ 2D transform visualization saved: {outputPath}");
                Console.WriteLine($"   Resolution: 1200x900, Points: {embedding.GetLength(0):N0}");
                Console.WriteLine($"   📊 Full path: {Path.GetFullPath(outputPath)}");

                // Auto-open only the HNSW transform visualization
                if (name == "mnist_2d_transform")
                {
                    try
                    {
                        Process.Start(new ProcessStartInfo
                        {
                            FileName = Path.GetFullPath(outputPath),
                            UseShellExecute = true
                        });
                        Console.WriteLine($"   📂 Opened {name} transform visualization");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"   ⚠️ Could not open {name}: {ex.Message}");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error creating transform plot: {ex.Message}");
            }
        }

        /// <summary>
        /// Creates neighborMNSTI experiments with k=5 to 40 (1 increment)
        /// </summary>
        private static void CreateNeighborMNSTI_Experiments(double[,] data, byte[] labels)
        {
            Console.WriteLine();
            Console.WriteLine("🔄 Creating neighborMNSTI Experiments...");
            Console.WriteLine("=======================================");

            try
            {
                var totalStopwatch = System.Diagnostics.Stopwatch.StartNew();
                var experimentsCreated = 0;

                for (int k = 5; k <= 60; k += 2)
                {
                    Console.WriteLine($"\n🎯 Experiment {experimentsCreated + 1}/28: k={k}");

                    // Create HNSW embedding only
                    CreateMnistEmbedding(
                        data: data,
                        labels: labels,
                        nNeighbors: k,
                        mnRatio: 0.5f,
                        fpRatio: 2.0f,
                        name: $"k_{k:D2}",
                        folderName: "neighborMNSTI",
                        directKNN: false
                    );
                    experimentsCreated++;

                    // Show progress
                    var percentComplete = (experimentsCreated * 100.0) / 28;
                    Console.WriteLine($"   Progress: {experimentsCreated}/28 ({percentComplete:F1}%)");
                }

                totalStopwatch.Stop();

                Console.WriteLine();
                Console.WriteLine($"✅ All neighborMNSTI experiments completed!");
                Console.WriteLine($"   Total experiments: {experimentsCreated} (28 k-values from 5-60, HNSW only)");
                Console.WriteLine($"   Total time: {totalStopwatch.Elapsed.TotalMinutes:F1} minutes");
                Console.WriteLine($"   Average time per experiment: {totalStopwatch.Elapsed.TotalSeconds / experimentsCreated:F1}s");
                Console.WriteLine($"   📁 Results saved to: {Path.GetFullPath(Path.Combine("Results", "neighborMNSTI"))}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error in neighborMNSTI experiments: {ex.Message}");
            }
        }

        /// <summary>
        /// Creates MNSTMnRatio experiments with mnRatio from 0.5 to 2.0 (increments of 0.2)
        /// </summary>
        private static void CreateMNSTMnRatio_Experiments(double[,] data, byte[] labels)
        {
            Console.WriteLine();
            Console.WriteLine("🔄 Creating MNSTMnRatio Experiments...");
            Console.WriteLine("=====================================");

            try
            {
                var totalStopwatch = System.Diagnostics.Stopwatch.StartNew();
                var experimentsCreated = 0;

                for (float mnRatio = 0.5f; mnRatio <= 2.0f; mnRatio += 0.2f)
                {
                    Console.WriteLine($"\n🎯 MnRatio Experiment {experimentsCreated + 1}/8: mn={mnRatio:F1}");

                    // Create HNSW embedding with fixed k=30 and varying mnRatio
                    CreateMnistEmbedding(
                        data: data,
                        labels: labels,
                        nNeighbors: 30,
                        mnRatio: mnRatio,
                        fpRatio: 2.0f,
                        name: $"mn_{mnRatio:F1}",
                        folderName: "MNSTMnRatio",
                        directKNN: false
                    );
                    experimentsCreated++;

                    // Show progress
                    var percentComplete = (experimentsCreated * 100.0) / 8;
                    Console.WriteLine($"   Progress: {experimentsCreated}/8 ({percentComplete:F1}%)");
                }

                totalStopwatch.Stop();

                Console.WriteLine();
                Console.WriteLine($"✅ All MNSTMnRatio experiments completed!");
                Console.WriteLine($"   Total experiments: {experimentsCreated} (8 mnRatio values, k=30 fixed)");
                Console.WriteLine($"   Total time: {totalStopwatch.Elapsed.TotalMinutes:F1} minutes");
                Console.WriteLine($"   Average time per experiment: {totalStopwatch.Elapsed.TotalSeconds / experimentsCreated:F1}s");
                Console.WriteLine($"   📁 Results saved to: {Path.GetFullPath(Path.Combine("Results", "MNSTMnRatio"))}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error in MNSTMnRatio experiments: {ex.Message}");
            }
        }

        /// <summary>
        /// Creates MNSTfpRatio experiments with fpRatio from 0.5 to 4.0 (increments of 0.5)
        /// </summary>
        private static void CreateMNSTfpRatio_Experiments(double[,] data, byte[] labels)
        {
            Console.WriteLine();
            Console.WriteLine("🔄 Creating MNSTfpRatio Experiments...");
            Console.WriteLine("=====================================");

            try
            {
                var totalStopwatch = System.Diagnostics.Stopwatch.StartNew();
                var experimentsCreated = 0;

                for (float fpRatio = 0.5f; fpRatio <= 4.0f; fpRatio += 0.5f)
                {
                    Console.WriteLine($"\n🎯 FpRatio Experiment {experimentsCreated + 1}/8: fp={fpRatio:F1}");

                    // Create HNSW embedding with fixed k=30 and varying fpRatio
                    CreateMnistEmbedding(
                        data: data,
                        labels: labels,
                        nNeighbors: 30,
                        mnRatio: 0.5f,
                        fpRatio: fpRatio,
                        name: $"fp_{fpRatio:F1}",
                        folderName: "MNSTfpRatio",
                        directKNN: false
                    );
                    experimentsCreated++;

                    // Show progress
                    var percentComplete = (experimentsCreated * 100.0) / 8;
                    Console.WriteLine($"   Progress: {experimentsCreated}/8 ({percentComplete:F1}%)");
                }

                totalStopwatch.Stop();

                Console.WriteLine();
                Console.WriteLine($"✅ All MNSTfpRatio experiments completed!");
                Console.WriteLine($"   Total experiments: {experimentsCreated} (8 fpRatio values, k=30 fixed)");
                Console.WriteLine($"   Total time: {totalStopwatch.Elapsed.TotalMinutes:F1} minutes");
                Console.WriteLine($"   Average time per experiment: {totalStopwatch.Elapsed.TotalSeconds / experimentsCreated:F1}s");
                Console.WriteLine($"   📁 Results saved to: {Path.GetFullPath(Path.Combine("Results", "MNSTfpRatio"))}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error in MNSTfpRatio experiments: {ex.Message}");
            }
        }
    }
}