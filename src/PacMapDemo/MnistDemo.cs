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
    /// Data structure for digit probability classification
    /// </summary>
    public struct DigitProbability
    {
        public int Digit { get; }
        public double Probability { get; }

        public DigitProbability(int digit, double probability)
        {
            Digit = digit;
            Probability = probability;
        }
    }

    /// <summary>
    /// Complete classification result for a single digit
    /// </summary>
    public struct DigitClassificationResult
    {
        public int TrueLabel { get; }
        public int PredictedLabel { get; }
        public double[] Probabilities { get; }
        public bool IsCorrect { get; }
        public double Confidence { get; }

        public DigitClassificationResult(int trueLabel, int predictedLabel, double[] probabilities, bool isCorrect, double confidence)
        {
            TrueLabel = trueLabel;
            PredictedLabel = predictedLabel;
            Probabilities = probabilities;
            IsCorrect = isCorrect;
            Confidence = confidence;
        }
    }

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

                // NEW: Run TransformWithSafety with classification right after the two main embeddings
                Console.WriteLine();
                Console.WriteLine("🔄 Creating TransformWithSafety with Classification (HNSW only)...");
                Console.WriteLine("===============================================");

                // Transform using previously fitted HNSW model with TransformWithSafety
                Console.WriteLine($"\n🎯 TransformWithSafety Experiment: HNSW (reusing fitted model)");
                CreateMnistTransformEmbedding(
                    data: doubleData,
                    labels: labels,
                    fittedModel: pacmapHNSW,
                    originalFitTime: hnswFitTime,
                    name: "mnist_2d_transform",
                    folderName: ""
                );

                // Create Transform API experiments for DirectKNN (basic transform only)
                Console.WriteLine($"\n🎯 Transform Experiment: DirectKNN (reusing fitted model)");
                CreateMnistTransformEmbedding(
                    data: doubleData,
                    labels: labels,
                    fittedModel: pacmapKNN,
                    originalFitTime: knnFitTime,
                    name: "mnist_2d_KNN_transform",
                    folderName: ""
                );

                // Create neighborMNSTI folder with k=5 to 60 experiments (HNSW only)
                CreateNeighborMNSTI_Experiments(doubleData, labels);

                // Create MNSTMnRatio experiments with mnRatio from 0.5 to 2.0 (increments of 0.2)
                CreateMNSTMnRatio_Experiments(doubleData, labels);

                // Create MNSTfpRatio experiments with fpRatio from 0.5 to 4.0 (increments of 0.5)
                CreateMNSTfpRatio_Experiments(doubleData, labels);

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
        /// Helper function to create MNIST embedding using Transform API with pre-fitted model (HNSW only)
        /// </summary>
        private static void CreateMnistTransformEmbedding(double[,] data, byte[] labels, PacMapModel fittedModel, double originalFitTime, string name, string folderName = "")
        {
            Console.WriteLine($"🚀 Creating {name} transform with classification (using pre-fitted HNSW model)...");

            // Only use TransformWithSafety for HNSW model (not DirectKNN)
            if (fittedModel.ModelInfo.ForceExactKnn)
            {
                Console.WriteLine($"   ⚠️ Skipping classification for DirectKNN model - using basic Transform...");
                // Use basic transform for DirectKNN
                var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                var embedding = fittedModel.Transform(data);
                stopwatch.Stop();
                var basicTransformTime = stopwatch.Elapsed.TotalSeconds;
                CreateTransformScatterPlot(embedding, labels, fittedModel, originalFitTime, basicTransformTime, name, folderName);
                return;
            }

            // Use TransformWithSafety with classification for HNSW
            var transformStopwatch = System.Diagnostics.Stopwatch.StartNew();

            // Get TransformResults with safety information
            var transformResults = GetTransformResults(fittedModel, data);

            transformStopwatch.Stop();
            var transformTime = transformStopwatch.Elapsed.TotalSeconds;

            Console.WriteLine($"✅ {name} TransformWithSafety completed in {transformTime:F2}s");
            Console.WriteLine($"   Shape: [{transformResults.Length} samples with safety metrics]");
            Console.WriteLine($"   ⚡ Fast transform - no model training needed!");
            Console.WriteLine($"   📊 Original fit time: {originalFitTime:F2}s | Transform time: {transformTime:F2}s");

            // Classify all samples using nearest neighbor voting
            var classifications = ClassifyAllSamples(transformResults, labels);

            // Get difficult samples (misclassified) with filtering rules
            var (difficultSamples, filteredCounts) = GetDifficultSamples(classifications, transformResults);

            // Print classification statistics
            PrintClassificationStatistics(classifications, difficultSamples, filteredCounts);

            // Create bad samples visualization
            if (difficultSamples.Count > 0)
            {
                CreateBadSamplesVisualization(difficultSamples, data, classifications, "badSamples_{0:D2}.png");
            }

            // Create clean transform plot (without difficult samples)
            CreateCleanTransformPlot(transformResults, classifications, difficultSamples, originalFitTime, transformTime, fittedModel, name);

            // Create difficult samples transform plot (only difficult samples)
            if (difficultSamples.Count > 0)
            {
                CreateDifficultSamplesTransformPlot(transformResults, classifications, difficultSamples, originalFitTime, transformTime, fittedModel, name);
            }

            // Create regular transform scatter plot with all data
            CreateTransformScatterPlotWithSafety(transformResults, labels, fittedModel, originalFitTime, transformTime, name, folderName, classifications);
        }

        /// <summary>
        /// Creates 2D scatter plot for TransformWithSafety API with classification overlay
        /// </summary>
        private static void CreateTransformScatterPlotWithSafety(TransformResult[] transformResults, byte[] labels, PacMapModel fittedModel, double originalFitTime, double transformTime, string name, string folderName, DigitClassificationResult[] classifications)
        {
            Console.WriteLine("🎨 Creating 2D Transform Visualization with Classification...");

            try
            {
                // Count labels for title
                var labelCounts = new int[10];
                for (int i = 0; i < labels.Length; i++)
                {
                    labelCounts[labels[i]]++;
                }

                // Build title with classification statistics
                var modelInfo = fittedModel.ModelInfo;
                var version = PacMapModel.GetVersion().Replace(" (Corrected Gradients)", "").Replace("-CLEAN-OUTPUT", "");
                var knnMode = modelInfo.ForceExactKnn ? "Direct KNN" : "HNSW";
                var correctSamples = classifications.Count(c => c.IsCorrect);
                var accuracy = (correctSamples * 100.0) / classifications.Length;

                var title = $@"MNIST 2D Transform with Classification (PACMAP)
PACMAP v{version} | Sample: {transformResults.Length:N0} | {knnMode} | TRANSFORM
Classification: {correctSamples:N0}/{classifications.Length:N0} ({accuracy:F1}% accuracy)
k={modelInfo.Neighbors} | {modelInfo.Metric} | seed=42
mn={modelInfo.MN_ratio:F2} | fp={modelInfo.FP_ratio:F2}
Fit: {originalFitTime:F2}s | Transform: {transformTime:F2}s | Speedup: {(originalFitTime/transformTime):F1}x";

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
                    for (int i = 0; i < transformResults.Length; i++)
                    {
                        if (labels[i] == config.Digit)
                        {
                            var coords = transformResults[i].ProjectionCoordinates;
                            scatterSeries.Points.Add(new ScatterPoint(coords[0], coords[1], 4));
                        }
                    }

                    if (scatterSeries.Points.Count > 0)
                        plotModel.Series.Add(scatterSeries);
                }

                // Calculate bounds from TransformResults
                var allCoords = transformResults.SelectMany(r => r.ProjectionCoordinates).ToArray();
                var minX = allCoords[0];
                var maxX = allCoords[0];
                var minY = allCoords[1];
                var maxY = allCoords[1];

                for (int i = 2; i < allCoords.Length; i += 2)
                {
                    if (allCoords[i] < minX) minX = allCoords[i];
                    if (allCoords[i] > maxX) maxX = allCoords[i];
                    if (allCoords[i + 1] < minY) minY = allCoords[i + 1];
                    if (allCoords[i + 1] > maxY) maxY = allCoords[i + 1];
                }

                // Add padding
                double xPadding = (maxX - minX) * 0.2;

                // Configure axes
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

                // Determine output path
                string outputDir = string.IsNullOrEmpty(folderName) ? "Results" : Path.Combine("Results", folderName);
                Directory.CreateDirectory(outputDir);
                var outputPath = Path.Combine(outputDir, $"{name}.png");
                var exporter = new PngExporter { Width = 1200, Height = 900, Resolution = 300 };
                using var stream = File.Create(outputPath);
                exporter.Export(plotModel, stream);

                Console.WriteLine($"✅ 2D transform visualization with classification saved: {outputPath}");
                Console.WriteLine($"   Resolution: 1200x900, Points: {transformResults.Length:N0}");
                Console.WriteLine($"   Classification accuracy: {accuracy:F1}%");

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
                Console.WriteLine($"❌ Error creating transform plot with classification: {ex.Message}");
            }
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
        /// Weighted classification algorithm using nearest neighbor voting
        /// </summary>
        private static DigitClassificationResult ClassifyDigit(TransformResult result, byte[] trainingLabels)
        {
            return ClassifyDigitWithLimitedNeighbors(
                result.NearestNeighborIndices,
                result.NearestNeighborDistances,
                trainingLabels
            );
        }

        /// <summary>
        /// Weighted classification algorithm using specified nearest neighbor indices and distances
        /// </summary>
        private static DigitClassificationResult ClassifyDigitWithLimitedNeighbors(int[] neighborIndices, double[] neighborDistances, byte[] trainingLabels)
        {
            // Calculate distance weights (closer neighbors have more influence)
            var maxDistance = neighborDistances.Max();
            var labelVotes = new double[10]; // For digits 0-9

            for (int i = 0; i < neighborIndices.Length; i++)
            {
                var neighborIndex = neighborIndices[i];
                var neighborDistance = neighborDistances[i];
                var neighborLabel = trainingLabels[neighborIndex];

                // Weight by inverse distance (closer = more important)
                var weight = CalculateDistanceWeight(neighborDistance, maxDistance);
                labelVotes[neighborLabel] += weight;
            }

            // Convert to probabilities
            var totalWeight = labelVotes.Sum();
            var probabilities = labelVotes.Select(v => totalWeight > 0 ? v / totalWeight : 0.1).ToArray();

            // Find the predicted label (highest probability)
            var predictedLabel = probabilities
                .Select((prob, index) => new { Digit = index, Probability = prob })
                .OrderByDescending(x => x.Probability)
                .First().Digit;

            var confidence = probabilities[predictedLabel];
            var isCorrect = predictedLabel == (int)result.Severity; // Will be updated later

            return new DigitClassificationResult(0, predictedLabel, probabilities, isCorrect, confidence);
        }

        /// <summary>
        /// Calculate weight for a neighbor based on distance (closer = higher weight)
        /// </summary>
        private static double CalculateDistanceWeight(double distance, double maxDistance)
        {
            if (maxDistance == 0) return 1.0;

            // Inverse distance weighting with smoothing to avoid division by zero
            var weight = 1.0 / (1.0 + (distance / maxDistance));
            return weight;
        }

        /// <summary>
        /// Get TransformResult array using TransformWithSafety
        /// </summary>
        private static TransformResult[] GetTransformResults(PacMapModel model, double[,] data)
        {
            Console.WriteLine($"   🔄 Getting TransformWithSafety results for {data.GetLength(0)} samples...");
            return model.TransformWithSafety(data);
        }

        /// <summary>
        /// Classify all samples using TransformWithSafety results
        /// </summary>
        private static DigitClassificationResult[] ClassifyAllSamples(TransformResult[] results, byte[] labels)
        {
            Console.WriteLine($"   🏷️ Classifying {results.Length} samples using nearest neighbor voting...");
            var classifications = new DigitClassificationResult[results.Length];

            for (int i = 0; i < results.Length; i++)
            {
                var result = results[i];
                var classification = ClassifyDigit(result, labels);

                // Update the true label and isCorrect status
                var trueLabel = labels[i];
                var isCorrect = classification.PredictedLabel == trueLabel;

                classifications[i] = new DigitClassificationResult(
                    trueLabel,
                    classification.PredictedLabel,
                    classification.Probabilities,
                    isCorrect,
                    classification.Confidence
                );
            }

            return classifications;
        }

        /// <summary>
        /// Get list of difficult samples (misclassified digits) with enhanced filtering rules
        /// </summary>
        private static (List<int> difficultSamples, int[] filteredCounts) GetDifficultSamples(DigitClassificationResult[] classifications, TransformResult[] transformResults)
        {
            var difficultSamples = new List<int>();
            var filteredByRule1 = 0;
            var filteredByRule2 = 0;
            var filteredByRule3 = 0;

            for (int i = 0; i < classifications.Length; i++)
            {
                if (!classifications[i].IsCorrect)
                {
                    var classification = classifications[i];
                    var transformResult = transformResults[i];

                    // Rule 1: Skip if number of neighbors <= 3 (insufficient evidence)
                    if (transformResult.NearestNeighborIndices.Length <= 3)
                    {
                        filteredByRule1++;
                        continue; // Don't add to difficult samples list
                    }

                    // Rule 3: If more than 15 neighbors, use only closest 15 for classification
                    var neighborIndicesToUse = transformResult.NearestNeighborIndices;
                    var neighborDistancesToUse = transformResult.NearestNeighborDistances;
                    if (transformResult.NearestNeighborIndices.Length > 15)
                    {
                        // Sort by distance and take closest 15
                        var indexedNeighbors = transformResult.NearestNeighborIndices
                            .Select((idx, dist) => new { Index = idx, Distance = dist })
                            .OrderBy(x => x.Distance)
                            .Take(15)
                            .ToArray();

                        neighborIndicesToUse = indexedNeighbors.Select(x => x.Index).ToArray();
                        neighborDistancesToUse = indexedNeighbors.Select(x => x.Distance).ToArray();
                        filteredByRule3++;
                    }

                    // Recalculate classification with limited neighbors
                    var limitedClassification = ClassifyDigitWithLimitedNeighbors(
                        neighborIndicesToUse, neighborDistancesToUse, labels[i]);

                    // Rule 2: Add if true label probability < 0.3 (border case)
                    var trueLabelProbability = limitedClassification.Probabilities[classification.TrueLabel];
                    if (trueLabelProbability < 0.3)
                    {
                        difficultSamples.Add(i);
                    }
                    else
                    {
                        filteredByRule2++;
                    }
                }
            }

            Console.WriteLine($"   📊 Filtering Results:");
            Console.WriteLine($"      - Rule 1 (≤3 neighbors): Filtered {filteredByRule1:N0} samples");
            Console.WriteLine($"      - Rule 2 (true label < 30%): Filtered {filteredByRule2:N0} samples");
            Console.WriteLine($"      - Rule 3 (>15 neighbors, limited to 15): Filtered {filteredByRule3:N0} samples");
            Console.WriteLine($"      - Remaining difficult samples: {difficultSamples.Count:N0}");

            return (difficultSamples, new int[] { filteredByRule1, filteredByRule2, filteredByRule3 });
        }

        /// <summary>
        /// Print classification statistics to console (Enhanced with filtering effects)
        /// </summary>
        private static void PrintClassificationStatistics(DigitClassificationResult[] classifications, List<int> difficultSamples, int[] filteredCounts)
        {
            Console.WriteLine();
            Console.WriteLine("📊 ENHANCED CLASSIFICATION STATISTICS");
            Console.WriteLine("========================================");
            Console.WriteLine($"Total samples processed: {classifications.Length:N0}");
            Console.WriteLine($"Correctly classified: {classifications.Count(c => c.IsCorrect):N0} ({(classifications.Count(c => c.IsCorrect) * 100.0 / classifications.Length):F1}%)");
            Console.WriteLine($"Misclassified samples: {classifications.Count(c => !c.IsCorrect):N0} ({(classifications.Count(c => !c.IsCorrect) * 100.0 / classifications.Length):F1}%)");

            Console.WriteLine($"\n🎯 FILTERING RULES IMPACT:");
            Console.WriteLine($"   Rule 1 filtered (≤2 neighbors): {filteredCounts[0]:N0} samples removed");
            Console.WriteLine($"   Rule 2 filtered (true prob <0.4): {filteredCounts[1]:N0} samples removed");
            Console.WriteLine($"   Total filtered: {(filteredCounts[0] + filteredCounts[1]):N0} samples");
            Console.WriteLine($"   Final difficult samples: {difficultSamples.Count:N0} samples");

            var overallReduction = (filteredCounts[0] + filteredCounts[1]) * 100.0 / classifications.Length;
            Console.WriteLine($"   Overall reduction: {overallReduction:F1}% of total dataset");

            Console.WriteLine($"\n📈 CONFIDENCE ANALYSIS:");
            var correctlyClassified = classifications.Where(c => c.IsCorrect).ToArray();
            var misclassified = classifications.Where(c => !c.IsCorrect).ToArray();

            if (correctlyClassified.Length > 0)
            {
                var avgConfidenceCorrect = correctlyClassified.Average(c => c.Confidence);
                var highConfidenceCorrect = correctlyClassified.Count(c => c.Confidence > 0.8);
                Console.WriteLine($"   Avg confidence (correct): {avgConfidenceCorrect:F3}");
                Console.WriteLine($"   High confidence (>0.8): {highConfidenceCorrect:N0} ({(highConfidenceCorrect * 100.0 / correctlyClassified.Length):F1}%)");
            }

            if (misclassified.Length > 0)
            {
                var avgConfidenceIncorrect = misclassified.Average(c => c.Confidence);
                var lowConfidenceIncorrect = misclassified.Count(c => c.Confidence < 0.6);
                Console.WriteLine($"   Avg confidence (incorrect): {avgConfidenceIncorrect:F3}");
                Console.WriteLine($"   Low confidence (<0.6): {lowConfidenceIncorrect:N0} ({(lowConfidenceIncorrect * 100.0 / misclassified.Length):F1}%)");
            }

            Console.WriteLine($"\n🔍 MOST CONFUSED DIGIT PAIRS:");
            var confusions = misclassified
                .GroupBy(c => $"{c.TrueLabel} → {c.PredictedLabel}")
                .OrderByDescending(g => g.Count())
                .Take(5);

            foreach (var confusion in confusions)
            {
                var percentage = confusion.Count() * 100.0 / classifications.Length;
                Console.WriteLine($"   {confusion.Key}: {confusion.Count():N0} samples ({percentage:F2}%)");
            }

            Console.WriteLine($"\n✅ Enhanced classification with filtering rules completed successfully!");
        }

          /// <summary>
        /// Create visualization of bad (misclassified) samples with sorting by true label and enhanced labeling
        /// </summary>
        private static void CreateBadSamplesVisualization(List<int> difficultSamples, double[,] imageData, DigitClassificationResult[] classifications, string filenamePattern)
        {
            if (difficultSamples.Count == 0)
            {
                Console.WriteLine("   ✅ No difficult samples to visualize!");
                return;
            }

            Console.WriteLine($"   🎨 Creating bad sample visualizations for {difficultSamples.Count:N0} samples...");

            // Sort bad samples by true label (group all 3's, then 4's, etc.)
            var sortedDifficultSamples = difficultSamples
                .OrderBy(index => classifications[index].TrueLabel)
                .ToList();

            // Create bad samples folder
            var badSamplesDir = Path.Combine("Results", "badsampel");
            Directory.CreateDirectory(badSamplesDir);

            const int samplesPerImage = 20; // 4x5 grid
            const int gridCols = 5;
            const int gridRows = 4;

            var imageIndex = 1;
            for (int startIdx = 0; startIdx < sortedDifficultSamples.Count; startIdx += samplesPerImage)
            {
                var endIndex = Math.Min(startIdx + samplesPerImage, sortedDifficultSamples.Count);
                var currentBatch = sortedDifficultSamples.Skip(startIdx).Take(endIndex - startIdx).ToList();

                if (currentBatch.Count == 0) break;

                // Create plot model for this batch
                var plotModel = new PlotModel
                {
                    Title = $"Misclassified Digits - Batch {imageIndex} (Samples {startIdx + 1}-{endIndex})",
                    Background = OxyColors.White
                };

                // Add each sample to the grid
                for (int i = 0; i < currentBatch.Count; i++)
                {
                    var sampleIndex = currentBatch[i];
                    var classification = classifications[sampleIndex];

                    var row = i / gridCols;
                    var col = i % gridCols;

                    // Convert double image back to byte for visualization
                    var imageBytes = new byte[28, 28];
                    for (int h = 0; h < 28; h++)
                    {
                        for (int w = 0; w < 28; w++)
                        {
                            var pixelValue = imageData[sampleIndex, h * 28 + w];
                            imageBytes[h, w] = (byte)Math.Max(0, Math.Min(255, pixelValue));
                        }
                    }

                    // Create digit visualization
                    CreateDigitPixelScatter(plotModel, imageBytes, classification.TrueLabel, col, gridRows - 1 - row);

                    // Enhanced labeling: True→Pred, Index, and Confidence
                    var labelAnnotation = new TextAnnotation
                    {
                        Text = $"True: {classification.TrueLabel} → Pred: {classification.PredictedLabel}",
                        TextPosition = new DataPoint(col, -0.5 - row * 0.4),
                        TextHorizontalAlignment = HorizontalAlignment.Center,
                        TextVerticalAlignment = VerticalAlignment.Middle,
                        FontSize = 10,
                        FontWeight = FontWeights.Bold,
                        TextColor = OxyColors.Red
                    };
                    plotModel.Annotations.Add(labelAnnotation);

                    // Add index information
                    var indexAnnotation = new TextAnnotation
                    {
                        Text = $"Index: {sampleIndex:D5}",
                        TextPosition = new DataPoint(col, -0.7 - row * 0.4),
                        TextHorizontalAlignment = HorizontalAlignment.Center,
                        TextVerticalAlignment = VerticalAlignment.Middle,
                        FontSize = 8,
                        TextColor = OxyColors.DarkBlue
                    };
                    plotModel.Annotations.Add(indexAnnotation);

                    // Add confidence
                    var confidenceAnnotation = new TextAnnotation
                    {
                        Text = $"Conf: {classification.Confidence:F2}",
                        TextPosition = new DataPoint(col, -0.9 - row * 0.4),
                        TextHorizontalAlignment = HorizontalAlignment.Center,
                        TextVerticalAlignment = VerticalAlignment.Middle,
                        FontSize = 8,
                        TextColor = OxyColors.Gray
                    };
                    plotModel.Annotations.Add(confidenceAnnotation);
                }

                // Configure axes
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Minimum = -0.5, Maximum = gridCols - 0.5, Title = "" });
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Minimum = -1.5, Maximum = gridRows - 0.5, Title = "" });

                // Save to file
                var filename = string.Format(filenamePattern, imageIndex);
                var outputPath = Path.Combine(badSamplesDir, filename);
                var exporter = new PngExporter { Width = 1200, Height = 800, Resolution = 300 };
                using var stream = File.Create(outputPath);
                exporter.Export(plotModel, stream);

                Console.WriteLine($"   ✅ Bad samples batch {imageIndex} saved: {filename}");
                imageIndex++;
            }

            Console.WriteLine($"   📁 All bad sample visualizations saved to: {Path.GetFullPath(badSamplesDir)}");
        }

        /// <summary>
        /// Create clean transform plot without difficult samples
        /// </summary>
        private static void CreateCleanTransformPlot(TransformResult[] transformResults, DigitClassificationResult[] classifications, List<int> difficultSamples, double originalFitTime, double transformTime, PacMapModel model, string name)
        {
            Console.WriteLine($"   🎨 Creating clean transform plot (removing {difficultSamples.Count:N0} difficult samples)...");

            try
            {
                // Create plot model
                var plotModel = new PlotModel
                {
                    Title = $"MNIST 2D Transform - Clean (Difficult Samples Removed)",
                    Background = OxyColors.White
                };

                // Build title with model info
                var modelInfo = model.ModelInfo;
                var version = PacMapModel.GetVersion().Replace(" (Corrected Gradients)", "").Replace("-CLEAN-OUTPUT", "");
                var knnMode = modelInfo.ForceExactKnn ? "Direct KNN" : "HNSW";

                var title = $@"MNIST 2D Transform - Clean (PACMAP)
PACMAP v{version} | {transformResults.Length:N0} samples | {knnMode} | TRANSFORM
Clean: {transformResults.Length - difficultSamples.Count:N0} samples ({(100.0 * (transformResults.Length - difficultSamples.Count) / transformResults.Length):F1}% retained)
k={modelInfo.Neighbors} | mn={modelInfo.MN_ratio:F2} | fp={modelInfo.FP_ratio:F2}
Fit: {originalFitTime:F2}s | Transform: {transformTime:F2}s";

                plotModel.Title = title;

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

                // Count clean samples per digit
                var cleanLabelCounts = new int[10];
                for (int i = 0; i < classifications.Length; i++)
                {
                    if (!difficultSamples.Contains(i))
                    {
                        cleanLabelCounts[classifications[i].TrueLabel]++;
                    }
                }

                // Create scatter series for each digit (only clean samples)
                foreach (var config in digitConfigs)
                {
                    var scatterSeries = new ScatterSeries
                    {
                        Title = $"Digit {config.Digit} ({cleanLabelCounts[config.Digit]:D4}) - {config.Name}",
                        MarkerType = config.Marker,
                        MarkerSize = 4,
                        MarkerFill = config.Color,
                        MarkerStroke = config.Color,
                        MarkerStrokeThickness = 0.5
                    };

                    // Add clean points for this digit
                    for (int i = 0; i < transformResults.Length; i++)
                    {
                        if (!difficultSamples.Contains(i) && classifications[i].TrueLabel == config.Digit)
                        {
                            var coords = transformResults[i].ProjectionCoordinates;
                            scatterSeries.Points.Add(new ScatterPoint(coords[0], coords[1], 4));
                        }
                    }

                    if (scatterSeries.Points.Count > 0)
                        plotModel.Series.Add(scatterSeries);
                }

                // Calculate bounds for clean data only
                var cleanCoords = transformResults
                    .Where((r, i) => !difficultSamples.Contains(i))
                    .SelectMany(r => r.ProjectionCoordinates)
                    .ToArray();

                if (cleanCoords.Length < 2)
                {
                    Console.WriteLine("   ⚠️ No clean data to plot!");
                    return;
                }

                var minX = cleanCoords[0];
                var maxX = cleanCoords[0];
                var minY = cleanCoords[1];
                var maxY = cleanCoords[1];

                for (int i = 2; i < cleanCoords.Length; i += 2)
                {
                    if (cleanCoords[i] < minX) minX = cleanCoords[i];
                    if (cleanCoords[i] > maxX) maxX = cleanCoords[i];
                    if (cleanCoords[i + 1] < minY) minY = cleanCoords[i + 1];
                    if (cleanCoords[i + 1] > maxY) maxY = cleanCoords[i + 1];
                }

                // Add padding
                double xPadding = (maxX - minX) * 0.2;
                double yPadding = (maxY - minY) * 0.2;

                // Configure axes
                plotModel.Axes.Add(new LinearAxis
                {
                    Position = AxisPosition.Bottom,
                    Title = "X Coordinate",
                    Minimum = minX - xPadding,
                    Maximum = maxX + xPadding
                });
                plotModel.Axes.Add(new LinearAxis
                {
                    Position = AxisPosition.Left,
                    Title = "Y Coordinate",
                    Minimum = minY - yPadding,
                    Maximum = maxY + yPadding
                });

                // Add legend
                plotModel.Legends.Add(new Legend { LegendPosition = LegendPosition.TopRight });

                // Save to file
                var outputPath = Path.Combine("Results", $"{name}_clean.png");
                var exporter = new PngExporter { Width = 1200, Height = 900, Resolution = 300 };
                using var stream = File.Create(outputPath);
                exporter.Export(plotModel, stream);

                Console.WriteLine($"   ✅ Clean transform plot saved: {outputPath}");
                Console.WriteLine($"   📊 Clean samples: {transformResults.Length - difficultSamples.Count:N0} / {transformResults.Length:N0}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ❌ Error creating clean transform plot: {ex.Message}");
            }
        }

        /// <summary>
        /// Create transform plot with only difficult samples (misclassified samples)
        /// </summary>
        private static void CreateDifficultSamplesTransformPlot(TransformResult[] transformResults, DigitClassificationResult[] classifications, List<int> difficultSamples, double originalFitTime, double transformTime, PacMapModel model, string name)
        {
            Console.WriteLine($"   🎨 Creating difficult samples transform plot (only {difficultSamples.Count:N0} difficult samples)...");

            try
            {
                // Create plot model
                var plotModel = new PlotModel
                {
                    Title = $"MNIST 2D Transform - Difficult Samples Only",
                    Background = OxyColors.White
                };

                // Build title with model info
                var modelInfo = model.ModelInfo;
                var version = PacMapModel.GetVersion().Replace(" (Corrected Gradients)", "").Replace("-CLEAN-OUTPUT", "");
                var knnMode = modelInfo.ForceExactKnn ? "Direct KNN" : "HNSW";

                var title = $@"MNIST 2D Transform - Difficult Samples (PACMAP)
PACMAP v{version} | {transformResults.Length:N0} samples | {knnMode} | TRANSFORM
Difficult: {difficultSamples.Count:N0} samples ({(100.0 * difficultSamples.Count / transformResults.Length):F1}% of total)
k={modelInfo.Neighbors} | mn={modelInfo.MN_ratio:F2} | fp={modelInfo.FP_ratio:F2}
Fit: {originalFitTime:F2}s | Transform: {transformTime:F2}s";

                plotModel.Title = title;

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

                // Count difficult samples per digit
                var difficultLabelCounts = new int[10];
                for (int i = 0; i < difficultSamples.Count; i++)
                {
                    var sampleIndex = difficultSamples[i];
                    difficultLabelCounts[classifications[sampleIndex].TrueLabel]++;
                }

                // Create scatter series for each digit (only difficult samples)
                foreach (var config in digitConfigs)
                {
                    var scatterSeries = new ScatterSeries
                    {
                        Title = $"Digit {config.Digit} ({difficultLabelCounts[config.Digit]:D4}) - {config.Name}",
                        MarkerType = config.Marker,
                        MarkerSize = 6, // Larger markers for difficult samples
                        MarkerFill = config.Color,
                        MarkerStroke = config.Color,
                        MarkerStrokeThickness = 1.0
                    };

                    // Add difficult points for this digit
                    for (int i = 0; i < difficultSamples.Count; i++)
                    {
                        var sampleIndex = difficultSamples[i];
                        if (classifications[sampleIndex].TrueLabel == config.Digit)
                        {
                            var coords = transformResults[sampleIndex].ProjectionCoordinates;
                            scatterSeries.Points.Add(new ScatterPoint(coords[0], coords[1], 6));
                        }
                    }

                    if (scatterSeries.Points.Count > 0)
                        plotModel.Series.Add(scatterSeries);
                }

                // Calculate bounds for difficult data only
                var difficultCoords = difficultSamples
                    .Select(i => transformResults[i].ProjectionCoordinates)
                    .SelectMany(coords => coords)
                    .ToArray();

                if (difficultCoords.Length < 2)
                {
                    Console.WriteLine("   ⚠️ No difficult data to plot!");
                    return;
                }

                var minX = difficultCoords[0];
                var maxX = difficultCoords[0];
                var minY = difficultCoords[1];
                var maxY = difficultCoords[1];

                for (int i = 2; i < difficultCoords.Length; i += 2)
                {
                    if (difficultCoords[i] < minX) minX = difficultCoords[i];
                    if (difficultCoords[i] > maxX) maxX = difficultCoords[i];
                    if (difficultCoords[i + 1] < minY) minY = difficultCoords[i + 1];
                    if (difficultCoords[i + 1] > maxY) maxY = difficultCoords[i + 1];
                }

                // Add padding
                double xPadding = (maxX - minX) * 0.2;
                double yPadding = (maxY - minY) * 0.2;

                // Configure axes
                plotModel.Axes.Add(new LinearAxis
                {
                    Position = AxisPosition.Bottom,
                    Title = "X Coordinate",
                    Minimum = minX - xPadding,
                    Maximum = maxX + xPadding
                });
                plotModel.Axes.Add(new LinearAxis
                {
                    Position = AxisPosition.Left,
                    Title = "Y Coordinate",
                    Minimum = minY - yPadding,
                    Maximum = maxY + yPadding
                });

                // Add legend
                plotModel.Legends.Add(new Legend { LegendPosition = LegendPosition.TopRight });

                // Save to file
                var outputPath = Path.Combine("Results", $"{name}_difficult.png");
                var exporter = new PngExporter { Width = 1200, Height = 900, Resolution = 300 };
                using var stream = File.Create(outputPath);
                exporter.Export(plotModel, stream);

                Console.WriteLine($"   ✅ Difficult samples transform plot saved: {outputPath}");
                Console.WriteLine($"   📊 Difficult samples: {difficultSamples.Count:N0} / {transformResults.Length:N0}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ❌ Error creating difficult samples transform plot: {ex.Message}");
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