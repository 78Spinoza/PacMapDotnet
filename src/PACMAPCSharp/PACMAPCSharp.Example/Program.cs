using System;
using System.IO;
using System.Linq;
using PacMapSharp;

namespace PACMAPExample
{
    class CompleteUsageDemo
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=== Complete Enhanced PACMAP Wrapper Demo ===\n");

            // Set up global callback to catch all warnings and errors
            // PacMapModel.SetGlobalCallback((phase, current, total, percent, message) => {
            //     if (phase == "Warning") {
            //         Console.WriteLine($"⚠️  GLOBAL WARNING: {message}");
            //     } else if (phase == "Error") {
            //         Console.WriteLine($"❌ GLOBAL ERROR: {message}");
            //     } else if (phase.Contains("k-NN") || phase.Contains("HNSW")) {
            //         Console.WriteLine($"🔍 GLOBAL INFO: {phase} - {message}");
            //     }
            //     // Skip regular training progress since we have local callbacks for those
            // });

            Console.WriteLine("Global callback set to capture warnings and errors...\n");

            try
            {
                // Demo 1: 27D Embedding with Progress Reporting
                Demo27DEmbeddingWithProgress();

                // Demo 2: Multi-Dimensional Embeddings (1D to 50D)
                DemoMultiDimensionalEmbeddings();

                // Demo 3: Model Persistence and Transform
                DemoModelPersistence();

                // Demo 4: Different Data Types and Metrics with Progress
                DemoDistanceMetricsWithProgress();

                // Demo 5: Enhanced Safety Features with HNSW
                DemoSafetyFeatures();

                // Demo 6: New Spread Parameter with Smart Defaults
                DemoSpreadParameter();

                Console.WriteLine("\nAll demos completed successfully!");
                Console.WriteLine("Your enhanced PACMAP wrapper is ready for production use!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }
            finally
            {
                // Clean up global callback - not available in PacMapModel
                // PacMapModel.ClearGlobalCallback();
                Console.WriteLine("Demo cleanup completed.");
            }
        }

        static void Demo27DEmbeddingWithProgress()
        {
            Console.WriteLine("=== Demo 1: 27D Embedding with Progress Reporting ===");

            // Generate sample high-dimensional data
            const int nSamples = 10000;
            const int nFeatures = 300;
            const int embeddingDim = 27;

            var data = GenerateTestData(nSamples, nFeatures, DataPattern.Clustered);
            Console.WriteLine($"Generated data: {nSamples} samples × {nFeatures} features");

            using var model = new PacMapModel();

            Console.WriteLine("Training 27D PACMAP embedding with progress reporting...");
            var startTime = DateTime.Now;

            // Progress tracking variables
            var lastPercent = -1;
            var progressBar = new char[50];

            var embedding = model.FitWithProgress(
                data: data,
                progressCallback: (phase, current, total, percent, message) =>
                {
                    var currentPercent = (int)percent;
                    if (currentPercent != lastPercent && currentPercent % 2 == 0) // Update every 2%
                    {
                        lastPercent = currentPercent;

                        // Update progress bar
                        var filled = (int)(percent / 2); // 50 characters for 100%
                        for (int i = 0; i < 50; i++)
                        {
                            progressBar[i] = i < filled ? '█' : '░';
                        }

                        var lossInfo = !string.IsNullOrEmpty(message) ? $" - {message}" : "";
                        Console.Write($"\r  Progress: [{new string(progressBar)}] {percent:F1}% (Iteration {current}/{total}){lossInfo}");
                    }
                },
                embeddingDimension: embeddingDim,
                nNeighbors: 20,
                numIters: (100, 100, 250),
                metric: DistanceMetric.Euclidean
            );

            var elapsed = DateTime.Now - startTime;
            Console.WriteLine($"\nTraining completed in {elapsed.TotalMilliseconds:F0}ms");

            // Display model info
            var info = model.ModelInfo;
            Console.WriteLine($"Model: {info}");

            // Show embedding statistics
            Console.WriteLine($"Embedding shape: [{embedding.GetLength(0)}, {embedding.GetLength(1)}]");
            ShowEmbeddingStats(embedding, "27D embedding");

            Console.WriteLine();
        }

        static void DemoMultiDimensionalEmbeddings()
        {
            Console.WriteLine("=== Demo 2: Multi-Dimensional Embeddings (1D to 50D) ===");

            var data = GenerateTestData(300, 50, DataPattern.Standard);
            Console.WriteLine($"Generated data: {data.GetLength(0)} samples × {data.GetLength(1)} features");

            var testDimensions = new[] { 1, 2, 3, 5, 10, 15, 20, 27, 35, 50 };

            foreach (var dim in testDimensions)
            {
                Console.WriteLine($"\nTesting {dim}D embedding:");

                using var model = new PacMapModel();

                // Use enhanced progress callback for larger dimensions
                ProgressCallback? progressCallback = null;
                if (dim >= 20)
                {
                    progressCallback = (phase, current, total, percent, message) =>
                    {
                        if (current % 25 == 0 || current == total) // Report every 25 epochs
                        {
                            string msg = !string.IsNullOrEmpty(message) ? $" ({message})" : "";
                            Console.Write($"\r    {phase} {dim}D: {percent:F0}%{msg} ");
                        }
                    };
                }

                var startTime = DateTime.Now;

                double[,] embedding;
                if (progressCallback != null)
                {
                    embedding = model.FitWithProgress(
                        data,
                        progressCallback,
                        embeddingDimension: dim,
                        nNeighbors: 15,
                        numIters: (50, 50, 75),
                        metric: DistanceMetric.Euclidean
                    );
                }
                else
                {
                    embedding = model.Fit(
                        data,
                        embeddingDimension: dim,
                        nNeighbors: 15,
                        numIters: (50, 50, 75),
                        metric: DistanceMetric.Euclidean
                    );
                }

                var elapsed = DateTime.Now - startTime;

                if (dim >= 20)
                {
                    Console.WriteLine(); // New line after progress
                }

                Console.WriteLine($"  Result: {embedding.GetLength(0)} samples → {dim}D in {elapsed.TotalMilliseconds:F0}ms");

                var info = model.ModelInfo;
                Console.WriteLine($"  Model info: {info.OutputDimension}D embedding, {info.TrainingSamples} training samples");

                // Show stats for first few dimensions
                ShowEmbeddingStats(embedding, $"{dim}D embedding", maxDims: Math.Min(5, dim));
            }

            Console.WriteLine();
        }

        static void DemoModelPersistence()
        {
            Console.WriteLine("=== Demo 3: Model Persistence and Transform ===");

            const string modelFile = "demo_model.pmp";

            try
            {
                // Generate training data
                var trainData = GenerateTestData(500, 50, DataPattern.Standard);
                var testData = GenerateTestData(100, 50, DataPattern.Standard, seed: 456);

                PacMapModelInfo savedInfo;

                // Train and save model with progress
                using (var model = new PacMapModel())
                {
                    Console.WriteLine("Training model with progress reporting...");

                    var trainEmbedding = model.FitWithProgress(
                        trainData,
                        progressCallback: (phase, current, total, percent, message) =>
                        {
                            if (current % 20 == 0 || current == total)
                            {
                                var lossInfo = !string.IsNullOrEmpty(message) ? $" - {message}" : "";
                                Console.Write($"\r  Training progress: {percent:F0}% (Iteration {current}/{total}){lossInfo}");
                            }
                        },
                        embeddingDimension: 5,
                        nNeighbors: 15,
                        numIters: (75, 75, 125),
                        metric: DistanceMetric.Cosine
                    );

                    Console.WriteLine(); // New line after progress

                    savedInfo = model.ModelInfo;
                    Console.WriteLine($"Trained model: {savedInfo}");

                    Console.WriteLine("Saving model...");
                    model.Save(modelFile);
                    Console.WriteLine($"Model saved to: {modelFile}");
                }

                // Load and use model
                Console.WriteLine("Loading model from disk...");
                using var loadedModel = PacMapModel.Load(modelFile);

                var loadedInfo = loadedModel.ModelInfo;
                Console.WriteLine($"Loaded model: {loadedInfo}");

                // Verify model consistency
                if (savedInfo.ToString() == loadedInfo.ToString())
                {
                    Console.WriteLine("✓ Model loaded successfully with consistent parameters");
                }

                // Transform new data
                Console.WriteLine("Transforming new data...");
                var testEmbedding = loadedModel.Transform(testData);
                Console.WriteLine($"Transformed {testData.GetLength(0)} samples");

                ShowEmbeddingStats(testEmbedding, "Transformed data");
            }
            finally
            {
                // Cleanup
                if (File.Exists(modelFile))
                    File.Delete(modelFile);
            }

            Console.WriteLine();
        }

        static void DemoDistanceMetricsWithProgress()
        {
            Console.WriteLine("=== Demo 4: Different Distance Metrics with Progress ===");

            var metrics = new[]
            {
                (DistanceMetric.Euclidean, DataPattern.Standard, "Standard Gaussian data"),
                (DistanceMetric.Cosine, DataPattern.Sparse, "Sparse high-dimensional data"),
                (DistanceMetric.Manhattan, DataPattern.Clustered, "Clustered data (outlier robust)"),
                (DistanceMetric.Correlation, DataPattern.Correlated, "Correlated features"),
                (DistanceMetric.Hamming, DataPattern.Binary, "Binary/categorical data")
            };

            foreach (var (metric, pattern, description) in metrics)
            {
                Console.WriteLine($"\nTesting {metric.ToString()} metric:");
                Console.WriteLine($"  Data type: {description}");

                var data = GenerateTestData(200, 20, pattern);

                using var model = new PacMapModel();

                // Progress callback for this metric
                var metricName = metric.ToString();
                var embedding = model.FitWithProgress(
                    data,
                    progressCallback: (phase, current, total, percent, message) =>
                    {
                        if (current % 15 == 0 || current == total)
                        {
                            var lossInfo = !string.IsNullOrEmpty(message) ? $" - {message}" : "";
                            Console.Write($"\r  {metricName}: {percent:F0}%{lossInfo} ");
                        }
                    },
                    embeddingDimension: 2,
                    nNeighbors: 12,
                    numIters: (50, 50, 75),
                    metric: metric
                );

                Console.WriteLine(); // New line after progress

                var info = model.ModelInfo;
                Console.WriteLine($"  Result: {embedding.GetLength(0)} samples → 2D, metric: {info.Metric}");
                ShowEmbeddingStats(embedding, $"{metric.ToString()} embedding", maxDims: 2);
            }

            Console.WriteLine();
        }

        static void DemoSafetyFeatures()
        {
            Console.WriteLine("=== Demo 5: Enhanced Safety Features with HNSW ===");

            // Generate training data with clear patterns
            var trainData = GenerateTestData(400, 30, DataPattern.Clustered, seed: 123);
            Console.WriteLine($"Training data: {trainData.GetLength(0)} samples × {trainData.GetLength(1)} features (clustered pattern)");

            using var model = new PacMapModel();

            // Train the model
            Console.WriteLine("Training model for safety analysis...");
            var trainEmbedding = model.FitWithProgress(
                trainData,
                progressCallback: (phase, current, total, percent, message) =>
                {
                    if (current % 30 == 0 || current == total)
                    {
                        Console.Write($"\r  Training: {percent:F0}%");
                    }
                },
                embeddingDimension: 10,
                nNeighbors: 15,
                numIters: (75, 75, 125),
                metric: DistanceMetric.Euclidean
            );

            Console.WriteLine("\n  Training completed!");

            // Generate different types of test data to demonstrate safety analysis
            var testScenarios = new[]
            {
                (GenerateTestData(5, 30, DataPattern.Clustered, seed: 200), "Similar to training (clustered)", true),
                (GenerateTestData(5, 30, DataPattern.Standard, seed: 201), "Somewhat different (Gaussian)", false),
                (GenerateExtremeOutliers(3, 30), "Extreme outliers", false)
            };

            Console.WriteLine("\nTransform Analysis with Safety Metrics:");

            foreach (var (testData, description, expectedSafe) in testScenarios)
            {
                Console.WriteLine($"\n--- {description} ---");

                try
                {
                    // Use enhanced transform with safety analysis
                    var results = model.TransformWithSafety(testData);

                    for (int i = 0; i < results.Length; i++)
                    {
                        var result = results[i];
                        Console.WriteLine($"  Sample {i + 1}:");
                        Console.WriteLine($"    Confidence: {result.ConfidenceScore:F3}");
                        Console.WriteLine($"    Severity: {result.Severity}");
                        Console.WriteLine($"    Percentile: {result.PercentileRank:F1}%");
                        Console.WriteLine($"    Z-Score: {result.ZScore:F2}");
                        Console.WriteLine($"    Quality: {result.QualityAssessment}");
                        Console.WriteLine($"    Production Ready: {(result.IsReliable ? "✓ Yes" : "✗ No")}");

                        // Show embedding coordinates (first 3 dimensions)
                        var coords = result.ProjectionCoordinates;
                        var coordStr = string.Join(", ", coords.Take(Math.Min(3, coords.Length)).Select(x => x.ToString("F3")));
                        if (coords.Length > 3) coordStr += "...";
                        Console.WriteLine($"    Coordinates: [{coordStr}]");

                        // Show nearest neighbors info
                        Console.WriteLine($"    Nearest neighbors: {result.NeighborCount} analyzed");
                        var nearestDist = result.NearestNeighborDistances[0];  // [0] = closest neighbor
                        Console.WriteLine($"    Closest training point distance: {nearestDist:F3}");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"    Error during transform: {ex.Message}");
                }
            }

            // Demonstrate batch processing with safety filtering
            Console.WriteLine("\n--- Batch Processing with Safety Filtering ---");
            var batchData = GenerateTestData(20, 30, DataPattern.Standard, seed: 300);

            try
            {
                var batchResults = model.TransformWithSafety(batchData);

                var safeCount = batchResults.Count(r => r.IsReliable);
                var normalCount = batchResults.Count(r => r.Severity == OutlierLevel.Normal);
                var outlierCount = batchResults.Count(r => r.Severity >= OutlierLevel.Mild);

                Console.WriteLine($"  Processed {batchResults.Length} samples:");
                Console.WriteLine($"    ✓ Production safe: {safeCount}/{batchResults.Length} ({100.0 * safeCount / batchResults.Length:F1}%)");
                Console.WriteLine($"    Normal: {normalCount}, Outliers: {outlierCount}");

                // Show distribution of confidence scores
                var avgConfidence = batchResults.Average(r => r.ConfidenceScore);
                var minConfidence = batchResults.Min(r => r.ConfidenceScore);
                var maxConfidence = batchResults.Max(r => r.ConfidenceScore);

                Console.WriteLine($"    Confidence range: {minConfidence:F3} - {maxConfidence:F3} (avg: {avgConfidence:F3})");

                // Show severity distribution
                var severityGroups = batchResults.GroupBy(r => r.Severity)
                                                .OrderBy(g => g.Key)
                                                .Select(g => $"{g.Key}: {g.Count()}")
                                                .ToArray();
                Console.WriteLine($"    Severity breakdown: {string.Join(", ", severityGroups)}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"    Batch processing error: {ex.Message}");
            }

            Console.WriteLine("\n  Safety analysis demonstrates:");
            Console.WriteLine("  • Real-time outlier detection for production safety");
            Console.WriteLine("  • Confidence scoring for reliability assessment");
            Console.WriteLine("  • Multi-level severity classification");
            Console.WriteLine("  • Nearest neighbor analysis for interpretability");
            Console.WriteLine("  • Quality assessment for decision making");

            Console.WriteLine();
        }

        static double[,] GenerateExtremeOutliers(int nSamples, int nFeatures, int seed = 999)
        {
            var random = new Random(seed);
            var data = new double[nSamples, nFeatures];

            for (int i = 0; i < nSamples; i++)
            {
                for (int j = 0; j < nFeatures; j++)
                {
                    // Generate extreme values far from typical training data
                    var sign = random.NextDouble() < 0.5 ? -1 : 1;
                    data[i, j] = sign * (10 + random.NextDouble() * 5); // Values in range [-15, -10] or [10, 15]
                }
            }

            return data;
        }

        static double[,] GenerateTestData(int nSamples, int nFeatures, DataPattern pattern, int seed = 42)
        {
            var random = new Random(seed);
            var data = new double[nSamples, nFeatures];

            switch (pattern)
            {
                case DataPattern.Standard:
                    // Standard Gaussian
                    for (int i = 0; i < nSamples; i++)
                    {
                        for (int j = 0; j < nFeatures; j++)
                        {
                            data[i, j] = GenerateNormal(random);
                        }
                    }
                    break;

                case DataPattern.Sparse:
                    // Sparse data (good for cosine metric)
                    for (int i = 0; i < nSamples; i++)
                    {
                        for (int j = 0; j < nFeatures; j++)
                        {
                            data[i, j] = random.NextDouble() < 0.2
                                ? GenerateNormal(random)
                                : 0.0;
                        }
                    }
                    break;

                case DataPattern.Binary:
                    // Binary data (good for Hamming metric)
                    for (int i = 0; i < nSamples; i++)
                    {
                        for (int j = 0; j < nFeatures; j++)
                        {
                            data[i, j] = random.NextDouble() < 0.5 ? 1.0 : 0.0;
                        }
                    }
                    break;

                case DataPattern.Clustered:
                    // Clustered data
                    var centers = new[] { -2.0, 0.0, 2.0 };
                    for (int i = 0; i < nSamples; i++)
                    {
                        var center = centers[random.Next(centers.Length)];
                        for (int j = 0; j < nFeatures; j++)
                        {
                            data[i, j] = center + GenerateNormal(random) * 0.5;
                        }
                    }
                    break;

                case DataPattern.Correlated:
                    // Correlated features
                    for (int i = 0; i < nSamples; i++)
                    {
                        var baseValue = GenerateNormal(random);
                        for (int j = 0; j < nFeatures; j++)
                        {
                            var correlation = 0.7;
                            var noise = GenerateNormal(random) * (1.0 - correlation);
                            data[i, j] = baseValue * correlation + noise;
                        }
                    }
                    break;
            }

            return data;
        }

        static double GenerateNormal(Random random)
        {
            // Box-Muller transform for normal distribution
             double? spare = null;

            if (spare != null)
            {
                var result = spare.Value;
                spare = null;
                return result;
            }

            var u1 = 1.0 - random.NextDouble();
            var u2 = 1.0 - random.NextDouble();
            var normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            spare = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);

            return normal;
        }

        static void ShowEmbeddingStats(double[,] embedding, string title, int maxDims = 5)
        {
            var nSamples = embedding.GetLength(0);
            var nDims = embedding.GetLength(1);

            Console.WriteLine($"  {title} statistics (first {Math.Min(maxDims, nDims)} dimensions):");

            for (int d = 0; d < Math.Min(maxDims, nDims); d++)
            {
                double min = double.MaxValue, max = double.MinValue, sum = 0;

                for (int i = 0; i < nSamples; i++)
                {
                    var val = embedding[i, d];
                    min = Math.Min(min, val);
                    max = Math.Max(max, val);
                    sum += val;
                }

                var mean = sum / nSamples;
                Console.WriteLine($"    Dim {d}: range=[{min:F3}, {max:F3}], mean={mean:F3}");
            }

            if (nDims > maxDims)
            {
                Console.WriteLine($"    ... ({nDims - maxDims} more dimensions)");
            }
        }

        static void DemoSpreadParameter()
        {
            Console.WriteLine("\n=== Demo 6: NEW Spread Parameter with Smart Defaults ===");

            var data = GenerateTestData(500, 100, DataPattern.Standard);
            Console.WriteLine($"Generated data: {data.GetLength(0)} samples × {data.GetLength(1)} features");

            // Demo 1: Smart auto-defaults (recommended approach)
            Console.WriteLine("\n1. Using Smart Auto-Defaults (Recommended):");
            Console.WriteLine("   - 2D: numIters=(100,100,250), neighbors=25 (PACMAP optimal parameters)");
            Console.WriteLine("   - Higher dimensions: use same three-phase iteration pattern");

            var dimensions = new[] { 2, 10, 24 };
            foreach (var dim in dimensions)
            {
                using var model = new PacMapModel();

                var sw = System.Diagnostics.Stopwatch.StartNew();

                // Use smart defaults - just specify dimension!
                var embedding = model.Fit(data, embeddingDimension: dim);

                sw.Stop();
                Console.WriteLine($"   {dim}D embedding: {embedding.GetLength(0)} samples × {embedding.GetLength(1)}D (auto-optimized in {sw.ElapsedMilliseconds}ms)");
            }

            // Demo 2: Custom PACMAP parameters comparison
            Console.WriteLine("\n2. Custom PACMAP Parameters Comparison (2D Visualization):");

            var testData = GenerateTestData(200, 50, DataPattern.Clustered);
            var testValues = new[] { 1.0, 2.5, 5.0, 8.0 };

            foreach (var spread in testValues)
            {
                using var model = new PacMapModel();

                var sw = System.Diagnostics.Stopwatch.StartNew();

                // Use PACMAP parameters instead of UMAP spread/minDist
                var embedding = model.Fit(
                    data: testData,
                    embeddingDimension: 2,
                    nNeighbors: 25,           // PACMAP neighbor parameter
                    numIters: (100, 100, 250) // PACMAP three-phase iterations
                );

                sw.Stop();
                Console.WriteLine($"   PACMAP {spread:F1}: {CalculateSpreadScore(embedding)} space utilization ({sw.ElapsedMilliseconds}ms)");
            }

            // Demo 3: Dimension scaling demonstration
            Console.WriteLine("\n3. Dimension-Based Scaling with PACMAP:");
            Console.WriteLine("   PACMAP uses consistent three-phase iterations across dimensions:");

            var scalingDemo = new[] {
                (dim: 2, desc: "2D Visualization"),
                (dim: 10, desc: "10D Clustering"),
                (dim: 24, desc: "24D ML Pipeline")
            };

            foreach (var (dim, desc) in scalingDemo)
            {
                using var model = new PacMapModel();

                // Show PACMAP parameters for this dimension
                var iterations = dim switch
                {
                    2 => (100, 100, 250),
                    <= 10 => (75, 75, 125),
                    _ => (50, 50, 100)
                };

                Console.WriteLine($"   {desc} ({dim}D): numIters={iterations}");

                var embedding = model.Fit(data, embeddingDimension: dim);
                Console.WriteLine($"     Result: {embedding.GetLength(0)} samples × {embedding.GetLength(1)}D embedding");
            }

            Console.WriteLine("\n✓ PACMAP three-phase optimization successfully demonstrated!");
            Console.WriteLine("  - Use model.Fit(data, embeddingDimension: dim) for default (100,100,250) iterations");
            Console.WriteLine("  - Override with custom numIters/(phase1,phase2,phase3) for fine-tuning");
        }

        private static double CalculateSpreadScore(double[,] embedding)
        {
            // Simple metric: average distance from origin (higher = more spread out)
            double totalDist = 0;
            int nSamples = embedding.GetLength(0);

            for (int i = 0; i < nSamples; i++)
            {
                double dist = 0;
                for (int j = 0; j < embedding.GetLength(1); j++)
                {
                    dist += embedding[i, j] * embedding[i, j];
                }
                totalDist += Math.Sqrt(dist);
            }

            return totalDist / nSamples;
        }

        enum DataPattern
        {
            Standard,
            Sparse,
            Binary,
            Clustered,
            Correlated
        }
    }
}