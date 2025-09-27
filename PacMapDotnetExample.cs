// Example usage of PacMAP Enhanced with HNSW Auto-scaling from C#
using System;
using PacMapDotnet;

class Program
{
    static void Main()
    {
        Console.WriteLine("PacMAP Enhanced C# Integration Example");
        Console.WriteLine($"Library Version: {PacMapModel.GetVersion()}");
        Console.WriteLine();

        // Example 1: Basic usage with auto-scaling
        BasicUsageExample();

        Console.WriteLine();

        // Example 2: High-accuracy configuration for research
        HighAccuracyExample();

        Console.WriteLine();

        // Example 3: Memory-optimized for large datasets
        MemoryOptimizedExample();

        Console.WriteLine();

        // Example 4: Model persistence
        ModelPersistenceExample();
    }

    static void BasicUsageExample()
    {
        Console.WriteLine("=== Example 1: Basic Usage with Auto-scaling ===");

        // Create sample data (1000 samples, 10 features)
        var data = GenerateSampleData(1000, 10);
        Console.WriteLine($"Generated sample data: {data.GetLength(0)} samples, {data.GetLength(1)} features");

        // Create configuration with auto-scaling
        var config = PacMapConfig.Default();
        config.HnswConfig = HnswConfig.AutoScale(HnswUseCase.Balanced);
        config.NormalizationMode = (int)NormalizationMode.Auto;

        Console.WriteLine("Configuration:");
        Console.WriteLine($"  HNSW: Auto-scaling with Balanced use case");
        Console.WriteLine($"  Normalization: Auto-detect");
        Console.WriteLine($"  Neighbors: {config.NNeighbors}");
        Console.WriteLine($"  Embedding dimensions: {config.EmbeddingDimensions}");

        // Fit model with progress tracking
        using var model = new PacMapModel();
        model.ProgressChanged += (sender, e) =>
        {
            Console.WriteLine($"  [{e.Percent:F1}%] {e.Phase}: {e.Current}/{e.Total} - {e.Message}");
        };

        var embedding = model.FitTransform(data, config);
        Console.WriteLine($"Embedding shape: {embedding.GetLength(0)} x {embedding.GetLength(1)}");

        // Display model information
        var modelInfo = model.GetModelInfo();
        Console.WriteLine("Model Information:");
        Console.WriteLine($"  Training samples: {modelInfo.NSamples}");
        Console.WriteLine($"  Features: {modelInfo.NFeatures}");
        Console.WriteLine($"  Normalization: {modelInfo.NormalizationMode}");
        Console.WriteLine($"  HNSW parameters: M={modelInfo.HnswM}, ef_construction={modelInfo.HnswEfConstruction}, ef_search={modelInfo.HnswEfSearch}");
        Console.WriteLine($"  Estimated HNSW memory: {modelInfo.MemoryUsageMb} MB");

        Console.WriteLine("✅ Basic usage completed successfully");
    }

    static void HighAccuracyExample()
    {
        Console.WriteLine("=== Example 2: High-Accuracy Configuration ===");

        // Create sample data (smaller dataset for higher accuracy)
        var data = GenerateSampleData(500, 20);
        Console.WriteLine($"Generated sample data: {data.GetLength(0)} samples, {data.GetLength(1)} features");

        // Create high-accuracy configuration
        var config = PacMapConfig.Default();
        config.HnswConfig = HnswConfig.AutoScale(HnswUseCase.HighAccuracy);
        config.NormalizationMode = (int)NormalizationMode.ZScore; // Force Z-score normalization
        config.NNeighbors = 15; // More neighbors for better quality

        Console.WriteLine("High-accuracy configuration:");
        Console.WriteLine($"  HNSW: Auto-scaling optimized for high accuracy");
        Console.WriteLine($"  Normalization: Z-score (forced)");
        Console.WriteLine($"  Neighbors: {config.NNeighbors}");

        using var model = new PacMapModel();
        model.ProgressChanged += (sender, e) =>
        {
            if (e.Percent % 20 == 0 || e.Percent == 100) // Reduce output
                Console.WriteLine($"  [{e.Percent:F1}%] {e.Phase}");
        };

        var embedding = model.FitTransform(data, config);

        var modelInfo = model.GetModelInfo();
        Console.WriteLine($"High-accuracy HNSW parameters: M={modelInfo.HnswM}, ef_construction={modelInfo.HnswEfConstruction}, ef_search={modelInfo.HnswEfSearch}");
        Console.WriteLine($"Memory usage: {modelInfo.MemoryUsageMb} MB");

        Console.WriteLine("✅ High-accuracy example completed");
    }

    static void MemoryOptimizedExample()
    {
        Console.WriteLine("=== Example 3: Memory-Optimized for Large Datasets ===");

        // Simulate large dataset
        var data = GenerateSampleData(5000, 50);
        Console.WriteLine($"Generated large dataset: {data.GetLength(0)} samples, {data.GetLength(1)} features");

        // Create memory-optimized configuration
        var config = PacMapConfig.Default();
        config.HnswConfig = HnswConfig.AutoScale(HnswUseCase.MemoryOptimized, memoryLimitMb: 100); // 100MB limit
        config.NormalizationMode = (int)NormalizationMode.Robust; // Robust to outliers
        config.NNeighbors = 8; // Fewer neighbors to save memory

        Console.WriteLine("Memory-optimized configuration:");
        Console.WriteLine($"  HNSW: Auto-scaling with 100MB memory limit");
        Console.WriteLine($"  Normalization: Robust (outlier-resistant)");
        Console.WriteLine($"  Neighbors: {config.NNeighbors}");

        using var model = new PacMapModel();
        model.ProgressChanged += (sender, e) =>
        {
            if (e.Phase == "Complete" || e.Percent % 25 == 0)
                Console.WriteLine($"  [{e.Percent:F1}%] {e.Phase}: {e.Message}");
        };

        var embedding = model.FitTransform(data, config);

        var modelInfo = model.GetModelInfo();
        Console.WriteLine($"Memory-optimized HNSW: M={modelInfo.HnswM}, Memory={modelInfo.MemoryUsageMb}MB");

        if (modelInfo.MemoryUsageMb <= 100)
        {
            Console.WriteLine("✅ Successfully stayed within memory limit");
        }
        else
        {
            Console.WriteLine($"⚠️ Exceeded memory limit (estimated {modelInfo.MemoryUsageMb}MB)");
        }

        Console.WriteLine("✅ Memory-optimized example completed");
    }

    static void ModelPersistenceExample()
    {
        Console.WriteLine("=== Example 4: Model Persistence ===");

        // Create and fit a model
        var data = GenerateSampleData(200, 5);
        var config = PacMapConfig.Default();

        using var originalModel = new PacMapModel();
        Console.WriteLine("Fitting original model...");
        originalModel.ProgressChanged += (sender, e) =>
        {
            if (e.Phase == "Complete")
                Console.WriteLine($"  {e.Phase}: {e.Message}");
        };

        var originalEmbedding = originalModel.FitTransform(data, config);
        var originalInfo = originalModel.GetModelInfo();

        // Save model
        string modelPath = "pacmap_example_model.bin";
        Console.WriteLine($"Saving model to {modelPath}...");
        originalModel.Save(modelPath, quantize: true); // Use quantization for smaller file size

        // Load model
        Console.WriteLine("Loading model from file...");
        using var loadedModel = PacMapModel.Load(modelPath);
        var loadedInfo = loadedModel.GetModelInfo();

        // Verify model consistency
        Console.WriteLine("Verifying model consistency:");
        Console.WriteLine($"  Original samples: {originalInfo.NSamples}, Loaded: {loadedInfo.NSamples}");
        Console.WriteLine($"  Original features: {originalInfo.NFeatures}, Loaded: {loadedInfo.NFeatures}");
        Console.WriteLine($"  Original HNSW M: {originalInfo.HnswM}, Loaded: {loadedInfo.HnswM}");

        bool isConsistent = originalInfo.NSamples == loadedInfo.NSamples &&
                           originalInfo.NFeatures == loadedInfo.NFeatures &&
                           originalInfo.HnswM == loadedInfo.HnswM;

        if (isConsistent)
        {
            Console.WriteLine("✅ Model saved and loaded successfully");
        }
        else
        {
            Console.WriteLine("❌ Model consistency check failed");
        }

        // Transform new data with loaded model
        var newData = GenerateSampleData(50, 5); // Same feature count
        Console.WriteLine("Transforming new data with loaded model...");
        var transformedEmbedding = loadedModel.Transform(newData);
        Console.WriteLine($"Transformed {newData.GetLength(0)} new samples");

        Console.WriteLine("✅ Model persistence example completed");
    }

    static double[,] GenerateSampleData(int rows, int cols)
    {
        var random = new Random(42); // Fixed seed for reproducibility
        var data = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // Generate normally distributed data with some structure
                double value = random.NextGaussian() * 2.0 + (i / 100.0) * j;
                data[i, j] = value;
            }
        }

        return data;
    }
}

// Extension method for generating Gaussian random numbers
public static class RandomExtensions
{
    public static double NextGaussian(this Random random, double mean = 0.0, double stdDev = 1.0)
    {
        // Box-Muller transform
        if (random.NextDouble() > 0.5)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return mean + stdDev * randStdNormal;
        }
        else
        {
            return mean + stdDev * Math.Sqrt(-2.0 * Math.Log(random.NextDouble())) *
                   Math.Cos(2.0 * Math.PI * random.NextDouble());
        }
    }
}