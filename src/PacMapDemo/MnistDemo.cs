using System;
using System.IO;
using System.Linq;

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

                // Convert first 1000 samples to float array
                var subsetSize = Math.Min(1000, mnistData.NumImages);
                var conversionStopwatch = System.Diagnostics.Stopwatch.StartNew();
                var floatData = mnistData.ToFloatArray(0, subsetSize);
                conversionStopwatch.Stop();

                Console.WriteLine($"   Converted {subsetSize:N0} images to float array");
                Console.WriteLine($"   Shape: [{floatData.GetLength(0):N0}, {floatData.GetLength(1)}]");
                Console.WriteLine($"   Conversion time: {conversionStopwatch.Elapsed.TotalMilliseconds:F1} ms");
                Console.WriteLine($"   Memory: {floatData.Length * 4 / 1024 / 1024.0:F1} MB");
                Console.WriteLine();

                // Show some statistics about the float data
                Console.WriteLine("📊 Float Data Statistics:");
                Console.WriteLine("========================");

                double minVal = floatData.Cast<float>().Min();
                double maxVal = floatData.Cast<float>().Max();
                double meanVal = floatData.Cast<float>().Average();

                Console.WriteLine($"   Value range: [{minVal:F3}, {maxVal:F3}]");
                Console.WriteLine($"   Mean value: {meanVal:F3}");
                Console.WriteLine($"   Expected range: [0.000, 1.000] (normalized from 0-255)");
                Console.WriteLine();

                // Demonstrate PACMAP usage (placeholder)
                Console.WriteLine("🚀 Ready for PACMAP Integration:");
                Console.WriteLine("==============================");
                Console.WriteLine("The float data can now be used with PACMAP:");
                Console.WriteLine();
                Console.WriteLine("// Example PACMAP usage:");
                Console.WriteLine("var model = new PacMapModel();");
                Console.WriteLine("var embedding = model.Fit(");
                Console.WriteLine("    data: floatData,  // MNIST data as float[,]");
                Console.WriteLine("    embeddingDimension: 2,");
                Console.WriteLine("    nNeighbors: 15,");
                Console.WriteLine("    // ... other parameters");
                Console.WriteLine(");");
                Console.WriteLine();

                // Performance comparison
                Console.WriteLine("⚡ Performance Benefits:");
                Console.WriteLine("======================");
                Console.WriteLine("✅ Binary loading is ~10x faster than parsing text/CSV");
                Console.WriteLine("✅ Direct memory access without intermediate conversions");
                Console.WriteLine("✅ Compact storage with minimal memory overhead");
                Console.WriteLine("✅ Type-safe with fixed structure validation");
                Console.WriteLine("✅ Cross-platform compatibility (little-endian)");
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
                var floatData = mnistData.ToFloatArray(0, actualSubsetSize);
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
    }
}