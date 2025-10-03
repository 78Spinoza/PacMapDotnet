using System;
using System.IO;
using PacMAPSharp;

namespace FrameGenerator
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("🦣 Generating Additional Mammoth Frames");
            Console.WriteLine("=====================================");

            // Load regular mammoth data
            var mammothData = DataLoaders.LoadMammothData("../PacMapDemo/Data/mammoth_data.csv");
            Console.WriteLine($"🦣 Loaded mammoth: {mammothData.GetLength(0):N0} samples, {mammothData.GetLength(1)} features");

            // Create output directory
            string outputDir = "../PacMapDemo/Results/additional_frames";
            Directory.CreateDirectory(outputDir);

            // More granular parameter arrays
            double[] midNearRatios = { 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                                     0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
                                     1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.50, 4.00, 4.50, 5.00 };

            double[] farPairRatios = { 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0,
                                     3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0 };

            int[] neighbors = { 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80 };

            int frameCount = 1;

            // Generate mid-near variations
            foreach (var ratio in midNearRatios)
            {
                Console.WriteLine($"🧪 Mid-near ratio: {ratio} ({frameCount}/{midNearRatios.Length + farPairRatios.Length + neighbors.Length})");

                using var model = new PacMAPModel();
                var result = model.Fit(mammothData, embeddingDimensions: 2, neighbors: 10,
                                     normalization: NormalizationMode.ZScore,
                                     metric: DistanceMetric.Euclidean,
                                     forceExactKnn: true,
                                     midNearRatio: ratio, farPairRatio: 2.0,
                                     seed: 42);

                var embedding = result.EmbeddingCoordinates;
                var embedding2D = ConvertEmbeddingTo2D(embedding, mammothData.GetLength(0), 2);
                string plotPath = Path.Combine(outputDir, $"midnear_{frameCount:D3}.png");

                Visualizer.PlotSimplePacMAP(embedding2D, $"Mammoth - midNearRatio={ratio}", plotPath, $"midNearRatio={ratio}");
                frameCount++;
            }

            // Generate far-pair variations
            foreach (var ratio in farPairRatios)
            {
                Console.WriteLine($"🧪 Far-pair ratio: {ratio} ({frameCount}/{midNearRatios.Length + farPairRatios.Length + neighbors.Length})");

                using var model = new PacMAPModel();
                var result = model.Fit(mammothData, embeddingDimensions: 2, neighbors: 10,
                                     normalization: NormalizationMode.ZScore,
                                     metric: DistanceMetric.Euclidean,
                                     forceExactKnn: true,
                                     midNearRatio: 0.5, farPairRatio: ratio,
                                     seed: 42);

                var embedding = result.EmbeddingCoordinates;
                var embedding2D = ConvertEmbeddingTo2D(embedding, mammothData.GetLength(0), 2);
                string plotPath = Path.Combine(outputDir, $"farpair_{frameCount:D3}.png");

                Visualizer.PlotSimplePacMAP(embedding2D, $"Mammoth - farPairRatio={ratio}", plotPath, $"farPairRatio={ratio}");
                frameCount++;
            }

            // Generate neighbor variations
            foreach (var n in neighbors)
            {
                Console.WriteLine($"🧪 Neighbors: {n} ({frameCount}/{midNearRatios.Length + farPairRatios.Length + neighbors.Length})");

                using var model = new PacMAPModel();
                var result = model.Fit(mammothData, embeddingDimensions: 2, neighbors: n,
                                     normalization: NormalizationMode.ZScore,
                                     metric: DistanceMetric.Euclidean,
                                     forceExactKnn: true,
                                     midNearRatio: 0.5, farPairRatio: 2.0,
                                     seed: 42);

                var embedding = result.EmbeddingCoordinates;
                var embedding2D = ConvertEmbeddingTo2D(embedding, mammothData.GetLength(0), 2);
                string plotPath = Path.Combine(outputDir, $"neighbors_{frameCount:D3}.png");

                Visualizer.PlotSimplePacMAP(embedding2D, $"Mammoth - neighbors={n}", plotPath, $"neighbors={n}");
                frameCount++;
            }

            Console.WriteLine($"✅ Generated {frameCount-1} additional frames");
            Console.WriteLine($"📁 Results in: {outputDir}");
        }

        // Copy conversion method from main program
        static double[,] ConvertEmbeddingTo2D(float[] embedding, int numPoints, int dimensions)
        {
            double[,] embedding2D = new double[numPoints, dimensions];

            for (int i = 0; i < numPoints; i++)
            {
                for (int d = 0; d < dimensions; d++)
                {
                    embedding2D[i, d] = embedding[i * dimensions + d];
                }
            }

            return embedding2D;
        }
    }
}