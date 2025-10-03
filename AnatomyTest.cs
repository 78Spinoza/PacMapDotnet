using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using ScottPlot;

namespace PacMapDemo
{
    /// <summary>
    /// Separate test program to perfect mammoth anatomical classification
    /// </summary>
    public static class AnatomyTest
    {
        /// <summary>
        /// Test different anatomical classification algorithms
        /// </summary>
        /// <param name="originalData">Original 3D mammoth data</param>
        /// <param name="version">Algorithm version to test</param>
        /// <returns>Array of part names for each point</returns>
        public static string[] TestAnatomicalClassification(double[,] originalData, int version)
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
            double zCenter = (minZ + maxZ) / 2;

            Console.WriteLine($"Testing Classification Version {version}:");
            Console.WriteLine($"X range: [{minX:F1}, {maxX:F1}] (range: {xRange:F1}, center: {xCenter:F1})");
            Console.WriteLine($"Y range: [{minY:F1}, {maxY:F1}] (range: {yRange:F1}, center: {yCenter:F1})");
            Console.WriteLine($"Z range: [{minZ:F1}, {maxZ:F1}] (range: {zRange:F1}, center: {zCenter:F1})");

            switch (version)
            {
                case 1: // Original approach - Z based
                    ClassifyVersion1(originalData, parts, minX, maxX, minY, maxY, minZ, maxZ, xRange, yRange, zRange, xCenter, yCenter, zCenter);
                    break;
                case 2: // Better trunk detection - look for hanging extension
                    ClassifyVersion2(originalData, parts, minX, maxX, minY, maxY, minZ, maxZ, xRange, yRange, zRange, xCenter, yCenter, zCenter);
                    break;
                case 3: // Focus on head compactness vs trunk extension
                    ClassifyVersion3(originalData, parts, minX, maxX, minY, maxY, minZ, maxZ, xRange, yRange, zRange, xCenter, yCenter, zCenter);
                    break;
                case 4: // Analyze actual mammoth shape from 3D views
                    ClassifyVersion4(originalData, parts, minX, maxX, minY, maxY, minZ, maxZ, xRange, yRange, zRange, xCenter, yCenter, zCenter);
                    break;
            }

            return parts;
        }

        private static void ClassifyVersion1(double[,] originalData, string[] parts, double minX, double maxX, double minY, double maxY, double minZ, double maxZ, double xRange, double yRange, double zRange, double xCenter, double yCenter, double zCenter)
        {
            Console.WriteLine("Version 1: Original Z-based classification");

            // Simple Z-based classification
            double lowZThreshold = minZ + zRange * 0.25;
            double highZThreshold = minZ + zRange * 0.75;

            for (int i = 0; i < originalData.GetLength(0); i++)
            {
                double x = originalData[i, 0];
                double y = originalData[i, 1];
                double z = originalData[i, 2];

                if (z < lowZThreshold)
                    parts[i] = "legs";
                else if (z > highZThreshold)
                    parts[i] = "head";
                else
                    parts[i] = "body";
            }
        }

        private static void ClassifyVersion2(double[,] originalData, string[] parts, double minX, double maxX, double minY, double maxY, double minZ, double maxZ, double xRange, double yRange, double zRange, double xCenter, double yCenter, double zCenter)
        {
            Console.WriteLine("Version 2: Trunk as hanging extension");

            // Legs at bottom
            double legZThreshold = minZ + zRange * 0.2;

            // Head at top
            double headZThreshold = minZ + zRange * 0.8;

            // Trunk detection: look for points that hang down from head area
            double trunkZMax = minZ + zRange * 0.7; // Trunk doesn't go to very top
            double trunkZMin = minZ + zRange * 0.3; // Trunk doesn't go to very bottom

            // Trunk should be forward (high X) and narrow in Y
            double trunkXThreshold = minX + xRange * 0.6;
            double trunkYRange = yRange * 0.4;

            for (int i = 0; i < originalData.GetLength(0); i++)
            {
                double x = originalData[i, 0];
                double y = originalData[i, 1];
                double z = originalData[i, 2];

                double yDistFromCenter = Math.Abs(y - yCenter);

                if (z < legZThreshold)
                {
                    parts[i] = "legs";
                }
                else if (z > headZThreshold)
                {
                    parts[i] = "head";
                }
                else if (x > trunkXThreshold && yDistFromCenter < trunkYRange && z >= trunkZMin && z <= trunkZMax)
                {
                    parts[i] = "trunk";
                }
                else
                {
                    parts[i] = "body";
                }
            }
        }

        private static void ClassifyVersion3(double[,] originalData, string[] parts, double minX, double maxX, double minY, double maxY, double minZ, double maxZ, double xRange, double yRange, double zRange, double xCenter, double yCenter, double zCenter)
        {
            Console.WriteLine("Version 3: Head compactness vs trunk length");

            // Legs clearly at bottom
            double legZThreshold = minZ + zRange * 0.15;

            // Head: high Z + not too extended in X (compact)
            double headZThreshold = minZ + zRange * 0.75;
            double headXMax = minX + xRange * 0.7; // Head not too forward

            // Trunk: forward extension + medium height
            double trunkXThreshold = minX + xRange * 0.7; // Forward
            double trunkZMin = minZ + zRange * 0.25;
            double trunkZMax = minZ + zRange * 0.8;
            double trunkYRange = yRange * 0.5; // Moderate width

            for (int i = 0; i < originalData.GetLength(0); i++)
            {
                double x = originalData[i, 0];
                double y = originalData[i, 1];
                double z = originalData[i, 2];

                double yDistFromCenter = Math.Abs(y - yCenter);

                if (z < legZThreshold)
                {
                    parts[i] = "legs";
                }
                else if (z > headZThreshold && x < headXMax)
                {
                    parts[i] = "head";
                }
                else if (x > trunkXThreshold && yDistFromCenter < trunkYRange && z >= trunkZMin && z <= trunkZMax)
                {
                    parts[i] = "trunk";
                }
                else
                {
                    parts[i] = "body";
                }
            }
        }

        private static void ClassifyVersion4(double[,] originalData, string[] parts, double minX, double maxX, double minY, double maxY, double minZ, double maxZ, double xRange, double yRange, double zRange, double xCenter, double yCenter, double zCenter)
        {
            Console.WriteLine("Version 4: Based on actual 3D mammoth shape analysis");

            // From 3D view analysis:
            // - Legs are clearly at bottom (low Z)
            // - Head is the bulky part at top-back (high Z, not maximum X)
            // - Trunk hangs down from head area (high X, medium Z)
            // - Body connects everything

            double legZThreshold = minZ + zRange * 0.18;

            // Head: high Z but NOT the most forward X (that's trunk)
            double headZThreshold = minZ + zRange * 0.7;
            double headXMax = minX + xRange * 0.6; // Head is behind trunk

            // Trunk: most forward X + hanging down from head level
            double trunkXThreshold = minX + xRange * 0.75; // Most forward
            double trunkZMin = minZ + zRange * 0.2;
            double trunkZMax = minZ + zRange * 0.75; // Hangs down from head
            double trunkYRange = yRange * 0.35; // Narrow

            for (int i = 0; i < originalData.GetLength(0); i++)
            {
                double x = originalData[i, 0];
                double y = originalData[i, 1];
                double z = originalData[i, 2];

                double yDistFromCenter = Math.Abs(y - yCenter);

                if (z < legZThreshold)
                {
                    parts[i] = "legs";
                }
                else if (x > trunkXThreshold && yDistFromCenter < trunkYRange && z >= trunkZMin && z <= trunkZMax)
                {
                    parts[i] = "trunk"; // Check trunk first (most forward)
                }
                else if (z > headZThreshold && x <= headXMax)
                {
                    parts[i] = "head"; // Head behind trunk area
                }
                else
                {
                    parts[i] = "body";
                }
            }
        }

        /// <summary>
        /// Create 3D visualization with specific anatomical classification version
        /// </summary>
        public static void TestVisualization(double[,] originalData, int version, string outputPath)
        {
            var parts = TestAnatomicalClassification(originalData, version);

            // Count parts
            var partCounts = parts.GroupBy(p => p).ToDictionary(g => g.Key, g => g.Count());
            Console.WriteLine($"Part counts: {string.Join(", ", partCounts.Select(kv => $"{kv.Key}: {kv.Value}"))}");

            // Create visualization
            CreateTestVisualization(originalData, parts, $"Mammoth Anatomy Test - Version {version}", outputPath);
        }

        private static void CreateTestVisualization(double[,] originalData, string[] parts, string title, string outputPath)
        {
            try
            {
                Console.WriteLine($"Creating test visualization: {title}");

                int numPoints = originalData.GetLength(0);

                // Define colors for each part
                var partColors = new Dictionary<string, Color>
                {
                    { "legs", Color.Blue },
                    { "body", Color.Green },
                    { "head", Color.Purple },
                    { "trunk", Color.Red }
                };

                var x = new double[numPoints];
                var y = new double[numPoints];
                var z = new double[numPoints];

                for (int i = 0; i < numPoints; i++)
                {
                    x[i] = originalData[i, 0];
                    y[i] = originalData[i, 1];
                    z[i] = originalData[i, 2];
                }

                // Group points by part
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

                // Normalize coordinates
                double xMin = x.Min(), xMax = x.Max(), xRange = xMax - xMin;
                double yMin = y.Min(), yMax = y.Max(), yRange = yMax - yMin;
                double zMin = z.Min(), zMax = z.Max(), zRange = zMax - zMin;

                // Create three-view visualization
                var plt = new Plot(2400, 800);

                // XY projection (left)
                foreach (var kvp in partGroups)
                {
                    var part = kvp.Key;
                    var (xPoints, yPoints, zPoints) = kvp.Value;
                    if (xPoints.Count > 0)
                    {
                        var normX = xPoints.Select(x => (x - xMin) / xRange * 600 + 50).ToArray();
                        var normY = yPoints.Select(y => (y - yMin) / yRange * 600 + 100).ToArray();
                        plt.AddScatter(normX, normY,
                            color: partColors[part], lineWidth: 0, markerSize: 2,
                            label: part == "legs" ? $"{char.ToUpper(part[0]) + part.Substring(1)}" : null);
                    }
                }

                // XZ projection (middle)
                foreach (var kvp in partGroups)
                {
                    var part = kvp.Key;
                    var (xPoints, yPoints, zPoints) = kvp.Value;
                    if (xPoints.Count > 0)
                    {
                        var normX = xPoints.Select(x => (x - xMin) / xRange * 600 + 850).ToArray();
                        var normZ = zPoints.Select(z => (z - zMin) / zRange * 600 + 100).ToArray();
                        plt.AddScatter(normX, normZ, color: partColors[part], lineWidth: 0, markerSize: 2);
                    }
                }

                // YZ projection (right)
                foreach (var kvp in partGroups)
                {
                    var part = kvp.Key;
                    var (xPoints, yPoints, zPoints) = kvp.Value;
                    if (xPoints.Count > 0)
                    {
                        var normY = yPoints.Select(y => (y - yMin) / yRange * 600 + 1650).ToArray();
                        var normZ = zPoints.Select(z => (z - zMin) / zRange * 600 + 100).ToArray();
                        plt.AddScatter(normY, normZ, color: partColors[part], lineWidth: 0, markerSize: 2);
                    }
                }

                plt.Title(title);
                plt.XLabel("XY View | XZ View | YZ View");
                plt.YLabel("Mammoth 3D Projections");
                plt.Legend(location: Alignment.UpperRight);
                plt.Grid(enable: false);

                Directory.CreateDirectory(Path.GetDirectoryName(outputPath));
                plt.SaveFig(outputPath, width: 2400, height: 800);

                Console.WriteLine($"Test visualization saved: {outputPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to create test visualization: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Main test function - runs all versions
        /// </summary>
        public static void RunAnatomyTests(double[,] originalData, string outputDir)
        {
            Console.WriteLine("=== MAMMOTH ANATOMY CLASSIFICATION TESTS ===\n");

            Directory.CreateDirectory(outputDir);

            for (int version = 1; version <= 4; version++)
            {
                Console.WriteLine($"\n--- Testing Version {version} ---");
                string outputPath = Path.Combine(outputDir, $"anatomy_test_v{version}.png");
                TestVisualization(originalData, version, outputPath);
                Console.WriteLine();
            }

            Console.WriteLine("=== All anatomy tests completed ===");
        }
    }
}