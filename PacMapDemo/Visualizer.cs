using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using ScottPlot;

namespace PacMapDemo
{
    /// <summary>
    /// Visualization utilities for PacMAP results
    /// </summary>
    public static class Visualizer
    {
        /// <summary>
        /// Color palette for different classes/clusters
        /// </summary>
        private static readonly Color[] ClassColors = new Color[]
        {
            Color.Red,       // 0
            Color.Blue,      // 1
            Color.Green,     // 2
            Color.Orange,    // 3
            Color.Purple,    // 4
            Color.Brown,     // 5
            Color.Pink,      // 6
            Color.Gray,      // 7
            Color.Olive,     // 8
            Color.Cyan       // 9
        };

        /// <summary>
        /// Plot MNIST PacMAP results with color-coded digits
        /// </summary>
        /// <param name="embedding">2D embedding from PacMAP</param>
        /// <param name="labels">MNIST digit labels (0-9)</param>
        /// <param name="title">Plot title</param>
        /// <param name="outputPath">Path to save the plot image</param>
        public static void PlotMNIST(float[,] embedding, int[] labels, string title, string outputPath)
        {
            try
            {
                Console.WriteLine($"Creating MNIST plot: {title}");

                var plt = new Plot(1200, 800);

                // Group points by digit class
                var classesSeparated = new Dictionary<int, (List<double> x, List<double> y)>();

                for (int i = 0; i < 10; i++)
                {
                    classesSeparated[i] = (new List<double>(), new List<double>());
                }

                // Separate points by class
                int numPoints = embedding.GetLength(0);
                for (int i = 0; i < numPoints; i++)
                {
                    int digit = labels[i];
                    if (digit >= 0 && digit <= 9)
                    {
                        classesSeparated[digit].x.Add(embedding[i, 0]);
                        classesSeparated[digit].y.Add(embedding[i, 1]);
                    }
                }

                // Plot each digit class with different color
                for (int digit = 0; digit < 10; digit++)
                {
                    var (xPoints, yPoints) = classesSeparated[digit];
                    if (xPoints.Count > 0)
                    {
                        plt.AddScatter(
                            xPoints.ToArray(),
                            yPoints.ToArray(),
                            color: ClassColors[digit],
                            lineWidth: 0, // No lines
                            markerSize: 5,
                            label: $"Digit {digit}"
                        );
                    }
                }

                // Customize plot
                plt.Title(title);
                plt.XLabel("PacMAP Dimension 1");
                plt.YLabel("PacMAP Dimension 2");
                plt.Legend(location: Alignment.UpperRight);
                plt.Grid(enable: true);

                // Set axis margins
                plt.AxisAuto(horizontalMargin: 0.1, verticalMargin: 0.1);

                // Save plot
                Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
                plt.SaveFig(outputPath, width: 1200, height: 800);

                Console.WriteLine($"✅ MNIST plot saved: {outputPath}");
                Console.WriteLine($"   Total points plotted: {numPoints}");

                // Print class distribution
                for (int digit = 0; digit < 10; digit++)
                {
                    int count = classesSeparated[digit].x.Count;
                    if (count > 0)
                    {
                        Console.WriteLine($"   Digit {digit}: {count} points");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Failed to create MNIST plot: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Plot mammoth PacMAP results
        /// </summary>
        /// <param name="embedding">2D embedding from PacMAP</param>
        /// <param name="title">Plot title</param>
        /// <param name="outputPath">Path to save the plot image</param>
        public static void PlotMammoth(float[,] embedding, string title, string outputPath)
        {
            try
            {
                Console.WriteLine($"Creating mammoth plot: {title}");

                var plt = new Plot(1200, 800);

                int numPoints = embedding.GetLength(0);
                var x = new double[numPoints];
                var y = new double[numPoints];

                for (int i = 0; i < numPoints; i++)
                {
                    x[i] = embedding[i, 0];
                    y[i] = embedding[i, 1];
                }

                // Create scatter plot for mammoth data
                plt.AddScatter(x, y, color: Color.DarkBlue, lineWidth: 0, markerSize: 4);

                // Customize plot
                plt.Title(title);
                plt.XLabel("PacMAP Dimension 1");
                plt.YLabel("PacMAP Dimension 2");
                plt.Grid(enable: true);

                // Set equal aspect ratio to preserve mammoth shape
                plt.SetAxisLimits();
                plt.AxisAuto(horizontalMargin: 0.1, verticalMargin: 0.1);

                // Save plot
                Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
                plt.SaveFig(outputPath, width: 1200, height: 800);

                Console.WriteLine($"✅ Mammoth plot saved: {outputPath}");
                Console.WriteLine($"   Total points plotted: {numPoints}");

                // Print coordinate ranges
                double minX = double.MaxValue, maxX = double.MinValue;
                double minY = double.MaxValue, maxY = double.MinValue;

                for (int i = 0; i < numPoints; i++)
                {
                    minX = Math.Min(minX, x[i]);
                    maxX = Math.Max(maxX, x[i]);
                    minY = Math.Min(minY, y[i]);
                    maxY = Math.Max(maxY, y[i]);
                }

                Console.WriteLine($"   X range: [{minX:F3}, {maxX:F3}]");
                Console.WriteLine($"   Y range: [{minY:F3}, {maxY:F3}]");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Failed to create mammoth plot: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Assign anatomical parts to mammoth points based on 3D coordinates
        /// </summary>
        /// <param name="originalData">Original 3D mammoth data</param>
        /// <returns>Array of part names for each point</returns>
        public static string[] AssignMammothParts(double[,] originalData)
        {
            int numPoints = originalData.GetLength(0);
            var parts = new string[numPoints];

            // Compute dynamic thresholds based on coordinate ranges
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

            // Calculate thresholds for anatomical classification
            double midZ = (minZ + maxZ) / 2;
            double lowZThreshold = minZ + (maxZ - minZ) * 0.3;  // Bottom 30% for legs
            double highZThreshold = minZ + (maxZ - minZ) * 0.7; // Top 30% for head
            double trunkXThreshold = maxX * 0.6;  // Trunk extends to higher X coordinates
            double yMidRange = Math.Abs(maxY - minY) * 0.4;     // Trunk should be centered in Y

            // Assign parts based on coordinate rules
            for (int i = 0; i < numPoints; i++)
            {
                double x = originalData[i, 0];
                double y = originalData[i, 1];
                double z = originalData[i, 2];

                if (z < lowZThreshold)
                {
                    parts[i] = "legs";  // Low height = legs
                }
                else if (z > highZThreshold)
                {
                    parts[i] = "head";  // High height = head
                }
                else if (x > trunkXThreshold && Math.Abs(y - (minY + maxY) / 2) < yMidRange)
                {
                    parts[i] = "trunk"; // Extended X, centered Y = trunk
                }
                else
                {
                    parts[i] = "body";  // Everything else = body
                }
            }

            return parts;
        }

        /// <summary>
        /// Plot mammoth PacMAP results with anatomical part coloring
        /// </summary>
        /// <param name="embedding">2D embedding from PacMAP</param>
        /// <param name="originalData">Original 3D mammoth data for part classification</param>
        /// <param name="title">Plot title</param>
        /// <param name="outputPath">Path to save the plot image</param>
        public static void PlotMammothWithParts(float[,] embedding, double[,] originalData, string title, string outputPath)
        {
            try
            {
                Console.WriteLine($"Creating mammoth plot with anatomical parts: {title}");

                var plt = new Plot(1200, 800);

                int numPoints = embedding.GetLength(0);

                // Assign anatomical parts based on 3D coordinates
                var parts = AssignMammothParts(originalData);

                // Define colors for each part
                var partColors = new Dictionary<string, Color>
                {
                    { "legs", Color.Blue },
                    { "body", Color.Green },
                    { "head", Color.Orange },
                    { "trunk", Color.Red }
                };

                // Group points by anatomical part
                var partGroups = new Dictionary<string, (List<double> x, List<double> y)>();
                foreach (var part in partColors.Keys)
                {
                    partGroups[part] = (new List<double>(), new List<double>());
                }

                // Separate points by part
                for (int i = 0; i < numPoints; i++)
                {
                    string part = parts[i];
                    partGroups[part].x.Add(embedding[i, 0]);
                    partGroups[part].y.Add(embedding[i, 1]);
                }

                // Plot each part with different color
                foreach (var kvp in partGroups)
                {
                    var part = kvp.Key;
                    var (xPoints, yPoints) = kvp.Value;
                    if (xPoints.Count > 0)
                    {
                        plt.AddScatter(
                            xPoints.ToArray(),
                            yPoints.ToArray(),
                            color: partColors[part],
                            lineWidth: 0,
                            markerSize: 4,
                            label: char.ToUpper(part[0]) + part.Substring(1)
                        );
                    }
                }

                // Customize plot
                plt.Title(title);
                plt.XLabel("PacMAP Dimension 1");
                plt.YLabel("PacMAP Dimension 2");
                plt.Legend(location: Alignment.UpperRight);
                plt.Grid(enable: true);

                // Set equal aspect ratio to preserve mammoth shape
                plt.SetAxisLimits();
                plt.AxisAuto(horizontalMargin: 0.1, verticalMargin: 0.1);

                // Save plot
                Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
                plt.SaveFig(outputPath, width: 1200, height: 800);

                Console.WriteLine($"✅ Mammoth plot with parts saved: {outputPath}");
                Console.WriteLine($"   Total points plotted: {numPoints}");

                // Print part distribution
                var partCounts = partGroups.Select(kvp => $"{kvp.Key}: {kvp.Value.x.Count}").ToArray();
                Console.WriteLine($"   Part distribution: {string.Join(", ", partCounts)}");

                // Print coordinate ranges
                double minX = double.MaxValue, maxX = double.MinValue;
                double minY = double.MaxValue, maxY = double.MinValue;

                for (int i = 0; i < numPoints; i++)
                {
                    minX = Math.Min(minX, embedding[i, 0]);
                    maxX = Math.Max(maxX, embedding[i, 0]);
                    minY = Math.Min(minY, embedding[i, 1]);
                    maxY = Math.Max(maxY, embedding[i, 1]);
                }

                Console.WriteLine($"   X range: [{minX:F3}, {maxX:F3}]");
                Console.WriteLine($"   Y range: [{minY:F3}, {maxY:F3}]");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Failed to create mammoth plot with parts: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Plot original 3D mammoth data with anatomical part coloring - three 2D projections
        /// </summary>
        /// <param name="originalData">Original 3D mammoth data</param>
        /// <param name="title">Plot title</param>
        /// <param name="outputPath">Path to save the plot image</param>
        public static void PlotOriginalMammoth3DWithParts(double[,] originalData, string title, string outputPath)
        {
            try
            {
                Console.WriteLine($"Creating original 3D mammoth plot with parts: {title}");

                int numPoints = originalData.GetLength(0);

                // Assign anatomical parts
                var parts = AssignMammothParts(originalData);

                // Define colors for each part
                var partColors = new Dictionary<string, Color>
                {
                    { "legs", Color.Blue },
                    { "body", Color.Green },
                    { "head", Color.Orange },
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
                    partGroups[part].x.Add(x[i]);
                    partGroups[part].y.Add(y[i]);
                    partGroups[part].z.Add(z[i]);
                }

                // Create one image with three side-by-side projections
                var plt = new Plot(2400, 800); // Wide format for 3 plots

                // XY projection (left)
                var xyXOffset = 0;
                var xyYOffset = 0;
                foreach (var kvp in partGroups)
                {
                    var part = kvp.Key;
                    var (xPoints, yPoints, zPoints) = kvp.Value;
                    if (xPoints.Count > 0)
                    {
                        var adjustedX = xPoints.Select(x => x / 3 + xyXOffset).ToArray(); // Scale and position
                        var adjustedY = yPoints.Select(y => y / 3 + xyYOffset).ToArray();
                        plt.AddScatter(adjustedX, adjustedY,
                            color: partColors[part], lineWidth: 0, markerSize: 1,
                            label: part == "head" ? char.ToUpper(part[0]) + part.Substring(1) : null); // Only label once
                    }
                }

                // XZ projection (middle)
                var xzXOffset = 800;
                var xzYOffset = 0;
                foreach (var kvp in partGroups)
                {
                    var part = kvp.Key;
                    var (xPoints, yPoints, zPoints) = kvp.Value;
                    if (xPoints.Count > 0)
                    {
                        var adjustedX = xPoints.Select(x => x / 3 + xzXOffset).ToArray();
                        var adjustedZ = zPoints.Select(z => z / 3 + xzYOffset).ToArray();
                        plt.AddScatter(adjustedX, adjustedZ,
                            color: partColors[part], lineWidth: 0, markerSize: 1);
                    }
                }

                // YZ projection (right)
                var yzXOffset = 1600;
                var yzYOffset = 0;
                foreach (var kvp in partGroups)
                {
                    var part = kvp.Key;
                    var (xPoints, yPoints, zPoints) = kvp.Value;
                    if (xPoints.Count > 0)
                    {
                        var adjustedY = yPoints.Select(y => y / 3 + yzXOffset).ToArray();
                        var adjustedZ = zPoints.Select(z => z / 3 + yzYOffset).ToArray();
                        plt.AddScatter(adjustedY, adjustedZ,
                            color: partColors[part], lineWidth: 0, markerSize: 1);
                    }
                }

                // Customize plot
                plt.Title(title);
                plt.XLabel("XY View | XZ View | YZ View");
                plt.YLabel("Mammoth 3D Projections");
                plt.Legend(location: Alignment.UpperRight);
                plt.Grid(enable: false);

                // Save plot
                Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
                plt.SaveFig(outputPath, width: 2400, height: 800);

                Console.WriteLine($"✅ Original 3D mammoth plot with parts saved: {outputPath}");
                Console.WriteLine($"   Total points plotted: {numPoints} (3 views in one image)");

                // Print part distribution
                var partCounts = partGroups.Select(kvp => $"{kvp.Key}: {kvp.Value.x.Count}").ToArray();
                Console.WriteLine($"   Part distribution: {string.Join(", ", partCounts)}");

                // Print coordinate ranges
                double minX = x.Min(), maxX = x.Max();
                double minY = y.Min(), maxY = y.Max();
                double minZ = z.Min(), maxZ = z.Max();

                Console.WriteLine($"   X range: [{minX:F3}, {maxX:F3}]");
                Console.WriteLine($"   Y range: [{minY:F3}, {maxY:F3}]");
                Console.WriteLine($"   Z range: [{minZ:F3}, {maxZ:F3}]");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Failed to create original 3D mammoth plot with parts: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Plot original 3D mammoth data with multiple 2D projections (X-Y, X-Z, Y-Z)
        /// </summary>
        /// <param name="originalData">Original 3D mammoth data</param>
        /// <param name="title">Plot title</param>
        /// <param name="outputPath">Path to save the plot image</param>
        public static void PlotOriginalMammoth3D(double[,] originalData, string title, string outputPath)
        {
            try
            {
                Console.WriteLine($"Creating original 3D mammoth plot: {title}");

                var plt = new Plot(2400, 800); // Wider for 3 side-by-side projections

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

                // Create three separate 2D projections arranged horizontally
                // Calculate proper offsets to separate the three views clearly

                // Find coordinate ranges for proper scaling
                var xRange = x.Max() - x.Min();
                var yRange = y.Max() - y.Min();
                var zRange = z.Max() - z.Min();
                var maxRange = Math.Max(Math.Max(xRange, yRange), zRange);

                // Create horizontal offsets (place views side by side)
                var spacing = maxRange * 1.5; // Space between projections

                // Projection 1: X-Y view (left) - original position
                plt.AddScatter(x, y, color: Color.DarkGreen, lineWidth: 0, markerSize: 2, label: "X-Y View");

                // Projection 2: X-Z view (center) - offset right
                var xOffset1 = x.Select(xi => xi + spacing).ToArray();
                plt.AddScatter(xOffset1, z, color: Color.DarkBlue, lineWidth: 0, markerSize: 2, label: "X-Z View");

                // Projection 3: Y-Z view (right) - offset further right
                var yOffset = y.Select(yi => yi + spacing * 2).ToArray();
                plt.AddScatter(yOffset, z, color: Color.DarkRed, lineWidth: 0, markerSize: 2, label: "Y-Z View");

                // Customize plot
                plt.Title($"{title} - Three 2D Projections");
                plt.XLabel("← X-Y View  |  X-Z View  |  Y-Z View →");
                plt.YLabel("Y / Z Coordinates");
                plt.Legend();
                plt.Grid(enable: true);

                plt.AxisAuto(horizontalMargin: 0.05, verticalMargin: 0.05);

                // Save plot
                Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
                plt.SaveFig(outputPath, width: 2400, height: 800);

                Console.WriteLine($"✅ Original 3D mammoth plot saved: {outputPath}");
                Console.WriteLine($"   Total points plotted: {numPoints} (3 projections)");

                // Print 3D coordinate ranges
                double minX = x.Min(), maxX = x.Max();
                double minY = y.Min(), maxY = y.Max();
                double minZ = z.Min(), maxZ = z.Max();

                Console.WriteLine($"   X range: [{minX:F3}, {maxX:F3}]");
                Console.WriteLine($"   Y range: [{minY:F3}, {maxY:F3}]");
                Console.WriteLine($"   Z range: [{minZ:F3}, {maxZ:F3}]");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Failed to create original 3D mammoth plot: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Create separate 3D projection plots (X-Y, X-Z, Y-Z views)
        /// </summary>
        /// <param name="originalData">Original 3D mammoth data</param>
        /// <param name="outputDir">Directory to save projection plots</param>
        public static void CreateMammoth3DProjections(double[,] originalData, string outputDir)
        {
            try
            {
                Console.WriteLine("Creating separate 3D projection plots...");

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

                // X-Y projection
                var xyPlot = new Plot(800, 600);
                xyPlot.AddScatter(x, y, color: Color.DarkGreen, lineWidth: 0, markerSize: 2);
                xyPlot.Title("Mammoth 3D Data - X-Y Projection");
                xyPlot.XLabel("X Coordinate");
                xyPlot.YLabel("Y Coordinate");
                xyPlot.Grid(enable: true);
                xyPlot.AxisAuto(horizontalMargin: 0.1, verticalMargin: 0.1);

                string xyPath = Path.Combine(outputDir, "mammoth_3d_projection_xy.png");
                xyPlot.SaveFig(xyPath, width: 800, height: 600);
                Console.WriteLine($"✅ X-Y projection saved: {xyPath}");

                // X-Z projection
                var xzPlot = new Plot(800, 600);
                xzPlot.AddScatter(x, z, color: Color.DarkBlue, lineWidth: 0, markerSize: 2);
                xzPlot.Title("Mammoth 3D Data - X-Z Projection");
                xzPlot.XLabel("X Coordinate");
                xzPlot.YLabel("Z Coordinate");
                xzPlot.Grid(enable: true);
                xzPlot.AxisAuto(horizontalMargin: 0.1, verticalMargin: 0.1);

                string xzPath = Path.Combine(outputDir, "mammoth_3d_projection_xz.png");
                xzPlot.SaveFig(xzPath, width: 800, height: 600);
                Console.WriteLine($"✅ X-Z projection saved: {xzPath}");

                // Y-Z projection
                var yzPlot = new Plot(800, 600);
                yzPlot.AddScatter(y, z, color: Color.DarkRed, lineWidth: 0, markerSize: 2);
                yzPlot.Title("Mammoth 3D Data - Y-Z Projection");
                yzPlot.XLabel("Y Coordinate");
                yzPlot.YLabel("Z Coordinate");
                yzPlot.Grid(enable: true);
                yzPlot.AxisAuto(horizontalMargin: 0.1, verticalMargin: 0.1);

                string yzPath = Path.Combine(outputDir, "mammoth_3d_projection_yz.png");
                yzPlot.SaveFig(yzPath, width: 800, height: 600);
                Console.WriteLine($"✅ Y-Z projection saved: {yzPath}");

                Console.WriteLine($"✅ All 3D projections completed: 3 individual projection plots");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Failed to create 3D projections: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Create a comparison plot showing before/after embeddings
        /// </summary>
        /// <param name="originalEmbedding">Original/baseline embedding</param>
        /// <param name="pacmapEmbedding">PacMAP embedding</param>
        /// <param name="labels">Data labels (optional)</param>
        /// <param name="title">Plot title</param>
        /// <param name="outputPath">Path to save the plot image</param>
        public static void PlotComparison(double[,] originalEmbedding, double[,] pacmapEmbedding,
            int[]? labels, string title, string outputPath)
        {
            try
            {
                Console.WriteLine($"Creating comparison plot: {title}");

                var plt = new Plot(1600, 600);

                // Create single plot for comparison - simplified for ScottPlot 4.x
                // TODO: Implement proper subplot functionality

                int numPoints = originalEmbedding.GetLength(0);

                // Left plot: Original
                var origX = new double[numPoints];
                var origY = new double[numPoints];
                for (int i = 0; i < numPoints; i++)
                {
                    origX[i] = originalEmbedding[i, 0];
                    origY[i] = originalEmbedding[i, 1];
                }

                // Original data in blue
                plt.AddScatter(origX, origY, color: Color.Blue, lineWidth: 0, markerSize: 3, label: "Original");

                // PacMAP data in red (offset for visibility)
                var pacX = new double[numPoints];
                var pacY = new double[numPoints];
                for (int i = 0; i < numPoints; i++)
                {
                    pacX[i] = pacmapEmbedding[i, 0] + 10; // Offset for comparison
                    pacY[i] = pacmapEmbedding[i, 1];
                }

                plt.AddScatter(pacX, pacY, color: Color.Red, lineWidth: 0, markerSize: 3, label: "PacMAP");

                plt.Title(title);
                plt.Legend();

                // Save plot
                Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
                plt.SaveFig(outputPath, width: 1600, height: 600);

                Console.WriteLine($"✅ Comparison plot saved: {outputPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Failed to create comparison plot: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Save embedding data as CSV for external analysis
        /// </summary>
        /// <param name="embedding">2D embedding</param>
        /// <param name="labels">Optional labels</param>
        /// <param name="outputPath">CSV output path</param>
        public static void SaveEmbeddingAsCSV(float[,] embedding, int[]? labels, string outputPath)
        {
            try
            {
                Console.WriteLine($"Saving embedding as CSV: {outputPath}");

                using var writer = new StreamWriter(outputPath);

                // Header
                if (labels != null)
                {
                    writer.WriteLine("x,y,label");
                }
                else
                {
                    writer.WriteLine("x,y");
                }

                // Data
                int numPoints = embedding.GetLength(0);
                for (int i = 0; i < numPoints; i++)
                {
                    if (labels != null)
                    {
                        writer.WriteLine($"{embedding[i, 0]:F6},{embedding[i, 1]:F6},{labels[i]}");
                    }
                    else
                    {
                        writer.WriteLine($"{embedding[i, 0]:F6},{embedding[i, 1]:F6}");
                    }
                }

                Console.WriteLine($"✅ Embedding saved as CSV: {outputPath} ({numPoints} points)");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Failed to save embedding as CSV: {ex.Message}");
                throw;
            }
        }
    }
}