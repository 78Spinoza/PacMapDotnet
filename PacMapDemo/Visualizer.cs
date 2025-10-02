using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.Axes;
using OxyPlot.SkiaSharp;

namespace PacMapDemo
{
    public static class Visualizer
    {
        /// <summary>
        /// Assign anatomical parts to mammoth points based on 3D coordinates
        /// </summary>
        public static string[] AssignMammothParts(double[,] originalData)
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

            // Initialize all points as "body" first
            for (int i = 0; i < numPoints; i++)
            {
                parts[i] = "body";
            }

            // 1. FEET: Bottom 8% (very bottom)
            double feetZThreshold = minZ + zRange * 0.08;
            for (int i = 0; i < numPoints; i++)
            {
                double z = originalData[i, 2];
                if (z < feetZThreshold)
                    parts[i] = "feet";
            }

            // 2. LEGS: Bottom 28% but excluding feet (8% to 28% Z range)
            double legZThreshold = minZ + zRange * 0.28;
            for (int i = 0; i < numPoints; i++)
            {
                double z = originalData[i, 2];
                if (z >= feetZThreshold && z < legZThreshold)
                    parts[i] = "legs";
            }

            // 3. HEAD: Middle 28% X + First 41.5% Y + Top 50% Z (optimized from testing)
            double firstXThreshold = minX + xRange * 0.36;  // First 36% of X
            double lastXThreshold = maxX - xRange * 0.36;   // Last 36% of X (middle 28%)
            double yThreshold = minY + yRange * 0.415; // First 41.5% of Y direction
            double zThreshold = minZ + zRange * 0.5; // Top 50% of Z (upper half)

            for (int i = 0; i < numPoints; i++)
            {
                double x = originalData[i, 0];
                double y = originalData[i, 1];
                double z = originalData[i, 2];

                if (x >= firstXThreshold && x <= lastXThreshold && y < yThreshold && z > zThreshold)
                    parts[i] = "head";
            }

            // 4. TRUNK: Inverse of X (First 36% + Last 36%) + First 41.5% Y (extremities)
            for (int i = 0; i < numPoints; i++)
            {
                if (parts[i] == "body") // Don't override feet, legs, or head
                {
                    double x = originalData[i, 0];
                    double y = originalData[i, 1];

                    if ((x < firstXThreshold || x > lastXThreshold) && y < yThreshold)
                        parts[i] = "trunk";
                }
            }

            // 5. TUSKS: Very forward + very high + very narrow (refine from trunk)
            double tuskXThreshold = minX + xRange * 0.9; // Ultra forward
            double tuskZThreshold = minZ + zRange * 0.85; // Very high
            double tuskYWidth = yRange * 0.1; // Ultra narrow
            double yCenter = (minY + maxY) / 2;

            for (int i = 0; i < numPoints; i++)
            {
                if (parts[i] == "trunk") // Can refine trunk to tusks
                {
                    double x = originalData[i, 0];
                    double y = originalData[i, 1];
                    double z = originalData[i, 2];
                    double yDist = Math.Abs(y - yCenter);

                    if (x > tuskXThreshold && z > tuskZThreshold && yDist < tuskYWidth)
                        parts[i] = "tusks";
                }
            }

            return parts;
        }

        /// <summary>
        /// Create real 3D visualization using OxyPlot with anatomical part coloring
        /// </summary>
        public static void PlotOriginalMammoth3DReal(double[,] originalData, string title, string outputPath)
        {
            try
            {
                Console.WriteLine($"Creating real 3D mammoth plot: {title}");

                int numPoints = originalData.GetLength(0);
                var parts = AssignMammothParts(originalData);

                var partColors = new Dictionary<string, OxyColor>
                {
                    { "feet", OxyColors.Orange },
                    { "legs", OxyColors.Blue },
                    { "body", OxyColors.Green },
                    { "head", OxyColors.Purple },
                    { "tusks", OxyColors.Yellow },
                    { "trunk", OxyColors.Red }
                };

                // Create a composite plot with 3 views (XY, XZ, YZ)
                var plotModel = new PlotModel
                {
                    Title = title,
                    Background = OxyColors.White,
                    PlotAreaBorderColor = OxyColors.Black
                };

                // Configure for side-by-side layout - we'll create separate plots
                // This main plot will show XY view (top view)
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "X Coordinate (Left-Right)" });
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Y Coordinate (Front-Back)" });

                // Group points by anatomical part
                var partGroups = new Dictionary<string, (List<double> x, List<double> y, List<double> z)>();
                foreach (var part in partColors.Keys)
                {
                    partGroups[part] = (new List<double>(), new List<double>(), new List<double>());
                }

                for (int i = 0; i < numPoints; i++)
                {
                    string part = parts[i];
                    partGroups[part].x.Add(originalData[i, 0]);
                    partGroups[part].y.Add(originalData[i, 1]);
                    partGroups[part].z.Add(originalData[i, 2]);
                }

                // Calculate Z bounds for depth mapping
                double minZ = partGroups.Values.SelectMany(g => g.z).Min();
                double maxZ = partGroups.Values.SelectMany(g => g.z).Max();
                double zRange = maxZ - minZ;

                // Add scatter series for each part
                foreach (var kvp in partGroups)
                {
                    var part = kvp.Key;
                    var (xPoints, yPoints, zPoints) = kvp.Value;

                    if (xPoints.Count > 0)
                    {
                        var scatterSeries = new ScatterSeries
                        {
                            Title = $"{char.ToUpper(part[0]) + part.Substring(1)} ({xPoints.Count} points)",
                            MarkerType = MarkerType.Circle,
                            MarkerFill = partColors[part],
                            MarkerStroke = partColors[part]
                        };

                        // Add XY view points (top view)
                        for (int i = 0; i < xPoints.Count; i++)
                        {
                            scatterSeries.Points.Add(new ScatterPoint(xPoints[i], yPoints[i], 3));
                        }

                        plotModel.Series.Add(scatterSeries);
                    }
                }

                // Display statistics
                var partCounts = partGroups.ToDictionary(kvp => kvp.Key, kvp => kvp.Value.x.Count);
                Console.WriteLine("Anatomical part distribution:");
                foreach (var part in partCounts.OrderByDescending(p => p.Value))
                {
                    double percentage = (part.Value * 100.0) / numPoints;
                    Console.WriteLine($"   {char.ToUpper(part.Key[0]) + part.Key.Substring(1)}: {part.Value} points ({percentage:F1}%)");
                }

                // Export XY view (top view)
                var exporter = new OxyPlot.SkiaSharp.PngExporter { Width = 800, Height = 600 };
                string xyPath = outputPath.Replace(".png", "_XY_TopView.png");
                using (var stream = File.Create(xyPath))
                {
                    exporter.Export(plotModel, stream);
                }

                // Create XZ view (side view)
                var xzPlotModel = CreateViewPlot(partGroups, partColors, "XZ View (Side)", "X Coordinate (Left-Right)", "Z Coordinate (Height)", "xz");
                string xzPath = outputPath.Replace(".png", "_XZ_SideView.png");
                using (var stream = File.Create(xzPath))
                {
                    exporter.Export(xzPlotModel, stream);
                }

                // Create YZ view (front view)
                var yzPlotModel = CreateViewPlot(partGroups, partColors, "YZ View (Front)", "Y Coordinate (Front-Back)", "Z Coordinate (Height)", "yz");
                string yzPath = outputPath.Replace(".png", "_YZ_FrontView.png");
                using (var stream = File.Create(yzPath))
                {
                    exporter.Export(yzPlotModel, stream);
                }

                Console.WriteLine($"SUCCESS: Multiple view mammoth plots saved:");
                Console.WriteLine($"  XY (Top): {xyPath}");
                Console.WriteLine($"  XZ (Side): {xzPath}");
                Console.WriteLine($"  YZ (Front): {yzPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: Failed to create 3D plot: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Create PacMAP 2D embedding visualization with anatomical part coloring and parameter display
        /// </summary>
        public static void PlotMammothPacMAP(float[,] embedding, double[,] originalData, string title, string outputPath,
            string? parameterInfo = null)
        {
            try
            {
                Console.WriteLine($"Creating PacMAP embedding plot: {title}");

                int numPoints = originalData.GetLength(0);
                var parts = AssignMammothParts(originalData);

                var partColors = new Dictionary<string, OxyColor>
                {
                    { "feet", OxyColors.Orange },
                    { "legs", OxyColors.Blue },
                    { "body", OxyColors.Green },
                    { "head", OxyColors.Purple },
                    { "tusks", OxyColors.Yellow },
                    { "trunk", OxyColors.Red }
                };

                // Enhanced title with parameter information
                string enhancedTitle = title;
                if (!string.IsNullOrEmpty(parameterInfo))
                {
                    enhancedTitle = $"{title}\n{parameterInfo}";
                }

                var plotModel = new PlotModel
                {
                    Title = enhancedTitle,
                    Background = OxyColors.White,
                    PlotAreaBorderColor = OxyColors.Black
                };

                // Add X-Y axes for 2D embedding
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "X Coordinate (PacMAP Dimension 1)" });
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Y Coordinate (PacMAP Dimension 2)" });

                // Group points by anatomical part
                var partGroups = new Dictionary<string, (List<double> x, List<double> y)>();
                foreach (var part in partColors.Keys)
                {
                    partGroups[part] = (new List<double>(), new List<double>());
                }

                for (int i = 0; i < numPoints; i++)
                {
                    string part = parts[i];
                    partGroups[part].x.Add(embedding[i, 0]);
                    partGroups[part].y.Add(embedding[i, 1]);
                }

                // Add scatter series for each part
                foreach (var kvp in partGroups)
                {
                    var part = kvp.Key;
                    var (xPoints, yPoints) = kvp.Value;

                    if (xPoints.Count > 0)
                    {
                        var scatterSeries = new ScatterSeries
                        {
                            Title = $"{char.ToUpper(part[0]) + part.Substring(1)} ({xPoints.Count} points)",
                            MarkerType = MarkerType.Circle,
                            MarkerFill = partColors[part],
                            MarkerStroke = partColors[part],
                            MarkerSize = 3
                        };

                        for (int i = 0; i < xPoints.Count; i++)
                        {
                            scatterSeries.Points.Add(new ScatterPoint(xPoints[i], yPoints[i]));
                        }

                        plotModel.Series.Add(scatterSeries);
                    }
                }

                // Export
                var exporter = new OxyPlot.SkiaSharp.PngExporter { Width = 1600, Height = 1200 };
                using (var stream = File.Create(outputPath))
                {
                    exporter.Export(plotModel, stream);
                }

                Console.WriteLine($"SUCCESS: PacMAP plot saved to: {outputPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: Failed to create PacMAP plot: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Plot PacMAP embedding without labels (single color for unlabeled data)
        /// </summary>
        public static void PlotSimplePacMAP(float[,] embedding, string title, string outputPath, string? parameterInfo = null)
        {
            try
            {
                Console.WriteLine($"Creating simple PacMAP plot: {title}");

                int numPoints = embedding.GetLength(0);

                // Enhanced title with parameter information
                string enhancedTitle = title;
                if (!string.IsNullOrEmpty(parameterInfo))
                {
                    enhancedTitle = $"{title}\n{parameterInfo}";
                }

                var plotModel = new PlotModel
                {
                    Title = enhancedTitle,
                    Background = OxyColors.White,
                    PlotAreaBorderColor = OxyColors.Black
                };

                // Add X-Y axes for 2D embedding
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "X Coordinate (PacMAP Dimension 1)" });
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Y Coordinate (PacMAP Dimension 2)" });

                // Create single scatter series with all points - BLACK color
                var scatterSeries = new ScatterSeries
                {
                    Title = $"{numPoints:N0} points",
                    MarkerType = MarkerType.Circle,
                    MarkerFill = OxyColors.Black,
                    MarkerStroke = OxyColors.Black,
                    MarkerSize = 2
                };

                for (int i = 0; i < numPoints; i++)
                {
                    scatterSeries.Points.Add(new ScatterPoint(embedding[i, 0], embedding[i, 1]));
                }

                plotModel.Series.Add(scatterSeries);

                // Export
                var exporter = new OxyPlot.SkiaSharp.PngExporter { Width = 1600, Height = 1200 };
                using (var stream = File.Create(outputPath))
                {
                    exporter.Export(plotModel, stream);
                }

                Console.WriteLine($"SUCCESS: Simple PacMAP plot saved to: {outputPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: Failed to create simple PacMAP plot: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Save embedding data as CSV
        /// </summary>
        public static void SaveEmbeddingAsCSV(float[,] embedding, string[]? labels, string outputPath)
        {
            try
            {
                int numPoints = embedding.GetLength(0);

                using (var writer = new StreamWriter(outputPath))
                {
                    // Write header
                    if (labels != null)
                        writer.WriteLine("x,y,label");
                    else
                        writer.WriteLine("x,y");

                    // Write data
                    for (int i = 0; i < numPoints; i++)
                    {
                        if (labels != null)
                            writer.WriteLine($"{embedding[i, 0]},{embedding[i, 1]},{labels[i]}");
                        else
                            writer.WriteLine($"{embedding[i, 0]},{embedding[i, 1]}");
                    }
                }

                Console.WriteLine($"SUCCESS: Embedding saved as CSV: {outputPath} ({numPoints} points)");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: Failed to save embedding as CSV: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Create a specific view plot (XY, XZ, or YZ)
        /// </summary>
        private static PlotModel CreateViewPlot(
            Dictionary<string, (List<double> x, List<double> y, List<double> z)> partGroups,
            Dictionary<string, OxyColor> partColors,
            string title,
            string xAxisTitle,
            string yAxisTitle,
            string viewType)
        {
            var plotModel = new PlotModel
            {
                Title = title,
                Background = OxyColors.White,
                PlotAreaBorderColor = OxyColors.Black
            };

            plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = xAxisTitle });
            plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = yAxisTitle });

            foreach (var kvp in partGroups)
            {
                var part = kvp.Key;
                var (xPoints, yPoints, zPoints) = kvp.Value;

                if (xPoints.Count > 0)
                {
                    var scatterSeries = new ScatterSeries
                    {
                        Title = $"{char.ToUpper(part[0]) + part.Substring(1)} ({xPoints.Count} points)",
                        MarkerType = MarkerType.Circle,
                        MarkerFill = partColors[part],
                        MarkerStroke = partColors[part],
                        MarkerSize = 3
                    };

                    for (int i = 0; i < xPoints.Count; i++)
                    {
                        double x, y;
                        switch (viewType.ToLower())
                        {
                            case "xy":
                                x = xPoints[i];
                                y = yPoints[i];
                                break;
                            case "xz":
                                x = xPoints[i];
                                y = zPoints[i];
                                break;
                            case "yz":
                                x = yPoints[i];
                                y = zPoints[i];
                                break;
                            default:
                                x = xPoints[i];
                                y = yPoints[i];
                                break;
                        }

                        scatterSeries.Points.Add(new ScatterPoint(x, y));
                    }

                    plotModel.Series.Add(scatterSeries);
                }
            }

            return plotModel;
        }
    }
}