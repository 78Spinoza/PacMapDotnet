using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;

namespace PacMapDemo
{
    /// <summary>
    /// MNIST Binary Data Reader
    /// Reads compact binary MNIST format created by mnist_converter.py
    /// </summary>
    public class MnistReader
    {
        /// <summary>
        /// MNIST Binary File Header Structure (32 bytes)
        /// </summary>
        public struct MnistHeader
        {
            public string Magic;        // 4 bytes: "MNIST"
            public int Version;         // 4 bytes: version number
            public int NumImages;       // 4 bytes: number of images
            public int ImageHeight;     // 4 bytes: image height (28)
            public int ImageWidth;      // 4 bytes: image width (28)
            public int NumLabels;       // 4 bytes: number of labels
            public long Reserved;       // 8 bytes: reserved for future use

            public override string ToString()
            {
                return $"MNIST v{Version}: {NumImages:N0} images ({ImageHeight}x{ImageWidth}), {NumLabels:N0} labels";
            }
        }

        /// <summary>
        /// Complete MNIST Dataset
        /// </summary>
        public class MnistData
        {
            public MnistHeader Header { get; set; } = default;
            public byte[,,]? Images { get; set; }    // [num_images, 28, 28]
            public byte[]? Labels { get; set; }      // [num_images]

            public int NumImages => Header.NumImages;
            public int NumLabels => Header.NumLabels;
            public int ImageSize => Header.ImageHeight * Header.ImageWidth;

            /// <summary>
            /// Get a single image as 2D array
            /// </summary>
            public byte[,] GetImage(int index)
            {
                if (Images == null)
                    throw new InvalidOperationException("Images data not loaded");
                if (index < 0 || index >= NumImages)
                    throw new ArgumentOutOfRangeException(nameof(index));

                var image = new byte[Header.ImageHeight, Header.ImageWidth];
                for (int h = 0; h < Header.ImageHeight; h++)
                {
                    for (int w = 0; w < Header.ImageWidth; w++)
                    {
                        image[h, w] = Images[index, h, w];
                    }
                }
                return image;
            }

            /// <summary>
            /// Get a single image as flattened 1D array
            /// </summary>
            public byte[] GetImageFlattened(int index)
            {
                if (Images == null)
                    throw new InvalidOperationException("Images data not loaded");
                if (index < 0 || index >= NumImages)
                    throw new ArgumentOutOfRangeException(nameof(index));

                var flattened = new byte[ImageSize];
                Buffer.BlockCopy(Images, index * ImageSize, flattened, 0, ImageSize);
                return flattened;
            }

            /// <summary>
            /// Convert all images to float array for PACMAP
            /// </summary>
            public float[,] ToFloatArray()
            {
                if (Images == null)
                    throw new InvalidOperationException("Images data not loaded");

                var result = new float[NumImages, ImageSize];
                for (int i = 0; i < NumImages; i++)
                {
                    for (int j = 0; j < ImageSize; j++)
                    {
                        result[i, j] = Images[i, j / Header.ImageWidth, j % Header.ImageWidth] / 255.0f;
                    }
                }
                return result;
            }

            /// <summary>
            /// Convert subset of images to float array for PACMAP
            /// </summary>
            public float[,] ToFloatArray(int startIndex, int count)
            {
                if (Images == null)
                    throw new InvalidOperationException("Images data not loaded");
                if (startIndex < 0 || count <= 0 || startIndex + count > NumImages)
                    throw new ArgumentException("Invalid range");

                var result = new float[count, ImageSize];
                for (int i = 0; i < count; i++)
                {
                    for (int j = 0; j < ImageSize; j++)
                    {
                        int imgIndex = startIndex + i;
                        result[i, j] = Images[imgIndex, j / Header.ImageWidth, j % Header.ImageWidth] / 255.0f;
                    }
                }
                return result;
            }
        }

        /// <summary>
        /// Read MNIST binary file (supports both .dat and .dat.zip formats)
        /// </summary>
        /// <param name="filePath">Path to the binary file</param>
        /// <returns>MNIST dataset</returns>
        public static MnistData Read(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"MNIST binary file not found: {filePath}");

            string extension = Path.GetExtension(filePath).ToLowerInvariant();

            if (extension == ".zip")
            {
                return ReadFromZip(filePath);
            }
            else
            {
                return ReadFromBinary(filePath);
            }
        }

        /// <summary>
        /// Read from ZIP compressed file
        /// </summary>
        private static MnistData ReadFromZip(string zipFilePath)
        {
            Console.WriteLine($"Reading compressed MNIST data from: {zipFilePath}");

            using var zipStream = new FileStream(zipFilePath, FileMode.Open, FileAccess.Read);
            using var zip = new ZipArchive(zipStream, ZipArchiveMode.Read);

            // Look for the binary data file inside the ZIP
            var entry = zip.Entries.FirstOrDefault(e => e.Name.EndsWith(".dat"));
            if (entry == null)
                throw new InvalidDataException("No .dat file found in the ZIP archive");

            using var entryStream = entry.Open();
            using var reader = new BinaryReader(entryStream);

            // Read header (32 bytes)
            var header = ReadHeader(reader);

            // Validate header
            ValidateHeader(header);

            // Read image data
            var images = ReadImageData(reader, header);

            // Read label data
            var labels = ReadLabelData(reader, header);

            return new MnistData
            {
                Header = header,
                Images = images,
                Labels = labels
            };
        }

        /// <summary>
        /// Read from uncompressed binary file
        /// </summary>
        private static MnistData ReadFromBinary(string filePath)
        {
            Console.WriteLine($"Reading uncompressed MNIST data from: {filePath}");

            using var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
            using var reader = new BinaryReader(stream);

            // Read header (32 bytes)
            var header = ReadHeader(reader);

            // Validate header
            ValidateHeader(header);

            // Read image data
            var images = ReadImageData(reader, header);

            // Read label data
            var labels = ReadLabelData(reader, header);

            return new MnistData
            {
                Header = header,
                Images = images,
                Labels = labels
            };
        }

        /// <summary>
        /// Read header from binary stream
        /// </summary>
        private static MnistHeader ReadHeader(BinaryReader reader)
        {
            // Read magic number (4 bytes)
            var magicBytes = reader.ReadBytes(4);
            var magic = Encoding.ASCII.GetString(magicBytes);

            // Read remaining header fields (little endian)
            var version = reader.ReadInt32();
            var numImages = reader.ReadInt32();
            var imageHeight = reader.ReadInt32();
            var imageWidth = reader.ReadInt32();
            var numLabels = reader.ReadInt32();
            var reserved = reader.ReadInt64();

            return new MnistHeader
            {
                Magic = magic,
                Version = version,
                NumImages = numImages,
                ImageHeight = imageHeight,
                ImageWidth = imageWidth,
                NumLabels = numLabels,
                Reserved = reserved
            };
        }

        /// <summary>
        /// Validate header integrity
        /// </summary>
        private static void ValidateHeader(MnistHeader header)
        {
            if (header.Magic != "MNIST")
                throw new InvalidDataException($"Invalid magic number: {header.Magic}, expected 'MNIST'");

            if (header.Version != 1)
                throw new InvalidDataException($"Unsupported version: {header.Version}, expected 1");

            if (header.NumImages <= 0)
                throw new InvalidDataException($"Invalid number of images: {header.NumImages}");

            if (header.ImageHeight != 28 || header.ImageWidth != 28)
                throw new InvalidDataException($"Invalid image dimensions: {header.ImageHeight}x{header.ImageWidth}, expected 28x28");

            if (header.NumLabels != header.NumImages)
                throw new InvalidDataException($"Label count mismatch: {header.NumLabels} labels vs {header.NumImages} images");
        }

        /// <summary>
        /// Read image data from binary stream
        /// </summary>
        private static byte[,,] ReadImageData(BinaryReader reader, MnistHeader header)
        {
            var images = new byte[header.NumImages, header.ImageHeight, header.ImageWidth];
            var imageSize = header.ImageHeight * header.ImageWidth;

            for (int i = 0; i < header.NumImages; i++)
            {
                // Read flattened image data
                var flattenedData = reader.ReadBytes(imageSize);

                if (flattenedData.Length != imageSize)
                    throw new EndOfStreamException($"Unexpected end of file while reading image {i}");

                // Reshape to 2D and store in 3D array
                for (int h = 0; h < header.ImageHeight; h++)
                {
                    for (int w = 0; w < header.ImageWidth; w++)
                    {
                        int flatIndex = h * header.ImageWidth + w;
                        images[i, h, w] = flattenedData[flatIndex];
                    }
                }
            }

            return images;
        }

        /// <summary>
        /// Read label data from binary stream
        /// </summary>
        private static byte[] ReadLabelData(BinaryReader reader, MnistHeader header)
        {
            var labels = reader.ReadBytes(header.NumLabels);

            if (labels.Length != header.NumLabels)
                throw new EndOfStreamException($"Unexpected end of file while reading labels");

            // Validate label values
            for (int i = 0; i < labels.Length; i++)
            {
                if (labels[i] > 9)
                    throw new InvalidDataException($"Invalid label value {labels[i]} at index {i}, expected 0-9");
            }

            return labels;
        }

        /// <summary>
        /// Print dataset information
        /// </summary>
        public static void PrintInfo(MnistData data)
        {
            Console.WriteLine("📊 MNIST Dataset Information:");
            Console.WriteLine($"   {data.Header}");
            Console.WriteLine($"   Image size: {data.ImageSize} pixels");
            Console.WriteLine($"   Memory usage: {data.NumImages * data.ImageSize / 1024 / 1024.0:F1} MB (images)");
            Console.WriteLine($"                 + {data.NumLabels / 1024.0:F1} KB (labels)");

            // Show label distribution
            var labelCounts = new int[10];
            if (data.Labels != null)
            {
                for (int i = 0; i < data.Labels.Length; i++)
                {
                    labelCounts[data.Labels[i]]++;
                }
            }

            Console.WriteLine("\n📈 Label Distribution:");
            for (int digit = 0; digit < 10; digit++)
            {
                var count = labelCounts[digit];
                var percentage = (count * 100.0) / data.NumImages;
                Console.WriteLine($"   Digit {digit}: {count,6:N0} samples ({percentage,5:F1}%)");
            }
        }

        /// <summary>
        /// Get random sample indices for each digit
        /// </summary>
        public static int[] GetRandomSamples(MnistData data, int samplesPerDigit = 10, int? seed = null)
        {
            var random = new Random(seed ?? 42);
            var samples = new List<int>();

            for (int digit = 0; digit < 10; digit++)
            {
                var digitIndices = new List<int>();
                if (data.Labels != null)
                {
                    for (int i = 0; i < data.Labels.Length; i++)
                    {
                        if (data.Labels[i] == digit)
                            digitIndices.Add(i);
                    }
                }

                // Randomly select samples for this digit
                var shuffled = digitIndices.OrderBy(x => random.Next()).Take(samplesPerDigit);
                samples.AddRange(shuffled);
            }

            return samples.OrderBy(x => random.Next()).ToArray();
        }
    }
}