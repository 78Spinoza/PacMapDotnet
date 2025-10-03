using System;
using System.IO;
using PacMapDemo;

class TestAnatomyProgram
{
    static void Main(string[] args)
    {
        Console.WriteLine("Mammoth Anatomy Classification Test Program");
        Console.WriteLine("==========================================\n");

        try
        {
            // Load mammoth data
            string dataPath = @"C:\PacMAN\PacMapDemo\Data\mammoth_data.csv";
            Console.WriteLine($"Loading mammoth data from: {dataPath}");

            var mammothData = DataLoaders.LoadMammothData(dataPath);
            Console.WriteLine($"Loaded {mammothData.GetLength(0)} points with {mammothData.GetLength(1)} dimensions\n");

            // Run anatomy tests
            string outputDir = @"C:\PacMAN\AnatomyTests";
            AnatomyTest.RunAnatomyTests(mammothData, outputDir);

            Console.WriteLine($"\nTest results saved to: {outputDir}");
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }
    }
}