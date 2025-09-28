using System;
using System.Runtime.InteropServices;

public class DllTest
{
    private const string WindowsDll = "pacmap_enhanced.dll";

    [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_get_version")]
    private static extern IntPtr GetVersion();

    [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pacmap_config_default")]
    private static extern IntPtr ConfigDefault();

    public static void TestDll()
    {
        try
        {
            Console.WriteLine("🔍 Testing DLL exports...");

            // Test version function
            try
            {
                var versionPtr = GetVersion();
                if (versionPtr != IntPtr.Zero)
                {
                    var version = Marshal.PtrToStringAnsi(versionPtr);
                    Console.WriteLine($"✅ pacmap_get_version: {version}");
                }
                else
                {
                    Console.WriteLine("❌ pacmap_get_version returned null");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ pacmap_get_version failed: {ex.Message}");
            }

            // Test config function
            try
            {
                var configPtr = ConfigDefault();
                Console.WriteLine($"✅ pacmap_config_default: {configPtr}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ pacmap_config_default failed: {ex.Message}");
            }

            Console.WriteLine("🔍 DLL test completed");
        }
        catch (DllNotFoundException)
        {
            Console.WriteLine("❌ DLL not found: pacmap_enhanced.dll");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ DLL test failed: {ex.Message}");
        }
    }
}