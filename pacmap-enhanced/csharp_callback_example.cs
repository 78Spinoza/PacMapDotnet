using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

/// <summary>
/// Example C# code demonstrating the safe FFI callback pattern using queue+poll
/// This is the recommended approach for multi-threaded Rust applications
/// </summary>
class RustCallbackExample
{
    // ============================================================================
    // Queue+Poll Pattern (Recommended for multi-threaded applications)
    // ============================================================================

    [DllImport("pacmap_enhanced", CallingConvention = CallingConvention.Cdecl)]
    private static extern void pacmap_enqueue_message([MarshalAs(UnmanagedType.LPStr)] string msg);

    [DllImport("pacmap_enhanced", CallingConvention = CallingConvention.Cdecl)]
    private static extern UIntPtr pacmap_poll_next_message(IntPtr buffer, UIntPtr bufLen);

    [DllImport("pacmap_enhanced", CallingConvention = CallingConvention.Cdecl)]
    private static extern bool pacmap_has_messages();

    [DllImport("pacmap_enhanced", CallingConvention = CallingConvention.Cdecl)]
    private static extern void pacmap_clear_messages();

    // ============================================================================
    // Legacy Direct Callback (Use with caution - not thread-safe)
    // ============================================================================

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void TextCallback(IntPtr userData, IntPtr dataPtr, UIntPtr len);

    private static TextCallback _legacyCallbackDelegate;
    private static IntPtr _legacyCallbackPtr;

    [DllImport("pacmap_enhanced", CallingConvention = CallingConvention.Cdecl)]
    private static extern void pacmap_register_text_callback(IntPtr cb, IntPtr user_data);

    /// <summary>
    /// Initialize the new thread-safe callback system
    /// </summary>
    public static void InitializeThreadSafeCallbacks()
    {
        Console.WriteLine("Initializing thread-safe callback system...");
        // No registration needed for queue+poll pattern
        Console.WriteLine("Thread-safe callback system ready.");
    }

    /// <summary>
    /// Poll for messages from Rust (thread-safe)
    /// Call this periodically from your main thread or a dedicated polling thread
    /// </summary>
    public static void PollMessages()
    {
        byte[] buffer = new byte[2048]; // Adjust size as needed

        while (pacmap_has_messages())
        {
            GCHandle handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
            try
            {
                UIntPtr messageLen = pacmap_poll_next_message(handle.AddrOfPinnedObject(), (UIntPtr)buffer.Length);
                int len = (int)messageLen;

                if (len > 0)
                {
                    string message = Encoding.UTF8.GetString(buffer, 0, len);
                    Console.WriteLine($"[Thread-Safe] Rust message: {message}");
                }
                else
                {
                    break; // No more messages
                }
            }
            finally
            {
                handle.Free();
            }
        }
    }

    /// <summary>
    /// Example: Run a message polling loop on a background thread
    /// </summary>
    public static void StartMessagePollingThread(CancellationToken cancellationToken)
    {
        Thread pollingThread = new Thread(() =>
        {
            Console.WriteLine("Message polling thread started.");

            try
            {
                while (!cancellationToken.IsCancellationRequested)
                {
                    PollMessages();
                    Thread.Sleep(50); // Poll every 50ms (adjust as needed)
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Message polling thread error: {ex.Message}");
            }
            finally
            {
                Console.WriteLine("Message polling thread stopped.");
            }
        })
        {
            IsBackground = true,
            Name = "RustMessagePoller"
        };

        pollingThread.Start();
    }

    /// <summary>
    /// Initialize the legacy direct callback system (for backward compatibility)
    /// WARNING: Not recommended for multi-threaded Rust applications
    /// </summary>
    public static void InitializeLegacyCallbacks()
    {
        Console.WriteLine("Initializing legacy callback system (not recommended for multi-threading)...");

        _legacyCallbackDelegate = new TextCallback(OnLegacyCallback);
        _legacyCallbackPtr = Marshal.GetFunctionPointerForDelegate(_legacyCallbackDelegate);

        // Register callback with null user_data for this example
        pacmap_register_text_callback(_legacyCallbackPtr, IntPtr.Zero);

        Console.WriteLine("Legacy callback system initialized.");
    }

    /// <summary>
    /// Legacy callback handler (will be called from Rust threads)
    /// WARNING: This can cause threading issues with multi-threaded Rust applications
    /// </summary>
    private static void OnLegacyCallback(IntPtr userData, IntPtr dataPtr, UIntPtr len)
    {
        try
        {
            int length = (int)len;
            if (length == 0) return;

            byte[] buffer = new byte[length];
            Marshal.Copy(dataPtr, buffer, 0, length);

            string text = Encoding.UTF8.GetString(buffer);
            Console.WriteLine($"[Legacy Callback] Rust message: {text}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Exception in legacy callback: {ex}");
        }
    }

    /// <summary>
    /// Cleanup resources
    /// </summary>
    public static void Cleanup()
    {
        // Clear any pending messages
        pacmap_clear_messages();

        // Unregister legacy callback
        if (_legacyCallbackPtr != IntPtr.Zero)
        {
            pacmap_register_text_callback(IntPtr.Zero, IntPtr.Zero);
            _legacyCallbackDelegate = null;
            _legacyCallbackPtr = IntPtr.Zero;
        }

        Console.WriteLine("Callback cleanup completed.");
    }

    /// <summary>
    /// Example usage
    /// </summary>
    public static void Main()
    {
        Console.WriteLine("PacMAP Enhanced FFI Callback Example");
        Console.WriteLine("=====================================");

        // Initialize the recommended thread-safe system
        InitializeThreadSafeCallbacks();

        // Optionally initialize legacy system for backward compatibility
        // InitializeLegacyCallbacks();

        // Start message polling thread
        using (var cts = new CancellationTokenSource())
        {
            StartMessagePollingThread(cts.Token);

            // Simulate some work that would trigger Rust callbacks
            Console.WriteLine("Simulating work with Rust callbacks...");

            // Test the queue system directly
            pacmap_enqueue_message("Test message from C#");
            pacmap_enqueue_message("Another test message");

            // Let the polling thread run for a bit
            Thread.Sleep(2000);

            // Stop polling
            cts.Cancel();
            Thread.Sleep(500); // Give thread time to stop
        }

        // Cleanup
        Cleanup();

        Console.WriteLine("Example completed.");
    }
}

/// <summary>
/// Summary of the two callback approaches:
///
/// 1. Queue+Poll Pattern (Recommended):
///    - Thread-safe for multi-threaded Rust applications
///    - No reverse P/Invoke overhead
///    - C# controls when messages are processed
///    - Safe for UI thread usage
///    - Use: pacmap_enqueue_message(), pacmap_poll_next_message(), pacmap_has_messages()
///
/// 2. Legacy Direct Callback (Use with caution):
///    - Direct reverse P/Invoke from Rust threads
///    - Higher overhead per callback
///    - Potential threading issues with managed code
///    - Use: pacmap_register_text_callback() + delegate
///
/// For multi-threaded applications like PacMAP with HNSW + Rayon,
/// the Queue+Poll pattern is strongly recommended.
/// </summary>