#pragma once

#include <string>

// Forward declaration of callback type to avoid circular dependency
typedef void (*pacmap_progress_callback_v2)(const char* phase, int current, int total, float percent, const char* message);

// Global variables for passing information to v2 callbacks
extern thread_local float g_current_epoch_loss;
extern thread_local pacmap_progress_callback_v2 g_v2_callback;


// Cross-platform file utilities
namespace temp_utils {
    bool safe_remove_file(const std::string& filename);
}

// Enhanced progress reporting utilities
namespace progress_utils {
    // Format time duration in human-readable format
    std::string format_duration(double seconds);

    // Estimate remaining time based on current progress
    std::string estimate_remaining_time(int current, int total, double elapsed_seconds);

    // Generate complexity-based time warnings
    std::string generate_complexity_warning(int n_obs, int n_dim, const std::string& operation);

    // Safe callback invoker - handles null callbacks gracefully
    void safe_callback(pacmap_progress_callback_v2 callback,
        const char* phase, int current, int total, float percent,
        const char* message = nullptr);
}