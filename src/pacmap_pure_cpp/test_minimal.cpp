#include <iostream>
#include <windows.h>

// Define the same types as in the header to avoid including it
using PacMapModel = struct PacMapModel;

// Error codes
#define PACMAP_SUCCESS 0
#define PACMAP_ERROR_INVALID_PARAMS (-1)

// Distance metrics
typedef enum {
    PACMAP_METRIC_EUCLIDEAN = 0,
    PACMAP_METRIC_COSINE = 1,
    PACMAP_METRIC_MANHATTAN = 2,
    PACMAP_METRIC_CORRELATION = 3,
    PACMAP_METRIC_HAMMING = 4
} PacMapMetric;

// Progress callback type
typedef void (*pacmap_progress_callback_v2)(
    const char* phase,
    int current,
    int total,
    float percent,
    const char* message
);

// Function pointer types
typedef PacMapModel* (*pacmap_create_func)();
typedef void (*pacmap_destroy_func)(PacMapModel* model);
typedef const char* (*pacmap_get_version_func)();
typedef int (*pacmap_is_fitted_func)(PacMapModel* model);
typedef int (*pacmap_get_n_samples_func)(PacMapModel* model);
typedef int (*pacmap_test_minimal_fit_func)(PacMapModel* model);

int main() {
    std::cout << "=== Minimal PACMAP Test ===" << std::endl;

    // Load the DLL
    std::cout << "Loading pacmap.dll..." << std::endl;
    HMODULE hDll = LoadLibraryA("pacmap.dll");
    if (!hDll) {
        std::cout << "FAILED: Could not load pacmap.dll (Error: " << GetLastError() << ")" << std::endl;
        return 1;
    }
    std::cout << "SUCCESS: DLL loaded" << std::endl;

    // Get function pointers
    pacmap_create_func p_create = (pacmap_create_func)GetProcAddress(hDll, "pacmap_create");
    pacmap_destroy_func p_destroy = (pacmap_destroy_func)GetProcAddress(hDll, "pacmap_destroy");
    pacmap_get_version_func p_get_version = (pacmap_get_version_func)GetProcAddress(hDll, "pacmap_get_version");
    pacmap_is_fitted_func p_is_fitted = (pacmap_is_fitted_func)GetProcAddress(hDll, "pacmap_is_fitted");
    pacmap_get_n_samples_func p_get_n_samples = (pacmap_get_n_samples_func)GetProcAddress(hDll, "pacmap_get_n_samples");
    pacmap_test_minimal_fit_func p_test_minimal_fit = (pacmap_test_minimal_fit_func)GetProcAddress(hDll, "pacmap_test_minimal_fit");

    if (!p_create || !p_destroy || !p_get_version || !p_is_fitted) {
        std::cout << "FAILED: Could not get required function pointers" << std::endl;
        FreeLibrary(hDll);
        return 1;
    }

    // Create model
    std::cout << "Creating model..." << std::endl;
    PacMapModel* model = p_create();
    if (!model) {
        std::cout << "FAILED: Could not create model" << std::endl;
        FreeLibrary(hDll);
        return 1;
    }
    std::cout << "SUCCESS: Model created at address: " << (void*)model << std::endl;

    // Check version
    std::cout << "Version: " << p_get_version() << std::endl;

    // Test basic model access
    std::cout << "Testing basic model access..." << std::endl;
    std::cout << "Is fitted: " << p_is_fitted(model) << std::endl;
    std::cout << "N samples: " << p_get_n_samples(model) << std::endl;

    // Try to access model fields directly to see if it's valid
    std::cout << "Model address: " << (void*)model << std::endl;

    // Test simple memory access
    try {
        // Just test that we can read from the model structure
        int* n_samples_ptr = (int*)model;
        std::cout << "First field (n_samples): " << *n_samples_ptr << std::endl;
        std::cout << "SUCCESS: Basic model memory access works" << std::endl;
    } catch (...) {
        std::cout << "FAILED: Cannot access model memory" << std::endl;
    }

    // Test the minimal fit function
    if (p_test_minimal_fit) {
        std::cout << "Testing minimal fit function..." << std::endl;
        int result = p_test_minimal_fit(model);
        if (result == PACMAP_SUCCESS) {
            std::cout << "SUCCESS: Minimal fit test passed" << std::endl;
            std::cout << "Updated n_samples: " << p_get_n_samples(model) << std::endl;
        } else {
            std::cout << "FAILED: Minimal fit test returned error " << result << std::endl;
        }
    } else {
        std::cout << "WARNING: Could not find pacmap_test_minimal_fit function" << std::endl;
    }

    // Cleanup
    p_destroy(model);
    FreeLibrary(hDll);

    std::cout << "Minimal test completed successfully" << std::endl;
    return 0;
}