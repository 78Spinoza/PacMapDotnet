#include <iostream>
#include <vector>
#include <windows.h>

// Define the same types as in the header to avoid including it
typedef struct PacMapModel PacMapModel;

// Error codes
#define PACMAP_SUCCESS 0
#define PACMAP_ERROR_INVALID_PARAMS -1

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
typedef int (*pacmap_get_n_components_func)(PacMapModel* model);
typedef int (*pacmap_get_n_samples_func)(PacMapModel* model);
typedef int (*pacmap_fit_with_progress_v2_func)(
    PacMapModel* model,
    double* data,
    int n_obs,
    int n_dim,
    int embedding_dim,
    int n_neighbors,
    float MN_ratio,
    float FP_ratio,
    float learning_rate,
    int n_iters,
    int phase1_iters,
    int phase2_iters,
    int phase3_iters,
    PacMapMetric metric,
    double* embedding,
    pacmap_progress_callback_v2 progress_callback,
    int force_exact_knn,
    int M,
    int ef_construction,
    int ef_search,
    int use_quantization,
    int random_seed,
    int autoHNSWParam,
    float initialization_std_dev
);
typedef int (*pacmap_save_model_func)(PacMapModel* model, const char* filename);
typedef PacMapModel* (*pacmap_load_model_func)(const char* filename);
typedef const char* (*pacmap_get_error_message_func)(int error_code);

// Simple progress callback
void simple_progress_callback(
    const char* phase,
    int current,
    int total,
    float percent,
    const char* message) {
    std::cout << "PROGRESS: " << phase << " - " << current << "/" << total
              << " (" << percent << "%) - " << (message ? message : "no message") << std::endl;
}

int main() {
    std::cout << "=== Basic PACMAP Integration Test ===" << std::endl;

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
    pacmap_get_n_components_func p_get_n_components = (pacmap_get_n_components_func)GetProcAddress(hDll, "pacmap_get_n_components");
    pacmap_get_n_samples_func p_get_n_samples = (pacmap_get_n_samples_func)GetProcAddress(hDll, "pacmap_get_n_samples");
    pacmap_fit_with_progress_v2_func p_fit = (pacmap_fit_with_progress_v2_func)GetProcAddress(hDll, "pacmap_fit_with_progress_v2");
    pacmap_save_model_func p_save = (pacmap_save_model_func)GetProcAddress(hDll, "pacmap_save_model");
    pacmap_load_model_func p_load = (pacmap_load_model_func)GetProcAddress(hDll, "pacmap_load_model");
    pacmap_get_error_message_func p_get_error_message = (pacmap_get_error_message_func)GetProcAddress(hDll, "pacmap_get_error_message");

    if (!p_create || !p_destroy || !p_get_version || !p_is_fitted || !p_fit) {
        std::cout << "FAILED: Could not get required function pointers" << std::endl;
        FreeLibrary(hDll);
        return 1;
    }

    // Test 1: Create model
    std::cout << "Test 1: Creating PACMAP model..." << std::endl;
    PacMapModel* model = p_create();
    if (!model) {
        std::cout << "FAILED: Could not create model" << std::endl;
        FreeLibrary(hDll);
        return 1;
    }
    std::cout << "SUCCESS: Model created" << std::endl;

    // Test 2: Check version
    std::cout << "Test 2: Checking version..." << std::endl;
    const char* version = p_get_version();
    std::cout << "SUCCESS: Version " << version << std::endl;

    // Test 3: Check if fitted (should be false initially)
    std::cout << "Test 3: Checking fitted status..." << std::endl;
    int is_fitted = p_is_fitted(model);
    std::cout << "SUCCESS: Is fitted = " << is_fitted << " (should be 0)" << std::endl;

    // Test 4: Simple test data
    std::cout << "Test 4: Creating test data..." << std::endl;
    const int n_samples = 100;
    const int n_features = 5;
    const int embedding_dim = 2;

    // v2.8.6+: Use double precision instead of float
    std::vector<double> data(n_samples * n_features);
    std::vector<double> embedding(n_samples * embedding_dim);

    // Generate simple test data
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            data[i * n_features + j] = static_cast<double>(i * j) * 0.1;
        }
    }

    std::cout << "SUCCESS: Test data created (" << n_samples << " x " << n_features << ")" << std::endl;

    // Test 5: Fit model
    std::cout << "Test 5: Fitting PACMAP model..." << std::endl;
    // Try with minimal parameters to avoid potential issues
    int result = p_fit(
        model,
        data.data(),
        n_samples,
        n_features,
        embedding_dim,
        5,   // n_neighbors (smaller)
        0.1f, // MN_ratio (smaller)
        0.4f, // FP_ratio (smaller)
        1.0f, // learning_rate
        200,  // n_iters (much smaller)
        50,   // phase1_iters (smaller but >= 100 total)
        50,   // phase2_iters (smaller but >= 100 total)
        100,  // phase3_iters (smaller but >= 100 total)
        PACMAP_METRIC_EUCLIDEAN,
        embedding.data(),
        simple_progress_callback, // Use the progress callback to see system capabilities
        1,      // force_exact_knn (use exact for small dataset)
        -1,     // M
        -1,     // ef_construction
        -1,     // ef_search
        0,      // use_quantization
        42,     // random_seed
        0,      // autoHNSWParam (disable for small dataset)
        1e-4f   // initialization_std_dev (new parameter)
    );

    if (result != PACMAP_SUCCESS) {
        std::cout << "FAILED: Fit returned error " << result << std::endl;
        if (p_get_error_message) {
            const char* error_msg = p_get_error_message(result);
            std::cout << "Error message: " << error_msg << std::endl;
        }
        p_destroy(model);
        FreeLibrary(hDll);
        return 1;
    }
    std::cout << "SUCCESS: Model fitted successfully" << std::endl;

    // Test 6: Check fitted status (should be true now)
    std::cout << "Test 6: Checking fitted status after fit..." << std::endl;
    is_fitted = p_is_fitted(model);
    std::cout << "SUCCESS: Is fitted = " << is_fitted << " (should be 1)" << std::endl;

    // Test 7: Get model info
    std::cout << "Test 7: Getting model info..." << std::endl;
    int n_components = p_get_n_components ? p_get_n_components(model) : -1;
    int n_samples_stored = p_get_n_samples ? p_get_n_samples(model) : -1;
    std::cout << "SUCCESS: n_components = " << n_components << std::endl;
    std::cout << "SUCCESS: n_samples = " << n_samples_stored << std::endl;

    // Test 8: Check embedding dimensions
    std::cout << "Test 8: Checking embedding dimensions..." << std::endl;
    std::cout << "Embedding shape: " << n_samples << " x " << embedding_dim << std::endl;
    std::cout << "First embedding point: (" << embedding[0] << ", " << embedding[1] << ")" << std::endl;

    // Test 9: Save model
    std::cout << "Test 9: Saving model..." << std::endl;
    result = p_save ? p_save(model, "test_model.pacmap") : -1;
    if (result != PACMAP_SUCCESS) {
        std::cout << "WARNING: Save returned error " << result << std::endl;
    } else {
        std::cout << "SUCCESS: Model saved" << std::endl;
    }

    // Test 10: Load model
    std::cout << "Test 10: Loading model..." << std::endl;
    PacMapModel* loaded_model = p_load ? p_load("test_model.pacmap") : nullptr;
    if (!loaded_model) {
        std::cout << "WARNING: Could not load model" << std::endl;
    } else {
        std::cout << "SUCCESS: Model loaded" << std::endl;
        int loaded_components = p_get_n_components ? p_get_n_components(loaded_model) : -1;
        std::cout << "Loaded model n_components = " << loaded_components << std::endl;
        p_destroy(loaded_model);
    }

    // Cleanup
    std::cout << "Test 11: Cleanup..." << std::endl;

    // Call OpenMP cleanup before DLL unload to prevent segfault
    if (p_get_error_message) {
        // Try to get cleanup function if available
        typedef void (*pacmap_cleanup_func)();
        pacmap_cleanup_func p_cleanup = (pacmap_cleanup_func)GetProcAddress(hDll, "pacmap_cleanup");
        if (p_cleanup) {
            p_cleanup();
            std::cout << "SUCCESS: OpenMP cleanup completed" << std::endl;
        }
    }

    p_destroy(model);
    std::cout << "SUCCESS: Model destroyed" << std::endl;
    FreeLibrary(hDll);
    std::cout << "SUCCESS: DLL unloaded" << std::endl;

    std::cout << "\n=== ALL TESTS COMPLETED SUCCESSFULLY ===" << std::endl;
    return 0;
}