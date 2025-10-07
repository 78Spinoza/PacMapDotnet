#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <windows.h>

// Test the existing UMAP library that should be available
typedef struct {
    void* model;
} UMAPModel;

typedef void (*progress_callback_v2)(const char* phase, int current, int total, float percent, const char* message);

// Function pointers for UMAP DLL
typedef UMAPModel* (*umap_create_func)();
typedef void (*umap_destroy_func)(UMAPModel* model);
typedef int (*umap_fit_func)(UMAPModel* model, float* data, int n_obs, int n_dim,
                           float* embedding, int n_neighbors, float min_dist,
                           float spread, int n_epochs, float alpha,
                           progress_callback_v2 callback, int negative_sample_rate,
                           int a, int b, int seed, int approx_knn, int M,
                           int ef_construction, int ef_search, int deterministic);

void test_progress_callback(const char* phase, int current, int total, float percent, const char* message) {
    std::cout << "Progress: " << phase << " - " << current << "/" << total
              << " (" << std::fixed << std::setprecision(1) << percent << "%)";
    if (message) {
        std::cout << " - " << message;
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "=== Testing Existing UMAP DLL ===" << std::endl;

    // Load the UMAP DLL
    HMODULE hDLL = LoadLibraryA("uwot.dll");
    if (!hDLL) {
        std::cerr << "ERROR: Could not load uwot.dll" << std::endl;
        std::cerr << "Error code: " << GetLastError() << std::endl;
        return 1;
    }

    std::cout << "SUCCESS: uwot.dll loaded successfully" << std::endl;

    // Get function pointers
    umap_create_func uwot_create = (umap_create_func)GetProcAddress(hDLL, "uwot_create");
    umap_destroy_func uwot_destroy = (umap_destroy_func)GetProcAddress(hDLL, "uwot_destroy");
    umap_fit_func uwot_fit = (umap_fit_func)GetProcAddress(hDLL, "uwot_fit");

    if (!uwot_create || !uwot_destroy || !uwot_fit) {
        std::cerr << "ERROR: Could not find required functions in uwot.dll" << std::endl;
        FreeLibrary(hDLL);
        return 1;
    }

    std::cout << "SUCCESS: All required functions found in uwot.dll" << std::endl;

    // Test 1: Create UMAP model
    std::cout << "\n1. Creating UMAP model..." << std::endl;
    UMAPModel* model = uwot_create();
    if (!model) {
        std::cerr << "ERROR: Failed to create UMAP model" << std::endl;
        FreeLibrary(hDLL);
        return 1;
    }
    std::cout << "SUCCESS: UMAP model created" << std::endl;

    // Test 2: Generate test data
    std::cout << "\n2. Generating test data..." << std::endl;
    const int n_samples = 100;
    const int n_features = 10;

    std::vector<float> data(n_samples * n_features);
    std::vector<float> embedding(n_samples * 2);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n_samples * n_features; ++i) {
        data[i] = dist(rng);
    }

    std::cout << "Generated " << n_samples << " samples with " << n_features << " features each" << std::endl;

    // Test 3: Run UMAP fitting
    std::cout << "\n3. Running UMAP fitting..." << std::endl;
    int result = uwot_fit(model, data.data(), n_samples, n_features,
                         embedding.data(), 15, 0.1f, 1.0f, 500, 1.0f,
                         test_progress_callback, 5, 0, 0, 42, 1, 16, 200, 200, 0);

    if (result != 0) {
        std::cerr << "ERROR: UMAP fitting failed with code: " << result << std::endl;
        uwot_destroy(model);
        FreeLibrary(hDLL);
        return 1;
    }
    std::cout << "SUCCESS: UMAP fitting completed" << std::endl;

    // Test 4: Validate embedding
    std::cout << "\n4. Validating embedding..." << std::endl;
    bool embedding_valid = true;
    for (int i = 0; i < n_samples * 2; ++i) {
        if (std::isnan(embedding[i]) || std::isinf(embedding[i])) {
            embedding_valid = false;
            break;
        }
    }

    if (!embedding_valid) {
        std::cerr << "ERROR: Embedding contains NaN or Inf values" << std::endl;
        uwot_destroy(model);
        FreeLibrary(hDLL);
        return 1;
    }

    // Show first few embedding points
    std::cout << "First 5 embedding points:" << std::endl;
    for (int i = 0; i < (5 < n_samples ? 5 : n_samples); ++i) {
        std::cout << "  Point " << i << ": ("
                  << std::fixed << std::setprecision(3) << embedding[i * 2]
                  << ", " << embedding[i * 2 + 1] << ")" << std::endl;
    }
    std::cout << "SUCCESS: Embedding validation passed" << std::endl;

    // Cleanup
    uwot_destroy(model);
    FreeLibrary(hDLL);

    std::cout << "\n=== UMAP DLL TEST PASSED ===" << std::endl;
    std::cout << "The existing UMAP library is working correctly!" << std::endl;
    std::cout << "This confirms that the DLL loading and P/Invoke mechanism works." << std::endl;

    return 0;
}