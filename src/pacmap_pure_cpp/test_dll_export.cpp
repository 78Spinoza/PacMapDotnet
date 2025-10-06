#include <iostream>
#include <windows.h>
#include <iomanip>
#include <psapi.h>

// Test to enumerate all functions in the DLL
int main() {
    std::cout << "=== Checking DLL Exports ===" << std::endl;

    HMODULE hDLL = LoadLibraryA("uwot.dll");
    if (!hDLL) {
        std::cerr << "ERROR: Could not load uwot.dll" << std::endl;
        return 1;
    }

    std::cout << "SUCCESS: uwot.dll loaded" << std::endl;

    // Try some common UMAP function names
    const char* function_names[] = {
        "uwot_create",
        "uwot_destroy",
        "uwot_fit",
        "uwot_fit_with_progress",
        "uwot_fit_with_progress_v2",
        "uwot_transform",
        "uwot_transform_detailed",
        "uwot_save_model",
        "uwot_load_model",
        "uwot_get_n_neighbors",
        "uwot_get_min_dist",
        "uwot_get_spread",
        "uwot_get_model_info",
        "uwot_get_model_info_v2",
        "uwot_is_fitted",
        "uwot_get_version",
        "uwot_get_metric_name",
        "uwot_get_error_message",
        "uwot_clear_global_callback",
        "umap_create",
        "umap_destroy",
        "umap_fit",
        "create_umap",
        "destroy_umap",
        "fit_umap",
        "fit",
        "transform",
        "save",
        "load",
        NULL
    };

    std::cout << "\nChecking function exports:" << std::endl;
    int found_count = 0;
    for (int i = 0; function_names[i]; ++i) {
        FARPROC func = GetProcAddress(hDLL, function_names[i]);
        if (func) {
            std::cout << "  ✓ " << function_names[i] << " @ " << std::hex << func << std::endl;
            found_count++;
        } else {
            std::cout << "  ✗ " << function_names[i] << std::endl;
        }
    }

    std::cout << "\nFound " << found_count << " functions" << std::endl;

    // Try to find any function at all by checking if there are any exports
    std::cout << "\nChecking DLL size and basic info:" << std::endl;

    // Get module handle
    HMODULE hModule = GetModuleHandleA("uwot.dll");
    if (hModule) {
        MODULEINFO modInfo;
        if (GetModuleInformation(GetCurrentProcess(), hModule, &modInfo, sizeof(modInfo))) {
            std::cout << "  Base address: " << std::hex << modInfo.lpBaseOfDll << std::endl;
            std::cout << "  Image size: " << std::dec << modInfo.SizeOfImage << " bytes" << std::endl;
        }
    }

    FreeLibrary(hDLL);

    if (found_count == 0) {
        std::cout << "\nWARNING: No UMAP functions found. This might be a different type of DLL." << std::endl;
        std::cout << "Let's try to check if it's a C++ .NET assembly or another format." << std::endl;
    }

    return 0;
}