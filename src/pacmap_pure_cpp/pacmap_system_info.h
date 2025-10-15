#pragma once

#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <omp.h>
#include "pacmap_simd_utils.h"

// ERROR14 System Information and Performance Capabilities
namespace pacmap_system {

    struct SystemCapabilities {
        bool openmp_enabled = false;
        int max_threads = 1;
        bool simd_avx2 = false;
        bool simd_avx512 = false;
        std::string cpu_vendor;
        std::string cpu_model;
        size_t cache_l1 = 0;
        size_t cache_l2 = 0;
        size_t cache_l3 = 0;
    };

    // Detect system capabilities for performance optimization
    SystemCapabilities detect_system_capabilities() {
        SystemCapabilities caps;

        // OpenMP detection
        #ifdef _OPENMP
            caps.openmp_enabled = true;
            caps.max_threads = omp_get_max_threads();
        #endif

        // SIMD detection
        caps.simd_avx2 = pacmap_simd::supports_avx2();

        // TODO: Add AVX512 detection when needed
        // caps.simd_avx512 = supports_avx512();

        // Basic CPU info (Windows specific)
        #ifdef _WIN32
            int cpu_info[4] = {-1};
            __cpuid(cpu_info, 0);
            if (cpu_info[0] >= 1) {
                // Get vendor string
                char vendor[13] = {0};
                *((int*)vendor) = cpu_info[1];   // EBX
                *((int*)(vendor + 4)) = cpu_info[3];  // EDX
                *((int*)(vendor + 8)) = cpu_info[2];  // ECX
                caps.cpu_vendor = vendor;

                // Get model info (basic)
                __cpuid(cpu_info, 1);
                // Additional model detection could be added here
            }
        #endif

        return caps;
    }

    // Generate human-readable capability report
    std::string generate_capability_report(const SystemCapabilities& caps) {
        std::ostringstream report;

        report << "PACMAP v2.8.17 System Performance Report\n";
        report << "==========================================\n";

        // Parallel Processing
        report << "PARALLEL PROCESSING:\n";
        if (caps.openmp_enabled) {
            report << "   OpenMP: ENABLED (Max threads: " << caps.max_threads << ")\n";
            report << "   Multi-threading: ACTIVE for triplet sampling and gradient computation\n";
        } else {
            report << "   OpenMP: DISABLED (Single-threaded only)\n";
        }

        // SIMD Capabilities
        report << "\nSIMD ACCELERATION:\n";
        if (caps.simd_avx2) {
            report << "   AVX2: ENABLED (8x float/vector parallelism)\n";
            report << "   Eigen SIMD: ACTIVE for distance calculations and gradient updates\n";
            report << "   Performance boost: ~2-3x for vector operations\n";
        } else {
            report << "   AVX2: NOT AVAILABLE (Using scalar fallback)\n";
            report << "   Fallback: Standard loops (slower but compatible)\n";
        }

        // CPU Info
        report << "\nSYSTEM INFORMATION:\n";
        report << "   CPU Vendor: " << (caps.cpu_vendor.empty() ? "Unknown" : caps.cpu_vendor) << "\n";
        report << "   Logical Cores: " << caps.max_threads << "\n";

        // Performance Recommendations
        report << "\nPERFORMANCE OPTIMIZATIONS:\n";
        if (caps.openmp_enabled && caps.max_threads > 1) {
            report << "   Parallel triplet sampling with thread-local storage\n";
            report << "   Thread-safe gradient accumulation with OpenMP\n";
        } else {
            report << "   Consider enabling OpenMP for better performance\n";
        }

        if (caps.simd_avx2) {
            report << "   Eigen SIMD vectorization for distance calculations\n";
            report << "   Hardware-accelerated mathematical operations\n";
        } else {
            report << "   Consider upgrading to a CPU with AVX2 support\n";
        }

        // Thread Safety Status
        report << "\nTHREAD SAFETY STATUS:\n";
        report << "   Thread-local gradient accumulation (prevents race conditions)\n";
        report << "   Thread-safe RNG with per-thread seeding\n";
        report << "   Critical sections only where necessary\n";

        report << "\n==========================================\n";

        return report.str();
    }

    // Print capability report to console and via callback
    void report_system_capabilities(void (*progress_callback)(const char*, int, int, float, const char*)) {
        SystemCapabilities caps = detect_system_capabilities();
        std::string report = generate_capability_report(caps);

        // Print to console
        std::cout << report << std::endl;

        // Send via progress callback if available
        if (progress_callback) {
            // Split report into lines and send as progress messages
            std::istringstream stream(report);
            std::string line;
            int line_num = 0;
            int total_lines = std::count(report.begin(), report.end(), '\n') + 1;

            while (std::getline(stream, line)) {
                float percent = (float)++line_num / total_lines * 100.0f;
                progress_callback("System Capabilities", line_num, total_lines, percent, line.c_str());
            }
        }
    }

} // namespace pacmap_system