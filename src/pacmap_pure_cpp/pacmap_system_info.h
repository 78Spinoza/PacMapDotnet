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

        // OpenMP detection with proper core detection and thread management
        #ifdef _OPENMP
            // Detect available cores dynamically
            int num_cores = omp_get_num_procs();
            int logical_cores = omp_get_max_threads();

            // Use all available physical cores for maximum performance
            caps.max_threads = num_cores;

            // Force OpenMP to use detected thread count
            omp_set_num_threads(caps.max_threads);

            caps.openmp_enabled = true;

            // Ensure OpenMP is properly initialized with detected thread count
            #pragma omp parallel
            {
                #pragma omp single
                {
                    // Force thread pool creation with proper thread count
                }
            }
        #else
            caps.openmp_enabled = false;
            caps.max_threads = 1;
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

    // Generate minimal capability report
    std::string generate_capability_report(const SystemCapabilities& caps) {
        std::ostringstream report;

        report << "System: ";
        if (caps.openmp_enabled && caps.max_threads > 1) {
            report << "OpenMP " << caps.max_threads << " threads (detected " << omp_get_num_procs() << " cores), ";
        } else {
            report << "Single thread (" << omp_get_num_procs() << " cores detected), ";
        }
        if (caps.simd_avx2) {
            report << "AVX2 enabled";
        } else {
            report << "Scalar mode";
        }
        report << "\n";

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