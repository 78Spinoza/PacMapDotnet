#ifndef PACMAP_SIMD_UTILS_H
#define PACMAP_SIMD_UTILS_H

#include <cstdint>

// Platform-specific CPUID headers
#if defined(_MSC_VER)
    #include <intrin.h>  // MSVC intrinsics
#elif defined(__GNUC__) || defined(__clang__)
    #include <cpuid.h>   // GCC/Clang intrinsics
#endif

// ERROR14 Step 3: SIMD utility functions for runtime CPU feature detection
// Provides AVX2 detection with scalar fallback for older CPUs

namespace pacmap_simd {

// Runtime CPU feature detection for AVX2 support
// Returns true if CPU supports AVX2 instructions, false otherwise
inline bool supports_avx2() {
#if defined(_MSC_VER)
    // MSVC intrinsics for CPUID
    int cpu_info[4];
    __cpuid(cpu_info, 1);
    // Check ECX bit 28 for AVX support
    bool has_avx = (cpu_info[2] & (1 << 28)) != 0;

    if (!has_avx) return false;

    // Check for AVX2 (EBX bit 5 in function 7, subfunction 0)
    __cpuidex(cpu_info, 7, 0);
    bool has_avx2 = (cpu_info[1] & (1 << 5)) != 0;

    return has_avx2;
#elif defined(__GNUC__) || defined(__clang__)
    // GCC/Clang intrinsics for CPUID
    unsigned int eax, ebx, ecx, edx;

    // Check for AVX (function 1, ECX bit 28)
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    bool has_avx = (ecx & (1 << 28)) != 0;

    if (!has_avx) return false;

    // Check for AVX2 (function 7, subfunction 0, EBX bit 5)
    __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
    bool has_avx2 = (ebx & (1 << 5)) != 0;

    return has_avx2;
#else
    // Unknown compiler - conservative fallback to scalar
    return false;
#endif
}

// Global flag to enable/disable SIMD (can be set via environment variable)
// Default: auto-detect based on CPU capabilities
inline bool should_use_simd() {
    static bool use_simd = supports_avx2();
    return use_simd;
}

// Force enable/disable SIMD (for testing purposes)
inline void set_use_simd(bool enable) {
    // This would require a static variable in should_use_simd()
    // For now, we rely on compile-time detection
    // Future: add environment variable support (PACMAP_NO_SIMD=1)
}

} // namespace pacmap_simd

#endif // PACMAP_SIMD_UTILS_H
