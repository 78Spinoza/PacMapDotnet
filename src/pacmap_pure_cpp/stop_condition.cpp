#include "stop_condition.h"

namespace hnswlib {

// Initialize function pointers for inner product distance functions
#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
    float (*InnerProductSIMD16Ext)(const void* a, const void* b, const void* dim_ptr) = nullptr;
    float (*InnerProductSIMD4Ext)(const void* a, const void* b, const void* dim_ptr) = nullptr;
    float (*InnerProductDistanceSIMD16Ext)(const void* a, const void* b, const void* dim_ptr) = nullptr;
    float (*InnerProductDistanceSIMD4Ext)(const void* a, const void* b, const void* dim_ptr) = nullptr;
    float (*InnerProductDistanceSIMD16ExtResiduals)(const void* a, const void* b, const void* dim_ptr) = nullptr;
    float (*InnerProductDistanceSIMD4ExtResiduals)(const void* a, const void* b, const void* dim_ptr) = nullptr;

    #if defined(USE_AVX512)
        float (*InnerProductSIMD16ExtAVX512)(const void* a, const void* b, const void* dim_ptr) = nullptr;
        float (*InnerProductDistanceSIMD16ExtAVX512)(const void* a, const void* b, const void* dim_ptr) = nullptr;
    #endif

    #if defined(USE_AVX)
        float (*InnerProductSIMD16ExtAVX)(const void* a, const void* b, const void* dim_ptr) = nullptr;
        float (*InnerProductDistanceSIMD16ExtAVX)(const void* a, const void* b, const void* dim_ptr) = nullptr;
        float (*InnerProductSIMD4ExtAVX)(const void* a, const void* b, const void* dim_ptr) = nullptr;
        float (*InnerProductDistanceSIMD4ExtAVX)(const void* a, const void* b, const void* dim_ptr) = nullptr;
    #endif

    // Initialize L2 function pointers based on CPU capabilities
    #if defined(USE_AVX512)
        float (*L2SqrSIMD16ExtAVX512_Ptr)(const void* a, const void* b, const void* dim_ptr) = nullptr;
    #endif

    #if defined(USE_AVX)
        float (*L2SqrSIMD16ExtAVX_Ptr)(const void* a, const void* b, const void* dim_ptr) = nullptr;
    #endif

    float (*L2SqrSIMD16Ext_Ptr)(const void* a, const void* b, const void* dim_ptr) = nullptr;
    float (*L2SqrSIMD4Ext_Ptr)(const void* a, const void* b, const void* dim_ptr) = nullptr;
    float (*L2SqrSIMD16ExtResiduals_Ptr)(const void* a, const void* b, const void* dim_ptr) = nullptr;
    float (*L2SqrSIMD4ExtResiduals_Ptr)(const void* a, const void* b, const void* dim_ptr) = nullptr;

    // Function to initialize function pointers - called during library initialization
    void initialize_hnsw_function_pointers() {
        // Initialize inner product function pointers based on CPU capabilities
        #if defined(USE_AVX512)
            if (AVX512Capable()) {
                InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX512;
                InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX512;
            } else if (AVXCapable()) {
                InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
                InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
            }
        #elif defined(USE_AVX)
            if (AVXCapable()) {
                InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
                InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
                InnerProductSIMD4Ext = InnerProductSIMD4ExtAVX;
                InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtAVX;
            }
        #endif

        // Initialize L2 function pointers based on CPU capabilities
        #if defined(USE_AVX512)
            if (AVX512Capable()) {
                L2SqrSIMD16Ext_Ptr = L2SqrSIMD16ExtAVX512_Ptr;
            } else if (AVXCapable()) {
                L2SqrSIMD16Ext_Ptr = L2SqrSIMD16ExtAVX_Ptr;
            }
        #elif defined(USE_AVX)
            if (AVXCapable()) {
                L2SqrSIMD16Ext_Ptr = L2SqrSIMD16ExtAVX_Ptr;
            }
        #endif

        // Default to SSE implementation if no AVX available
        if (!L2SqrSIMD16Ext_Ptr) {
            L2SqrSIMD16Ext_Ptr = L2SqrSIMD16ExtSSE;
        }

        // Assign residual function implementations
        L2SqrSIMD4Ext_Ptr = L2SqrSIMD4Ext_impl;
        L2SqrSIMD16ExtResiduals_Ptr = L2SqrSIMD16ExtResiduals_impl;
        L2SqrSIMD4ExtResiduals_Ptr = L2SqrSIMD4ExtResiduals_impl;
    }

    // Initialize function pointers at library startup
    static struct HnswInitializer {
        HnswInitializer() {
            initialize_hnsw_function_pointers();
        }
    } hnsw_initializer;

#else
    // Stub implementations when SIMD is not available
    void initialize_hnsw_function_pointers() {
        // No SIMD available, use scalar implementations
    }
#endif

} // namespace hnswlib