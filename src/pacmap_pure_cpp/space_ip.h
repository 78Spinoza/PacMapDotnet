#pragma once

#include <cmath>
#include "hnswlib.h"

namespace hnswlib {

class IPSpace : public SpaceInterface<float> {
public:
    IPSpace(size_t dim) : dim_(dim) {}

    size_t get_data_size() override {
        return dim_ * sizeof(float);
    }

    DISTFUNC<float> get_dist_func() override {
        return InnerProductDistance;
    }

    void* get_dist_func_param() override {
        return &dim_;
    }

    static float InnerProductDistance(const void* a, const void* b, const void* dim_ptr) {
        size_t dim = *((size_t*)dim_ptr);
        const float* fa = (const float*)a;
        const float* fb = (const float*)b;

        // Inner product distance = 1 - cosine similarity
        float dot_product = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;

        for (size_t i = 0; i < dim; ++i) {
            dot_product += fa[i] * fb[i];
            norm_a += fa[i] * fa[i];
            norm_b += fb[i] * fb[i];
        }

        norm_a = std::sqrt(norm_a);
        norm_b = std::sqrt(norm_b);

        if (norm_a == 0.0f || norm_b == 0.0f) {
            return 1.0f; // Maximum distance if either vector is zero
        }

        float cosine_similarity = dot_product / (norm_a * norm_b);
        return 1.0f - std::max(-1.0f, std::min(1.0f, cosine_similarity));
    }

private:
    size_t dim_;
};

// Add alias for compatibility with existing code
using InnerProductSpace = IPSpace;

} // namespace hnswlib