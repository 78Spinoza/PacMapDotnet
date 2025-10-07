#pragma once

#include "pacmap_model.h"
#include "pacmap_hnsw_utils.h"
#include "pacmap_progress_utils.h"
#include <iostream>
#include <fstream>

namespace persistence_utils {

    // Main persistence functions
    int save_model(PacMapModel* model, const char* filename);
    PacMapModel* load_model(const char* filename);

    // HNSW compression utilities
    void save_hnsw_to_stream_compressed(std::ostream& output, hnswlib::HierarchicalNSW<float>* hnsw_index);
    void load_hnsw_from_stream_compressed(std::istream& input, hnswlib::HierarchicalNSW<float>* hnsw_index,
        hnswlib::SpaceInterface<float>* space);

}