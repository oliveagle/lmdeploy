// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_runtime.h>

namespace turbomind {

enum class RopeType
{
    kNull,
    kDefault,
    kLinear,
    kDynamic,
    kYarn,
    kLlama3,
    kMrope,
};

struct YarnRopeKernelParam {
    float scale_factor;
    float attention_factor;
    float ramp_inv_factor_div_2;
    float ramp_inv_factor_mul_min;
};

struct Llama3RopeKernelParam {
    float scale_factor;
    float alpha;
    float beta;
};

struct MropeRopeKernelParam {
    int3 section;

    int  stride{};
    int* position_ids{};
    int* position_delta{};
    int* length{};
};

struct RopeKernelParam {
    RopeType type;

    float* base{};  // for dynamic ntk
    int    dim;
    float  scale_factor;
    float  inv_factor;

    YarnRopeKernelParam   yarn;
    Llama3RopeKernelParam llama3;
    MropeRopeKernelParam  mrope;
};

struct YarnRopeParam {
    float attention_factor = 0.f;
    float beta_fast        = 0.f;
    float beta_slow        = 0.f;
};

struct Llama3RopeParam {
    float low_freq_factor                  = 0.f;
    float high_freq_factor                 = 0.f;
    int   original_max_position_embeddings = 0;
};

struct MropeParam {
    int3 section{};
};

struct RopeParam {
    RopeType type = RopeType::kDefault;

    float base = 0.f;
    int   dim  = 0;

    float factor                  = 0.f;
    int   max_position_embeddings = 0;

    YarnRopeParam   yarn{};
    Llama3RopeParam llama3{};
    MropeParam      mrope{};
};

}  // namespace turbomind
