//
//  shader.h
//  ComputeTexture
//
//  Created by larryhou on 2024/7/10.
//

#ifndef shader_h
#define shader_h
#include <simd/simd.h>

struct Uniform {
    simd_float2 brightness;
    simd_float2 screen;
    int threshold;
    int srgb;
};

#endif /* shader_h */
