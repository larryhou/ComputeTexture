//
//  shader.metal
//  ComputeTexture
//
//  Created by larryhou on 2024/7/10.
//

#include <metal_stdlib>
#include <metal_atomic>
#include "shader.h"
using namespace metal;

constexpr sampler defaultSampler(address::clamp_to_edge, filter::linear, max_anisotropy(4));


kernel void compute(texture2d<half, access::read >  source [[texture(0)]],
                    texture2d<half, access::write>  target [[texture(1)]],
                    texture2d<half, access::sample> mra    [[texture(2)]],
                    constant Uniform     *uniform [[buffer(0)]],
                    device atomic_uint   *putback [[buffer(1)]],
                    constant bool        *hasmra  [[buffer(2)]],
                    
                    ushort2 position [[thread_position_in_grid]])
{
    if (position.x > uniform->screen.x || position.y > uniform->screen.y) return;
    half4 color = source.read(position);
    half4 value = color * uniform->brightness.y;
    
    half4 mraColor(0);
    if (hasmra) {
        mraColor = mra.sample(defaultSampler, float2(position)/uniform->screen);
    }
    
    uint hit(0);
    float maxValue = fmax3(value.r, value.g, value.b) * 255.0f;
    if (mraColor.r <= 0.5 && maxValue >= uniform->threshold) {
        hit = 1;
    }
    
    atomic_fetch_max_explicit(&putback[0], maxValue * hit, memory_order_relaxed);
    atomic_fetch_add_explicit(&putback[1], hit, memory_order_relaxed);
    
    if (hit) {
        color = half4(1,0,1,1);
    }
    
    target.write(color, position);
}
