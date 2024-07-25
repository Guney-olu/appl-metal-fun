#include <metal_stdlib>
using namespace metal;

kernel void matrix_multiplication(const device float* inA [[ buffer(0) ]],
                                  const device float* inB [[ buffer(1) ]],
                                  device float* result [[ buffer(2) ]],
                                  uint2 gid [[ thread_position_in_grid ]],
                                  constant uint& M [[ buffer(3) ]],
                                  constant uint& N [[ buffer(4) ]],
                                  constant uint& K [[ buffer(5) ]]) {
    uint row = gid.x;
    uint col = gid.y;
    if (row < M && col < N) {
        float sum = 0.0;
        for (uint i = 0; i < K; ++i) {
            sum += inA[row * K + i] * inB[i * N + col];
        }
        result[row * N + col] = sum;
    }
}

