__global__ void vecAdd(const float* a, const float* b, float* c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__device__ float sq(float x) {
    return x * x;
}

struct Vec3 {
    float x, y, z;
};
