__kernel void vec_add(__global const float* a, __global const float* b, __global float* c) {
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}

float sq(float x) {
    return x * x;
}

struct Vec3 {
    float x, y, z;
};
