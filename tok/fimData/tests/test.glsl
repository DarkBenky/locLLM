#version 450

layout(location = 0) in vec3 aPos;
out vec4 fragColor;

float add(float a, float b) {
    return a + b;
}

struct Light {
    vec3 pos;
};

void main() {
    gl_Position = vec4(aPos, 1.0);
    fragColor = vec4(1.0);
}
