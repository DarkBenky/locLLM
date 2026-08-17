@compute @workgroup_size(8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
}

struct Vertex {
    pos : vec3<f32>,
}

fn add(a : f32, b : f32) -> f32 {
    return a + b;
}
