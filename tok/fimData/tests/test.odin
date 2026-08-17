add :: proc(a, b: int) -> int {
    return a + b
}

Vec3 :: struct {
    x, y, z: f32,
}

main :: proc() {
    x := add(1, 2)
}
