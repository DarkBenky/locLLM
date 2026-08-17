fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub struct Calculator {
    base: i32,
}

impl Calculator {
    pub fn multiply(&self, x: i32, y: i32) -> i32 {
        self.base * x * y
    }
}

enum Color {
    Red,
    Green,
    Blue,
}
