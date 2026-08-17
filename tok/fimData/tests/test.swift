func add(_ a: Int, _ b: Int) -> Int {
    return a + b
}

class Calculator {
    func multiply(_ x: Int, _ y: Int) -> Int {
        return x * y
    }
}

struct Point {
    var x: Int
    var y: Int
}

protocol Shape {
    func area() -> Double
}
