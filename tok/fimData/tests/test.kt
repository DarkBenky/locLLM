fun add(a: Int, b: Int): Int {
    return a + b
}

class Calculator {
    fun multiply(x: Int, y: Int): Int {
        return x * y
    }
}

data class Point(val x: Int, val y: Int)

interface Shape {
    fun area(): Double
}
