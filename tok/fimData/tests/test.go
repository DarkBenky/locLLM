package main

func add(a int, b int) int {
	return a + b
}

type Calculator struct {
	base int
}

func (c Calculator) multiply(x int, y int) int {
	return c.base * x * y
}
