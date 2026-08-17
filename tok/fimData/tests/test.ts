export function add(a: number, b: number): number {
    return a + b;
}

export class Calculator {
    multiply(x: number, y: number): number {
        return x * y;
    }
}

interface Shape {
    area(): number;
}
