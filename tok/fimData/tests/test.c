#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

struct Point {
    int x;
    int y;
};

static void print_point(struct Point p) {
    printf("%d %d\n", p.x, p.y);
}
