"""Synthetic eval cases for locLLM (data server not required) + small persist helpers.

Deterministic, hand-written code samples in the target languages (go / c /
python / opencl) used in both the plain-LM and FIM eval sets. They are
independent of the data server so eval is always comparable run-to-run.

Escaping note: entries use raw triple-quoted strings (r'''...''') so that
escapes like \\n inside target-language string literals stay intact.
"""
import json
import os

# synth cases: (lang, code)
SYNTH_EVAL_CASES = [
    ("go", r'''package main

import (
    "fmt"
    "sync"
    "time"
)

type Task struct {
    ID   int
    Data string
}

type Result struct {
    TaskID int
    Output string
    Err    error
}

func worker(id int, jobs <-chan Task, results chan<- Result, wg *sync.WaitGroup) {
    defer wg.Done()
    for job := range jobs {
        time.Sleep(10 * time.Millisecond)
        if job.Data == "" {
            results <- Result{TaskID: job.ID, Err: fmt.Errorf("empty data")}
            continue
        }
        results <- Result{TaskID: job.ID, Output: fmt.Sprintf("worker-%d:%s", id, job.Data)}
    }
}

func main() {
    const numWorkers = 4
    jobs := make(chan Task, 10)
    results := make(chan Result, 10)
    var wg sync.WaitGroup
    for w := 1; w <= numWorkers; w++ {
        wg.Add(1)
        go worker(w, jobs, results, &wg)
    }
    for i := 1; i <= 10; i++ {
        jobs <- Task{ID: i, Data: fmt.Sprintf("task-%d", i)}
    }
    close(jobs)
    go func() {
        wg.Wait()
        close(results)
    }()
    for r := range results {
        if r.Err != nil {
            fmt.Printf("task %d failed: %v\n", r.TaskID, r.Err)
            continue
        }
        fmt.Printf("task %d -> %s\n", r.TaskID, r.Output)
    }
}
'''),
    ("go", r'''package main

import (
    "encoding/json"
    "errors"
    "fmt"
    "time"
)

type User struct {
    ID        int       `json:"id"`
    Name      string    `json:"name"`
    CreatedAt time.Time `json:"created_at"`
    Tags      []string  `json:"tags,omitempty"`
}

func (u *User) Validate() error {
    if u.ID <= 0 {
        return errors.New("id must be positive")
    }
    if u.Name == "" {
        return fmt.Errorf("user %d: name is required", u.ID)
    }
    return nil
}

func parseUser(data []byte) (*User, error) {
    var u User
    if err := json.Unmarshal(data, &u); err != nil {
        return nil, fmt.Errorf("parse user: %w", err)
    }
    return &u, nil
}

func main() {
    raw := `{"id": 42, "name": "alice", "tags": ["admin", "beta"]}`
    u, err := parseUser([]byte(raw))
    if err != nil {
        fmt.Println("error:", err)
        return
    }
    if err := u.Validate(); err != nil {
        fmt.Println("validation:", err)
        return
    }
    out, _ := json.MarshalIndent(u, "", "  ")
    fmt.Println(string(out))
}
'''),
    ("go", r'''package main

import (
    "encoding/json"
    "log"
    "net/http"
    "strings"
    "time"
)

func logging(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        next.ServeHTTP(w, r)
        log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
    })
}

type apiResponse struct {
    OK    bool        `json:"ok"`
    Data  interface{} `json:"data,omitempty"`
    Error string      `json:"error,omitempty"`
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    _ = json.NewEncoder(w).Encode(v)
}

func main() {
    mux := http.NewServeMux()
    mux.HandleFunc("/api/health", func(w http.ResponseWriter, r *http.Request) {
        writeJSON(w, http.StatusOK, apiResponse{OK: true})
    })
    mux.HandleFunc("/api/echo", func(w http.ResponseWriter, r *http.Request) {
        msg := strings.TrimSpace(r.URL.Query().Get("msg"))
        writeJSON(w, http.StatusOK, apiResponse{OK: true, Data: map[string]string{"msg": msg}})
    })
    log.Println("listening on :8080")
    if err := http.ListenAndServe(":8080", logging(mux)); err != nil {
        log.Fatal(err)
    }
}
'''),
    ("go", r'''package main

import (
    "fmt"
    "sort"
)

type Item struct {
    Name  string
    Price float64
}

func main() {
    items := []Item{
        {"apple", 1.2},
        {"banana", 0.8},
        {"cherry", 3.1},
        {"date", 2.4},
    }
    sort.Slice(items, func(i, j int) bool {
        if items[i].Price == items[j].Price {
            return items[i].Name < items[j].Name
        }
        return items[i].Price < items[j].Price
    })
    for _, it := range items {
        fmt.Printf("%-10s %.2f\n", it.Name, it.Price)
    }
    total := 0.0
    for _, it := range items {
        total += it.Price
    }
    fmt.Printf("total: %.2f (n=%d)\n", total, len(items))
}
'''),
    ("go", r'''package main

import (
    "context"
    "fmt"
    "io"
    "net/http"
    "time"
)

func fetchWithRetry(ctx context.Context, client *http.Client, url string, attempts int) ([]byte, error) {
    var lastErr error
    for i := 0; i < attempts; i++ {
        req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
        if err != nil {
            return nil, err
        }
        resp, err := client.Do(req)
        if err != nil {
            lastErr = err
            select {
            case <-ctx.Done():
                return nil, ctx.Err()
            case <-time.After(time.Duration(i+1) * 200 * time.Millisecond):
            }
            continue
        }
        body, err := io.ReadAll(resp.Body)
        resp.Body.Close()
        if err != nil {
            lastErr = err
            continue
        }
        if resp.StatusCode >= 400 {
            lastErr = fmt.Errorf("status %d: %s", resp.StatusCode, url)
            continue
        }
        return body, nil
    }
    return nil, fmt.Errorf("fetch failed after %d attempts: %w", attempts, lastErr)
}

func main() {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    client := &http.Client{Timeout: 2 * time.Second}
    body, err := fetchWithRetry(ctx, client, "https://example.com", 3)
    if err != nil {
        fmt.Println("error:", err)
        return
    }
    fmt.Printf("downloaded %d bytes\n", len(body))
}
'''),
    ("go", r'''package main

import (
    "fmt"
    "sort"
)

func lowerBound(nums []int, target int) int {
    lo, hi := 0, len(nums)
    for lo < hi {
        mid := lo + (hi-lo)/2
        if nums[mid] < target {
            lo = mid + 1
        } else {
            hi = mid
        }
    }
    return lo
}

func upperBound(nums []int, target int) int {
    lo, hi := 0, len(nums)
    for lo < hi {
        mid := lo + (hi-lo)/2
        if nums[mid] <= target {
            lo = mid + 1
        } else {
            hi = mid
        }
    }
    return lo
}

func countRange(nums []int, lo, hi int) int {
    return upperBound(nums, hi) - lowerBound(nums, lo)
}

func main() {
    nums := []int{1, 3, 3, 5, 7, 9, 9, 11}
    sort.Ints(nums)
    fmt.Println("count in [3, 9]:", countRange(nums, 3, 9))
    fmt.Println("index of 7:", lowerBound(nums, 7))
}
'''),
    ("c", r'''#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int value;
    struct Node *next;
} Node;

Node *node_new(int value) {
    Node *n = (Node *)malloc(sizeof(Node));
    if (n == NULL) {
        return NULL;
    }
    n->value = value;
    n->next = NULL;
    return n;
}

void list_push(Node **head, int value) {
    Node *n = node_new(value);
    if (n == NULL) {
        return;
    }
    n->next = *head;
    *head = n;
}

int list_remove(Node **head, int value) {
    Node *prev = NULL;
    Node *cur = *head;
    while (cur != NULL) {
        if (cur->value == value) {
            if (prev == NULL) {
                *head = cur->next;
            } else {
                prev->next = cur->next;
            }
            free(cur);
            return 1;
        }
        prev = cur;
        cur = cur->next;
    }
    return 0;
}

void list_free(Node *head) {
    while (head != NULL) {
        Node *next = head->next;
        free(head);
        head = next;
    }
}

int main(void) {
    Node *head = NULL;
    for (int i = 0; i < 10; i++) {
        list_push(&head, i * i);
    }
    list_remove(&head, 25);
    for (Node *cur = head; cur != NULL; cur = cur->next) {
        printf("%d ", cur->value);
    }
    printf("\n");
    list_free(head);
    return 0;
}
'''),
    ("c", r'''#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int *data;
    size_t len;
    size_t cap;
} IntVec;

int vec_init(IntVec *v) {
    v->data = NULL;
    v->len = 0;
    v->cap = 0;
    return 0;
}

int vec_push(IntVec *v, int x) {
    if (v->len == v->cap) {
        size_t newcap = (v->cap == 0) ? 4 : v->cap * 2;
        int *nd = (int *)realloc(v->data, newcap * sizeof(int));
        if (nd == NULL) {
            return -1;
        }
        v->data = nd;
        v->cap = newcap;
    }
    v->data[v->len++] = x;
    return 0;
}

int cmp_int(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return (ia > ib) - (ia < ib);
}

int main(void) {
    IntVec v;
    vec_init(&v);
    int values[] = {5, 3, 9, 1, 7, 2, 8};
    size_t n = sizeof(values) / sizeof(values[0]);
    for (size_t i = 0; i < n; i++) {
        vec_push(&v, values[i]);
    }
    qsort(v.data, v.len, sizeof(int), cmp_int);
    for (size_t i = 0; i < v.len; i++) {
        printf("%d ", v.data[i]);
    }
    printf("\n");
    free(v.data);
    return 0;
}
'''),
    ("c", r'''#include <stdio.h>
#include <string.h>
#include <ctype.h>

size_t str_reverse(char *s, size_t len) {
    for (size_t i = 0, j = len - 1; i < j; i++, j--) {
        char tmp = s[i];
        s[i] = s[j];
        s[j] = tmp;
    }
    return len;
}

int is_palindrome(const char *s) {
    size_t len = strlen(s);
    for (size_t i = 0; i < len / 2; i++) {
        if (tolower((unsigned char)s[i]) != tolower((unsigned char)s[len - 1 - i])) {
            return 0;
        }
    }
    return 1;
}

int word_count(const char *s) {
    int count = 0;
    int in_word = 0;
    for (const char *p = s; *p; p++) {
        if (isspace((unsigned char)*p)) {
            in_word = 0;
        } else if (!in_word) {
            in_word = 1;
            count++;
        }
    }
    return count;
}

int main(void) {
    char buf[] = "abcdef";
    str_reverse(buf, strlen(buf));
    printf("%s\n", buf);
    printf("palindrome: %d\n", is_palindrome("Racecar"));
    printf("words: %d\n", word_count("hello brave new world"));
    return 0;
}
'''),
    ("c", r'''#include <stdio.h>
#include <stdlib.h>

void matmul(int m, int n, int k,
            const double *a, const double *b, double *c) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int p = 0; p < k; p++) {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

int main(void) {
    const int M = 3, N = 3, K = 4;
    double a[M * K] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    double b[K * N] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
    double c[M * N] = {0};
    matmul(M, N, K, a, b, c);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%8.2f", c[i * N + j]);
        }
        printf("\n");
    }
    return 0;
}
'''),
    ("c", r'''#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HT_SIZE 64

typedef struct Entry {
    char *key;
    int value;
    struct Entry *next;
} Entry;

typedef struct {
    Entry *buckets[HT_SIZE];
} HashTable;

unsigned int hash_key(const char *key) {
    unsigned int h = 5381;
    while (*key) {
        h = (h * 33) ^ (unsigned char)(*key++);
    }
    return h % HT_SIZE;
}

void ht_put(HashTable *ht, const char *key, int value) {
    unsigned int idx = hash_key(key);
    Entry *e = (Entry *)malloc(sizeof(Entry));
    e->key = strdup(key);
    e->value = value;
    e->next = ht->buckets[idx];
    ht->buckets[idx] = e;
}

int ht_get(HashTable *ht, const char *key, int *out) {
    unsigned int idx = hash_key(key);
    for (Entry *e = ht->buckets[idx]; e != NULL; e = e->next) {
        if (strcmp(e->key, key) == 0) {
            *out = e->value;
            return 1;
        }
    }
    return 0;
}

void ht_free(HashTable *ht) {
    for (int i = 0; i < HT_SIZE; i++) {
        Entry *e = ht->buckets[i];
        while (e != NULL) {
            Entry *next = e->next;
            free(e->key);
            free(e);
            e = next;
        }
    }
}

int main(void) {
    HashTable ht = {{0}};
    ht_put(&ht, "apple", 10);
    ht_put(&ht, "banana", 20);
    ht_put(&ht, "cherry", 30);
    int v;
    if (ht_get(&ht, "banana", &v)) {
        printf("banana -> %d\n", v);
    }
    if (!ht_get(&ht, "durian", &v)) {
        printf("durian not found\n");
    }
    ht_free(&ht);
    return 0;
}
'''),
    ("c", r'''#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int *data;
    int top;
    int cap;
} Stack;

int st_init(Stack *s, int cap) {
    s->data = (int *)malloc(cap * sizeof(int));
    if (s->data == NULL) return -1;
    s->top = 0;
    s->cap = cap;
    return 0;
}

int st_push(Stack *s, int v) {
    if (s->top == s->cap) {
        int newcap = s->cap * 2;
        int *nd = (int *)realloc(s->data, newcap * sizeof(int));
        if (nd == NULL) return -1;
        s->data = nd;
        s->cap = newcap;
    }
    s->data[s->top++] = v;
    return 0;
}

int st_pop(Stack *s, int *out) {
    if (s->top == 0) return 0;
    *out = s->data[--s->top];
    return 1;
}

int is_balanced(const char *expr) {
    Stack s;
    if (st_init(&s, 16) != 0) return -1;
    for (const char *p = expr; *p; p++) {
        if (*p == '(') {
            st_push(&s, 1);
        } else if (*p == ')') {
            int v;
            if (!st_pop(&s, &v)) {
                free(s.data);
                return 0;
            }
        }
    }
    int ok = (s.top == 0);
    free(s.data);
    return ok;
}

int main(void) {
    printf("balanced: %d\n", is_balanced("(a + (b * c))"));
    printf("balanced: %d\n", is_balanced("((a + b)"));
    return 0;
}
'''),
    ("python", r'''from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Account:
    owner: str
    balance: float = 0.0
    transactions: list[float] = field(default_factory=list)

    def deposit(self, amount: float) -> bool:
        if amount <= 0:
            raise ValueError("amount must be positive")
        self.balance += amount
        self.transactions.append(amount)
        return True

    def withdraw(self, amount: float) -> bool:
        if amount <= 0:
            raise ValueError("amount must be positive")
        if amount > self.balance:
            return False
        self.balance -= amount
        self.transactions.append(-amount)
        return True

    def summary(self) -> str:
        return f"{self.owner}: {self.balance:.2f} ({len(self.transactions)} txns)"


def main() -> None:
    acc = Account("alice", 100.0)
    acc.deposit(50.0)
    acc.withdraw(25.0)
    print(acc.summary())
    try:
        acc.withdraw(9999.0)
    except ValueError as exc:
        print(f"error: {exc}")


if __name__ == "__main__":
    main()
'''),
    ("python", r'''import asyncio
import random
import time

async def fetch(url: str, delay: float = 0.2) -> tuple[str, float]:
    await asyncio.sleep(delay)
    return url, random.random()


async def fetch_all(urls: list[str]) -> dict[str, float]:
    tasks = [asyncio.create_task(fetch(u)) for u in urls]
    results: dict[str, float] = {}
    for task in asyncio.as_completed(tasks):
        url, score = await task
        results[url] = score
    return results


async def main() -> None:
    t0 = time.perf_counter()
    urls = [f"https://example.com/{i}" for i in range(10)]
    results = await fetch_all(urls)
    elapsed = time.perf_counter() - t0
    print(f"fetched {len(results)} urls in {elapsed:.2f}s")
    for url, score in sorted(results.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {url}: {score:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
'''),
    ("python", r'''import argparse
import csv
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process a CSV of records")
    parser.add_argument("input", type=Path, help="input CSV file")
    parser.add_argument("--output", type=Path, default=Path("out.csv"))
    parser.add_argument("--limit", type=int, default=0, help="max rows (0 = all)")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def load_rows(path: Path, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            rows.append(row)
    logger.info("loaded %d rows from %s", len(rows), path)
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    if not args.input.exists():
        logger.error("input not found: %s", args.input)
        return 1
    rows = load_rows(args.input, args.limit)
    print(f"read {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''),
    ("python", r'''from contextlib import contextmanager
from typing import Iterator

@contextmanager
def timer(label: str) -> Iterator[None]:
    import time
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[{label}] {elapsed * 1000:.1f} ms")


def chunked(data: list[int], size: int) -> Iterator[list[int]]:
    if size <= 0:
        raise ValueError("size must be positive")
    for i in range(0, len(data), size):
        yield data[i:i + size]


def fibonacci(n: int) -> Iterator[int]:
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b


def main() -> None:
    nums = list(range(1, 21))
    with timer("chunked sum"):
        for chunk in chunked(nums, 4):
            print(chunk, sum(chunk))
    print(list(fibonacci(10)))


if __name__ == "__main__":
    main()
'''),
    ("python", r'''from __future__ import annotations
import re
import sys
from collections import Counter
from pathlib import Path

WORD_RE = re.compile(r"[A-Za-z']+")


def count_words(path: Path, top: int = 10) -> Counter[str]:
    counts: Counter[str] = Counter()
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            for word in WORD_RE.findall(line):
                counts[word.lower()] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    if not args:
        print("usage: wordcount.py <file> [top]", file=sys.stderr)
        return 2
    path = Path(args[0])
    top = int(args[1]) if len(args) > 1 else 10
    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        return 1
    counts = count_words(path, top)
    for word, n in counts.most_common(top):
        print(f"{word:20s} {n:6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''),
    ("python", r'''from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def dist2(self, other: "Point") -> int:
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2


@lru_cache(maxsize=128)
def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


def k_nearest(points: list[Point], origin: Point, k: int) -> list[Point]:
    return sorted(points, key=lambda p: p.dist2(origin))[:k]


def main() -> None:
    pts = [Point(x, y) for x in range(-2, 3) for y in range(-2, 3)]
    near = k_nearest(pts, Point(0, 0), 3)
    for p in near:
        print(p)
    print([fib(i) for i in range(12)])


if __name__ == "__main__":
    main()
'''),
    ("opencl", r'''#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

const char *kernel_src =
    "__kernel void vec_add(__global const float *a,\n"
    "                     __global const float *b,\n"
    "                     __global float *c,\n"
    "                     const int n) {\n"
    "    int i = get_global_id(0);\n"
    "    if (i < n) {\n"
    "        c[i] = a[i] + b[i];\n"
    "    }\n"
    "}\n";

int main(void) {
    const int n = 1024;
    float *a = malloc(n * sizeof(float));
    float *b = malloc(n * sizeof(float));
    float *c = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }

    cl_int err;
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
    cl_program program = clCreateProgramWithSource(ctx, 1, &kernel_src, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    cl_mem d_a = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), a, &err);
    cl_mem d_b = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), b, &err);
    cl_mem d_c = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n * sizeof(float), NULL, &err);

    cl_kernel kernel = clCreateKernel(program, "vec_add", &err);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &n);

    size_t gws = n;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gws, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, n * sizeof(float), c, 0, NULL, NULL);

    float checksum = 0.0f;
    for (int i = 0; i < n; i++) {
        checksum += c[i];
    }
    printf("checksum = %.2f\n", checksum);

    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(a);
    free(b);
    free(c);
    return 0;
}
'''),
    ("opencl", r'''#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

const char *mm_kernel_src =
    "__kernel void matmul(const int M, const int N, const int K,\n"
    "                     __global const float *A,\n"
    "                     __global const float *B,\n"
    "                     __global float *C) {\n"
    "    int row = get_global_id(0);\n"
    "    int col = get_global_id(1);\n"
    "    if (row >= M || col >= N) return;\n"
    "    float sum = 0.0f;\n"
    "    for (int k = 0; k < K; k++) {\n"
    "        sum += A[row * K + k] * B[k * N + col];\n"
    "    }\n"
    "    C[row * N + col] = sum;\n"
    "}\n";

int main(void) {
    const int M = 64, N = 64, K = 64;
    float *A = malloc(M * K * sizeof(float));
    float *B = malloc(K * N * sizeof(float));
    float *C = malloc(M * N * sizeof(float));
    for (int i = 0; i < M * K; i++) A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) B[i] = 1.0f;

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
    cl_program program = clCreateProgramWithSource(ctx, 1, &mm_kernel_src, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, M * K * sizeof(float), A, &err);
    cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, K * N * sizeof(float), B, &err);
    cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, M * N * sizeof(float), NULL, &err);

    cl_kernel kernel = clCreateKernel(program, "matmul", &err);
    clSetKernelArg(kernel, 0, sizeof(cl_int), &M);
    clSetKernelArg(kernel, 1, sizeof(cl_int), &N);
    clSetKernelArg(kernel, 2, sizeof(cl_int), &K);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &dA);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &dB);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &dC);

    size_t gws[2] = {M, N};
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, gws, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, dC, CL_TRUE, 0, M * N * sizeof(float), C, 0, NULL, NULL);
    printf("C[0][0] = %f (expected %f)\n", C[0], (float)K);
    clReleaseMemObject(dA);
    clReleaseMemObject(dB);
    clReleaseMemObject(dC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(A);
    free(B);
    free(C);
    return 0;
}
'''),
    ("opencl", r'''#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

const char *reduce_src =
    "__kernel void reduce(const int n, __global const float *in,\n"
    "                     __global float *out) {\n"
    "    int tid = get_global_id(0);\n"
    "    int lid = get_local_id(0);\n"
    "    int lsize = get_local_size(0);\n"
    "    __local float scratch[256];\n"
    "    float val = 0.0f;\n"
    "    if (tid < n) val = in[tid];\n"
    "    scratch[lid] = val;\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    for (int s = lsize / 2; s > 0; s >>= 1) {\n"
    "        if (lid < s) scratch[lid] += scratch[lid + s];\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "    if (lid == 0) out[get_group_id(0)] = scratch[0];\n"
    "}\n";

int main(void) {
    const int n = 4096;
    float *in = malloc(n * sizeof(float));
    float *partials = malloc((n / 256 + 1) * sizeof(float));
    for (int i = 0; i < n; i++) in[i] = 1.0f;

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
    cl_program program = clCreateProgramWithSource(ctx, 1, &reduce_src, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    cl_mem dIn = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), in, &err);
    cl_mem dOut = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (n / 256 + 1) * sizeof(float), NULL, &err);

    cl_kernel kernel = clCreateKernel(program, "reduce", &err);
    clSetKernelArg(kernel, 0, sizeof(cl_int), &n);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &dIn);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &dOut);

    size_t lws = 256;
    size_t gws = ((n + lws - 1) / lws) * lws;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gws, &lws, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, dOut, CL_TRUE, 0, (n / 256 + 1) * sizeof(float), partials, 0, NULL, NULL);

    float total = 0.0f;
    for (int i = 0; i < n / 256 + 1; i++) total += partials[i];
    printf("sum = %f (expected %f)\n", total, (float)n);

    clReleaseMemObject(dIn);
    clReleaseMemObject(dOut);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(in);
    free(partials);
    return 0;
}
'''),
    ("opencl", r'''#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

const char *saxpy_src =
    "__kernel void saxpy(const float alpha,\n"
    "                    __global const float *x,\n"
    "                    __global const float *y,\n"
    "                    __global float *z,\n"
    "                    const int n) {\n"
    "    int i = get_global_id(0);\n"
    "    if (i < n) {\n"
    "        z[i] = alpha * x[i] + y[i];\n"
    "    }\n"
    "}\n";

int main(void) {
    const int n = 2048;
    const float alpha = 2.5f;
    float *x = malloc(n * sizeof(float));
    float *y = malloc(n * sizeof(float));
    float *z = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) { x[i] = (float)i; y[i] = 1.0f; }

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
    cl_program program = clCreateProgramWithSource(ctx, 1, &saxpy_src, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    cl_mem dX = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), x, &err);
    cl_mem dY = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), y, &err);
    cl_mem dZ = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n * sizeof(float), NULL, &err);

    cl_kernel kernel = clCreateKernel(program, "saxpy", &err);
    clSetKernelArg(kernel, 0, sizeof(cl_float), &alpha);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &dX);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &dY);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &dZ);
    clSetKernelArg(kernel, 4, sizeof(cl_int), &n);

    size_t gws = n;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gws, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, dZ, CL_TRUE, 0, n * sizeof(float), z, 0, NULL, NULL);
    printf("z[0] = %f (expected %f)\n", z[0], alpha * x[0] + y[0]);

    clReleaseMemObject(dX);
    clReleaseMemObject(dY);
    clReleaseMemObject(dZ);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(x);
    free(y);
    free(z);
    return 0;
}
'''),
    ("opencl", r'''#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

const char *conv_src =
    "__kernel void conv1d(const int n, const int k,\n"
    "                     __global const float *in,\n"
    "                     __global const float *mask,\n"
    "                     __global float *out) {\n"
    "    int i = get_global_id(0);\n"
    "    if (i < n) {\n"
    "        float sum = 0.0f;\n"
    "        for (int j = 0; j < k; j++) {\n"
    "            int idx = i + j - k / 2;\n"
    "            if (idx >= 0 && idx < n) {\n"
    "                sum += in[idx] * mask[j];\n"
    "            }\n"
    "        }\n"
    "        out[i] = sum;\n"
    "    }\n"
    "}\n";

int main(void) {
    const int n = 16, k = 3;
    float *in = malloc(n * sizeof(float));
    float *mask = malloc(k * sizeof(float));
    float *out = malloc(n * sizeof(float));
    float maskv[3] = {0.25f, 0.5f, 0.25f};
    for (int i = 0; i < n; i++) in[i] = (float)i;
    for (int j = 0; j < k; j++) mask[j] = maskv[j];

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
    cl_program program = clCreateProgramWithSource(ctx, 1, &conv_src, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    cl_mem dIn = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), in, &err);
    cl_mem dMask = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, k * sizeof(float), mask, &err);
    cl_mem dOut = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n * sizeof(float), NULL, &err);

    cl_kernel kernel = clCreateKernel(program, "conv1d", &err);
    clSetKernelArg(kernel, 0, sizeof(cl_int), &n);
    clSetKernelArg(kernel, 1, sizeof(cl_int), &k);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &dIn);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &dMask);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &dOut);

    size_t gws = n;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gws, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, dOut, CL_TRUE, 0, n * sizeof(float), out, 0, NULL, NULL);
    printf("out[0]=%.2f out[7]=%.2f\n", out[0], out[7]);

    clReleaseMemObject(dIn);
    clReleaseMemObject(dMask);
    clReleaseMemObject(dOut);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(in);
    free(mask);
    free(out);
    return 0;
}
'''),
    ("opencl", r'''#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

const char *hist_src =
    "__kernel void histogram(const int n, __global const uchar *data,\n"
    "                        __global int *hist) {\n"
    "    int i = get_global_id(0);\n"
    "    if (i < n) {\n"
    "        atomic_inc(&hist[data[i]]);\n"
    "    }\n"
    "}\n";

int main(void) {
    const int n = 1024;
    unsigned char *data = malloc(n);
    int *hist = calloc(256, sizeof(int));
    for (int i = 0; i < n; i++) data[i] = (unsigned char)(i % 256);

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
    cl_program program = clCreateProgramWithSource(ctx, 1, &hist_src, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    cl_mem dData = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n, data, &err);
    cl_mem dHist = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, 256 * sizeof(int), NULL, &err);

    cl_kernel kernel = clCreateKernel(program, "histogram", &err);
    clSetKernelArg(kernel, 0, sizeof(cl_int), &n);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &dData);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &dHist);

    size_t gws = n;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gws, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, dHist, CL_TRUE, 0, 256 * sizeof(int), hist, 0, NULL, NULL);
    int total = 0;
    for (int i = 0; i < 256; i++) total += hist[i];
    printf("total = %d (expected %d)\n", total, n);

    clReleaseMemObject(dData);
    clReleaseMemObject(dHist);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(data);
    free(hist);
    return 0;
}
'''),
]


def load_eval_items(path):
    """Load persisted eval items (list of dicts) or None if absent."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_eval_items(path, items):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(items, f)
    os.replace(tmp, path)
