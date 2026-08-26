import argparse
import json
import os
import sqlite3
import sys
import tempfile
from hashlib import sha256
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import CodeDB  # noqa: E402
from dedup import compute_norm_hash, normalize_code, DedupIndex  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}")


def test_normalization():
    print("[normalization]")
    a = "def foo(x):\n    return x + 1\n"
    b = "def foo(x):  # hello world\n    return x + 1\n"
    c = "def bar(y):\n    # renamed + comment\n    return y + 1\n"
    d = "def foo(x):\n    return x * 2\n"
    check("comment-only variant same hash", compute_norm_hash(a, "python") == compute_norm_hash(b, "python"))
    check("rename variant same hash", compute_norm_hash(a, "python") == compute_norm_hash(c, "python"))
    check("different body different hash", compute_norm_hash(a, "python") != compute_norm_hash(d, "python"))

    s1 = 'print("hello world", 42)'
    s2 = 'print("totally different", 999)'
    check("string/number literals masked", compute_norm_hash(s1, "python") == compute_norm_hash(s2, "python"))

    r1 = "f <- function(x) { x + 1 }"
    r2 = "f <- function(x) { x + 1 }  # comment"
    check("fallback (r) comment variant", compute_norm_hash(r1, "r") == compute_norm_hash(r2, "r"))

    py = "def add(a, b):\n    return a + b\n"
    js = "function add(a, b) {\n    return a + b;\n}"
    check("different lang different hash", compute_norm_hash(py, "python") != compute_norm_hash(js, "javascript"))

    print("  sample norm:", normalize_code(a, "python"))


def test_db(tmp):
    print("[db]")
    path = os.path.join(tmp, "t.db")
    db = CodeDB(path, dim=16)
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info(items)")}
    check("norm_hash column exists", "norm_hash" in cols)

    emb1 = np.ones(16, dtype=np.float32)
    emb2 = np.full(16, 2.0, dtype=np.float32)
    emb3 = np.full(16, 3.0, dtype=np.float32)
    r = db.add_batch_dedup(
        [
            ("code1", "python", sha256(b"code1").hexdigest(), emb1, "nh1"),
            ("code2", "python", sha256(b"code2").hexdigest(), emb2, "nh1"),
        ]
    )
    check("norm dup skipped in batch", r["added"] == 1 and r["dup_norm"] == 1)
    check("dropped reason norm", r["dropped"] == [("python", "norm")])
    check("exists_norm", db.exists_norm("nh1") and not db.exists_norm("nope"))
    r = db.add_batch_dedup(
        [("code1b", "python", sha256(b"code1").hexdigest(), emb3, "nh3")]
    )
    check("exact hash dup counted", r["dup_hash"] == 1 and r["added"] == 0)
    rows = list(db.iter_all())
    check("iter_all one row", len(rows) == 1 and rows[0][0] == 1)
    check("norm-hash pre-check against DB", db.exists_norm("nh1"))

    path2 = os.path.join(tmp, "old.db")
    c0 = sqlite3.connect(path2)
    c0.execute("CREATE TABLE items (hash TEXT PRIMARY KEY, code TEXT, lang TEXT)")
    c0.execute("INSERT INTO items (hash, code, lang) VALUES ('h', 'c', 'l')")
    c0.commit()
    c0.close()
    db2 = CodeDB(path2, dim=16)
    cols2 = {r[1] for r in db2.conn.execute("PRAGMA table_info(items)")}
    check("old DB migrated", "norm_hash" in cols2 and db2.count() == 1)
    db2.close()
    db.close()

    nidx = os.path.join(tmp, "noidx.db")
    dn = CodeDB(nidx, dim=16, create_index=False)
    idxs = [r[1] for r in dn.conn.execute("PRAGMA index_list(items)")]
    check("index deferred", "idx_items_norm_hash" not in idxs)
    dn.conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_items_norm_hash ON items(norm_hash)"
    )
    dn.add_batch_dedup([("c", "python", "h", np.ones(16, dtype=np.float32), "n")])
    check("exists_norm after deferred index", dn.exists_norm("n"))
    dn.close()


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n


def test_dedup_index(tmp):
    print("[DedupIndex]")
    rng = np.random.default_rng(7)
    dim = 16
    path = os.path.join(tmp, "idx.db")
    db = CodeDB(path, dim=dim)

    base = _unit(rng.standard_normal(dim))
    perp = _unit(rng.standard_normal(dim))
    v1 = base
    v2 = _unit(base + 0.03 * perp)
    v3 = _unit(rng.standard_normal(dim))
    v4 = _unit(rng.standard_normal(dim))
    v5 = _unit(rng.standard_normal(dim))

    for i, (vec, lang) in enumerate(
        [(v1, "python"), (v2, "python"), (v3, "python"), (v4, "rust")], start=1
    ):
        cur = db.conn.execute(
            "INSERT OR IGNORE INTO items (hash, code, lang, norm_hash) VALUES (?,?,?,?)",
            (f"h{i}", f"code{i}", lang, f"nh{i}"),
        )
        db.conn.execute(
            "INSERT INTO vec_items (rowid, embedding) VALUES (?,?)",
            (cur.lastrowid, db._pack(vec)),
        )
    db.conn.commit()

    idx = DedupIndex(db.conn, dim=dim, threshold=0.95, cand_k=8, device=1)
    check("size", idx.size == 4)

    flags = idx.check_batch(np.stack([v1, v5, v1]), ["python", "python", "rust"])
    check("v1 dup (matches v2)", bool(flags[0]))
    check("v5 not dup (not in index)", not bool(flags[1]))
    check("cross-lang not dup", not bool(flags[2]))

    flags = idx.check_batch(np.stack([v2 * 7.0]), ["python"])
    check("unnormalized query still dup", bool(flags[0]))

    path2 = os.path.join(tmp, "idx2.db")
    db2 = CodeDB(path2, dim=dim)
    idx2 = DedupIndex(db2.conn, dim=dim, threshold=0.95, cand_k=8, device=1)
    check("empty index no dup", not bool(idx2.check_batch(v1, "python")))
    idx2.add_many([v1], ["python"])
    check("new buffer dup", bool(idx2.check_batch(v2, "python")))
    check("new buffer cross-lang", not bool(idx2.check_batch(v2, "rust")))
    db2.close()
    db.close()


def test_layer1_rebuild(tmp):
    print("[layer1 + rebuild]")
    dim = 16
    rng = np.random.default_rng(3)
    src = os.path.join(tmp, "src.db")
    db = CodeDB(src, dim=dim)

    texts = [
        ("def foo(x):\n    return x + 1\n", "python"),
        ("def foo(x):  # variant\n    return x + 1\n", "python"),
        ("fn add(a: i32) -> i32 { a }", "rust"),
        ("fn add(a: i32) -> i32 { a }  // dup", "rust"),
        ("x", "python"),
    ]
    for i, (code, lang) in enumerate(texts, start=1):
        db.add(code, lang, sha256(code.encode()).hexdigest(), rng.standard_normal(dim).astype(np.float32))
    check("5 rows inserted", db.count() == 5)

    import compact

    n = db.count()
    stage1, surv_nh, stats = compact.layer1(src, n, min_len=20, workers=1)
    check("stage1 = 2 (one per group, tiny dropped)", len(stage1) == 2)
    check("norm dups = 2", stats["dup_norm_total"] == 2)
    check("small dropped = 1", sum(stats["dropped_small"].values()) == 1)
    check("kept longest", stats["lens"][stage1[0]] >= 20)

    dst = os.path.join(tmp, "dst.db")
    added, items_ck, vec_ck = compact.rebuild(
        db.conn, dst, dim, np.asarray(stage1, dtype=np.int64), surv_nh
    )
    check("rebuild added 2", added == 2)
    check("dense items rowids", items_ck[0] == items_ck[1] == 2)
    check("dense vec rowids", vec_ck[0] == vec_ck[1] == 2)

    db2 = CodeDB(dst, dim=dim)
    check("dst count", db2.count() == 2)
    idx = DedupIndex(db2.conn, dim=dim, threshold=0.95, cand_k=8, device=1)
    check("dedup index loads on rebuilt DB", idx.size == 2)
    db2.close()
    db.close()


def test_layer2(tmp):
    print("[layer2 (GPU)]")
    import torch

    if not torch.cuda.is_available():
        print("  skipped (no CUDA)")
        return
    dim = 16
    rng = np.random.default_rng(5)
    src = os.path.join(tmp, "l2.db")
    db = CodeDB(src, dim=dim)
    base = _unit(rng.standard_normal(dim))
    perp = _unit(rng.standard_normal(dim))
    vecs = [
        base,
        _unit(base + 0.02 * perp),
        _unit(rng.standard_normal(dim)),
        _unit(rng.standard_normal(dim)),
    ]
    texts = [
        "def a():\n    return 1\n",
        "x = [1, 2, 3]",
        "print('hi')",
        "x",
    ]
    for i, (text, vec, lang) in enumerate(
        zip(texts, vecs, ["python", "python", "python", "python"]), start=1
    ):
        db.add(text, lang, f"h{i}", vec.astype(np.float32))
    n = db.count()
    import compact

    stage1, surv_nh, stats = compact.layer1(src, n, min_len=5, workers=1)
    check("layer2 stage1 = 3 (tiny dropped)", len(stage1) == 3)
    args = argparse.Namespace(
        dst=os.path.join(tmp, "l2out.db"),
        gpu=1,
        target_cluster=1000,
        iters=5,
        seed=0,
        cos=0.95,
        max_cluster=4000,
    )
    final, removed_l2 = compact.layer2(
        db.conn, n, dim, stage1, stats, args, os.path.join(tmp, "l2out.surv16.tmp")
    )
    check("layer2 merged near dup (3->2)", len(final) == 2)
    check("layer2 removed 1", sum(removed_l2.values()) == 1)
    db.close()


def test_gpu_topk_multichunk():
    print("[gpu topk multi-chunk merge]")
    import torch

    if not torch.cuda.is_available():
        print("  skipped (no CUDA)")
        return
    rng = np.random.default_rng(11)
    dim = 16
    n = 300000
    chunk = 100000
    X = rng.standard_normal((n, dim)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    q = _unit(rng.standard_normal((3, dim)))

    idx = DedupIndex.__new__(DedupIndex)
    idx.dim = dim
    idx.cand_k = 8
    idx.chunk = chunk
    idx.n = n
    idx._torch = torch
    idx.device = torch.device("cuda:1")
    idx._pinned = torch.empty((n, dim), dtype=torch.float16, pin_memory=True)
    idx._pinned.copy_(torch.from_numpy(X.astype(np.float16)))
    idx._resident = None
    idx._new_count = 0

    gi, gv = idx._gpu_topk(q.astype(np.float16))
    s = X @ q.T
    ref_idx = np.argpartition(-s, 8, axis=0)[:8]
    check("gpu topk sets match", np.array_equal(np.sort(gi, axis=0), np.sort(ref_idx, axis=0)))
    check("gpu topk values close", np.abs(gv - np.take_along_axis(s, gi, axis=0)).max() < 1e-2)
    check("gpu topk max per col", np.allclose(gv.max(axis=0), s.max(axis=0), atol=1e-2))


def test_stream_resume(tmp):
    print("[stream resume]")
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_dataset

    paths = []
    for shard, vals in enumerate(
        [list(range(2500)), list(range(2500, 5000)), list(range(5000, 7500))]
    ):
        p = os.path.join(tmp, f"part-{shard:05d}.parquet")
        pq.write_table(pa.table({"a": vals}), p)
        paths.append(p)

    ds = load_dataset("parquet", data_files=paths, split="train", streaming=True)
    it = iter(ds)
    for _ in range(3200):
        next(it)
    state = ds.state_dict()
    check("resume state json-safe", json.dumps(state) is not None)

    ds2 = load_dataset("parquet", data_files=paths, split="train", streaming=True)
    ds2.load_state_dict(state)
    it2 = iter(ds2)
    cont = [next(it2)["a"] for _ in range(5)]
    check("resume continues exactly", cont == [3200, 3201, 3202, 3203, 3204])


if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="dedup_test_")
    print(f"tmp: {tmp}")
    test_normalization()
    test_db(tmp)
    test_dedup_index(tmp)
    test_layer1_rebuild(tmp)
    test_layer2(tmp)
    test_gpu_topk_multichunk()
    test_stream_resume(tmp)
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
