import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import CodeDB  # noqa: E402
from dedup import compute_norm_hash  # noqa: E402

_VEC_CHUNK = 1024
_CPU_BLOCK = 16384


def _t(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _range_worker(args):
    path, a, b = args
    import sqlite3

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    out = []
    try:
        cur = conn.execute(
            "SELECT rowid, code, lang FROM items WHERE rowid BETWEEN ? AND ?",
            (a, b),
        )
        for rowid, code, lang in cur:
            out.append((rowid, compute_norm_hash(code, lang), len(code), lang))
    finally:
        conn.close()
    return out


def layer1(path, n, min_len, workers):
    rows_per = 5000
    ranges = [
        (path, a, min(a + rows_per - 1, n))
        for a in range(1, n + 1, rows_per)
    ]
    lens = np.zeros(n + 1, dtype=np.int64)
    lang_ids = np.zeros(n + 1, dtype=np.int16)
    lang_rev = {}
    total_lang = Counter()
    dropped_small = Counter()
    groups = defaultdict(list)

    def merge(out):
        for rowid, nh, length, lang in out:
            lens[rowid] = length
            lid = lang_rev.setdefault(lang, len(lang_rev))
            lang_ids[rowid] = lid
            total_lang[lid] += 1
            if length < min_len:
                dropped_small[lid] += 1
            else:
                groups[nh].append(rowid)

    _t(f"layer 1: scanning {n} rows ({len(ranges)} ranges, {workers} workers)")
    done_ranges = 0
    if workers > 1:
        with Pool(workers) as pool:
            for out in pool.imap_unordered(
                _range_worker, ranges, chunksize=2
            ):
                merge(out)
                done_ranges += 1
                if done_ranges % 100 == 0:
                    _t(f"  layer1 {done_ranges}/{len(ranges)} ranges")
    else:
        for rng in ranges:
            merge(_range_worker(rng))
            done_ranges += 1
            if done_ranges % 100 == 0:
                _t(f"  layer1 {done_ranges}/{len(ranges)} ranges")


    stage1 = []
    surv_nh = {}
    removed_norm = Counter()
    dup_norm_total = 0
    for nh, rows in groups.items():
        best = max(rows, key=lambda r: (lens[r], -r))
        stage1.append(best)
        surv_nh[best] = nh
        for r in rows:
            if r != best:
                dup_norm_total += 1
                removed_norm[lang_ids[r]] += 1
    stage1.sort()
    stats = {
        "langs": lang_rev,
        "lang_ids": lang_ids,
        "lens": lens,
        "total_lang": total_lang,
        "dropped_small": dropped_small,
        "removed_norm": removed_norm,
        "dup_norm_total": dup_norm_total,
    }
    return stage1, surv_nh, stats


def _load_survivor_embeddings(conn, n, dim, stage1, mem_path):
    n1 = len(stage1)
    stage1_arr = np.asarray(stage1, dtype=np.int64)
    x_mem = np.memmap(mem_path, dtype=np.float16, mode="w+", shape=(n1, dim))
    i = 0
    for (blob,) in conn.execute(
        "SELECT vectors FROM vec_items_vector_chunks00 ORDER BY rowid"
    ):
        c = min(_VEC_CHUNK, n - i * _VEC_CHUNK)
        if c <= 0:
            break
        rows = np.arange(i * _VEC_CHUNK + 1, i * _VEC_CHUNK + 1 + c)
        pos = np.searchsorted(stage1_arr, rows)
        ok = pos < n1
        valid = ok & (stage1_arr[np.where(ok, pos, 0)] == rows)
        if valid.any():
            block = np.frombuffer(blob, dtype=np.float32).reshape(-1, dim)[:c][valid]
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            block /= norms
            x_mem[pos[valid]] = block.astype(np.float16)
        i += 1
    x_mem.flush()
    return x_mem, stage1_arr


def _kmeans(torch, dev, x_mem, n1, dim, k, iters, seed):
    rng = np.random.default_rng(seed)
    init = np.sort(rng.choice(n1, size=k, replace=False))
    c = torch.from_numpy(x_mem[init].astype(np.float32)).to(dev)
    c = torch.nn.functional.normalize(c, dim=1)
    for _ in range(iters):
        sums = torch.zeros((k, dim), dtype=torch.float32, device=dev)
        counts = torch.zeros(k, dtype=torch.float32, device=dev)
        for start in range(0, n1, _CPU_BLOCK):
            xc = torch.from_numpy(
                x_mem[start:start + _CPU_BLOCK].astype(np.float32)
            ).to(dev)
            a = (xc @ c.T).argmax(dim=1)
            sums.index_add_(0, a, xc)
            counts.index_add_(0, a, torch.ones(a.shape[0], device=dev))
        valid = counts > 0
        newc = sums / counts.clamp(min=1).unsqueeze(1)
        newc = torch.nn.functional.normalize(newc, dim=1)
        empty = (~valid).nonzero(as_tuple=False).flatten().cpu().numpy()
        if len(empty):
            fill = x_mem[
                np.sort(rng.choice(n1, size=len(empty), replace=False))
            ].astype(np.float32)
            newc[empty] = torch.from_numpy(fill).to(dev)
        c = newc
    return c


def _assign(torch, dev, x_mem, n1, c):
    clusters = defaultdict(list)
    for start in range(0, n1, _CPU_BLOCK):
        xc = torch.from_numpy(
            x_mem[start:start + _CPU_BLOCK].astype(np.float32)
        ).to(dev)
        a = (xc @ c.T).argmax(dim=1).cpu().numpy()
        gpos = np.arange(start, min(start + _CPU_BLOCK, n1))
        for pos, lab in zip(gpos, a):
            clusters[int(lab)].append(int(pos))
    return clusters


def _bisect(torch, dev, x_mem, members, rng, iters=5):
    idx = np.asarray(members, dtype=np.int64)
    xt = torch.from_numpy(x_mem[idx].astype(np.float32)).to(dev)
    pick = rng.choice(len(idx), size=2, replace=False)
    c = xt[pick].clone()
    c = torch.nn.functional.normalize(c, dim=1)
    for _ in range(iters):
        a = (xt @ c.T).argmax(dim=1)
        n0 = (a == 0).sum().item()
        n1 = (a == 1).sum().item()
        if n0 == 0 or n1 == 0:
            half = len(idx) // 2
            return list(members[:half]), list(members[half:])
        s0 = xt[a == 0].sum(dim=0) / n0
        s1 = xt[a == 1].sum(dim=0) / n1
        c = torch.nn.functional.normalize(torch.stack([s0, s1]), dim=1)
    a = (xt @ c.T).argmax(dim=1).cpu().numpy()
    left = [members[j] for j in np.nonzero(a == 0)[0]]
    right = [members[j] for j in np.nonzero(a == 1)[0]]
    return left, right


def _split_clusters(torch, dev, x_mem, clusters, max_cluster, rng):
    out = []
    for members in clusters.values():
        queue = [members]
        while queue:
            mem = queue.pop()
            if len(mem) <= max_cluster:
                out.append(mem)
                continue
            a, b = _bisect(torch, dev, x_mem, mem, rng)
            queue.append(a)
            queue.append(b)
    return out


def _pairwise_dedup(torch, dev, x_mem, members, lens_by_pos, cos_thresh):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    idx = np.asarray(members, dtype=np.int64)
    xt = torch.from_numpy(x_mem[idx].astype(np.float32)).to(dev)
    s = (xt @ xt.T).cpu().numpy()
    np.fill_diagonal(s, -1.0)
    rows, cols = np.nonzero(s >= cos_thresh)
    keep = rows < cols
    rows, cols = rows[keep], cols[keep]
    m = len(idx)
    graph = csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(m, m)
    )
    ncomp, labels = connected_components(graph, directed=False, return_labels=True)
    survivors = []
    removed = 0
    for comp in range(ncomp):
        mask = labels == comp
        member_positions = idx[mask]
        best = max(member_positions, key=lambda p: (lens_by_pos[p], -p))
        survivors.append(best)
        removed += len(member_positions) - 1
    return survivors, removed


def layer2(conn, n, dim, stage1, stats, args, mem_path):
    n1 = len(stage1)
    if n1 == 0:
        return np.asarray([], dtype=np.int64), Counter()
    try:
        import torch
    except Exception as e:
        raise RuntimeError(f"layer 2 needs torch ({e}); use --skip-semantic")

    if not torch.cuda.is_available():
        raise RuntimeError("layer 2 needs CUDA; use --skip-semantic")
    dev = torch.device(f"cuda:{args.gpu}")
    _t(f"layer 2: {n1} survivors, GPU {torch.cuda.get_device_name(dev)}")

    x_mem, stage1_arr = _load_survivor_embeddings(conn, n, dim, stage1, mem_path)
    try:
        lens_by_pos = stats["lens"][stage1_arr]
        k = max(2, min(n1 // args.target_cluster, 65536))
        k = min(k, n1)
        _t(f"k-means: k={k}, iters={args.iters}")
        centroids = _kmeans(torch, dev, x_mem, n1, dim, k, args.iters, args.seed)
        clusters = _assign(torch, dev, x_mem, n1, centroids)
        nonempty = {cid: m for cid, m in clusters.items() if m}
        clusters = nonempty
        _t(f"{len(clusters)} non-empty clusters")
        _t(f"splitting clusters > {args.max_cluster} ...")
        split = _split_clusters(
            torch, dev, x_mem, clusters, args.max_cluster, np.random.default_rng(args.seed + 1)
        )
        _t(f"{len(split)} clusters after split")
        final_positions = []
        removed_l2 = Counter()
        lang_ids = stats["lang_ids"]
        done = 0
        for members in split:
            survivors, removed = _pairwise_dedup(
                torch, dev, x_mem, members, lens_by_pos, args.cos
            )
            surv_set = set(survivors)
            for p in members:
                if p not in surv_set:
                    removed_l2[lang_ids[stage1_arr[p]]] += 1
            final_positions.extend(survivors)
            done += 1
            if done % 200 == 0:
                _t(f"  pairwise {done}/{len(split)} clusters")
        final_positions.sort()
        final_rowids = stage1_arr[final_positions]
        return final_rowids, removed_l2
    finally:
        del x_mem
        try:
            os.remove(mem_path)
        except OSError:
            pass


class _ChunkCache:
    def __init__(self, conn, dim, keep=3):
        self.conn = conn
        self.dim = dim
        self.keep = keep
        self.cache = {}

    def vec(self, rowid):
        cid = (rowid - 1) // _VEC_CHUNK
        if cid not in self.cache:
            blob = self.conn.execute(
                "SELECT vectors FROM vec_items_vector_chunks00 WHERE rowid=?",
                (cid + 1,),
            ).fetchone()[0]
            self.cache[cid] = np.frombuffer(blob, dtype=np.float32).reshape(-1, self.dim)
            while len(self.cache) > self.keep:
                del self.cache[min(self.cache)]
        off = (rowid - 1) % _VEC_CHUNK
        return self.cache[cid][off].copy()


def rebuild(conn, dst, dim, final_rowids, surv_nh):
    keep = np.zeros(final_rowids.max() + 1 if len(final_rowids) else 1, dtype=bool)
    keep[final_rowids] = True
    dst_db = CodeDB(dst, dim=dim, create_index=False)
    dst_db.conn.execute("PRAGMA synchronous=OFF")
    dst_db.conn.execute("PRAGMA journal_mode=OFF")
    dst_db.conn.execute("PRAGMA cache_size=-2000000")
    dst_db.conn.execute("PRAGMA temp_store=MEMORY")
    cache = _ChunkCache(conn, dim)
    added = 0
    cur = conn.execute("SELECT rowid, hash, code, lang FROM items ORDER BY rowid")
    while True:
        rows = cur.fetchmany(4096)
        if not rows:
            break
        for rowid, h, code, lang in rows:
            if rowid >= keep.shape[0] or not keep[rowid]:
                continue
            emb = cache.vec(rowid)
            nh = surv_nh.get(rowid)
            if nh is None:
                nh = compute_norm_hash(code, lang)
            c = dst_db.conn.execute(
                "INSERT OR IGNORE INTO items (hash, code, lang, norm_hash) VALUES (?,?,?,?)",
                (h, code, lang, nh),
            )
            if c.rowcount:
                dst_db.conn.execute(
                    "INSERT INTO vec_items (rowid, embedding) VALUES (?,?)",
                    (c.lastrowid, dst_db._pack(emb)),
                )
                added += 1
        dst_db.conn.commit()
    dst_db.conn.commit()
    dst_db.conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_items_norm_hash ON items(norm_hash)"
    )
    dst_db.conn.execute("VACUUM")
    dst_db.conn.commit()
    n_items, max_items = dst_db.conn.execute(
        "SELECT COUNT(*), MAX(rowid) FROM items"
    ).fetchone()
    n_vec, max_vec = dst_db.conn.execute(
        "SELECT COUNT(*), MAX(rowid) FROM vec_items"
    ).fetchone()
    return added, (n_items, max_items), (n_vec, max_vec)


def main():
    ap = argparse.ArgumentParser(description="Compact near-duplicate code chunks")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--stage", default=None,
                    help="build in this dir, then move to --dst (for fast SSDs)")
    ap.add_argument("--gpu", type=int, default=1, help="torch CUDA device (1=3060)")
    ap.add_argument("--workers", type=int, default=None, help="layer-1 processes")
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--cos", type=float, default=0.95)
    ap.add_argument("--target-cluster", type=int, default=1000)
    ap.add_argument("--max-cluster", type=int, default=4000)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=1536)
    ap.add_argument("--dry-run", action="store_true", help="layer 1 only, report")
    ap.add_argument("--skip-semantic", action="store_true", help="skip layer 2")
    ap.add_argument("--swap", action="store_true", help="replace src with dst")
    ap.add_argument("--overwrite", action="store_true", help="allow existing dst")
    args = ap.parse_args()

    import sqlite3

    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        ap.error("--src and --dst must differ")
    build_path = (
        os.path.join(args.stage, os.path.basename(args.dst))
        if args.stage
        else args.dst
    )
    if os.path.exists(args.dst) and not args.overwrite:
        ap.error(f"{args.dst} exists; use --overwrite to replace it")
    if args.overwrite:
        for p in (args.dst, args.dst + "-journal"):
            if os.path.exists(p):
                os.remove(p)
    if build_path != args.dst:
        if os.path.exists(build_path):
            if args.overwrite:
                os.remove(build_path)
            else:
                ap.error(f"{build_path} exists; use --overwrite")
        os.makedirs(args.stage, exist_ok=True)
    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)

    workers = args.workers or min(8, os.cpu_count() or 1)
    conn = sqlite3.connect(f"file:{args.src}?mode=ro", uri=True)
    n, max_rowid = conn.execute("SELECT COUNT(*), MAX(rowid) FROM items").fetchone()
    n = int(n or 0)
    if n != (max_rowid or 0):
        _t(f"warning: items rowids not dense (count={n}, max={max_rowid})")
    _t(f"source: {args.src} ({n} rows)")

    t0 = time.time()
    stage1, surv_nh, stats = layer1(args.src, n, args.min_len, workers)
    _t(
        f"layer 1 done in {time.time()-t0:.0f}s: "
        f"{n} -> {len(stage1)} (dropped_small={sum(stats['dropped_small'].values())}, "
        f"norm_dups={stats['dup_norm_total']})"
    )

    if args.dry_run:
        print_report(stats, stage1, None, Counter(), conn, args)
        conn.close()
        return

    if args.skip_semantic:
        final_rowids = np.asarray(stage1, dtype=np.int64)
        removed_l2 = Counter()
    else:
        t1 = time.time()
        final_rowids, removed_l2 = layer2(
            conn, n, args.dim, stage1, stats, args, build_path + ".surv16.tmp"
        )
        _t(f"layer 2 done in {time.time()-t1:.0f}s: "
           f"{len(stage1)} -> {len(final_rowids)} (semantic dups={sum(removed_l2.values())})")

    t2 = time.time()
    added, items_ck, vec_ck = rebuild(conn, build_path, args.dim, final_rowids, surv_nh)
    _t(f"rebuild done in {time.time()-t2:.0f}s: {added} rows written")

    assert added == len(final_rowids), (added, len(final_rowids))
    assert items_ck[0] == items_ck[1], f"items rowids not dense: {items_ck}"
    assert vec_ck[0] == vec_ck[1], f"vec rowids not dense: {vec_ck}"

    if build_path != args.dst:
        shutil.move(build_path, args.dst)
        _t(f"moved {build_path} -> {args.dst}")

    print_report(stats, stage1, final_rowids, removed_l2, conn, args)
    conn.close()

    if args.swap:
        bak = args.src + ".bak"
        if os.path.exists(bak):
            ap.error(f"{bak} already exists; remove it first")
        shutil.move(args.src, bak)
        shutil.move(args.dst, args.src)
        _t(f"swapped: {args.src} -> {bak}; {args.dst} -> {args.src}")


def print_report(stats, stage1, final_rowids, removed_l2, conn, args):
    langs = stats["langs"]
    name = {v: k for k, v in langs.items()}
    rows = []
    for lid in sorted(stats["total_lang"]):
        total = stats["total_lang"][lid]
        small = stats["dropped_small"][lid]
        norm = stats["removed_norm"][lid]
        sem = removed_l2[lid] if removed_l2 else 0
        kept = total - small - norm - sem
        rows.append((name[lid], total, small, norm, sem, kept))
    rows.sort(key=lambda r: -r[1])
    print(f"\n{'lang':12s} {'total':>9s} {'small':>7s} {'norm':>7s} {'sem':>7s} {'kept':>9s}")
    for lang, total, small, norm, sem, kept in rows:
        print(f"{lang:12s} {total:9d} {small:7d} {norm:7d} {sem:7d} {kept:9d}")
    tot = sum(r[1] for r in rows)
    kept_tot = sum(r[5] for r in rows)
    print(f"\nTOTAL {tot} rows -> {kept_tot} kept "
          f"({100.0 * kept_tot / max(tot, 1):.1f}% retained)")
    src_size = os.path.getsize(args.src) / 2**30
    if os.path.exists(args.dst) and not args.dry_run:
        dst_size = os.path.getsize(args.dst) / 2**30
        print(f"size: {src_size:.1f} GiB -> {dst_size:.1f} GiB")
    else:
        print(f"source size: {src_size:.1f} GiB")
    report = {
        "src": args.src,
        "dst": args.dst,
        "total": tot,
        "kept": kept_tot,
        "dropped_small": sum(stats["dropped_small"].values()),
        "removed_norm": stats["dup_norm_total"],
        "removed_semantic": sum(removed_l2.values()) if removed_l2 else 0,
        "per_lang": [
            {"lang": l, "total": t, "small": s, "norm": n_, "sem": s_, "kept": k_}
            for l, t, s, n_, s_, k_ in rows
        ],
    }
    with open(args.dst + ".report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"report: {args.dst}.report.json")


if __name__ == "__main__":
    main()
