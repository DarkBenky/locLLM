import sqlite3
import struct

import sqlite_vec


class CodeDB:
    def __init__(self, path, dim=1536, create_index=True):
        self.dim = dim
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(embedding float[{dim}])"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS items (hash TEXT PRIMARY KEY, code TEXT, lang TEXT, norm_hash TEXT)"
        )
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(items)")}
        if "norm_hash" not in cols:
            self.conn.execute("ALTER TABLE items ADD COLUMN norm_hash TEXT")
        if create_index:
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_items_norm_hash ON items(norm_hash)"
            )
        self.conn.commit()

    def _pack(self, embedding):
        return struct.pack(f"{self.dim}f", *embedding)

    def add(self, code, lang, _hash, embedding):
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO items (hash, code, lang) VALUES (?,?,?)",
            (_hash, code, lang),
        )
        if not cur.rowcount:
            return False
        self.conn.execute(
            "INSERT INTO vec_items (rowid, embedding) VALUES (?,?)",
            (cur.lastrowid, self._pack(embedding)),
        )
        self.conn.commit()
        return True

    def add_result(self, r):
        return self.add(r.code, r.lang, r._hash, r.embedding)

    def add_many(self, results):
        added = 0
        for r in results:
            if self.add_result(r):
                added += 1
        return added

    def add_batch(self, items):
        added = 0
        for code, lang, _hash, embedding in items:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO items (hash, code, lang) VALUES (?,?,?)",
                (_hash, code, lang),
            )
            if not cur.rowcount:
                continue
            self.conn.execute(
                "INSERT INTO vec_items (rowid, embedding) VALUES (?,?)",
                (cur.lastrowid, self._pack(embedding)),
            )
            added += 1
        self.conn.commit()
        return added

    def search(self, embedding, k=10, lang=None):
        rows = self.conn.execute(
            "SELECT rowid, distance FROM vec_items WHERE embedding MATCH ? AND k=?",
            (self._pack(embedding), k),
        ).fetchall()
        out = []
        for rowid, distance in rows:
            item = self.conn.execute(
                "SELECT hash, code, lang FROM items WHERE rowid=?", (rowid,)
            ).fetchone()
            if item is None:
                continue
            h, code, lng = item
            if lang is not None and lng != lang:
                continue
            out.append({"hash": h, "code": code, "lang": lng, "distance": distance})
        return out

    def exists(self, _hash):
        return self.conn.execute(
            "SELECT 1 FROM items WHERE hash=?", (_hash,)
        ).fetchone() is not None

    def exists_norm(self, norm_hash):
        return self.conn.execute(
            "SELECT 1 FROM items WHERE norm_hash=? LIMIT 1", (norm_hash,)
        ).fetchone() is not None

    def add_batch_dedup(self, items):
        added = 0
        dup_norm = 0
        dup_hash = 0
        dropped = []
        accepted = []
        seen = set()
        for code, lang, _hash, embedding, norm_hash in items:
            if norm_hash is not None:
                if norm_hash in seen or self.exists_norm(norm_hash):
                    dup_norm += 1
                    dropped.append((lang, "norm"))
                    continue
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO items (hash, code, lang, norm_hash) VALUES (?,?,?,?)",
                (_hash, code, lang, norm_hash),
            )
            if not cur.rowcount:
                dup_hash += 1
                dropped.append((lang, "hash"))
                continue
            self.conn.execute(
                "INSERT INTO vec_items (rowid, embedding) VALUES (?,?)",
                (cur.lastrowid, self._pack(embedding)),
            )
            if norm_hash is not None:
                seen.add(norm_hash)
            accepted.append((lang, embedding))
            added += 1
        self.conn.commit()
        return {
            "added": added,
            "dup_norm": dup_norm,
            "dup_hash": dup_hash,
            "dropped": dropped,
            "accepted": accepted,
        }

    def iter_all(self):
        cur = self.conn.execute(
            "SELECT rowid, hash, code, lang FROM items ORDER BY rowid"
        )
        for row in cur:
            yield row

    def get_item(self, rowid):
        r = self.conn.execute(
            "SELECT hash, code, lang FROM items WHERE rowid=?", (rowid,)
        ).fetchone()
        if r is None:
            return None
        return {"hash": r[0], "code": r[1], "lang": r[2]}

    def count(self):
        return self.conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    import os
    import random

    from parseCode import SampleUnparsed, parseCodeSample

    path = "/tmp/test_codedb.db"
    if os.path.exists(path):
        os.remove(path)

    db = CodeDB(path, dim=16)
    samples = [
        ("def hello(name):\n    return name", "python"),
        ("def goodbye(name):\n    return name", "python"),
        ("fn add(a: i32) -> i32 { a }", "rust"),
    ]
    added = 0
    for code, lang in samples:
        r = parseCodeSample(SampleUnparsed(code, lang))
        r.embedding = [random.random() for _ in range(16)]
        if db.add_result(r):
            added += 1
    print("added:", added)
    print("dup add:", db.add(samples[0][0], "python", __import__("hashlib").sha256(samples[0][0].encode()).hexdigest(), [0.0] * 16))
    print("count:", db.count())
    res = db.search([random.random() for _ in range(16)], k=3)
    for r in res:
        print(f"  {r['lang']:6s} {r['distance']:.4f} {r['code'][:30]!r}")
    db.close()
