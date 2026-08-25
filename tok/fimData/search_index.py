import numpy as np


class InMemoryIndex:
    def __init__(self, conn, dim=1536):
        self.dim = dim
        self._load(conn)

    def _load(self, conn):
        n, _ = conn.execute("SELECT COUNT(*), MAX(rowid) FROM vec_items").fetchone()
        chunks = [b for (b,) in conn.execute(
            "SELECT vectors FROM vec_items_vector_chunks00 ORDER BY rowid")]
        blob = b"".join(chunks)
        self.X = np.frombuffer(blob, dtype=np.float32).reshape(-1, self.dim)[:n]
        self.norms_sq = (self.X * self.X).sum(axis=1)
        self.norms_sq[self.norms_sq == 0] = 1.0

    def search(self, query, k=10):
        q = np.asarray(query, dtype=np.float32)
        nq_sq = float(q @ q)
        if nq_sq == 0.0:
            nq_sq = 1.0
        dots = self.X @ q
        l2_sq = self.norms_sq + nq_sq - 2.0 * dots
        np.maximum(l2_sq, 0.0, out=l2_sq)
        k = min(k, self.X.shape[0])
        idx = np.argpartition(l2_sq, k - 1)[:k]
        idx = idx[np.argsort(l2_sq[idx])]
        return [(int(i) + 1, float(d)) for i, d in zip(idx, np.sqrt(l2_sq[idx]))]
