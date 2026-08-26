import re
from hashlib import sha256

import numpy as np

COS_THRESHOLD = 0.95
MIN_CHUNK_LEN = 20
CAND_K = 512

_PARSER_CACHE = {}

_NIM_BLOCK_COMMENT_RE = re.compile(r"#\[.*?\]#", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"#[^\n]*")
_WHITESPACE_RE = re.compile(r"\s+")


def _get_parser(ts_name):
    parser = _PARSER_CACHE.get(ts_name)
    if parser is None:
        from tree_sitter_language_pack import get_parser

        parser = get_parser(ts_name)
        _PARSER_CACHE[ts_name] = parser
    return parser


def _tokenize(code, ts_name):
    try:
        parser = _get_parser(ts_name)
    except Exception:
        return None
    try:
        tree = parser.parse(code.encode())
    except Exception:
        return None
    root = tree.root_node
    if root.has_error:
        return None

    out = []
    stack = [root]
    while stack:
        node = stack.pop()
        t = node.type
        lt = t.lower()
        if "comment" in lt:
            continue
        if "string" in lt or "char" in lt:
            out.append("STR")
            continue
        if t == "identifier" or t.endswith("_identifier"):
            out.append("ID")
            continue
        if "int" in lt or "float" in lt or t == "number":
            out.append("NUM")
            continue
        children = node.children
        if not children:
            tok = code[node.start_byte:node.end_byte].strip()
            if tok:
                out.append(tok)
        else:
            stack.extend(reversed(children))
    return out


def _normalize_fallback(code):
    text = code
    text = _NIM_BLOCK_COMMENT_RE.sub(" ", text)
    text = _LINE_COMMENT_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_code(code, lang, ts_name=None):
    if ts_name is None:
        from parseCode import TS_LANG_NAMES

        ts_name = TS_LANG_NAMES.get(lang)
    if ts_name:
        tokens = _tokenize(code, ts_name)
        if tokens is not None:
            return " ".join(tokens)
    return _normalize_fallback(code)


def compute_norm_hash(code, lang, ts_name=None):
    return sha256(normalize_code(code, lang, ts_name).encode("utf-8")).hexdigest()


def _merge_topk(torch, v1, i1, v2, i2, k):
    if v1 is None:
        return v2, i2
    cv = torch.cat([v1, v2], dim=0)
    ci = torch.cat([i1, i2], dim=0)
    kk = min(k, cv.shape[0])
    v, i = torch.topk(cv, kk, dim=0)
    return torch.gather(cv, 0, i), torch.gather(ci, 0, i)


_VEC_CHUNK = 1024


class DedupIndex:
    def __init__(self, conn, dim=1536, threshold=COS_THRESHOLD, cand_k=CAND_K,
                 device=None, chunk=262144, verbose=True):
        self.conn = conn
        self.dim = dim
        self.threshold = float(threshold)
        self.cand_k = int(cand_k)
        self.chunk = int(chunk)
        self.verbose = verbose

        row = conn.execute("SELECT COUNT(*), MAX(rowid) FROM vec_items").fetchone()
        n, max_rowid = int(row[0] or 0), int(row[1] or 0)
        if n != max_rowid:
            raise ValueError(
                f"vec_items rowids are not dense (count={n}, max={max_rowid}); "
                "rebuild the DB with compact.py first"
            )
        self.n = n

        lang_rows = conn.execute(
            "SELECT rowid, lang FROM items ORDER BY rowid"
        ).fetchall()
        if len(lang_rows) != n:
            raise ValueError(
                f"items count ({len(lang_rows)}) != vec_items count ({n})"
            )
        langs = [r[1] for r in lang_rows]
        self._lang_map = {l: i for i, l in enumerate(dict.fromkeys(langs))}
        self.lang_ids = np.array([self._lang_map[l] for l in langs], dtype=np.int16)

        self.X16 = np.empty((n, dim), dtype=np.float16)
        self._load_embeddings(conn, n, dim)

        self._torch = None
        self.device = None
        self._init_torch(device)

        self._new_cap = 1 << 20
        self._new16 = np.empty((self._new_cap, dim), dtype=np.float16)
        self._new_lang_ids = np.empty(self._new_cap, dtype=np.int16)
        self._new_count = 0

        self._pinned = None
        self._new_pinned = None
        self._resident = None
        if self.device is not None:
            self._pinned = self._torch.empty(
                (n, dim), dtype=self._torch.float16, pin_memory=True
            )
            self._pinned.copy_(self._torch.from_numpy(self.X16))
            self.X16 = None
            self._new_pinned = self._torch.empty(
                (self._new_cap, dim), dtype=self._torch.float16, pin_memory=True
            )
            total = self._torch.cuda.get_device_properties(self.device).total_memory
            need = n * dim * 2 + 4 * 2**30
            if need < total:
                self._resident = self._pinned.to(self.device)
                self._pinned = None
                if self.verbose:
                    print(
                        f"DedupIndex: matrix resident on GPU "
                        f"({need / 2**30:.1f} GiB of {total / 2**30:.1f} GiB)"
                    )

    def _load_embeddings(self, conn, n, dim):
        pos = 0
        for (blob,) in conn.execute(
            "SELECT vectors FROM vec_items_vector_chunks00 ORDER BY rowid"
        ):
            vecs = np.frombuffer(blob, dtype=np.float32).reshape(-1, dim)
            take = min(vecs.shape[0], n - pos)
            if take <= 0:
                break
            block = vecs[:take].copy()
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            block /= norms
            self.X16[pos:pos + take] = block.astype(np.float16)
            pos += take
        if pos != n:
            raise ValueError(f"shadow table yields {pos} vectors, expected {n}")

    def _init_torch(self, device):
        try:
            import torch
        except Exception:
            torch = None
        self._torch = torch
        if torch is not None and torch.cuda.is_available():
            if device is None:
                device = torch.cuda.current_device()
            self.device = torch.device(f"cuda:{device}")
            if self.verbose:
                print(f"DedupIndex using {torch.cuda.get_device_name(self.device)}")
        elif self.verbose:
            print("DedupIndex: no CUDA, using CPU fallback scoring")

    @staticmethod
    def _normalize_rows(x32):
        x = np.asarray(x32, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return x / norms

    def _lang_id_for(self, lang):
        if lang not in self._lang_map:
            self._lang_map[lang] = len(self._lang_map)
        return self._lang_map[lang]

    def _topk(self, s, k, base=0):
        k = min(k, s.shape[0])
        if k <= 0:
            return (np.zeros((0, s.shape[1]), dtype=np.int64),
                    np.zeros((0, s.shape[1]), dtype=np.float32))
        if s.shape[0] <= k:
            part = np.tile(np.arange(s.shape[0])[:, None], (1, s.shape[1]))
        else:
            part = np.argpartition(-s, k - 1, axis=0)[:k]
        vals = np.take_along_axis(s, part, axis=0).astype(np.float32)
        idx = part.astype(np.int64) + base
        return idx, vals

    def _gpu_topk(self, q16):
        torch = self._torch
        k = self.cand_k
        b = q16.shape[0]
        qt = torch.from_numpy(q16).to(self.device)
        best_vals = None
        best_idx = None
        for start in range(0, self.n, self.chunk):
            if self._resident is not None:
                xt = self._resident[start:start + self.chunk]
            else:
                xt = self._pinned[start:start + self.chunk].to(
                    self.device, non_blocking=True
                )
            s = (xt @ qt.T).float()
            v, i = torch.topk(s, min(k, s.shape[0]), dim=0)
            i = i.to(torch.int64) + start
            best_vals, best_idx = _merge_topk(torch, best_vals, best_idx, v, i, k)
        if self._new_count:
            xt = self._new_pinned[:self._new_count].to(
                self.device, non_blocking=True
            )
            s = (xt @ qt.T).float()
            v, i = torch.topk(s, min(k, s.shape[0]), dim=0)
            i = i.to(torch.int64) + self.n
            best_vals, best_idx = _merge_topk(torch, best_vals, best_idx, v, i, k)
        if best_vals is None:
            return (np.zeros((0, b), dtype=np.int64),
                    np.zeros((0, b), dtype=np.float32))
        return best_idx.cpu().numpy(), best_vals.cpu().numpy()

    def _cpu_scores(self, q32):
        parts = []
        for start in range(0, self.n, self.chunk):
            xc = self.X16[start:start + self.chunk].astype(np.float32)
            parts.append(xc @ q32.T)
        if not parts:
            return np.zeros((0, q32.shape[0]), dtype=np.float32)
        return np.concatenate(parts, axis=0)

    def _new_scores(self, q16):
        if self._new_count == 0:
            return np.zeros((0, q16.shape[0]), dtype=np.float32)
        xnew = self._new16[:self._new_count]
        parts = []
        for start in range(0, self._new_count, self.chunk):
            xc = xnew[start:start + self.chunk].astype(np.float32)
            parts.append(xc @ q16.astype(np.float32).T)
        return np.concatenate(parts, axis=0)

    def check_batch(self, embs, langs):
        embs = np.asarray(embs, dtype=np.float32)
        if embs.size == 0:
            return np.zeros(0, dtype=bool)
        single = embs.ndim == 1
        if single:
            embs = embs[None, :]
        if isinstance(langs, str):
            langs = [langs]
        b = embs.shape[0]
        q32 = self._normalize_rows(embs)
        q16 = q32.astype(np.float16)

        if self._torch is not None and self.device is not None:
            best_idx, best_vals = self._gpu_topk(q16)
        else:
            s = self._cpu_scores(q32)
            best_idx, best_vals = self._topk(s, self.cand_k, base=0)
            if self._new_count:
                s2 = self._new_scores(q16)
                idx2, vals2 = self._topk(s2, self.cand_k, base=self.n)
                best_idx = np.concatenate([best_idx, idx2], axis=0)
                best_vals = np.concatenate([best_vals, vals2], axis=0)
                if best_idx.shape[0] > self.cand_k:
                    part = np.argpartition(-best_vals, self.cand_k - 1, axis=0)[:self.cand_k]
                    best_idx = np.take_along_axis(best_idx, part, axis=0)
                    best_vals = np.take_along_axis(best_vals, part, axis=0)

        if self._new_count:
            lang_all = np.concatenate(
                [self.lang_ids, self._new_lang_ids[:self._new_count]]
            )
        else:
            lang_all = self.lang_ids

        tgt = np.array([self._lang_map.get(l, -1) for l in langs], dtype=np.int16)
        flags = np.zeros(b, dtype=bool)
        for i in range(b):
            if tgt[i] < 0 or best_idx.shape[0] == 0:
                continue
            mask = lang_all[best_idx[:, i]] == tgt[i]
            if mask.any() and float(best_vals[mask, i].max()) >= self.threshold:
                flags[i] = True
        return flags[0] if single else flags

    def add_many(self, embs, langs):
        embs = np.asarray(embs, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs[None, :]
        if isinstance(langs, str):
            langs = [langs]
        if embs.shape[0] == 0 or embs.size == 0:
            return
        q = self._normalize_rows(embs).astype(np.float16)
        ids = np.array([self._lang_id_for(l) for l in langs], dtype=np.int16)
        need = self._new_count + q.shape[0]
        if need > self._new_cap:
            cap = max(self._new_cap * 2, need)
            buf = np.empty((cap, self.dim), dtype=np.float16)
            buf[:self._new_count] = self._new16
            self._new16 = buf
            lbuf = np.empty(cap, dtype=np.int16)
            lbuf[:self._new_count] = self._new_lang_ids
            self._new_lang_ids = lbuf
            if self._new_pinned is not None:
                pbuf = self._torch.empty(
                    (cap, self.dim), dtype=self._torch.float16, pin_memory=True
                )
                pbuf[:self._new_count].copy_(self._new_pinned[:self._new_count])
                self._new_pinned = pbuf
            self._new_cap = cap
        self._new16[self._new_count:need] = q
        self._new_lang_ids[self._new_count:need] = ids
        if self._new_pinned is not None:
            self._new_pinned[self._new_count:need].copy_(self._torch.from_numpy(q))
        self._new_count = need

    @property
    def size(self):
        return self.n + self._new_count
