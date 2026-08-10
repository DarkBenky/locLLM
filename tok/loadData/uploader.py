import json
import os
import random
import signal
import threading
import time
from collections import deque
from queue import Empty, Queue

import requests
from requests.exceptions import ConnectionError, RequestException, Timeout


def send_with_retry(url, payload, max_retries=5, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            res = requests.post(url, json=payload, timeout=30)
            if res.status_code == 200:
                return res
            if res.status_code >= 500:
                raise RequestException(f"HTTP {res.status_code}")
            return res
        except (ConnectionError, Timeout, RequestException) as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"[retry] attempt {attempt+1}/{max_retries} after {delay:.1f}s: {e}")
            time.sleep(delay)
    return None


class DatasetPipeline:
    def __init__(self, api, checkpoint_file="checkpoint.json", dry_run=False):
        self.api = api.rstrip("/") + "/"
        self.checkpoint_file = checkpoint_file
        self.dry_run = dry_run

        self._sources = {}
        self._order = []
        self._buffers = {}
        self._source_done = {}
        self._stats = {}

        self._upload_queue = Queue(maxsize=256)
        self._stop = threading.Event()
        self._fatal_error = None
        self._total_iter = 0
        self._total_tokens = 0
        self._last_ckpt_iter = 0
        self._stats_lock = threading.Lock()
        self._ckpt_lock = threading.Lock()

        self._checkpoint_every = 1000
        self._progress_every = 16
        self._uploaders = []

    def register(self, name, fn=None, prefetch=16, weight=1):
        if fn is not None:
            self.add(name, fn, prefetch=prefetch, weight=weight)
            return fn

        def deco(f):
            self.add(name, f, prefetch=prefetch, weight=weight)
            return f
        return deco

    def add(self, name, fn, prefetch=16, weight=1):
        if name in self._sources:
            raise ValueError(f"source {name!r} already registered")
        self._sources[name] = {
            "fn": fn,
            "prefetch": max(1, int(prefetch)),
            "weight": max(1, int(weight)),
        }
        self._order.append(name)
        self._buffers[name] = deque()
        self._source_done[name] = False
        self._stats[name] = {"iter": 0, "tokens": 0}

    def save_checkpoint(self):
        with self._stats_lock:
            data = {
                "version": 2,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sources": {n: dict(s) for n, s in self._stats.items()},
                "total_iter": self._total_iter,
                "total_tokens": self._total_tokens,
            }
        tmp = self.checkpoint_file + ".tmp"
        try:
            with self._ckpt_lock:
                with open(tmp, "w") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp, self.checkpoint_file)
        except OSError as e:
            print(f"[checkpoint] save failed: {e}")

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file):
            return
        try:
            with open(self.checkpoint_file) as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            print(f"[checkpoint] unreadable, starting fresh: {e}")
            return
        sources = data.get("sources")
        if not isinstance(sources, dict):
            print("[checkpoint] old format (global-only) ignored — per-source progress starts fresh")
            return
        for name, st in sources.items():
            if name in self._stats:
                self._stats[name] = {"iter": int(st.get("iter", 0)), "tokens": int(st.get("tokens", 0))}
        self._total_iter = int(data.get("total_iter", 0))
        self._total_tokens = int(data.get("total_tokens", 0))
        self._last_ckpt_iter = self._total_iter
        print(f"[checkpoint] resuming: total_iter={self._total_iter:_} tokens={self._total_tokens:_}")

    def _producer(self, name):
        src = self._sources[name]
        buf = self._buffers[name]
        try:
            gen = iter(src["fn"]())
        except Exception as e:
            print(f"[{name}] FATAL: generator failed to start: {e}")
            self._source_done[name] = True
            return

        skip = self._stats[name]["iter"]
        if skip:
            t0 = time.time()
            try:
                for i in range(skip):
                    next(gen)
                    if i and i % 1024 == 0:
                        print(f"[{name}] resume: skipped {i:_} / {skip:_} ...")
            except StopIteration:
                pass
            print(f"[{name}] resume skip complete ({skip:_} records, {time.time() - t0:.1f}s)")

        while not self._stop.is_set():
            try:
                rec = next(gen)
            except StopIteration:
                break
            except Exception as e:
                print(f"[{name}] row error: {e}")
                continue
            while len(buf) >= src["prefetch"] and not self._stop.is_set():
                time.sleep(0.005)
            if self._stop.is_set():
                break
            buf.append(rec)
        self._source_done[name] = True

    def _dispatch(self):
        last_log = [0, time.time(), 0]
        while not self._stop.is_set():
            emitted = 0
            for name in self._order:
                src = self._sources[name]
                buf = self._buffers[name]
                for _ in range(src["weight"]):
                    if buf:
                        self._upload_queue.put((name, buf.popleft()))
                        emitted += 1
                    else:
                        break
            if emitted == 0:
                if all(self._source_done.values()) and all(not b for b in self._buffers.values()):
                    break
                self._log_progress(last_log)
                time.sleep(0.005)
                continue
            self._log_progress(last_log)

    def _uploader(self, tid):
        while not (self._stop.is_set() and self._upload_queue.empty()):
            try:
                name, rec = self._upload_queue.get(timeout=0.5)
            except Empty:
                continue
            if self.dry_run:
                token_count = max(1, len(str(rec.get("text", ""))) // 8)
            else:
                try:
                    res = send_with_retry(self.api + "api/receive-data", rec)
                except Exception as e:
                    print(f"[fatal] send failed after retries: {e}")
                    self.save_checkpoint()
                    self._fail(e)
                    return
                if res.status_code != 200:
                    print(f"[error] server returned status {res.status_code}: {res.text}")
                    continue
                token_count = res.json().get("token_count", 0)
            with self._stats_lock:
                self._stats[name]["iter"] += 1
                self._stats[name]["tokens"] += token_count
                self._total_iter += 1
                self._total_tokens += token_count
                due = self._total_iter - self._last_ckpt_iter >= self._checkpoint_every
                if due:
                    self._last_ckpt_iter = self._total_iter
            if due:
                self.save_checkpoint()

    def _log_progress(self, last_log):
        now = time.time()
        with self._stats_lock:
            it, tok = self._total_iter, self._total_tokens
        if it - last_log[0] >= self._progress_every and now - last_log[1] >= 0.5:
            dt = now - last_log[1]
            rate = (tok - last_log[2]) / dt if dt > 0 else 0
            print(f"iter {it:_} tokenCount {tok:_} tokensPerSec {rate:_.0f}")
            last_log[:] = [it, now, tok]

    def _fail(self, exc):
        if self._fatal_error is None:
            self._fatal_error = exc
        self._stop.set()

    def run(self, upload_threads=2, checkpoint_every=1000, progress_every=16):
        self._checkpoint_every = max(1, int(checkpoint_every))
        self._progress_every = max(1, int(progress_every))
        if not self._sources:
            raise RuntimeError("no sources registered")

        self.load_checkpoint()
        resume = ", ".join(f"{n}={s['iter']:_}" for n, s in self._stats.items())
        print(f"resuming (records already uploaded per source): {resume}")

        def _on_sigint(sig, frame):
            print("\n[SIGINT] saving checkpoint and stopping...")
            self.save_checkpoint()
            self._stop.set()

        signal.signal(signal.SIGINT, _on_sigint)

        for name in self._order:
            t = threading.Thread(target=self._producer, args=(name,), daemon=True,
                                 name=f"src-{name}")
            t.start()

        for i in range(upload_threads):
            t = threading.Thread(target=self._uploader, args=(i,), daemon=True,
                                 name=f"upload-{i}")
            self._uploaders.append(t)
            t.start()

        self._dispatch()
        self._stop.set()

        for t in self._uploaders:
            t.join()

        self.save_checkpoint()
        if self._fatal_error is not None:
            raise self._fatal_error
        print("Done")
