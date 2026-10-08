"""Stream per-frame training data (hnod.frames) from the Hugging Face Hub with bounded disk use.

Every frame shard holds whole episodes, so samples can be cut from one shard at a time.  Each data-loader
worker owns a disjoint share of the shards of every source; it downloads a shard, reads it into memory,
deletes the file at once, and draws samples at random from the shards it holds (`buffer_shards` per
source, the next one prefetched in a thread).  Sources are mixed by weight (`frames_mix`: equal, or
proportional to their frame counts).  Disk use is at most one shard per worker in flight; memory about
workers x sources x (buffer_shards + 1) shards.
"""
import os
import queue
import random
import tempfile
import threading
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, get_worker_info

from hnod.windows import FrameWindows


class _Source:
    def __init__(self, repo, split):
        from huggingface_hub import HfApi
        files = HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(repo, repo_type="dataset")
        self.repo = repo
        self.frames = sorted(f for f in files if f.startswith(f"data/frames/{split}-"))
        self.episode_files = sorted(f for f in files if f.startswith(f"data/episodes/{split}-"))
        self.sample_files = sorted(f for f in files if f.startswith(f"data/samples/{split}-"))  # a published sample list
        self.annotation_files = sorted(f for f in files if f.startswith(f"data/annotations/{split}-"))
        self.sizes = None  # frame counts, filled by load_episodes
        self.allowed = None
        self.descriptions = {}  # (episode_id, frame_index) -> {description, place, interaction}, from data/annotations

    def load_episodes(self):
        """All episode rows of the split (small), as a list of dicts."""
        from huggingface_hub import hf_hub_download
        rows = []
        with tempfile.TemporaryDirectory(dir=_tmp_root()) as d:
            for f in self.episode_files:
                p = hf_hub_download(self.repo, f, repo_type="dataset", local_dir=d)
                rows += pq.read_table(p).to_pylist()
                os.remove(p)
        self.sizes = sum(r["num_frames"] for r in rows)
        if self.sample_files:  # deduplicated repo: only the listed samples are cut
            from huggingface_hub import hf_hub_download
            allowed = set()
            with tempfile.TemporaryDirectory(dir=_tmp_root()) as d:
                for f in self.sample_files:
                    t = pq.read_table(hf_hub_download(self.repo, f, repo_type="dataset", local_dir=d), columns=["episode_id", "frame_index"])
                    allowed |= set(zip(t.column("episode_id").to_pylist(), t.column("frame_index").to_pylist()))
            self.allowed = allowed
            self.sizes = len(allowed)
        if self.annotation_files:
            with tempfile.TemporaryDirectory(dir=_tmp_root()) as d:
                for f in self.annotation_files:
                    t = pq.read_table(hf_hub_download(self.repo, f, repo_type="dataset", local_dir=d),
                                      columns=["episode_id", "frame_index", "description", "place", "interaction"]).to_pydict()
                    for e, i, s, pl, inter in zip(t["episode_id"], t["frame_index"], t["description"], t["place"], t["interaction"]):
                        if s:
                            self.descriptions[(e, i)] = dict(description=s, place=pl or "", interaction=inter or "")
        return rows


def _tmp_root():
    root = os.environ.get("STREAM_TMP", "/dev/shm/stream_tmp")
    Path(root).mkdir(parents=True, exist_ok=True)
    return root


def read_shard(repo, path):
    """Download one frames shard, load it into memory as a datasets.Dataset, delete the file."""
    from datasets import Dataset
    from huggingface_hub import hf_hub_download
    with tempfile.TemporaryDirectory(dir=_tmp_root()) as d:
        for k in range(6):
            try:
                p = hf_hub_download(repo, path, repo_type="dataset", local_dir=d)
                break
            except Exception:  # 429s and network hiccups
                import time
                time.sleep(10 * 2 ** k)
        else:
            raise RuntimeError(f"could not download {repo}/{path}")
        table = pq.read_table(p)  # into memory; the file goes with the directory
    return Dataset(table)  # features come from the parquet metadata (Image columns decode lazily)


class StreamFrames(IterableDataset):
    """Endless (train) or single-pass (`finite`) stream of FrameNavDataset-style items.

    item_fn(windows, k) -> item dict, e.g. vla.data.FrameNavDataset's item layout.
    """

    def __init__(self, repos, split, window_cfg, item_fn, mix="equal", buffer_shards=1, seed=0, finite=False,
                 max_items=0):
        self.sources = [s for s in (_Source(r, split) for r in repos) if s.frames]
        self.episodes = {s.repo: s.load_episodes() for s in self.sources}
        sizes = np.array([s.sizes for s in self.sources], float)
        w = {"equal": np.ones(len(sizes)), "sqrt": np.sqrt(sizes), "proportional": sizes}[mix]
        self.weights = w / w.sum()
        self.window_cfg, self.item_fn = window_cfg, item_fn
        self.buffer_shards, self.seed, self.finite, self.max_items = buffer_shards, seed, finite, max_items
        self.epoch = 0

    def describe(self):
        return ", ".join(f"{s.repo.split('/')[-1]}: {s.sizes} frames in {len(s.frames)} shards, weight {w:.2f}"
                         for s, w in zip(self.sources, self.weights))

    def _windows(self, src, path):
        ds = read_shard(src.repo, path)
        win = FrameWindows(ds, self.episodes[src.repo], self.window_cfg, allowed=src.allowed)
        win.descriptions = src.descriptions
        return win

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(f"{self.seed}-{self.epoch}-{wid}")
        self.epoch += 1
        # this worker's shards of every source; a source with fewer shards than workers is shared
        mine = []
        for s in self.sources:
            own = s.frames[wid::nw] or [s.frames[wid % len(s.frames)]]
            mine.append(list(own))
        if self.finite:
            yield from self._single_pass(mine)
            return
        queues = [self._prefetch(s, own, rng) for s, own in zip(self.sources, mine)]
        buffers = [[q.get() for _ in range(self.buffer_shards)] for q in queues]
        w = self.weights / self.weights.sum()
        while True:
            i = int(np.searchsorted(np.cumsum(w), rng.random() * 0.999999))
            j = rng.randrange(len(buffers[i]))
            win, left = buffers[i][j]
            if not left:  # this shard is used up: swap in the next one
                buffers[i][j] = queues[i].get()
                continue
            k = left.pop()
            yield self.item_fn(win, k)

    def _prefetch(self, src, own, rng):
        """Background thread: (FrameWindows, shuffled sample indices) per shard, cycling over `own`."""
        q = queue.Queue(maxsize=1)

        def run():
            while True:
                order = own[:]
                rng.shuffle(order)
                for path in order:
                    win = self._windows(src, path)
                    idx = list(range(len(win)))
                    rng.shuffle(idx)
                    q.put((win, idx))
        threading.Thread(target=run, daemon=True).start()
        return q

    def _single_pass(self, mine):
        n = 0
        for src, own in zip(self.sources, mine):
            for path in own:
                win = self._windows(src, path)
                for k in range(0, len(win), max(1, self.window_cfg.stride)):
                    yield self.item_fn(win, k)
                    n += 1
                    if self.max_items and n >= self.max_items:
                        return
