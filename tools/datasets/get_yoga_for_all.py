"""Fetch the Tadasana, Marjariasana (Cat-Cow) and Bhujangasana images from the
open "Yoga for all" dataset (Zenodo 7818789, CC BY 4.0, Mendeley jc4mmnvcdk)
without downloading the whole 23 GB archive.

The zip is read through an HTTP-range file with a block cache: the central
directory comes from the tail of the archive and the wanted folders, which are
stored contiguously, arrive in a handful of multi-megabyte requests instead of
one request per image (Zenodo rate-limits the latter to a crawl).

Each pose has "Right Steps" (the correct build-up, step by step) and "Wrong
Steps" (named by what is wrong: Legs, Hand, Head ...), so the dataset is
labelled for *correctness*, not just identity.  Layout written here:

    <pose>/right/step_<n>/*.jpg      <pose>/wrong/step_<n>-_<fault>/*.jpg

    python tools/datasets/get_yoga_for_all.py            # -> data/datasets/yoga_for_all/
"""
from __future__ import annotations

import io
import os
import sys
import time
import urllib.request
import zipfile

URL = "https://zenodo.org/api/records/7818789/files/Yoga%20Postures%20Dataset.zip/content"
WANT = {"Tadasana": "tadasana", "Marjariasana": "marjaryasana", "Bhujangasana": "bhujangasana"}
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "data", "datasets", "yoga_for_all")
BLOCK = 512 * 1024
PER_RIGHT_STEP = 24      #: images kept per correct step folder (evenly spaced)
PER_WRONG_FAULT = 12     #: images kept per labelled-fault folder
MAX_SIDE = 1024          #: longest side after downscaling


class RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file using cached HTTP range blocks."""

    def __init__(self, url: str) -> None:
        self.url, self.pos, self.cache = url, 0, {}
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = (off if whence == 0 else self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self.cache:
            lo, hi = i * BLOCK, min(self.size, (i + 1) * BLOCK) - 1
            for attempt in range(8):
                try:
                    req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        self.cache[i] = r.read()
                    break
                except Exception:                       # 429 / reset: back off and retry
                    time.sleep(3 + attempt * 4)
            else:
                raise IOError(f"could not fetch bytes {lo}-{hi}")
            if len(self.cache) > 10:                     # keep memory bounded
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        out, got = memoryview(b), 0
        while got < n:
            blk = self._block((self.pos + got) // BLOCK)
            off = (self.pos + got) % BLOCK
            take = min(n - got, len(blk) - off)
            out[got:got + take] = blk[off:off + take]
            got += take
        self.pos += n
        return n


def _spread(items: list, k: int) -> list:
    """k evenly spaced items - consecutive shots are near-duplicates, so spread."""
    if len(items) <= k:
        return items
    return [items[int(j * len(items) / k)] for j in range(k)]


def _save(data: bytes, target: str) -> None:
    from PIL import Image, ImageOps
    # Phone photos carry their rotation in EXIF; dropping it leaves them sideways.
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    im.save(target, quality=90)


def main() -> int:
    from concurrent.futures import ThreadPoolExecutor
    zf = zipfile.ZipFile(RangeFile(URL))
    folders: dict[tuple, list] = {}
    for info in sorted(zf.infolist(), key=lambda i: i.header_offset):
        parts = info.filename.split("/")
        if len(parts) < 6 or parts[1] != "Images" or info.is_dir() or parts[2] not in WANT:
            continue
        kind = "right" if "Right" in parts[3] else "wrong"
        step = parts[4].split(" Step ", 1)[-1].strip().replace(" ", "_")   # "3", "1-_Legs"
        folders.setdefault((WANT[parts[2]], kind, "step_" + step), []).append(info)
    jobs = []
    for (pose, kind, step), infos in folders.items():
        dest = os.path.join(OUT, pose, kind, step)
        keep = _spread(infos, PER_RIGHT_STEP if kind == "right" else PER_WRONG_FAULT)
        for info in keep:
            target = os.path.join(dest, os.path.splitext(os.path.basename(info.filename))[0] + ".jpg")
            if not os.path.exists(target):
                jobs.append((info.filename, dest, target))
    print(f"{len(jobs)} images to fetch from {len(folders)} folders", flush=True)

    import threading
    local, done = threading.local(), [0]

    def fetch(job) -> None:
        name, dest, target = job
        if not hasattr(local, "zf"):
            local.zf = zipfile.ZipFile(RangeFile(URL))
        os.makedirs(dest, exist_ok=True)
        _save(local.zf.read(name), target)
        done[0] += 1
        if done[0] % 25 == 0:
            print(f"  {done[0]} / {len(jobs)}", flush=True)

    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(fetch, jobs))
    print(f"done: {done[0]} new images in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
