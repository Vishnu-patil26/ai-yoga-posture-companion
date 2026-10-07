"""Extract the poses this project needs from the public 107-pose yoga dataset.

Source: Hugging Face `rotemvahava/yoga-poses-107` (5,994 web photographs of 107
asanas, ~56 per class, many different people).  The dataset sheet in
`dataset.pdf` names the Kaggle "Yoga Posture Dataset" (Triangle, Chair, Warrior
...) - this is the same 107-class collection, downloadable without an account.
Images are used for fitting/validation only and are not redistributed here.

Only the classes the library needs are written out, to
`data/datasets/yoga107/<class_slug>/NNN.jpg` (EXIF-rotated, longest side 1024).

    python tools/datasets/get_yoga107.py                 # downloads ~1.1 GB, then extracts
    python tools/datasets/get_yoga107.py --parquet-dir D # reuse already-downloaded parquet files
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = "https://huggingface.co/datasets/rotemvahava/yoga-poses-107"
PARTS = [f"data/train-0000{i}-of-00003.parquet" for i in range(3)]
OUT = os.path.join(HERE, "data", "datasets", "yoga107")

#: dataset class name -> folder slug used by tools/fitting/build_library.py
WANT = {
    "adho mukha svanasana": "adho_mukha_svanasana",
    "balasana": "balasana",
    "bhujangasana": "bhujangasana",
    "bitilasana": "bitilasana",                    # cow pose
    "marjaryasana": "marjaryasana",                # cat pose
    "sukhasana": "sukhasana",
    "tadasana": "tadasana",
    "urdhva hastasana": "urdhva_hastasana",        # tadasana with the arms overhead
    "utkatasana": "utkatasana",
    "utthita trikonasana": "utthita_trikonasana",
    "virabhadrasana ii": "virabhadrasana_ii",
}


def class_names() -> list[str]:
    with urllib.request.urlopen(f"{BASE}/raw/main/README.md", timeout=60) as r:
        text = r.read().decode("utf-8")
    pairs = re.findall(r"^\s+'(\d+)': (.+)$", text, flags=re.M)
    return [name.strip() for _, name in sorted(pairs, key=lambda p: int(p[0]))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parquet-dir", default=os.path.join(HERE, "data", "datasets", "_yoga107_parquet"))
    args = ap.parse_args(argv)

    import pyarrow.parquet as pq
    from PIL import Image, ImageOps

    names = class_names()
    wanted = {i: WANT[n] for i, n in enumerate(names) if n in WANT}
    missing = [n for n in WANT if n not in names]
    if missing:
        print("not in the dataset:", missing)
    os.makedirs(args.parquet_dir, exist_ok=True)
    counts = {slug: 0 for slug in wanted.values()}

    for part in PARTS:
        local = os.path.join(args.parquet_dir, os.path.basename(part).replace("-of-00003", ""))
        if not os.path.isfile(local):
            print("downloading", part, flush=True)
            urllib.request.urlretrieve(f"{BASE}/resolve/main/{part}", local)
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(batch_size=64, columns=["image", "label"]):
            for img, label in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
                if label not in wanted:
                    continue
                slug = wanted[label]
                dest = os.path.join(OUT, slug)
                os.makedirs(dest, exist_ok=True)
                counts[slug] += 1
                path = os.path.join(dest, f"{counts[slug]:03d}.jpg")
                im = ImageOps.exif_transpose(Image.open(io.BytesIO(img["bytes"]))).convert("RGB")
                im.thumbnail((1024, 1024))
                im.save(path, quality=90)
        print(f"{os.path.basename(local)} done", flush=True)
    print("images per class:", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
