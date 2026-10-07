"""Fetch openly licensed photos of Trikonasana, Balasana and Sukhasana from
Wikimedia Commons categories, for the poses that none of the freely
downloadable pose datasets cover.

Every file's author and licence is recorded in data/datasets/commons/credits.json
(copyright: attribution is a licence condition for the CC BY / BY-SA photos).

    python tools/get_commons_poses.py
"""
from __future__ import annotations

import html
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

CATS = {
    "trikonasana": ["Category:Trikoṇāsana"],
    "balasana": ["Category:Bālāsana"],
    "sukhasana": ["Category:Sukhāsana"],
}
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "datasets", "commons")
UA = {"User-Agent": "YogaCompanionProject/1.0 (student project; GitHub Vishnu-patil26)"}


def get(url: str) -> bytes:
    for attempt in range(8):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
        except Exception:
            time.sleep(4 + attempt * 5)           # Commons answers 429 when pushed
    raise IOError(url)


def api(**kw):
    kw["format"] = "json"
    return json.loads(get("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(kw)))


def main() -> int:
    from PIL import Image
    strip = lambda s: html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()
    credits: dict = {}
    for pose, cats in CATS.items():
        os.makedirs(os.path.join(OUT, pose), exist_ok=True)
        for cat in cats:
            d = api(action="query", generator="categorymembers", gcmtitle=cat, gcmtype="file",
                    gcmlimit=100, prop="imageinfo", iiprop="url|extmetadata",
                    iiurlwidth=1024, iiextmetadatafilter="LicenseShortName|Artist|LicenseUrl")
            for pg in d.get("query", {}).get("pages", {}).values():
                t = pg["title"]
                if not t.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                ii = pg["imageinfo"][0]
                m = ii["extmetadata"]
                lic = m.get("LicenseShortName", {}).get("value", "")
                if not lic.startswith(("CC", "Public")):
                    continue
                name = re.sub(r"[^A-Za-z0-9._-]+", "_", t[5:])
                path = os.path.join(OUT, pose, os.path.splitext(name)[0] + ".jpg")
                if not os.path.exists(path):
                    try:
                        Image.open(io.BytesIO(get(ii["thumburl"]))).convert("RGB").save(path, quality=90)
                    except Exception:
                        print("  skipped", t, flush=True)
                        continue
                    time.sleep(1.5)
                credits[f"{pose}/{os.path.basename(path)}"] = {
                    "title": t, "artist": strip(m.get("Artist", {}).get("value")),
                    "license": lic, "license_url": m.get("LicenseUrl", {}).get("value", ""),
                    "page": ii["descriptionurl"]}
        print(pose, sum(1 for k in credits if k.startswith(pose + "/")), flush=True)
    with open(os.path.join(OUT, "credits.json"), "w", encoding="utf-8") as fh:
        json.dump(credits, fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
