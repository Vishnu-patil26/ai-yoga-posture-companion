# Datasets in this repository

All four datasets used to fit and validate the pose library are included here. Three carry licences that
allow redistribution with attribution; the fourth (`yoga107/`) is a collection of web photographs whose host
states no licence - it is included by the project owner for research and education, with its true
provenance recorded below. Everything here is used for this college project only.

| Folder | Source | Licence | What is here |
|---|---|---|---|
| `yoga_poses/` | TensorFlow / Laurence Moroney yoga-pose dataset: <https://laurencemoroney.com/2021/08/23/yogapose-dataset.html> | Apache-2.0, Copyright 2021 The TensorFlow Authors (see `NOTICE`) | Unchanged `train/` and `test/` splits for chair, cobra, dog, tree, warrior (1,495 files) |
| `yoga_for_all/` | Kanorewala B., Suryawanshi Y., Patil K., Gunjal N. (2023). *Yoga for all: A Comprehensive Collection of Yoga Images and Videos dataset.* Zenodo. <https://doi.org/10.5281/zenodo.7818789> (Mendeley `jc4mmnvcdk`) | CC BY 4.0 | **Changed:** a sample of 468 of the original 4,223 Tadasana, Cat-Cow and Cobra photos, EXIF-rotated, downscaled to at most 1,024 px, folders renamed to `<pose>/right\|wrong/step_*` |
| `yoga107/` | Hugging Face `rotemvahava/yoga-poses-107` (5,994 web photographs of 107 asanas; it appears to be the same collection as the Kaggle "Yoga Posture Dataset" named in `dataset.pdf`) | **Not stated by the host.** Photographs belong to their original photographers | The 697 photos of the 11 classes the project uses (Tadasana, Urdhva Hastasana, Trikonasana, Warrior II, Utkatasana, Down Dog, Cobra, Cat, Cow, Balasana, Sukhasana), EXIF-rotated, downscaled to at most 1,024 px |
| `commons/` | Wikimedia Commons category Trikoṇāsana (only the photos the fit actually uses) | Per-file Creative Commons: author, licence and source page for every photo are in `commons/credits.json` | `trikonasana/`, 29 photos downscaled to at most 1,024 px |

`class_signatures.json` and `fitted_vrikshasana.json` are small derived files from fitting.

## Provenance of `yoga107/` and takedown

The host (Hugging Face `rotemvahava/yoga-poses-107`) publishes no licence, and the photographs were collected
from the web, so their copyright belongs to the original photographers. They are included here unmodified in
content (only resized) for research and education. **If you are a rights holder and want a photo removed,
open an issue and it will be deleted from the repository and the derived tables.** The other 4,994 photos in
that collection are not used and not included; the raw 1.1 GB download (`_yoga107_parquet/`) is git-ignored.

## Not included

* The raw 1.1 GB parquet download, and the 96 other pose classes of the 107-pose collection.
* **Volunteer photos** from `collect.py` (`data/collected/`) are personal data and are never committed.

## Re-downloading

`python bootstrap.py --dataset` (TF set), `python tools/datasets/get_yoga_for_all.py`,
`python tools/datasets/get_commons_poses.py`, `python tools/datasets/get_yoga107.py`.
Which photos fit which pose is `tools/fitting/build_library.py`.
