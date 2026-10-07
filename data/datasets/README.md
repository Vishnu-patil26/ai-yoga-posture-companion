# Datasets in this repository

Three of the four datasets used to fit and validate the pose library are included here, because
their licences allow redistribution with attribution. The fourth is **not** included and is
fetched by script. Everything here is used for research and education in this college project.

| Folder | Source | Licence | What is here |
|---|---|---|---|
| `yoga_poses/` | TensorFlow / Laurence Moroney yoga-pose dataset: <https://laurencemoroney.com/2021/08/23/yogapose-dataset.html> | Apache-2.0, Copyright 2021 The TensorFlow Authors (see `NOTICE`) | Unchanged `train/` and `test/` splits for chair, cobra, dog, tree, warrior (1,495 files) |
| `yoga_for_all/` | Kanorewala B., Suryawanshi Y., Patil K., Gunjal N. (2023). *Yoga for all: A Comprehensive Collection of Yoga Images and Videos dataset.* Zenodo. <https://doi.org/10.5281/zenodo.7818789> (Mendeley `jc4mmnvcdk`) | CC BY 4.0 | **Changed:** a sample of 468 of the original 4,223 Tadasana, Cat-Cow and Cobra photos, EXIF-rotated, downscaled to at most 1,024 px, folders renamed to `<pose>/right\|wrong/step_*` |
| `commons/` | Wikimedia Commons category Trikoṇāsana (only the photos the fit actually uses) | Per-file Creative Commons: author, licence and source page for every photo are in `commons/credits.json` | `trikonasana/`, 29 photos downscaled to at most 1,024 px |

`class_signatures.json` and `fitted_vrikshasana.json` are small derived files from fitting.

## Not included

* **107-pose yoga set** (Hugging Face `rotemvahava/yoga-poses-107`, 5,994 web photographs). The host
  states no licence and the photographs belong to their original photographers, so it is not
  redistributed in this public repository. It supplies the fits for Tadasana, Trikonasana,
  Warrior II, Balasana, Sukhasana and Cat-Cow. Fetch it with
  `python tools/datasets/get_yoga107.py` (about 1.1 GB; extracted to `yoga107/`, git-ignored).
  The fitted result is already committed in `data/asana_fits.json`, so the app does not need it.
* **Volunteer photos** from `collect.py` (`data/collected/`) are personal data and are never committed.

## Re-downloading

`python bootstrap.py --dataset` (TF set), `python tools/datasets/get_yoga_for_all.py`,
`python tools/datasets/get_commons_poses.py`, `python tools/datasets/get_yoga107.py`.
Which photos fit which pose is `tools/fitting/build_library.py`.
