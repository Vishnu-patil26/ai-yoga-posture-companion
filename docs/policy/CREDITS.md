# Credits, licences and copyright

This project uses third-party images and datasets. Everything below is used under the licence stated; attribution is a condition of those licences.

## Datasets used to fit and validate the pose references

| Dataset | Used for | Licence / terms |
|---|---|---|
| TensorFlow / Moroney 5-class yoga set (tree, warrior, chair, dog, cobra) | Fitting + held-out validation of Vrikshasana, Virabhadrasana, Utkatasana, Adho Mukha, Bhujangasana | Public download used for research/education; images are not redistributed in this repository |
| *Yoga for all: A Comprehensive Collection of Yoga Images and Videos dataset* (Zenodo 7818789, Mendeley jc4mmnvcdk) | Tadasana, Marjaryasana-Bitilasana, Bhujangasana right/wrong labels - fitting, validation, angle-deviation study | CC BY 4.0. Only a sample is downloaded by `tools/datasets/get_yoga_for_all.py`; not redistributed here |
| Wikimedia Commons categories Trikoṇāsana, Bālāsana, Sukhāsana | Fitting + hold-out validation of Trikonasana, Balasana, Sukhasana | Per-file CC licences, recorded in `data/datasets/commons/credits.json` (written by `tools/datasets/get_commons_poses.py`) |
| Yoga-82, Kaggle sets named in `dataset.pdf` | **Not used** - need a signed-in account / request form | - |

## Gallery photographs (`assets/gallery/`)

| Pose | Author | Licence | Source |
|---|---|---|---|
| tadasana | Mr. Yoga | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [File:Mr-yoga-mountain-pose-2.jpg](https://commons.wikimedia.org/wiki/File:Mr-yoga-mountain-pose-2.jpg) |
| trikonasana | Kennguru | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0) | [File:Trikonasana Yoga-Asana Nina-Mel.jpg](https://commons.wikimedia.org/wiki/File:Trikonasana_Yoga-Asana_Nina-Mel.jpg) |
| virabhadrasana | lululemon athletica | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0) | [File:Yoga Warrior I.jpg](https://commons.wikimedia.org/wiki/File:Yoga_Warrior_I.jpg) |
| utkatasana | Kennguru | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0) | [File:Utkatasana Yoga-Asana Nina-Mel.jpg](https://commons.wikimedia.org/wiki/File:Utkatasana_Yoga-Asana_Nina-Mel.jpg) |
| vrikshasana | Kennguru | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0) | [File:Vriksasana Yoga-Asana Nina-Mel.jpg](https://commons.wikimedia.org/wiki/File:Vriksasana_Yoga-Asana_Nina-Mel.jpg) |
| adho_mukha | Iveto | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0) | [File:Downward-Facing-Dog.JPG](https://commons.wikimedia.org/wiki/File:Downward-Facing-Dog.JPG) |
| bhujangasana | Kennguru | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0) | [File:Bhujangasana Yoga-Asana Nina-Mel.jpg](https://commons.wikimedia.org/wiki/File:Bhujangasana_Yoga-Asana_Nina-Mel.jpg) |
| marjaryasana | Mary O'Neill | [Public domain]() | [File:Yoga at Your Park - Bitilasana.jpg](https://commons.wikimedia.org/wiki/File:Yoga_at_Your_Park_-_Bitilasana.jpg) |
| balasana | Daniel Case | [CC BY-SA 3.0](http://creativecommons.org/licenses/by-sa/3.0/) | [File:Hatha yoga child pose.jpg](https://commons.wikimedia.org/wiki/File:Hatha_yoga_child_pose.jpg) |
| sukhasana | UmaPrykhodko | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [File:Sukhaasana.png](https://commons.wikimedia.org/wiki/File:Sukhaasana.png) |

## Volunteer photographs

Photographs the team collects with `collect.py` stay with the volunteer who took part: each volunteer keeps their copyright, grants a licence for this project's fitting and validation only, and can withdraw by ID at any time. See `docs/policy/COLLECTION_PROTOCOL.md`.

## Software

MediaPipe (Apache-2.0), OpenCV (Apache-2.0), NumPy, Pillow, pyttsx3 - used under their own licences.
