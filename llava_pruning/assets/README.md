# VisA official test split

`visa_1cls_test.csv` is derived from the VisA authors' official 1-class split:
https://github.com/amazon-science/spot-diff/blob/main/split_csv/1cls.csv

Authors: Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, Onkar Dabeer.
Paper: SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection
and Segmentation (ECCV 2022), https://arxiv.org/abs/2207.14315.

Changes: retained only rows whose `split` is `test`; retained the `object`,
`split`, `label`, and `image` columns, dropped the unused mask column, and
serialized CSV with quoting. Image paths and labels are unchanged. This
snapshot has 2,162 images (962 normal, 1,200 anomaly), across all 12 categories.
No images or masks are redistributed. This is not a new random split.

VisA dataset attribution/license: CC BY 4.0,
https://github.com/amazon-science/spot-diff/blob/main/LICENSE-DATASET
and https://creativecommons.org/licenses/by/4.0/.

The loader prefers a local `split_csv/1cls.csv` (or `1cls.csv`) over existing
test folders. It uses this offline snapshot only if both are absent. A selected
CSV with missing images is an error, never silently skipped or replaced by a
directory scan. Use `--visa-split-csv` to select an explicit, documented split.
