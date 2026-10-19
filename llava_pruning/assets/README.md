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

The loader uses this offline snapshot only if existing test folders and a
local split CSV are absent. A missing listed image is an error, never silently
skipped. Use `--visa-split-csv` to select an explicit, documented alternative.
