## Data setup for YOLO multiclass

For YOLO scripts to work, there has to be a directory parallel to `image_privacy`
called `image_privacy_data`.
Data related to general object classification and segmentation goes in `image_privacy_data/multiclass`.
The `yolo_data_setup.py` script takes files in `image_privacy_data/multiclass/unpartitioned`
in `images` and `labels` subdirectories and partitions them into
`train`, `val`, and `test` subdirectories in `image_privacy_data/multiclass`.
If run with data already partitioned in this way, it will redo the partition.
`yolo_data_setup.py` can be imported as a module, and the `display_counts()` function
used to display counts for the different object classes.

## Data setup for YOLO legible text

Data related to segmenting legible text goes in `image_privacy_data/text`,
pulled from [Kaggle](https://www.kaggle.com/datasets/c7934597/cocotext-v20?resource=download).

## training

Train by running `train.py`, which will also evaluate on the val set.
Run `evaluate.py` to do just evaluation.
`train.py` takes a positional argument for YOLO model to use,
defaults to `yolo11n-seg.yaml`.

TODO: experiment with using `yolo11m.pt`, and `yolo11l.pt`.

## Pathing notes

For some reason, I had to use absolute paths in the yaml file for YOLO to work.
Additionally, be sure to edit .config/Ultralytics/settings.json so that it points at `image_privacy_data`,
Or alternatively put contents of `image_privacy_data` in default data directory for YOLO.
