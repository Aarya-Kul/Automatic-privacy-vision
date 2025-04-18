## Data setup for YOLO

For YOLO scripts to work, there has to be a directory parallel to `image_privacy`
called `image_privacy_data`.
The `yolo_data_setup.py` script takes files in `image_privacy_data/unpartitioned`
in `images` and `labels` subdirectories and partitions them into
`train`, `val`, and `test` subdirectories in `image_privacy_data`.
If run with data already partitioned in this way, it will redo the partition.
`yolo_data_setup.py` can be imported as a module, and the `display_counts()` function
used to display counts for the different object classes.
