# script to separate data saved in train file into train, test, and val sets
# running this on already partitioned data will redo partition by gathering files in /train, /val, and /test
# partition is random, save for ensuring that least represented object classes are present in all partitions


import warnings
from pathlib import Path
import numpy as np

np.random.seed(1234)

TRAIN_PART = 0.8
VAL_PART = 0.1
TEST_PART = 0.1

DATA_DIR = Path("../../image_privacy_data")

CLASS_MAPPINGS = {
    0: "address",
    1: "advertisement",
    2: "business_sign",
    3: "electronicscreens",
    4: "face",
    5: "legible_text",
    6: "license_plate",
    7: "personal_document",
    8: "photo",
    9: "street_name",
}


def pair_files(image_files: list, label_files: list) -> tuple:
    i_paths = sorted(image_files)
    l_paths = sorted(label_files)
    paired_data = [(image, label) for image, label in zip(i_paths, l_paths)]

    # make sure filenames match between jpg and txt files
    for pair in paired_data:
        assert Path(pair[0]).stem == Path(pair[1]).stem, (
            f"{pair[0]} does not match {pair[1]}"
        )

    if len(paired_data) != len(i_paths) and len(paired_data) != len(l_paths):
        warnings.warn("The number of jpg files does not match the number of txt files")

    image_list, label_list = list(zip(*paired_data))

    return image_list, label_list


def count_objects(label_files: list) -> tuple[dict, list]:
    """
    Returns dictionary sorted from lowest to highest count,
    and a list of lists with object labels for each image.
    """
    count_dict = {}
    object_classes = []
    for txt in label_files:
        label_objects = []
        with open(txt, "r") as f:
            for line in f:
                label_objects.append(int(line.split(" ")[0]))
                count_dict[label_objects[-1]] = count_dict.get(label_objects[-1], 0) + 1
            object_classes.append(label_objects)  # list of lists
    count_dict = dict(sorted(count_dict.items(), key=lambda item: item[1]))
    return count_dict, object_classes


def split_data(image_files: list, label_files: list) -> tuple:
    counts, object_classes = count_objects(label_files)
    num_classes = len(counts.keys())

    # encoding array with columns for each object class,
    # encodes with 1 or 0 whether object class is present in image
    sampling_array = np.zeros((len(image_files), num_classes + 1), dtype=np.int32)
    sampling_array[:, 0] = np.arange(len(image_files), dtype=sampling_array.dtype)

    # set encodings
    for i, label_objects in enumerate(object_classes):
        sampling_array[i, 1:] = [
            1 if n in label_objects else 0 for n in range(num_classes)
        ]

    # construct train/val/test partition
    # by sampling from least represented classes first
    val_set_indices = []
    test_set_indices = []
    train_set_indices = []

    for object_class, count in counts.items():
        print(
            f"Partitioning images with object class {CLASS_MAPPINGS[object_class]} present, with {count} such objects present in the dataset."
        )
        to_partition = sampling_array[sampling_array[:, object_class + 1] == 1]
        print(f"These objects are present in {len(to_partition)} images.")
        class_indices = to_partition[:, 0]
        val_length = int(to_partition.shape[0] * VAL_PART)
        test_length = int(to_partition.shape[0] * TEST_PART)

        val_indices = np.random.choice(
            np.arange(to_partition.shape[0]), val_length, replace=False
        )
        val_set_indices.extend([int(row[0]) for row in to_partition[val_indices]])
        to_partition = np.delete(to_partition, val_indices, axis=0)

        test_indices = np.random.choice(
            np.arange(to_partition.shape[0]), test_length, replace=False
        )
        test_set_indices.extend([int(row[0]) for row in to_partition[test_indices]])
        to_partition = np.delete(to_partition, test_indices, axis=0)

        train_set_indices.extend([int(row[0]) for row in to_partition])
        assert len(to_partition) == len(class_indices) - int(
            len(class_indices) * VAL_PART
        ) - int(len(class_indices) * TEST_PART), (
            f"Lenths of train({len(to_partition)}), val({int(len(class_indices) * VAL_PART)}), and test({int(len(class_indices) * TEST_PART)}) partitions don't align"
        )
        assert (
            len(to_partition) >= len(class_indices) * TRAIN_PART
            and len(to_partition) < len(class_indices) * TRAIN_PART + 3
        ), "Lenths of train, val, and test partitions don't align"
        class_indices = set(class_indices)
        deletion_mask = sampling_array[:, 0]

        def membership_test(x):
            if x in class_indices:
                return True
            return False

        vec_membership_test = np.vectorize(membership_test)
        deletion_mask = vec_membership_test(deletion_mask)
        sampling_array = sampling_array[~deletion_mask]

    unique_val = set(val_set_indices)
    unique_test = set(test_set_indices)
    unique_train = set(train_set_indices)
    assert len(unique_train) == len(train_set_indices), "duplication in partition"
    assert len(unique_test) == len(test_set_indices), "duplication in test partition"
    assert len(unique_val) == len(val_set_indices), "duplication in val partition"
    assert len(unique_train.intersection(unique_test)) == 0, (
        "duplication between train and test partitions"
    )
    assert len(unique_train.intersection(unique_val)) == 0, (
        "duplication between train and val partitions"
    )
    assert len(unique_val.intersection(unique_test)) == 0, (
        "duplication between val and test partitions"
    )

    train_set = [(image_files[i], label_files[i]) for i in train_set_indices]
    val_set = [(image_files[i], label_files[i]) for i in val_set_indices]
    test_set = [(image_files[i], label_files[i]) for i in test_set_indices]

    return train_set, val_set, test_set


def get_filenames(dir_path, extension: str, exclusions=None) -> list:
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    assert dir_path.is_dir()

    files = []

    if exclusions is None:
        exclusions = []

    elif isinstance(exclusions, str):
        exclusions = [exclusions]

    for file in dir_path.glob(f"*.{extension}"):
        files.append(file)

    for content in dir_path.iterdir():
        if not content.is_dir():
            continue
        if content.stem in exclusions:
            continue
        files.extend(get_filenames(content, extension, exclusions))

    return files


path_train = DATA_DIR / "train"
(path_train / "images").mkdir(parents=True, exist_ok=True)
(path_train / "labels").mkdir(exist_ok=True)
path_val = DATA_DIR / "val"
(path_val / "images").mkdir(parents=True, exist_ok=True)
(path_val / "labels").mkdir(exist_ok=True)
path_test = DATA_DIR / "test"
(path_test / "images").mkdir(parents=True, exist_ok=True)
(path_test / "labels").mkdir(exist_ok=True)

images_files = (
    get_filenames(path_train / "images", "jpg")
    + get_filenames(path_val / "images", "jpg")
    + get_filenames(path_test / "images", "jpg")
)
labels_files = (
    get_filenames(path_train / "labels", "txt")
    + get_filenames(path_val / "labels", "txt")
    + get_filenames(path_test / "labels", "txt")
)

images, labels = pair_files(images_files, labels_files)
train_set, val_set, test_set = split_data(images, labels)

train_images = [pair[0] for pair in train_set]
train_labels = [pair[1] for pair in train_set]
val_images = [pair[0] for pair in val_set]
val_labels = [pair[1] for pair in val_set]
test_images = [pair[0] for pair in test_set]
test_labels = [pair[1] for pair in test_set]

print(
    f"There are {len(train_images)} images in the training set, {len(val_images)} in the validation set, and {len(test_images)} in the test set"
)

for image_file, label_file in zip(train_images, train_labels):
    file_stem = image_file.stem
    new_image_file = DATA_DIR / f"train/images/{file_stem}"
    new_label_file = DATA_DIR / f"train/labels/{file_stem}"
    if not new_image_file.exists():
        image_file.rename(new_image_file)
        label_file.rename(new_label_file)

for image_file, label_file in zip(val_images, val_labels):
    file_stem = image_file.stem
    new_image_file = DATA_DIR / f"val/images/{file_stem}"
    new_label_file = DATA_DIR / f"val/labels/{file_stem}"
    if not new_image_file.exists():
        image_file.rename(new_image_file)
        label_file.rename(new_label_file)

for image_file, label_file in zip(test_images, test_labels):
    file_stem = image_file.stem
    new_image_file = DATA_DIR / f"test/images/{file_stem}"
    new_label_file = DATA_DIR / f"test/labels/{file_stem}"
    if not new_image_file.exists():
        image_file.rename(new_image_file)
        label_file.rename(new_label_file)
