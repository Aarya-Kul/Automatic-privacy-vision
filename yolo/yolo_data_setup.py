import warnings
import yaml
from pathlib import Path
from shutil import rmtree
import numpy as np

np.random.seed(1234)


def get_labelled_data(image_files: list, label_files: list):
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

    image_list = []
    label_list = []

    image_list.extend([pair[0] for pair in paired_data])
    label_list.extend([pair[1] for pair in paired_data])

    return image_list, label_list


def split_data(image_files: list, label_files: list, train_proportion=0.8) -> tuple:
    val_proportion = 1 - train_proportion
    val_length = int(len(image_files) * val_proportion)
    val_indices = np.random.choice(
        np.arange(0, len(image_files)), val_length, replace=False
    )
    train_indices = np.arange(0, len(image_files))
    train_indices = np.delete(train_indices, val_indices)
    val_set = [(image_files[i], label_files[i]) for i in val_indices]
    train_set = [(image_files[i], label_files[i]) for i in train_indices]

    return train_set, val_set


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


def make_symlinks(symlink_from, symlink_to, path_mod=None):
    """
    Takes a directory path (symlink_from) and a list of file paths (symlink_to).
    Creates symbolic links in the directory path to all the file paths in the list.
    """
    if isinstance(symlink_from, str):
        symlink_from = Path(symlink_from)

    if not isinstance(symlink_to, list):
        symlink_to = [symlink_from]

    if isinstance(path_mod, str):
        path_mod = Path(path_mod)

    for i, link in enumerate(symlink_to):
        if path_mod:
            symlink_to[i] = path_mod / str(link)
            continue
        if isinstance(link, str):
            symlink_to[i] = Path(link)

    counter = 0

    for file in symlink_to:
        new_path = symlink_from / file.name
        new_path.symlink_to(file)
        counter += 1

    print(f"{counter} symlinks created in {symlink_from}")


if Path("./images").exists():
    rmtree(Path("./images"))
Path("./images").mkdir()
path_i_train = Path("./images/train")
path_i_train.mkdir()
path_i_val = Path("./images/val")
path_i_val.mkdir()

if Path("./labels").exists():
    rmtree(Path("./labels"))
Path("./labels").mkdir()
path_l_train = Path("./labels/train")
path_l_train.mkdir()
path_l_val = Path("./labels/val")
path_l_val.mkdir()

images_source = Path("../data/images/image_pool")
labels_source = Path("../data/labels/image_pool")

images_files = get_filenames(images_source, "jpg", exclusions="unbounded")
labels_files = get_filenames(labels_source, "txt", exclusions="unbounded")

images, labels = get_labelled_data(images_files, labels_files)
train_set, val_set = split_data(images, labels)

# create symlinks to image and label files
train_images = [pair[0] for pair in train_set]
train_labels = [pair[1] for pair in train_set]
val_images = [pair[0] for pair in val_set]
val_labels = [pair[1] for pair in val_set]

make_symlinks(path_i_train, train_images, "../..")
make_symlinks(path_l_train, train_labels, "../..")
make_symlinks(path_i_val, val_images, "../..")
make_symlinks(path_l_val, val_labels, "../..")

# create yaml file
yaml_path = Path("./dataset.yaml")
yaml_path.unlink(missing_ok=True)

yaml_data = {
    "path": ".",
    "train": "images/train",
    "val": "images/val",
    # optional "test": ...,
    "names": {0: "face", 1: "license plate", 2: "picture", 3: "street sign"},
}

with open(yaml_path, "w") as f:
    yaml.dump(yaml_data, f)

print(f"data written to {yaml_path}")
