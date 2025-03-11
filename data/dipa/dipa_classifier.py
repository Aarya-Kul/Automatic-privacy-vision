from pathlib import Path
import re
import shutil
import cv2


def is_class(category: str, description: str) -> bool:
    is_present = []
    for word in re.split(" |_", category):
        if word.lower() in description.lower():
            is_present.append(True)
        else:
            is_present.append(False)
    if all(is_present):
        return True
    return False


def sort_annotations(categories: list, annotations: list) -> dict:
    bins = {}
    for category in categories:
        bins[category] = []
    for ann in annotations:
        for category in categories:
            for key in ann["defaultAnnotation"]:
                if is_class(category, key):
                    bins[category].append(ann)
                    break
            for value in list(ann["manualAnnotation"].values()):
                description = value["category"]
                if is_class(category, description):
                    bins[category].append(ann)
                    break

    return bins


def get_manual(annotations: dict, categories=None) -> dict:
    if categories is None:
        categories = list(annotations.keys())
    if isinstance(categories, str):
        categories = [categories]
    bins = {}
    for category in categories:
        bins[category] = []
        for ann in annotations[category]:
            for value in list(ann["manualAnnotation"].values()):
                description = value["category"]
                if is_class(category, description):
                    bins[category].append(ann)
                    break
    for class_name in bins.keys():
        print(
            f"There are {len(bins[class_name])} manual annotations for class {class_name}"
        )
    return bins


def normalize_bbox(bbox: list, width, height) -> list:
    """
    Normalize by dividing x_center and width by image width, and y_center and height by image height.
    """
    bbox[0] /= width
    bbox[2] /= width
    bbox[1] /= height
    bbox[3] /= height

    return bbox


def ann_to_string(
    annotation: dict, class_name: str, class_map: dict, bbox_width, bbox_height
):
    man_ann = annotation["manualAnnotation"]
    class_number = class_map[class_name]
    bbox = None
    for key in man_ann:
        description = man_ann[key]["category"]
        if is_class(class_name, description):
            bbox = man_ann[key]["bbox"]
            break
    if bbox is None:
        return

    # normalize bbox values for YOLO
    # note that bbox is annotations is given in format [x_centre, y_centre, width, height], which is also YOLO format
    bbox = normalize_bbox(bbox, bbox_width, bbox_height)
    bbox = [str(el) for el in bbox]

    return f"{class_number} {' '.join(bbox)} \n"


def get_image_dimensions(filename, image_dir=Path("./images")) -> tuple:
    if isinstance(image_dir, str):
        image_dir = Path(image_dir)
    if not isinstance(filename, str):
        filename = str(filename)

    image = cv2.imread(str(image_dir / filename))
    height, width, _ = image.shape

    return (height, width)


# directory structure:
# ../images/image_pool/object_class/image.image,
# ../images/image_pool/object_class/unbounded/image.image,
# ../labels/image_pool/object_class/text.txt,
# ../labels/image_pool/object_class/unbounded/text.txt.


def save_image_and_txt(
    annotations: dict, class_map: dict, image_dir=None, dest_dir=None
):
    """
    Function to create/fill a directory with images and txt files suitable for use with YOLO.
    Expects a dict where each key is a class of objects and each value is a list of annotations.
    Can be passed a source directory for images and destination directory, otherwise defaults are
    ./images and ../training.
    """
    if image_dir is None:
        image_dir = Path("./images")
    elif isinstance(image_dir, str):
        image_dir = Path(image_dir)
    assert image_dir.exists(), "Image directory not found"

    if dest_dir is None:
        dest_dir = Path("..")
    elif isinstance(dest_dir, str):
        dest_dir = Path(dest_dir)

    image_dir_new = dest_dir / "images/image_pool"
    image_info_dir = dest_dir / "labels/image_pool"

    # create subdirectories for objects classes
    for key in annotations:
        class_image_dir = image_dir_new / key
        class_image_info_dir = image_info_dir / key
        # since only bounded images are useful, they go in the main directory, unbounded images go in subdirectory
        class_image_dir_unbounded = class_image_dir / "unbounded"
        class_image_info_dir_unbounded = class_image_info_dir / "unbounded"
        if not class_image_dir_unbounded.exists():
            class_image_dir_unbounded.mkdir(parents=True)
        if not class_image_info_dir_unbounded.exists():
            class_image_info_dir_unbounded.mkdir(parents=True)

    # loop through object categories to add images and labels
    for key in annotations:
        cur_image_dir = image_dir_new / key
        cur_info_dir = image_info_dir / key
        cur_manual = get_manual(annotations, key)

        # first we put in place images that do have manual annotations for the class in question, and therefore
        # bounding boxes (which we use to make the labels)
        for ann in cur_manual[key]:
            filename = ann["file"]
            label_filename = filename.split(".")[0] + ".txt"
            height, width = get_image_dimensions(filename)
            label_contents = ann_to_string(ann, key, class_map, width, height)
            cur_image_path = cur_image_dir / filename
            cur_label_path = cur_info_dir / label_filename
            if cur_label_path.exists():
                with cur_label_path.open("a") as f:
                    f.write(label_contents)
            elif not cur_image_path.exists():
                shutil.copy(image_dir / filename, cur_image_dir)
                cur_label_path.write_text(label_contents)

        # now we look through all the annotations by class, skipping any cases where we already have processed the annotation+class
        # for its manual annotation.
        for ann in annotations[key]:
            filename = ann["file"]
            if (cur_image_dir / filename).exists():
                continue
            unbounded_image_path = cur_image_dir / "unbounded"
            shutil.copy(image_dir / filename, unbounded_image_path)
