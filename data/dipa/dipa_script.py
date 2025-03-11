#!/usr/bin/env python

from pathlib import Path
import json

from dipa_classifier import (
    sort_annotations,
    save_image_and_txt,
)

prolific_dir = Path("./annotations/Prolific/labels")
crowdworks_dir = Path("./annotations/CrowdWorks/labels")
prolific_gen = prolific_dir.glob("*.json")
crowdworks_gen = crowdworks_dir.glob("*.json")

prolific_annotations = []
for file in prolific_gen:
    with open(file) as f:
        cur_ann = json.load(f)
        cur_ann["file"] = f"{file.stem.split('_')[0]}.jpg"
        prolific_annotations.append(cur_ann)

crowdworks_annotations = []
for file in crowdworks_gen:
    with open(file) as f:
        cur_ann = json.load(f)
        cur_ann["file"] = f"{file.stem.split('_')[0]}.jpg"
        crowdworks_annotations.append(cur_ann)

prolific_street_count = 0
for ann in prolific_annotations:
    for values in ann["manualAnnotation"].values():
        if (
            "street" in values["category"].lower()
            and "sign" in values["category"].lower()
        ):
            prolific_street_count += 1
            continue
    for key in ann["defaultAnnotation"].keys():
        if "street" in key.lower() and "sign" in key.lower():
            prolific_street_count += 1

crowdworks_street_count = 0
for ann in crowdworks_annotations:
    for values in ann["manualAnnotation"].values():
        if (
            "street" in values["category"].lower()
            and "sign" in values["category"].lower()
        ):
            crowdworks_street_count += 1
            continue
    for key in ann["defaultAnnotation"].keys():
        if "street" in key.lower() and "sign" in key.lower():
            crowdworks_street_count += 1

print(
    f"""
    prolific has {prolific_street_count} annotations with street signs.
    crowdworks has {crowdworks_street_count} annotations with street signs. 
    The total is {crowdworks_street_count + prolific_street_count}.
    """
)

class_map = {"face": 0, "license plate": 1, "picture": 2, "street sign": 3}
categories = list(class_map.keys())
sorted_annotations = sort_annotations(
    categories, prolific_annotations + crowdworks_annotations
)
save_image_and_txt(sorted_annotations, class_map)
