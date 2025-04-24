import cv2
import numpy as np
from pathlib import Path
import argparse
from ultralytics import YOLO
import easyocr
from shapely.geometry import Polygon
from shapely.ops import unary_union

DEST_DIR = Path("./pipeline_out")

## ALLOW USERS TO SET THESE
MERGE_IOU_THRESHOLD = 0.3
PRIVACY_SCORE_THRESHOLD = 0.5
GAUSSIAN_BLUR = True

model = YOLO("runs/yolo11s/weights/best.pt")

# Load OCR model
ocr_reader = easyocr.Reader(["en"])


YOLO_CLASSES = [
    "address",
    "advertisement",
    "business_sign",
    "electronicscreens",
    "face",
    "legible_text",
    "license_plate",
    "personal_document",
    "photo",
    "street_name",
]

CLASS_FEATURE_WEIGHTS = {
    "address": {"text_keywords": 0.4, "ocr_confidence": 0.3, "size": 0.2, "base": 0.1},
    "advertisement": {
        "text_keywords": 0.3,
        "ocr_confidence": 0.3,
        "size": 0.2,
        "base": 0.2,
    },
    "business_sign": {
        "text_keywords": 0.2,
        "ocr_confidence": 0.3,
        "size": 0.3,
        "base": 0.2,
    },
    "electronicscreens": {
        "text_keywords": 0.3,
        "ocr_confidence": 0.3,
        "size": 0.2,
        "base": 0.2,
    },
    "face": {"blurriness": 0.4, "face_count": 0.3, "center_focus": 0.2, "base": 0.1},
    "legible_text": {"text_keywords": 0.4, "ocr_confidence": 0.4, "base": 0.2},
    "license_plate": {
        "text_keywords": 0.3,
        "ocr_confidence": 0.3,
        "size": 0.2,
        "base": 0.2,
    },
    "personal_document": {
        "text_keywords": 0.4,
        "ocr_confidence": 0.2,
        "blurriness": 0.2,
        "size": 0.2,
    },
    "photo": {"blurriness": 0.3, "face_count": 0.3, "center_focus": 0.2, "base": 0.2},
    "street_name": {
        "text_keywords": 0.3,
        "ocr_confidence": 0.3,
        "size": 0.3,
        "base": 0.1,
    },
}


def call_llm_on_text(text):
    return 0.8  # pretend score returned by local LLM TODO OMKAR FILL THIS


def compute_privacy_score(poly, texts, image, cls_id):
    pass


def get_mask_polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        return Polygon(largest[:, 0, :])
    return None


def union_segments_and_boxes(mask_polygons, ocr_results, class_ids):
    """
    Merge each YOLO segment with overlapping OCR text boxes.
    Returns list of tuples: (Polygon, [texts associated with merged regions])
    """
    updated_polygons = []
    used_ocr = set()

    # go through each object mask and then try to see what words are in it
    for mask_poly, class_id in zip(mask_polygons, class_ids):
        union_poly = mask_poly
        merged_texts = []
        for idx, (bbox, text, conf) in enumerate(ocr_results):
            box_poly = Polygon(bbox)
            if box_poly.is_valid and union_poly.intersects(box_poly):
                intersection_area = union_poly.intersection(box_poly).area
                if intersection_area / box_poly.area > MERGE_IOU_THRESHOLD:
                    union_poly = unary_union([union_poly, box_poly])
                    merged_texts.append(text)
                    used_ocr.add(idx)
        updated_polygons.append(((union_poly, merged_texts), class_id))

    # add standalone OCR boxes that were not merged into any segment
    for idx, (bbox, text, conf) in enumerate(ocr_results):
        if idx not in used_ocr:
            box_poly = Polygon(bbox)
            if box_poly.is_valid:
                updated_polygons.append(
                    ((box_poly, [text]), 6)
                )  # LEGIBLE TEXT CLASS ASSOCIATED

    return updated_polygons


def blur_regions(image, region_tuples):
    output = image.copy()
    for (poly, texts), class_id in region_tuples:
        if not poly.is_valid:
            continue

        # texts is a list of words OCR picked out within the mask region
        llm_score = call_llm_on_text(texts) if texts else 0.0
        score = compute_privacy_score(poly, texts, image, class_id)

        if llm_score != 0.0:
            privacy_score = 0.7 * score + 0.3 * llm_score
        else:
            privacy_score = score

        if privacy_score > PRIVACY_SCORE_THRESHOLD:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            int_coords = np.array(poly.exterior.coords, dtype=np.int32)
            cv2.fillPoly(mask, [int_coords], 255)
            blurred = cv2.GaussianBlur(output, (31, 31), 0)
            output[mask == 255] = blurred[mask == 255]

    return output


def process_image(image_path, output_path):
    image = cv2.imread(image_path)

    # Step 1: YOLO Segmentation
    yolo_results = model.predict(image_path, verbose=False)[0]
    mask_polygons = []
    class_ids = []
    for seg, class_id in zip(yolo_results.masks.xy, yolo_results.boxes.cls):
        poly = Polygon(seg)
        if poly.is_valid:
            mask_polygons.append(poly)
            class_ids.append(class_id)

    ocr_results = ocr_reader.readtext(image)

    # Merge OCR results into segments, track texts
    merged_regions_with_texts = union_segments_and_boxes(
        mask_polygons, ocr_results, class_ids
    )

    # Blur only if sensitive
    output_image = blur_regions(image, merged_regions_with_texts)

    cv2.imwrite(output_path, output_image)
    print(f"Processed image saved to: {output_path}")


if __name__ == "__main__":
    msg = """
    Takes a CL argument for the path to the image to process. 
    TODO: allow directory as target.
    """
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument(
        "input",
        nargs="?",
        default="../image_privacy_data/multiclass/train/images/010d9a7f0d2e0622_jpg.rf.2ebd2db9331fe38d6a3b868fe56d93bb.jpg",
    )
    parser.add_argument("-t", "--target_directory", default=DEST_DIR)

    if not DEST_DIR.exists():
        DEST_DIR.mkdir(parents=True)

    args = parser.parse_args()

    input = Path(args.input)
    output_path = args.target_directory
    supported_types = [".jpg", ".png"]

    # handle directory as input
    # only iterates through files in directory itself:
    # does not check subdirectories.
    if input.is_dir():
        for dir_item in input.iterdir():
            if not dir_item.is_file():
                continue
            if dir_item.suffix not in supported_types:
                continue
            process_image(input, output_path)
        quit()

    # handle single file input
    if input.suffix not in supported_types:
        raise ValueError(f"Unsupported file type: {input.suffix}")
    process_image(input, output_path)
