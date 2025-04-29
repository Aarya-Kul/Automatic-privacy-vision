import cv2
import numpy as np
from pathlib import Path
import argparse
from ultralytics import YOLO
import easyocr
import requests
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

# LITERALLY IDK this is just random initial weights
CLASS_FEATURE_WEIGHTS = {
    "address": {"ocr_confidence": 0.4, "size": 0.2, "center_focus": 0.2, "base": 0.2},
    "advertisement": {"ocr_confidence": 0.3, "size": 0.2, "background_complexity": 0.3, "base": 0.2},
    "business_sign": {"ocr_confidence": 0.3, "size": 0.3, "background_complexity": 0.2, "base": 0.2},
    "electronicscreens": {"ocr_confidence": 0.3, "background_complexity": 0.3, "size": 0.2, "base": 0.2},
    "face": {"blurriness": 0.25, "center_focus": 0.25, "background_complexity": 0.2, "base": 0.2, "ocr_confidence": 0.1},
    "legible_text": {"ocr_confidence": 0.4, "background_complexity": 0.3, "base": 0.3},
    "license_plate": {"ocr_confidence": 0.3, "size": 0.3, "base": 0.4},
    "personal_document": {"ocr_confidence": 0.3, "blurriness": 0.2, "size": 0.2, "background_complexity": 0.2, "base": 0.1},
    "photo": {"blurriness": 0.3, "center_focus": 0.3, "background_complexity": 0.2, "base": 0.2},
    "street_name": {"ocr_confidence": 0.3, "size": 0.3,"center_focus": 0.2, "base": 0.2},
}


# def call_llm_on_text(text, class_id):
#     return 0.8  # pretend score returned by local LLM TODO OMKAR FILL THIS



def call_llm_on_text(text, class_id):
    """
    Calls local Ollama server running 'llama3' to get a privacy score between 0.0 and 1.0.
    If Ollama is not running, fallback to dummy score.
    """
    prompt = f"""
You are determining the privacy sensitivity of regions detected in images.

Region class: {YOLO_CLASSES[int(class_id)]}
Detected text: {text}

Output a single number between 0.0 (completely non-sensitive) and 1.0 (extremely private).
Only output the number, nothing else.
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3",
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0.0},
                "stream": False
            },
            timeout=10,
        )
        output = response.json()["message"]["content"].strip()
        return float(output)
    except Exception as e:
        print(f"[Warning] Could not connect to Ollama or parse output. Using dummy score. (Reason: {e})")
        return 0.5
    
def compute_privacy_score(poly, texts, image, class_id):
    class_name = YOLO_CLASSES[int(class_id)]
    weights = CLASS_FEATURE_WEIGHTS.get(class_name, {})

    score = 0.0
    total_weight = 0.0

    if "base" in weights:
        score += weights["base"] * 1.0
        total_weight += weights["base"]

    if "ocr_confidence" in weights:
        if texts:
            conf_score = np.mean([conf for _, conf in texts])
            score += weights["ocr_confidence"] * conf_score
            total_weight += weights["ocr_confidence"]

    if "blurriness" in weights or "background_complexity" in weights or "center_focus" in weights:
        x, y, w, h = map(int, poly.bounds)
        crop = image[y:h, x:w]
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if "blurriness" in weights:
            lap_var = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
            blur_score = min(1.0, max(0.0, 1.0 - lap_var / 100.0))
            score += weights["blurriness"] * blur_score
            total_weight += weights["blurriness"]

        if "background_complexity" in weights:
            edges = cv2.Canny(gray_crop, 100, 200)
            complexity_score = min(1.0, np.mean(edges) / 255.0)
            score += weights["background_complexity"] * complexity_score
            total_weight += weights["background_complexity"]

    if "center_focus" in weights:
        image_center = (image.shape[1] // 2, image.shape[0] // 2)
        poly_center = poly.centroid
        distance = np.linalg.norm(np.array([poly_center.x, poly_center.y]) - np.array(image_center))
        center_focus_score = 1.0 - min(1.0, distance / (max(image.shape[:2]) / 2))
        score += weights["center_focus"] * center_focus_score
        total_weight += weights["center_focus"]

    if "size" in weights:
        size_ratio = poly.area / (image.shape[0] * image.shape[1])
        score += weights["size"] * min(1.0, size_ratio * 5)
        total_weight += weights["size"]

    return score / total_weight if total_weight > 0 else 0.0


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
                    merged_texts.append((text, conf))
                    used_ocr.add(idx)
        updated_polygons.append(((union_poly, merged_texts), class_id))

    # add standalone OCR boxes that were not merged into any segment
    for idx, (bbox, text, conf) in enumerate(ocr_results):
        if idx not in used_ocr:
            box_poly = Polygon(bbox)
            if box_poly.is_valid:
                updated_polygons.append(((box_poly, [(text, conf)]), 6)) # LEGIBLE TEXT CLASS ASSOCIATED

    return updated_polygons


def blur_regions(image, region_tuples):
    output = image.copy()
    for (poly, texts), class_id in region_tuples:
        if not poly.is_valid:
            continue

        # texts is a list of words OCR picked out within the mask region
        llm_score = call_llm_on_text([t for t, _ in texts], class_id) if texts else 0.0
        score = compute_privacy_score(poly, texts, image, class_id)

        # if an llm score then use it. otherwise don't because there was no text for this then
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
