import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import easyocr
from shapely.geometry import Polygon
from shapely.ops import unary_union

MERGE_IOU_THRESHOLD = 0.3

model = YOLO("runs/segment/train/weights/best.pt")

# Load OCR model
ocr_reader = easyocr.Reader(['en'])


def get_mask_polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        return Polygon(largest[:, 0, :])
    return None


def union_segments_and_boxes(mask_polygons, ocr_results):
    """
    Merge each YOLO segment with overlapping OCR text boxes.
    Returns list of tuples: (Polygon, [texts associated with merged regions])
    """
    updated_polygons = []
    used_ocr = set()

    # go through each object mask and then try to see what words are in it
    for mask_poly in mask_polygons:
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
        updated_polygons.append((union_poly, merged_texts))

    # add standalone OCR boxes that were not merged into any segment
    for idx, (bbox, text, conf) in enumerate(ocr_results):
        if idx not in used_ocr:
            box_poly = Polygon(bbox)
            if box_poly.is_valid:
                updated_polygons.append((box_poly, [text]))

    return updated_polygons


def extract_sensitive_texts(ocr_results):
    """ privacy scoring and with LLM integration."""
    return True


def blur_regions(image, region_tuples):
    for poly, texts in region_tuples:
        if not poly.is_valid:
            continue
        # TODO : NEED TO IMPLEMENT THIS FUNCTION
        if not extract_sensitive_texts(texts):
            continue  # Skip blurring if region not sensitive or if a scale then modify this condition TODO
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        int_coords = np.array(poly.exterior.coords, dtype=np.int32)
        cv2.fillPoly(mask, [int_coords], 255)
        blurred = cv2.GaussianBlur(image, (31, 31), 0)
        image[mask == 255] = blurred[mask == 255]
    return image


def process_image(image_path, output_path):
    image = cv2.imread(image_path)

    # Step 1: YOLO Segmentation
    yolo_results = model.predict(image_path, verbose=False)[0]
    mask_polygons = []
    for seg, cls in zip(yolo_results.masks.xy, yolo_results.boxes.cls):
        poly = Polygon(seg)
        if poly.is_valid:
            mask_polygons.append(poly)

    ocr_results = ocr_reader.readtext(image)

    # Merge OCR results into segments, track texts
    merged_regions_with_texts = union_segments_and_boxes(mask_polygons, ocr_results)

    # Blur only if sensitive
    output_image = blur_regions(image, merged_regions_with_texts)

    cv2.imwrite(output_path, output_image)
    print(f"Processed image saved to: {output_path}")


if __name__ == "__main__":
    test_image = "C:/Aarya/umich/eecs545/final_project/image_privacy_data/multiclass/train/images/010d9a7f0d2e0622_jpg.rf.2ebd2db9331fe38d6a3b868fe56d93bb.jpg"
    output_path = "C:/Aarya/umich/eecs545/final_project/image_privacy/blurred_example5.jpg"
    process_image(test_image, output_path)
