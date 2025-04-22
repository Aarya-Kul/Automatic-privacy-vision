import cv2
import numpy as np
import os

# Class label map
LABEL_MAP = {
    0: "address",
    1: "advertisement",
    2: "business_sign",
    3: "electronicscreens",
    4: "face",
    5: "legible_text",
    6: "license_plate",
    7: "personal_document",
    8: "photo",
    9: "street_name"
}

def parse_annotation_line(line):
    parts = line.strip().split()
    if len(parts) < 4:
        return None
    try:
        class_id = int(parts[0])
        coords = list(map(float, parts[2:]))  # skip class_id and polygon_id
        return class_id, coords
    except ValueError:
        return None

def extract_polygon_patch(img, coords, width, height):
    points = np.array([
        [int(x * width), int(y * height)] for x, y in zip(coords[::2], coords[1::2])
    ], dtype=np.int32)

    # Create mask from polygon
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)

    # Apply mask to image
    masked = cv2.bitwise_and(img, img, mask=mask)

    # Crop to bounding box of polygon
    x, y, w_crop, h_crop = cv2.boundingRect(points)
    cropped = masked[y:y+h_crop, x:x+w_crop]
    return cropped

def process_image_and_annotations(image_path, annotation_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    with open(annotation_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parsed = parse_annotation_line(line)
        if parsed is None:
            print(f"Skipping malformed line {i}: {line.strip()}")
            continue

        class_id, coords = parsed
        class_name = LABEL_MAP.get(class_id, f"class_{class_id}")
        patch = extract_polygon_patch(img, coords, width, height)

        output_filename = f"{class_name}_{i}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, patch)
        print(f"Saved patch: {output_path}")

if __name__ == "__main__":
    # process_image_and_annotations(
    # image_path="/Users/hendrikmayer/Desktop/ImagePrivacyProject/image_privacy/example_text_and_data/0a7c597abf1e90d4_jpg.rf.0c48da804f75a8a12ef1ae9c81aa2166.jpg",
    # annotation_path="/Users/hendrikmayer/Desktop/ImagePrivacyProject/image_privacy/example_text_and_data/0a7c597abf1e90d4_jpg.rf.0c48da804f75a8a12ef1ae9c81aa2166.txt",
    # output_dir="patches_image_001"
    # )
    process_image_and_annotations(
    image_path="/Users/hendrikmayer/Desktop/ImagePrivacyProject/image_privacy/example_text_and_data/00e65d13159b498b_jpg.rf.bf431d2b995469e37b3941a9a9ead48b.jpg",
    annotation_path="/Users/hendrikmayer/Desktop/ImagePrivacyProject/image_privacy/example_text_and_data/00e65d13159b498b_jpg.rf.bf431d2b995469e37b3941a9a9ead48b.txt",
    output_dir="patches_image_001"
    )
    