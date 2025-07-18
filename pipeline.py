import cv2
import codecs
import numpy as np
from pathlib import Path
from shutil import rmtree
import argparse
from ultralytics import YOLO
import easyocr
import requests
from shapely.geometry import Polygon
from shapely.ops import unary_union

from obfuscation.lama_inpaint import inpaint

DEST_DIR = Path("./pipeline_out")

## ALLOW USERS TO SET THESE
MERGE_IOU_THRESHOLD = 0.3
PRIVACY_SCORE_THRESHOLD = 0.5
INPAINT_THRESHOLD = None
GAUSSIAN_BLUR = True

# applying custom YOLO confidence threshold to nonfaces
YOLO_CONF_NONFACE = 0.4  # if <= 0.25 (default yolo threshold), this will have no effect

LOGGING_PATH = Path("log_pipeline.txt")

# model = YOLO("runs/yolo11s/weights/best.pt")
model = YOLO("best_yolo_weights/best.pt")


# Load OCR model
ocr_reader = easyocr.Reader(["en"])

# note that class legible_text is not present in yolo model,
# but extracted from OCR model
YOLO_CLASSES = [
    "address",
    "business_sign",
    "electronicscreens",
    "face",
    "license_plate",
    "personal_document",
    "photo",
    "street_name",
    "legible_text",
]

CLASS_FEATURE_WEIGHTS = {
    "address": {"ocr_confidence": 0.4, "size": 0.2, "center_focus": 0.2, "base": 0.4},
    "advertisement": {
        "llm": 0.3,
        "ocr_confidence": 0.3,
        "size": 0.1,
        "background_complexity": 0.1,
        "base": 0.2,
    },
    "business_sign": {
        "llm": 0.3,
        "ocr_confidence": 0.3,
        "size": 0.1,
        "background_complexity": 0.1,
        "base": 0.2,
    },
    "electronicscreens": {
        "llm": 0.4,
        "ocr_confidence": 0.2,
        "background_complexity": 0.1,
        "base": 0.3,
    },
    "face": {
        "blurriness": 0.25,
        "center_focus": 0.2,
        "background_complexity": 0.1,
        "base": 0.5,
    },
    "legible_text": {
        "llm": 0.5,
        "ocr_confidence": 0.15,
        "background_complexity": 0.15,
        "base": 0.2,
    },
    "license_plate": {"ocr_confidence": 0.3, "size": 0.1, "base": 0.6},
    "personal_document": {"llm": 0.4, "ocr_confidence": 0.2, "size": 0.1, "base": 0.3},
    "photo": {
        "blurriness": 0.3,
        "center_focus": 0.25,
        "background_complexity": 0.25,
        "base": 0.2,
    },
    "street_name": {
        "ocr_confidence": 0.4,
        "size": 0.2,
        "center_focus": 0.2,
        "base": 0.3,
    },
}


def call_llm_on_text(text, class_id):
    """
    Calls local Ollama server running 'llama3' to get a privacy score between 0.0 and 1.0.
    If Ollama is not running, fallback to dummy score.
    """
    prompt = f"""
        You are determining the privacy sensitivity of the list of text detected in a region of an image.

        Region class: {YOLO_CLASSES[int(class_id)]}
        Detected text: {text}

        Output a single number between 0.0 (completely non-sensitive text) and 1.0 (extremely private information text) with regards to the class. 
        This number should be 1.0 or close to 1.0 if the text identifies a person or place, 
        and if the text contains passwords, numerical ids, or financial information. 
        Only output the number, nothing else.
        """
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3",
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0.0},
                "stream": False,
            },
            timeout=40,
        )
        output = response.json()["message"]["content"].strip()
        return float(output)
    except Exception as e:
        print(
            f"[Warning] Could not connect to Ollama or parse output. Using dummy score. (Reason: {e})"
        )
        with open(LOGGING_PATH, "a") as f:
            f.write(
                f"[Warning] Could not connect to Ollama or parse output. Using dummy score. (Reason: {e})"
                + "\n"
            )
        return 0.5


def compute_privacy_score(poly, texts, image, class_id, counter: list, filename=None):
    class_name = YOLO_CLASSES[int(class_id)]
    weights = CLASS_FEATURE_WEIGHTS.get(class_name, {})

    counter[-1] += 1

    message = f"\n=== IMAGE {len(counter)} OBJECT {counter[-1]}: " + class_name + "==="
    print(message)

    with open(LOGGING_PATH, "a") as f:
        f.write(message + "\n")

    if filename:
        if isinstance(filename, Path):
            filename = filename.name
        with open(LOGGING_PATH, "a") as f:
            f.write(f"file: {filename}" + "\n")
        print(f"file: {filename}")

    score = 0.0
    total_weight = 0.0

    if "base" in weights:
        score += weights["base"] * 1.0
        total_weight += weights["base"]

    if "ocr_confidence" in weights:
        if texts:
            text_info = "\n".join(
                [f'text: "{text_object[0]}"' for text_object in texts]
            )
            print(text_info)
            with codecs.open(str(LOGGING_PATH), "a", "utf-8") as f:
                f.write(text_info + "\n")
            conf_score = np.mean([conf for _, conf in texts])
            print("CONFIDENCE SCORE:", conf_score)
            with open(LOGGING_PATH, "a") as f:
                f.write("CONFIDENCE SCORE: " + str(conf_score) + "\n")
            score += weights["ocr_confidence"] * conf_score
            total_weight += weights["ocr_confidence"]

    if (
        "blurriness" in weights
        or "background_complexity" in weights
        or "center_focus" in weights
    ):
        x, y, w, h = map(int, poly.bounds)
        crop = image[y:h, x:w]
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if "blurriness" in weights:
            lap_var = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
            lap_var = np.clip(lap_var, 0, 500)
            blur_score = min(1.0, lap_var / 500.0)
            print("BLUR SCORE:", blur_score)
            with open(LOGGING_PATH, "a") as f:
                f.write("BLUR SCORE: " + str(blur_score) + "\n")
            score += weights["blurriness"] * blur_score
            total_weight += weights["blurriness"]

        if "background_complexity" in weights:
            edges = cv2.Canny(gray_crop, 100, 200)
            complexity_score = min(1.0, np.mean(edges) / 255.0)
            print("COMPLEXITY SCORE:", complexity_score)
            with open(LOGGING_PATH, "a") as f:
                f.write("COMPEXITY SCORE: " + str(complexity_score) + "\n")
            score += weights["background_complexity"] * complexity_score
            total_weight += weights["background_complexity"]

    if "center_focus" in weights:
        image_center = (image.shape[1] // 2, image.shape[0] // 2)
        poly_center = poly.centroid
        distance = np.linalg.norm(
            np.array([poly_center.x, poly_center.y]) - np.array(image_center)
        )
        center_focus_score = 1.0 - min(1.0, distance / (max(image.shape[:2]) / 2))
        print("CENTER FOCUS SCORE:", center_focus_score)
        with open(LOGGING_PATH, "a") as f:
            f.write("CENTER FOCUS SCORE: " + str(center_focus_score) + "\n")
        score += weights["center_focus"] * center_focus_score
        total_weight += weights["center_focus"]

    if "size" in weights:
        size_ratio = poly.area / (image.shape[0] * image.shape[1])
        print("SIZE RATIO:", size_ratio)
        with open(LOGGING_PATH, "a") as f:
            f.write("SIZE RATIO: " + str(size_ratio) + "\n")
        score += weights["size"] * min(1.0, size_ratio * 5)
        total_weight += weights["size"]

    if "llm" in weights:
        llm_score = call_llm_on_text([t for t, _ in texts], class_id) if texts else 0.0
        print("LLM SCORE:", llm_score)
        with open(LOGGING_PATH, "a") as f:
            f.write("LLM SCORE: " + str(llm_score) + "\n")
        score += weights["llm"] * llm_score
        total_weight += weights["llm"]

    return score
    # return score / total_weight if total_weight > 0 else 0.0


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
                updated_polygons.append(
                    ((box_poly, [(text, conf)]), 8)
                )  # 8 is label for legible_text class
    return updated_polygons


def blur_regions(image, region_tuples, counter: list, filename=None):
    output = image.copy()
    for (poly, texts), class_id in region_tuples:
        if not poly.is_valid:
            continue

        # texts is a list of words OCR picked out within the mask region
        privacy_score = compute_privacy_score(
            poly, texts, image, class_id, counter=counter, filename=filename
        )
        print("PRIVACY_SCORE:", privacy_score)
        with open(LOGGING_PATH, "a") as f:
            f.write("PRIVACY SCORE: " + str(privacy_score) + "\n")

        if privacy_score > PRIVACY_SCORE_THRESHOLD:
            # select Gaussian blurring or inpainting
            # always uses blurring for faces
            # case where inpainting threshold is provided:
            if (
                INPAINT_THRESHOLD
                and privacy_score > INPAINT_THRESHOLD
                and YOLO_CLASSES[int(class_id)] == "face"
            ):
                int_coords = np.array(poly.exterior.coords, dtype=np.int32)
                output = np.asarray(inpaint(output, [int_coords]), copy=True)  # type: ignore
                continue
            # case where inpainting generally preferred over blurring
            elif GAUSSIAN_BLUR is False and YOLO_CLASSES[int(class_id)] != "face":
                int_coords = np.array(poly.exterior.coords, dtype=np.int32)
                output = np.asarray(inpaint(output, [int_coords]), copy=True)  # type: ignore
                continue
            # case where blurring is used
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            int_coords = np.array(poly.exterior.coords, dtype=np.int32)
            cv2.fillPoly(mask, [int_coords], 255)
            obfuscated = cv2.GaussianBlur(
                output, (51, 51), 0
            )  # kernel size changed from (31, 31) since some text was still legible
            output[mask == 255] = obfuscated[mask == 255]

    return output


def process_image(image_path, output_path, counter=[]):
    counter.append(0)
    image = cv2.imread(image_path)

    # Step 1: YOLO Segmentation
    yolo_results = model.predict(image_path, verbose=False, save=True)[0]
    mask_polygons = []
    class_ids = []
    # conditional avoids throwing an error if there are no yolo predictions
    if yolo_results:
        assert yolo_results.masks, f"issue with yolo_results: {yolo_results}"
        assert yolo_results.boxes, f"issue with yolo_results: {yolo_results}"
        for seg, class_id, conf_score in zip(
            yolo_results.masks.xy, yolo_results.boxes.cls, yolo_results.boxes.conf
        ):
            # the point of taking conf scores here is to skip anything with conf score <= 0.4
            # unless it's a face, to control the false positive rate (for faces, the appropriate threshold is
            # around YOLO's default of 0.25)
            conf = float(conf_score.max())
            # max taken because one object may have multiple scores for different keypoints
            if YOLO_CLASSES[int(class_id)] != "face" and conf <= YOLO_CONF_NONFACE:
                continue
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
    output_image = blur_regions(
        image, merged_regions_with_texts, counter, filename=image_path
    )

    output_filename = output_path / (image_path.stem + "_blurred" + image_path.suffix)
    cv2.imwrite(str(output_filename), output_image)

    print(f"Processed image saved to: {output_path}")
    with open(LOGGING_PATH, "a") as f:
        f.write("Processed image saved to " + str(output_path) + "\n")


if __name__ == "__main__":
    # part of a workaround for yolo saving predictions to runs/segment
    if Path("runs/sgment").exists():
        rmtree(Path("runs/segment"))

    msg = f"""
    Takes a CL argument for the path to the image to process. 
    Path for images to process can be a directory.
    Option -t <path> to specify save directory, defaults to {DEST_DIR}.
    Option -i to use inpainting instead of Gaussian blurring.
    Option -p followed by one or two arguments: first argument is privacy score threshold,
    second argument (if given) is privacy score threshold for inpainting.
    """
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument(
        "input", nargs="*", default=["pipeline_in"], help="image(s) to process"
    )
    parser.add_argument("-t", "--target_directory", default=DEST_DIR, help="save dir")

    parser.add_argument(
        "-i", "--inpaint", action="store_true", help="Inpaint instead of blur"
    )

    parser.add_argument(
        "-p",
        "--pscore",
        nargs="*",
        help="Privacy score threshold, second arg if given is inpainting threshold",
    )

    args = parser.parse_args()

    if args.pscore and len(args.pscore) > 0:
        PRIVACY_SCORE_THRESHOLD = float(args.pscore[0])
        assert PRIVACY_SCORE_THRESHOLD >= 0 and PRIVACY_SCORE_THRESHOLD < 1, (
            f"Invalid privacy score threshold: {PRIVACY_SCORE_THRESHOLD}"
        )
        if len(args.pscore) > 1:
            INPAINT_THRESHOLD = float(args.pscore[1])
            assert PRIVACY_SCORE_THRESHOLD >= 0 and PRIVACY_SCORE_THRESHOLD < 1, (
                f"Invalid inpainting threshold: {INPAINT_THRESHOLD}"
            )

    # setting GAUSSIAN_BLUR to False makes the pipeline inpaint rather than blur
    if args.inpaint:
        GAUSSIAN_BLUR = False

    input = [Path(path) for path in args.input]
    output_path = Path(args.target_directory)
    supported_types = [".jpg", ".png"]

    if not output_path.exists():
        output_path.mkdir(parents=True)

    counter = []

    for path in input:
        # handle directory as input
        # only iterates through files in directory itself:
        # does not check subdirectories.
        if path.is_dir():
            for dir_item in path.iterdir():
                if not dir_item.is_file():
                    continue
                if dir_item.suffix not in supported_types:
                    continue
                process_image(dir_item, output_path, counter=counter)
            continue

        # handle single file path
        if path.suffix not in supported_types:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        process_image(path, output_path, counter=counter)

    # move yolo predictions from runs/segment to save dir
    default_pred_dir = Path("runs/segment")
    if default_pred_dir.exists():
        new_pred_dir = Path(output_path / "yolo_predictions")
        new_pred_dir.mkdir(parents=True, exist_ok=True)
        for file in default_pred_dir.glob("predict*/*.jpg"):
            file.rename(new_pred_dir / file.name)

        print(f"Moving YOLO predictions to {new_pred_dir} and deleting runs/segment")
        with open(LOGGING_PATH, "a") as f:
            f.write(f"Saving YOLO predictions to {new_pred_dir}")
        rmtree(default_pred_dir)

    # note: can time this from CLI by running e.g.
    # start=$(date +%s);
    # python pipeline.py debug_pipeline -i -t debug_out; end=$(date +%s);
    # duration=$((end - start)); echo "The script took $duration seconds to run."
