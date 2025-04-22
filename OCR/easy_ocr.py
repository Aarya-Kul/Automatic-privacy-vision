import easyocr

def extract_text_from_image(image_path):
    """
    Detect and output text from an image using EasyOCR.

    Input:
    image_path (str): Path to the image file.

    Output:
    list: A list of detected text with bounding boxes and confidence scores.
    """
    reader = easyocr.Reader(['en'])  # Load English OCR model
    result = reader.readtext(image_path)  # Process the image

    # Print detected text with confidence
    for detection in result:
        bbox, text, confidence = detection
        print(f"Detected: '{text}' (Confidence: {confidence:.2f})")

    return result  # Return the raw OCR result if needed
if __name__ == "__main__":
# Example usage
    image_path = "C:/Aarya/umich/eecs545/final_project/image_privacy_data/multiclass/train/images/00a0ab30f64c0096_jpg.rf.9bd0a48d7f0588dde926fe61f5c4b6ce.jpg"
    extract_text_from_image(image_path)