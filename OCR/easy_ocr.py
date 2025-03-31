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
    image_path = "/home/htmayer/image_privacy/data/images/image_pool/street sign/3758118486_1bee1c55e5_z_rf.jpg"
    extract_text_from_image(image_path)