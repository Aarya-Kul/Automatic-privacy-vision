#!/usr/bin/env python

import numpy as np
import cv2
from simple_lama_inpainting import SimpleLama
from PIL import Image

simple_lama = SimpleLama()


# Load the text file containing contour coordinates
def load_contours(txt_file_path):
    contours = []
    with open(txt_file_path, "r") as file:
        for line in file:
            # Assuming each line contains the coordinates of one object
            points = []
            coords = line.strip().split()[1:]
            for i in range(0, len(coords), 2):
                x = float(coords[i]) * width  # Scale x coordinate
                y = float(coords[i + 1]) * height  # Scale y coordinate
                points.append([int(x), int(y)])
            contours.append(np.array(points, np.int32))
    return contours


def inpaint(image: Image.Image, contours) -> Image.Image:
    height, width, _ = np.array(image).shape
    # Create an empty mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Fill the mask with the contour areas
    for contour in contours:
        cv2.fillPoly(mask, [contour], 255)  # Fill the polygon with white (255)
    return simple_lama(image, mask)


# main subroutine is just for debugging
if __name__ == "__main__":
    img_path = "obfusc_example.jpg"

    image = Image.open(img_path)
    height, width, _ = np.array(image).shape
    # Specify the text file path
    txt_file_path = "./obfusc_example.txt"
    contours = load_contours(txt_file_path)

    result = inpaint(image, contours)

    # Save the inpainted image
    result.save("inpainted.png")
