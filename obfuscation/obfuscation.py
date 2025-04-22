#!/usr/bin/env python

import numpy as np
import cv2
from simple_lama_inpainting import SimpleLama
from PIL import Image



simple_lama = SimpleLama()

img_path = "obfusc_example.jpg"

image = Image.open(img_path)
height, width, _ = np.array(image).shape

# Load the text file containing contour coordinates
def load_contours(txt_file_path):
    contours = []
    with open(txt_file_path, 'r') as file:
        for line in file:
            # Assuming each line contains the coordinates of one object
            points = []
            coords = line.strip().split()[1:]
            for i in range(0, len(coords), 2):
                x = float(coords[i]) * width  # Scale x coordinate
                y = float(coords[i+1]) * height # Scale y coordinate
                points.append([int(x), int(y)])
            contours.append(np.array(points, np.int32))
    return contours

# Specify the text file path
txt_file_path = './obfusc_example.txt'
contours = load_contours(txt_file_path)

# Create an empty mask
mask = np.zeros((height, width), dtype=np.uint8)

# Fill the mask with the contour areas
for contour in contours:
    cv2.fillPoly(mask, [contour], 255)  # Fill the polygon with white (255)

# Save or display the inpainted image
result = simple_lama(image, mask)
result.save("inpainted.png")
