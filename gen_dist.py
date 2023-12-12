"""
Given a greyscale image save a csv file containing approximate coordinates of
non-white points. The image and output file should be replaced in-place. The
points are normalized in a range to make the output legible. The range can be
customized using the custom_min and custom_max variables.
"""
import cv2
import csv
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def trim_image(img):
    # Find the first row with a pixel value not equal to 255
    top = np.argmax(np.any(img != 255, axis=1))

    # Find the last row with a pixel value not equal to 255
    bottom = len(img) - np.argmax(np.any(np.flipud(img) != 255, axis=1))

    # Find the first column with a pixel value not equal to 255
    left = np.argmax(np.any(img != 255, axis=0))

    # Find the last column with a pixel value not equal to 255
    right = len(img[0]) - np.argmax(np.any(np.fliplr(img) != 255, axis=0))

    # Trim the image
    return img[top:bottom, left:right]


def save_coords_to_json(image_path, out_file):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    np.set_printoptions(threshold=np.inf)

    trimmed_img = trim_image(img)

    # Get the coordinates of non-255 pixels
    non_white_points = np.column_stack(np.where(img != 255))

    # Define the custom range (min_value, max_value)
    custom_min = 0
    custom_max = 4

    # Normalize the array column-wise to the custom range
    normalized_array = custom_min + (
        non_white_points - non_white_points.min(axis=0)
    ) * (custom_max - custom_min) / (
        non_white_points.max(axis=0) - non_white_points.min(axis=0)
    )

    with open(out_file, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(normalized_array)


# Example usage
image_path = "/home1/navdeep/work/projects/gradi-mojo/hipc.png"
save_coords_to_json(image_path, "shapes/hipc.csv")
