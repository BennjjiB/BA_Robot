import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

# Rotation matrix around x-axis
def rotation_matrix_x(degrees):

    radians = np.radians(degrees)
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(radians), -np.sin(radians), 0],
        [0, np.sin(radians), np.cos(radians), 0],
        [0, 0, 0, 1]
    ])

# Rotation matrix around y-axis
def rotation_matrix_y(degrees):

    radians = np.radians(degrees)
    return np.array([
        [np.cos(radians), 0, np.sin(radians), 0],
        [0, 1, 0, 0],
        [-np.sin(radians), 0, np.cos(radians), 0],
        [0, 0, 0, 1]
    ])

# Rotation matrix around z-axis
def rotation_matrix_z(degrees):

    radians = np.radians(degrees)
    return np.array([
        [np.cos(radians), -np.sin(radians), 0, 0],
        [np.sin(radians), np.cos(radians), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

# Translation matrix
def translation_matrix(dx, dy, dz):

    return np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]
    ])

# computes difference between two images
def compute_image_difference(image1, image2, mask):
    kernel = np.ones((8, 8), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image1 = np.where(mask == 255, 0, image1)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image2 = np.where(mask == 255, 0, image2)

    epsilon = 1e-10  
    ssim_score, diff = ssim(image1, image2, full=True, data_range=image1.max() - image1.min() + epsilon)
    diff = (diff * 255).astype(np.uint8)

    _, ssim_image_thresholded = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    white_image = np.full_like(image1, 255, dtype=np.uint8)
    ssim_score, diff = ssim(ssim_image_thresholded, white_image, full=True)

    print("SSIM score: {:.4f}".format(ssim_score))

    return ssim_score

def compute_image_difference_without_mask(image1, image2):
    # Convert images to grayscale
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Handle potential differences in image ranges
    epsilon = 1e-10  
    ssim_score, diff = ssim(image1, image2, full=True, data_range=image1.max() - image1.min() + epsilon)

    # Scale the difference map to an 8-bit image
    diff = (diff * 255).astype(np.uint8)
    _, ssim_image_thresholded = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    # Create a white image for SSIM comparison
    white_image = np.full_like(image1, 255, dtype=np.uint8)
    ssim_score, _ = ssim(ssim_image_thresholded, white_image, full=True)
    return ssim_score