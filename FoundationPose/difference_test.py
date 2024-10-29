import pyrealsense2 as rs
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Start streaming
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Initialize variables for storing images
image1 = None
image2 = None

try:
    print("Press SPACE to capture the first image.")
    
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        
        # Display the current frame
        cv2.imshow('Live Stream', color_image)
        
        key = cv2.waitKey(1)
        
        if key == 32:  # SPACE key
            if image1 is None:
                image1 = color_image.copy()
                print("First image captured. Press SPACE to capture the second image.")
            elif image2 is None:
                image2 = color_image.copy()
                print("Second image captured. Processing the similarity...")
                break

    # Apply a larger Gaussian Blur to reduce micro changes
    image1_blur = cv2.GaussianBlur(image1, (9, 9), 0)
    image2_blur = cv2.GaussianBlur(image2, (9, 9), 0)

    # Convert images to HSV color space (ignore the Value channel)
    hsv1 = cv2.cvtColor(image1_blur, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(image2_blur, cv2.COLOR_BGR2HSV)

    # Use only Hue and Saturation channels for comparison
    h_diff = cv2.absdiff(hsv1[:, :, 0], hsv2[:, :, 0])  # Hue difference
    s_diff = cv2.absdiff(hsv1[:, :, 1], hsv2[:, :, 1])  # Saturation difference

    # Apply thresholds to the Hue and Saturation differences
    _, h_thresh = cv2.threshold(h_diff, 15, 255, cv2.THRESH_BINARY)
    _, s_thresh = cv2.threshold(s_diff, 30, 255, cv2.THRESH_BINARY)

    # Combine the thresholds
    combined_thresh = cv2.bitwise_or(h_thresh, s_thresh)

    # Apply morphological operations to remove small noise
    kernel = np.ones((5, 5), np.uint8)
    combined_thresh = cv2.erode(combined_thresh, kernel, iterations=1)
    combined_thresh = cv2.dilate(combined_thresh, kernel, iterations=1)

    # Calculate SSIM for a more robust similarity measure
    gray1 = cv2.cvtColor(image1_blur, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2_blur, cv2.COLOR_BGR2GRAY)
    similarity_score, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")

    # Use template matching as another similarity measure
    res = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
    template_match_score = res[0][0]  # Normalized cross-correlation score

    print(f"SSIM Similarity Score: {similarity_score:.4f}")
    print(f"Template Match Score: {template_match_score:.4f}")
    
    # Calculate similarity percentage based on unchanged pixels
    unchanged_pixels = np.sum(combined_thresh == 0)  # Count pixels where the combined threshold is 0 (unchanged)
    total_pixels = combined_thresh.size
    similarity_percentage = (unchanged_pixels / total_pixels) * 100

    print(f"Similarity Percentage: {similarity_percentage:.2f}%")

    # Display the original images, difference map, and threshold results
    cv2.imshow('First Image', image1)
    cv2.imshow('Second Image', image2)
    cv2.imshow('Differences (Threshold)', combined_thresh)
    cv2.imshow('SSIM Difference', diff)
    cv2.waitKey(0)

finally:
    # Stop streaming
    pipeline.stop()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
