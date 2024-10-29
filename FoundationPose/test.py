import pyrealsense2 as rs
import numpy as np
import cv2

def normalize_lighting(image):
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_equalized = clahe.apply(l)

    # Merge the channels back
    lab = cv2.merge((l_equalized, a, b))
    # Convert back to BGR color space
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return normalized

def get_edge_map(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use Canny Edge Detector
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    return edges

def remove_noise(diff_thresh, min_area=20):
    # Remove small white spots using morphological operations
    kernel = np.ones((1, 1), np.uint8)
    
    # Morphological opening to remove small noise (reduce iterations)
    diff_cleaned = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Morphological closing to close small gaps
    diff_cleaned = cv2.morphologyEx(diff_cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Visualize the result after morphological operations
    cv2.imshow('After Morphological Operations', diff_cleaned)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Find contours
    contours, _ = cv2.findContours(diff_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area to remove small contours
    filtered_image = np.zeros_like(diff_cleaned)
    for contour in contours:
        if cv2.contourArea(contour) > min_area:  # Lower the min_area threshold
            cv2.drawContours(filtered_image, [contour], -1, 255, thickness=cv2.FILLED)

    # Visualize the filtered contours
    cv2.imshow('Filtered Contours', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return filtered_image

def compare_edge_maps(edge1, edge2):
    # Compute absolute difference
    diff = cv2.absdiff(edge1, edge2)

    # Threshold the difference to get binary image of changes
    _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Visualize the difference before noise removal
    cv2.imshow('Raw Difference Image', diff_thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Remove small white noise
    diff_cleaned = remove_noise(diff_thresh)

    # Compute the number of changed pixels
    num_changed_pixels = np.sum(diff_cleaned) / 255

    # Compute total number of pixels
    total_pixels = diff_cleaned.shape[0] * diff_cleaned.shape[1]

    # Compute similarity score
    similarity_score = 100 * (1 - (num_changed_pixels / total_pixels))

    return similarity_score, diff_cleaned

def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Start streaming
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        print("Press 'c' to capture the first image.")
        image1_captured = False
        image2_captured = False
        image1 = None
        image2 = None

        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # Show the live stream
            cv2.imshow('RealSense', color_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                if not image1_captured:
                    image1 = color_image.copy()
                    image1_captured = True
                    print("First image captured. Press 'c' to capture the second image.")
                elif not image2_captured:
                    image2 = color_image.copy()
                    image2_captured = True
                    print("Second image captured. Processing...")
                    break
            elif key == ord('q'):
                print("Exiting without capturing both images.")
                break

        # Release resources
        cv2.destroyAllWindows()

        if image1_captured and image2_captured:
            # Normalize lighting
            image1_norm = normalize_lighting(image1)
            image2_norm = normalize_lighting(image2)

            # Get edge maps
            edges1 = get_edge_map(image1_norm)
            edges2 = get_edge_map(image2_norm)

            # Compare edge maps
            similarity_score, diff_image = compare_edge_maps(edges1, edges2)

            print(f"Similarity Score: {similarity_score:.2f}%")

            # Show the difference image
            cv2.imshow('Detected Changes', diff_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Images were not captured properly.")

    finally:
        # Stop streaming
        pipeline.stop()

if __name__ == "__main__":
    main()
