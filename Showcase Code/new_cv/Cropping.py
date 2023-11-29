import cv2
import numpy as np

def ExtractAndStraightenFromPath(path):
    img = cv2.imread(path)

    return _ExtractAndStraighten(img)

def ExtractAndStraightenFromImage(img):
    return _ExtractAndStraighten(img)

def _ExtractAndStraighten(img):
    # Convert to HSV for easier color thresholding
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Threshold for the color red
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Find the largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found!")
        return img
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon (quadrilateral in this case)
    epsilon = 0.05 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) != 4:
        print("Could not find four corners!")
        return img

    # Define the points for perspective transformation
    h, w = img.shape[:2]
    dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
    
    # Ensure the points are in the correct order (top-left, top-right, bottom-right, bottom-left)
    src = np.array([approx[i][0] for i in [0, 1, 2, 3]], dtype=np.float32)
    
    # Compute perspective transformation
    matrix = cv2.getPerspectiveTransform(src, dst)
    result = cv2.warpPerspective(img, matrix, (w, h))

    cv2.flip(result, 1, result)

    return result
#cropping results in a accurate representation of checkboard even in slanted order.
# Use the function
#result_img = ExtractAndStraightenFromPath("BoardPictures\Picture_1697036031.2633069.png")
#cv2.imwrite("result.jpg", result_img)
