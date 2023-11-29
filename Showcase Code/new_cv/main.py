import cv2
import os
import time

from Cropping import ExtractAndStraightenFromImage
from LocateGrid import DetectGrid
from LocateGrid import morph_gradient
from LocateGrid import bit_masking_grad

IMAGE_FILE_PATH = os.path.join("Capture", "BoardPictures")

# Create directory to save pictures if it doesn't exist
if not os.path.exists(IMAGE_FILE_PATH):
    os.makedirs(IMAGE_FILE_PATH)

# Start the video capture
vid = cv2.VideoCapture(0)  # ID 1 assumes a second camera (like your Orbbec Astra). Use 0 for default camera

is_automatic = False


while True:
    ret, frame = vid.read()
    if not ret:
        print("Failed to grab frame")
        break

    key = cv2.waitKey(1)

    boardImg = ExtractAndStraightenFromImage(frame)
    checkBoard = DetectGrid(boardImg)
    morphImg = morph_gradient(boardImg)
    bitResult = bit_masking_grad(morphImg, boardImg)

    cv2.imshow("Frame", frame)
    #cv2.imshow("Board img", boardImg)
    cv2.imshow("Check board", checkBoard)
    cv2.imshow("Morph Method", morphImg)
    cv2.imshow("Bit Result", bitResult)

    if key & 0xFF == ord('q'):
        break
            

vid.release()
cv2.destroyAllWindows()
