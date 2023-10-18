# this code works but still need to work on normalisation.
import numpy as np
import cv2
vid = cv2.VideoCapture(1) 
from matplotlib import pyplot as plt
import argparse
#webcam number for my laptop is 1, change according to device id number.


#locate from path, collect chess piece images.
king_black = "C:\\Users\\chukwunedu\\OneDrive\\Pictures\\white_chess_picece.png" 
queen_black = 'placeholder'
bishop_black = 'placeholder'
cross_black = 'placeholder'
rook_black = 'placeholder'
knight_black = 'placeholder'
pawn_black = 'placeholder'
king_white = 'placeholder'
queen_white = 'placeholder'
bishop_white = 'placeholder'
cross_white = 'placeholder'
rook_white = 'placeholder'
knight_white = 'placeholder'
pawn_white = 'placeholder'

#record individual chess piece in array
chess_pieces = [king_black, queen_black, bishop_black, cross_black, rook_black, knight_black, pawn_black, king_white, queen_white, bishop_white, cross_white, rook_white, knight_white, pawn_white]



while True:
    #read video
    ret, frame = vid.read()
    
    # Color-segmentation to get binary mask
    lwr = np.array([0, 0, 0])
    upr = np.array([100, 255, 50])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    msk = cv2.inRange(frame, lwr, upr)

    #show image frame on laptop
    cv2.imshow("mask", msk) 

    # Extract chess-board
    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
    dlt = cv2.dilate(msk, krn, iterations=5)
    res = 255 - cv2.bitwise_and(dlt, msk)

    res = np.uint8(res)
    ret, corners = cv2.findChessboardCorners(res, (7, 7),
                                            flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                cv2.CALIB_CB_FAST_CHECK +
                                                cv2.CALIB_CB_NORMALIZE_IMAGE)

    objp = np.zeros((7*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2) * 30

    
    
    cv2.imshow("frame", frame)

    #if found image draw chessboard corners etc.
    if ret:
        print("Found")
        #draw chessboard corners. Try cv2.findChessboardCorners to get the chessboard grids.
        fnl = cv2.drawChessboardCorners(frame, (7, 7), corners, ret) 
        #perform camera calibration. Returns camera matrix, distortion coefficients, rotation, translation vectors.
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], res.shape[::-1], None, None)

        #undistort image.
        h,  w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        dst = cv2.resize(dst, (300, 300))

        # mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
        # dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        # x, y, w, h = roi
        # dst = dst[y:y+h, x:x+w]
		
        #THIS IS USED TO DISPLAY THE IMAGES.
        cv2.imshow("fnl", fnl)
        cv2.imshow("dst", dst)
        
      # This is for matching the images to the pices on the board   
    #template = cv2.imread("C:\\Users\\chukwunedu\\OneDrive\\Pictures\\nedu.png", cv2.IMREAD_GRAYSCALE)
    #w, h = template.shape[::-1]
    #method = ['cv2.TM_CCOEFF']
    
    #for med in method:
        #img = king_black.copy()
        #method = eval(med)
       # res = cv2.matchTemplate(img,template,method)
      #  min_va, max_va, min_loc, max_loc = cv2.minMaxLoc(res)
        # this displasy the images that are being taken
    #    plt.subplot(121),plt.imshow(frame,cmap = 'gray')
        # this tries to capture the video of the image that it is looking for
     #   plt.title("matching result"), plt.xticks([]), plt.yticks([])
        # this is the image that is being compared to
      #  plt.subplot(122), plt.imshow(img, cmap = 'gray')
       # plt.title('detected point'), plt.xticks([]), plt.yticks([])
        #plt.suptitle(med)
        #plt.show()
        
        #this is may no be needed.
        #ap = argparse.ArgumentParser()
		#ap.add_argument("-i", "--image", type=str, required=True,
		#help="path to input image where we'll apply template matching")
		#ap.add_argument("-t", "--template", type=str, required=True,
		#help="path to template image")
		#args = vars(ap.parse_args())
        
        tracker = cv2.legacy.TrackerMOSSE_create()
        
        print("looking for images")
        #take image of the chessboard and map it here.s
        template = cv2.imread("C:\\Users\\chukwunedu\\OneDrive\\Pictures\\white_chess_picece.png")
        template_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        print("[INFO] performing template matching...")
        result = cv2.matchTemplate(msk, template_gray, cv2.TM_CCOEFF_NORMED)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
        
    else:
        print("No Chessdboard Found")
    
    #break look for 'q' key.
    key = cv2.waitKey(1)
    if key == ord('q'):
            break
    
    cv2.waitKey(1)
