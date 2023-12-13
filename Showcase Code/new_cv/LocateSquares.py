""" Detect invididual squares of chess board
    This program uses Hough Transform to 'see' the squares
"""
import sys
import math
import cv2
import numpy as np

def findLines(boardImg):
    
    gray = cv2.cvtColor(boardImg, cv2.COLOR_BGR2GRAY)

    #blur = cv2.GaussianBlur(gray, (5,5), 0)

    dst = cv2.Canny(gray, 50, 200, None, 3)

    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLines(dst, 1, np.pi / 180, 12, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            thetha = lines[i][0][1]
            a = math.cos(thetha)
            b = math.sin(thetha)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0+1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))  
            
            cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)   

    return cdst

def findLinesP(boardImg):
    
    gray = cv2.cvtColor(boardImg, cv2.COLOR_BGR2GRAY)

    dst = cv2.Canny(gray, 50, 200, None, 3)

    cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 15, 1)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    return cdstP

try:
    while True:
        image = cv2.imread('test.png', cv2.IMREAD_UNCHANGED)
        imageLines = findLines(image)
        imageLinesP = findLinesP(image)
        cv2.imshow("Raw Image", image)
        cv2.imshow("Lines from hough transform", imageLines)
        cv2.imshow("Lines from P hough transform", imageLinesP)
        key = cv2.waitKey(10)  
        if key == 27:  # esc key ascii code
            break  
finally:
    cv2.destroyAllWindows()
