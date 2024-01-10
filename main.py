import cv2
import keyboard
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance as dist

# to learn:
# cv2.threshold or cv2.adaptiveTreshold()
# image restoration
# optional: add the functionality to rotate a frame

def detect_contour():
    blurred = cv2.GaussianBlur(frame, (5, 5), 1)
    # apply automatic Canny edge detection using the computed median
    med = np.median(blurred)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * med))
    upper = int(min(255, (1.0 + sigma) * med))
    edges = cv2.Canny(blurred, lower, upper)

    # cv2.findContours detects change in the image color and marks it as a contour
    # the other value which it returns is hierarhy of contours if they are nested in each other (I am not interested in it)
    # RETR_EXTERNAL only the external contours are considered
    # CHAIN_APPROX_SIMPLE specifies that only the contours will be stored as a list of only the corner points rather than all the points in the contour
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # sorting contours by the area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    max_area = 0
    # By default the document contour are the frame borders
    scan_borders = np.array([[0,0],[width,0],[width,height],[0,height]])
    # finding the biggest rectangular contour
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > 2500:
            contour_len = cv2.arcLength(contour, True)
            #  the 0.02 * contourLen value is a maximum deistance between contour and its aproximation
            polygon = cv2.approxPolyDP(contour, 0.02*contour_len, True)
            # if the contour can be aproximated to polygon with the area bigger than
            if len(polygon) == 4 and contour_area > max_area:
                scan_borders = polygon
                max_area = contour_area

    cv2.drawContours(frame, [scan_borders], -1, (0, 255, 0), 3)
    #plt.imshow(edges)
    return scan_borders

#TO DO:
def closing(image):
    return image
def thresholding(image):
    #adjust values:
    blurred = cv2.GaussianBlur(frame, (3, 3), 1)
    #adjust the parameters
    thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 20, 2)
    return thresholded

# This function comes from the Imutils package
# https://github.com/PyImageSearch/imutils/blob/master/imutils/perspective.py
def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

#This function comes from the Imutils package
#https://github.com/PyImageSearch/imutils/blob/master/imutils/perspective.py
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

# change the argument to change the camera
cap = cv2.VideoCapture(0)

_, frame = cap.read()

height, width, _ = frame.shape

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

#what is fig?
fig, ax = plt.subplots()


scan_index = 1

while True:
    _, frame = cap.read()
    #turning to greyscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # perform all bluring operations on the original frame
    # and use frame copy for saving the scan
    frame_copy = frame.copy()

    scan_borders = detect_contour()

    #displaying the frame:
    plt.imshow(frame) #displays the current frame in the Matplotlib axis.
    plt.pause(0.2)
    ax.cla() #clears the previous plot to show the next frame.


    # Check if the Escape key is pressed
    if keyboard.is_pressed('esc'):
        break
    elif keyboard.is_pressed('w'):
        points = scan_borders.reshape(4, 2)
        transformed = four_point_transform(frame_copy, points)
        #restored = thresholding(transformed)
        cv2.imwrite("scans/scan_" + str(scan_index) + ".jpg", transformed)
        scan_index += 1
        plt.imshow(frame)
        plt.pause(2)
        print("scan saved")
cap.release()