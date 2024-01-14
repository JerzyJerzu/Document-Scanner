import cv2
import keyboard
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance as dist
from PIL import Image
import os

# TODO:
# Add the functionality to rotate a frame
# Tune the parameters for the higher resolution camera
# split the code to different files and improve it readability

def detect_contour(frame):
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
            polygon = cv2.approxPolyDP(contour, 0.1*contour_len, True)
            # if the contour can be aproximated to polygon with the area bigger than
            if len(polygon) == 4 and contour_area > max_area:
                scan_borders = polygon
                max_area = contour_area

    cv2.drawContours(frame, [scan_borders], -1, (0, 255, 0), 3)
    #plt.imshow(edges)
    return scan_borders

#TO DO:
def bolding(img):
    img = cv2.bitwise_not(img)
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.dilate(img, kernel)
    img = cv2.erode(img, kernel)
    img = cv2.dilate(img, kernel)
    img = cv2.erode(img, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    img = cv2.dilate(img, kernel)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel)
    img = cv2.dilate(img, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = cv2.erode(img, kernel)
    img = cv2.bitwise_not(img)
    return img
def denoising(img):
    img = cv2.bitwise_not(img)
    img = cv2.medianBlur(img, 3)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.medianBlur(img, 5)
    img = cv2.bitwise_not(img)
    return img
def thresholding(img, scale_factor):
    original_height, original_width = img.shape[:2]

    # Calculate the new dimensions after scaling
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    resized = cv2.resize(img, (new_width, new_height))
    #blurred = cv2.GaussianBlur(resized, (5,5), 1)
    #C: Constant subtracted from the mean or weighted mean.It is used to fine - tune the threshold value.
    thresholded = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
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
def save_image(demo_mode, scan_borders, frame_copy, frame, scan_index, scale_factor):
    points = scan_borders.reshape(4, 2)
    transformed = four_point_transform(frame_copy, points)
    thresholded = thresholding(transformed, scale_factor)
    bolded = bolding(thresholded)
    denoised = denoising(bolded)
    if demo_mode == True:
        cv2.imwrite("demo/scan_" + str(scan_index) + "_original.jpg", frame)
        cv2.imwrite("demo/scan_" + str(scan_index) + "_cropped.jpg", transformed)
        cv2.imwrite("demo/scan_" + str(scan_index) + "_thresholded.jpg", thresholded)
        cv2.imwrite("demo/scan_" + str(scan_index) + "_bolded.jpg", bolded)
    cv2.imwrite("output/scan_" + str(scan_index) + ".jpg", denoised)
    print("scan saved")

def read_images_from_folder():
    folder_path = './input'
    image_list = []

    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter only files with .jpg and .png extensions by checking the end of the filename
    image_files = [file for file in files if file.lower().endswith(('.jpg', '.png'))]

    # Read images using PIL
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        try:
            img = cv2.imread(image_path)
            image_list.append(img)
        except Exception as e:
            print(f"Error reading image '{image_file}': {e}")
    return image_list

# change the argument to change the camera
demo_mode = True
camera_mode = True
scan_index = 1
camera = 0

if camera_mode == True:
    cap = cv2.VideoCapture(camera)
    _, frame = cap.read()
    height, width, _ = frame.shape
    fig, ax = plt.subplots() # fig is a container and ax is the plotting area

    while True:
        _, frame = cap.read()
        # turning to greyscale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # perform all bluring and drawing operations on the original frame
        # and use frame copy for saving the scan
        frame_copy = frame.copy()

        scan_borders = detect_contour(frame)

        #displaying the frame:
        plt.imshow(frame) #displays the current frame in the Matplotlib axis.
        plt.pause(0.2)
        ax.cla() #clears the previous plot to show the next frame.

        # Check if the Escape key is pressed
        if keyboard.is_pressed('esc'):
            break
        elif keyboard.is_pressed('w'):
            save_image(demo_mode, scan_borders, frame_copy, frame, scan_index, 3)
            scan_index += 1
            plt.imshow(frame)
            plt.pause(2)
    cap.release()
else:
    # Not finished yet:
    images = read_images_from_folder()
    for image in images:
        height, width, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # perform all bluring and drawing operations on the original frame
        # and use frame copy for saving the scan
        image_copy = image.copy()
        borders = detect_contour(image)
        save_image(demo_mode, borders, image_copy, image, scan_index, 1)
        scan_index += 1