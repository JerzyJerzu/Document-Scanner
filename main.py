import cv2
import keyboard
import matplotlib.pyplot as plt
import numpy as np

# to learn:
# cv2.threshold
# image restoration
# optional: add the functionality to rotate a frame

# this function is part of this github repository:
#https://github.com/codegiovanni/Warp_perspective
def four_point_transform(image,contour):
    #TO DO:
    points = contour.reshape(4, 2)
    input_points = np.zeros((4, 2), dtype="float32")

    #The sum of coordinates for each point in the points array is calculated.
    #The top-left point is assigned the coordinates of the point with the smallest sum.
    #The bottom-right point is assigned the coordinates of the point with the largest sum.

    points_sum = points.sum(axis=1)
    input_points[0] = points[np.argmin(points_sum)]
    input_points[3] = points[np.argmax(points_sum)]

    #The differences between coordinates for each point in the points array are calculated.
    #The top-right point is assigned the coordinates of the point with the smallest difference.
    #The bottom-left point is assigned the coordinates of the point with the largest difference.

    points_diff = np.diff(points, axis=1)
    input_points[1] = points[np.argmin(points_diff)]
    input_points[2] = points[np.argmax(points_diff)]

    (top_left, top_right, bottom_right, bottom_left) = input_points
    bottom_width = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    top_width = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    right_height = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    left_height = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))

    # Output image size is just calculated as the biggest vertical and horizontal document contour
    max_width = max(int(bottom_width), int(top_width))
    max_height = max(int(right_height), int(left_height))
    #max_height = int(max_width * 1.414)  # for A4

    # Desired points values in the output image
    converted_points = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])

    # Perspective transformation
    matrix = cv2.getPerspectiveTransform(input_points, converted_points)
    img_output = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return img_output

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


def thresholding(image):
    #adjust values:
    _, threshold = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    return threshold

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

    document_contour = detect_contour()

    #displaying the frame:
    plt.imshow(frame) #displays the current frame in the Matplotlib axis.
    plt.pause(0.2)
    ax.cla() #clears the previous plot to show the next frame.


    # Check if the Escape key is pressed
    if keyboard.is_pressed('esc'):
        break
    elif keyboard.is_pressed('w'):
        transformed = four_point_transform(frame_copy, document_contour)
        #restored = thresholding(transformed)
        cv2.imwrite("scans/scan_" + str(scan_index) + ".jpg", transformed)
        scan_index += 1
        plt.imshow(frame)
        plt.pause(2)
        print("scan saved")
cap.release()