import cv2
import keyboard
import matplotlib.pyplot as plt
import numpy as np

def four_point_transform(image):
    #TO DO:
    return image

def detect_contour():
    #TO DO:
    document_contour = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    # use trackbars for canny
    #edges = cv2.Canny(blurred, 20, 200)
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                document_contour = approx
                max_area = area

    cv2.drawContours(frame, [document_contour], -1, (0, 255, 0), 3)
    return

def treasholding(image):
    # TO DO:
    return image

cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

WIDTH, HEIGHT = 800, 600
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

fig, ax = plt.subplots()

count = 1

while True:
    _, frame = cap.read()
    frame_copy = frame.copy()

    detect_contour()

    #displaying the frame:
    plt.imshow(frame) #displays the current frame in the Matplotlib axis.
    plt.pause(0.001)
    ax.cla() #clears the previous plot to show the next frame.

    transformed = four_point_transform(frame_copy)


    # Check if the Escape key is pressed
    if keyboard.is_pressed('esc'):
        break
    elif keyboard.is_pressed('w'):
        cv2.imwrite("scans/scan_" + str(count) + ".jpg", frame)
        count += 1
        plt.imshow(frame)
        plt.pause(2)
        print("scan saved")
cap.release()