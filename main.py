import cv2
import keyboard
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

WIDTH, HEIGHT = 800, 600
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

fig, ax = plt.subplots()

count = 1

def four_point_transform(image):
    #TO DO:
    return False

def detect_contour(image):
    #TO DO:
    return False

while True:
    _, frame = cap.read()
    frame_copy = frame.copy()

    #displaying the frame:
    plt.imshow(frame) #displays the current frame in the Matplotlib axis.
    plt.pause(0.001)
    ax.cla() #clears the previous plot to show the next frame.

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