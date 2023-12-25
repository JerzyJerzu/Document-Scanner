import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

WIDTH, HEIGHT = 800, 600
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

fig, ax = plt.subplots()

while True:
    _, frame = cap.read()
    frame_copy = frame.copy()

    #displaying the frame
    plt.imshow(frame) #displays the current frame in the Matplotlib axis.
    plt.pause(0.001)
    ax.cla() #clears the previous plot to show the next frame.
