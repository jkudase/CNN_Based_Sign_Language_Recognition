import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Change the label value manually for each Label class

def main():
    while True:
        ret, image = cap.read()
        # lower_blue = np.array([0, 10, 60])
        # upper_blue = np.array([20, 100, 255])
        lower_blue = np.array([0, 10, 60])
        upper_blue = np.array([20, 250, 255])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(image, lower_blue, upper_blue)
        image = cv2.bitwise_and(image,image, mask= mask)
        cv2.imshow("Mask", cv2.resize(image,(800,640)))
        if (cv2.waitKey(25) & 0xFF=='q'):
            cv2.destroyAllWindows()
            break

main()