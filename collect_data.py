import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Change the label value manually for each Label class
label = 'E'

def main():
    count = 1
    training_data = []
    while True:
        print(count)
        ret, image = cap.read()
        
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        lower_blue = np.array([0, 10, 60])
        upper_blue = np.array([20, 250, 255])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(image, lower_blue, upper_blue)
        image = cv2.bitwise_and(image,image, mask= mask)
        cv2.imshow("Test", cv2.resize(image,(800,640)))
        image = cv2.resize(image, (180, 180))
        training_data.append(image)
        count = count+1
        if (cv2.waitKey(25) & 0xFF=='q') or count == 500 :
            np.save("{}-data.npy".format(label), training_data)
            cv2.destroyAllWindows()
            break

main()