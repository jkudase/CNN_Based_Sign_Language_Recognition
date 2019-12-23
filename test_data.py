import numpy as np
import cv2
from keras.models import load_model
cap = cv2.VideoCapture(0)

#Check the labels in the labels array and modify them accordingly if they are not same.
labels=["A", "B", "C", "D", "E"]
def main():

    model = load_model("sign_model_5.h5")
    while True:
        
        count = [0,0,0,0,0]
        for k in range(1,101):
            ret, image = cap.read()
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            lower_blue = np.array([0, 10, 60])
            upper_blue = np.array([20, 250, 255])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(image, lower_blue, upper_blue)
            image = cv2.bitwise_and(image,image, mask= mask)
            cv2.imshow("Test", cv2.resize(image,(800,640)))
            image = cv2.resize(image, (180, 180))
            image = image.reshape((1,) + image.shape)
            output = model.predict(image, batch_size=1, verbose=1)
            
            output = (np.argmax(output[0]))
            if(output<=len(labels)):
                count[output] += 1
            else:
                print("Cannot Predict")

            if (cv2.waitKey(25) & 0xFF=='q') :
                # np.save("{}-data.npy".format(label), training_data)
                cv2.destroyAllWindows()
                break
        output = np.argmax(count)
        percent = count[output]/100
        print("{}-{} Percent".format(labels[output], percent))

main()