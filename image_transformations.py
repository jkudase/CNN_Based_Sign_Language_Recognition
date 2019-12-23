import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

#Check the labels in the labels array and modify them accordingly if they are not same.
labels=["A", "B", "C", "D", "E"]

def main():
    for label in labels:
        training_data = []
        data = np.load("{}-new_data.npy".format(label))
        print("Loaded data for {}".format(label))
        for dat in data:
            dat1 = dat
            x = dat1[0].reshape((1,) + dat1[0].shape)
            y  = dat[1]
            training_data.append([dat[0] , y])
            i=1
            for x_batch in datagen.flow(x, batch_size=1, shuffle=False):
                training_data.append([x_batch[0], y])
                i += 1
                if i > 10:
                    break
        np.save("{}_new_training_data_1.npy".format(label), training_data)
        print("Saved {} label data".format(label))


main()