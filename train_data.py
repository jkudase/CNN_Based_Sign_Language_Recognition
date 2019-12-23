from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
import numpy as np
from random import shuffle

WIDTH = 180
HEIGHT = 180
#Check the labels in the labels array and modify them accordingly if they are not same.
labels=["A", "B", "C", "D", "E"]

def main():

    training_data=[]
    for label in labels:
        data = np.load("{}_new_training_data_1.npy".format(label))
        for dat in data:
            training_data.append(dat)

    print("Training_data length", len(training_data))
    shuffle(training_data)
    shuffle(training_data)
    shuffle(training_data)

    t = int(0.9*len(training_data))
    train = training_data[:t]
    test = training_data[t:]
    training_data = []
    X_train = np.array([i[0] for i in train])
    Y_train = [i[1] for i in train]
    X_valid = np.array([i[0] for i in test])
    Y_valid = [i[1] for i in test]

    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(WIDTH, HEIGHT, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    #  let's add a fully-connected layer 
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(5, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    # model = load_model("sign_model_5.h5")
    print("Model Loaded!")
    for layer in base_model.layers:
        layer.trainable = False
    print("Compiling...")
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Compiled")
    print("Starting training")
    history = model.fit(X_train, Y_train, batch_size=32, epochs=5, shuffle=True, verbose=1, validation_data=(X_valid, Y_valid))
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, batch_size=32, epochs=5, shuffle=True, verbose=1, validation_data=(X_valid, Y_valid))
    model.save("sign_model_5.h5")
    del model
    print("model saved")

main()