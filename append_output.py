import numpy as np



#Check the labels in the labels array and modify them accordingly if they are not same. And also change the respective label in Elif conditions below.
labels=["A", "B", "C", "D", "E"]

def main():

    for label in labels:
        training_data = []
        image_data = np.load("{}-data.npy".format(label))
        #Change labels in the elif conditions beow if they are not the same.
        for data in image_data:
            if(label == "A"):
                output = [1,0,0,0,0]
            elif(label == "B"):
                output = [0,1,0,0,0]
            elif(label == "C"):
                output = [0,0,1,0,0]
            elif(label == "D"):
                output = [0,0,0,1,0]
            elif(label == "E"):
                output = [0,0,0,0,1]
            else:
                print("Please use correct Label")
                break
            training_data.append([data, output])
        
        np.save("{}_new_training_data_1.npy".format(label), training_data)
        print("Saved new {} data".format(label))
    print("Output appended for all labels")

main()