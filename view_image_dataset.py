import tensorflow
from keras.datasets import cifar10
import matplotlib.pyplot as plt

#The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
# There are 50000 training images and 10000 test images.

cifar10_class_names = {
    0: "Plane",
    1: "Car",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Boat",
    9: "Truck"
}

# Load the entire data set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#print(x_train[0].shape)  #--> (32, 32, 3)

#print(y_train.shape)         #(50000, 1) 50000 rows and one column contains the IDs of class names.

#Iterate trhough each picture(here 500 pictures) in dataset:

for i in range(500):
    sample_image = x_train[i]
    image_class_no = y_train[i][0]
    image_class_name = y_train[image_class_no]

    #Draw the image

    plt.imshow(sample_image)

    #Label the image

    plt.title(image_class_name)


    plt.show()

