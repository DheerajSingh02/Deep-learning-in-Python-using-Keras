import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from pathlib import Path
#Load the data
(x_train, y_train) , (x_test, y_test) = cifar10.load_data()

#Normalize data set to 0-to-1 range
x_train = x_train.reshape((50000, 32*32*3))
x_train = x_train.astype('float32')

x_test = x_test.reshape((10000, 32*32*3))
x_test = x_test.astype('float32')

x_train /= 255        #Pixel values are in range of (0,1]
x_test /= 255

#Convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train,10)    #Output labels are single value between 0-to-9,so we need a binary class where only one value will be 1 at a time.
y_test = keras.utils.to_categorical(y_test,10)

#Create a model and add layers
model = Sequential()

model.add(Dense(512,activation="relu",input_shape=(32*32*3,)))
model.add(Dense(10,activation="softmax"))

#Print the summary of the model
model.summary()

#Compile the model
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

#Fit the model
model.fit(x_train,y_train,epochs=4, batch_size=256)

test_loss, test_acc = model.evaluate(x_test, y_test)

#Print test accuracy

print("Accuracy is {}".format(test_acc))
