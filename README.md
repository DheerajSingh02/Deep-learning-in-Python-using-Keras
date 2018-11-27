# Deep-learning-in-Python-using-Keras
Image recognition in Keras
--------------------------------------------------------------------------------------------------------------------------
The fundamental difference between a densely connected layer and a convolution layer is this:

Dense Layers -----learns---> Global patterns in the input feature space, for a MNIST digit, patterns involving all pixels.
Convolution Layers -----learns ---> Local patterns, in case of image-edges, and textures. 

These important characteristics give convnets two properties:
1.	The pattern they learn are translational invariant.
2.	They can learn spatial hierarchies of patterns.
Introduction to Keras
Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.
It supports both convolutional networks and recurrent networks, as well as combinations of the two.
Some points to keep in mind:
Neural nets will work best with input values which are in range (0,1]
Therefore, we should preprocess the data in such a way to scale input values in above range.
1.	The core data structure of Keras is a model, a way to organize layers.
The simplest type of model is the ‘Sequential’ model, a linear stack of layers.
Below the example to create the Sequential Model:
from keras.models import Sequential

model = Sequential()

2.	Layers are stacked by using ‘.add()’ method.
from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

3.	Once required model has been set up, model need to the defined its learning process using the
‘.compile()’ method.

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
Here we have defined a ‘loss function’ which is supposed to get minimized while learning/training the model.
An optimizer is one of the two arguments required for compiling a Keras model.
Optimizer denotes the way in which above loss function will be minimized.
 A metric is a function that is used to judge the performance of your model. Metric functions are to be supplied in the metrics parameter when a model is compiled. E.g. ['mae', 'acc']

4.	Now we can fit the trained model with training data so that it will learn the weight parameters.
   #x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
   model.fit(x_train, y_train, epochs=5, batch_size=32)

epoch = No. of times the whole dataset will be fed to model for training. 
Batch_size = no. of samples at a time during each epoch, which will be used to train the model. Generally shuffling the training data gives good results in terms of accuracy of the model.

If batch_size is too small, model will start ‘memorizing’ the training data instead of learning and it will overfit. Similarly, if batch_size is too large, model will not learn all-important features and it will underfit.

It is generally good idea to keep batch_size in 32-to-128.

Finally, epoch will determine the time required for training the model. Good value to choose of epoch can be determined using some hints & trials. We should check if accuracy is increasing and loss is decreasing with each epoch, then we could assume that model is learning properly.

5.	Now above trained model is ready to do predictions of the new data or the test data.
classes = model.predict(x_test, batch_size=128)


Use of Maxpooling, Dropout and Flatten layers:

Maxpooling: It consists of extracting windows from the input feature maps and outputting the max value of each channel.
POOL layer will perform a downsampling operation along the spatial dimensions (width, height).

Dropout: Dropout is a technique where randomly selected neurons are ignored during training. They are “dropped-out” randomly. This means that their contribution to the activation of downstream neurons is temporally removed on the forward pass and any weight updates are not applied to the neuron on the backward pass. The effect is that the network becomes less sensitive to the specific weights of neurons. This in turn results in a network that is capable of better generalization and is less likely to overfit the training data.

Flatten: Flattens the input. Does not affect the batch size. The last stage of a convolutional neural network (CNN) is a classifier. It is called a dense layer, which is just an artificial neural network (ANN) classifier.

And an ANN classifier needs individual features, just like any other classifier. This means it needs a feature vector.

Therefore, you need to convert the output of the convolutional part of the CNN into a 1D feature vector, to be used by the ANN part of it. This operation is called flattening. It gets the output of the convolutional layers, flattens all its structure to create a single long feature vector to be used by the dense layer for the final classification.

References: https://keras.io/
            http://cs231n.github.io/convolutional-networks/ 


Your comments/suggentions are welcomed. 

Thanks. 



