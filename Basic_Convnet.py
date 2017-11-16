#This is a basic Convolutional Neural Network, essentialy the the model-type at a barebones stage
#This model ahieves a 99.3% accuracy on the MNIST dataset
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

# We build a sequential model consisting of 3 Conv2D layers and 2 MaxPooling layers

model = models.Sequential()
#The Conv Layer takes an image and then examines through N (in our case 32) seperate filters for the strongest patterns
#In the process outputting N seperate feature maps
#The layer examines the input in windows of dimensions (a x b) (in our case (3 x 3))
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
#Max Pooling Layers downsamples the data outputted by the layer above by taking the max values of each feature maps 
#and only passing the top half of the max values on to the next layer
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#We then Flatten our 3D tensors into 1D vectors so it can be feed into our Dense layers that can cheaply refine our resulsts
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
#We use summary to demonstrate the total amount of parameters our model trains on
#A lot more intsense than just simple dense models!
model.summary()

#We load and seperate the mnist dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#We reshape the training and test data + labels into a more managable and relevant form
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Call the model and see our resulsts
model.compile(optimizer='rmsprop',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print test_acc