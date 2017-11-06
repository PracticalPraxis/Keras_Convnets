#This is a convolutional neural network that trains on images of cats and dogs
#The non-standard tweaks in this program mainly come from the datagen implementation
#and the inclusion of a Dropout layer in the model
#Datagen in this program generates new maniplated images from our base dataset
#Using the dataset along with these manipulated images, this model achieves a rough accuracy of 77%
#No overfitting occurs thanks to the manipulated images keeping the model on its toes
#but this method is very computationally expensieve

#Using this dataset from Kaggle: www.kaggle.com/c/dogs-vs-cats/data

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import os, shutil
import matplotlib.pyplot as plt

train_dir = '/your/training/data/'
validation_dir = '/your/validation/data'

#Standard Keras Convnet model with additional Flatten and Dropout layer
#We use Flatten to turn the 3D tensors of our image data into 1D vectors so they can be efficently dealt with 
#by the Dropout layer
#Dropout randomly causes (according to a defined percentage) features of a layers output to equal to 0.
#On face value, this seems activley harmful to the model's training, however
#Using dropout allows the model to get a better grasp on the strongest patterns within the data, which are 
#stastically more likely to surivie being damaged by the droput than weaker patterns
#Therefore, Dropout helps cut through the misleading patterns our model might otherwise find in its computations
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
						input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#We'll be using binary_crossentropy as the model predicts from only two classes, cat or dog
#and our model's final output is a probability
model.compile(loss='binary_crossentropy',
			  optimizer=optimizers.RMSprop(lr=1e-4),
			  metrics=['acc'])

#Our image generator's parameters for the training data
#The various parameters, rotation, shear, flip etc. will change the image in minor but noticable ways
train_datagen = ImageDataGenerator(
	rescale=1./255,
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,)

#We call the generator on our images
train_generator = train_datagen.flow_from_directory(
	train_dir,
	target_size=(150, 150),
	batch_size=32,
	# We're using binary_crossentropy as our loss value, so we need to label our data in a binary way
	class_mode='binary')

#we process our validation images to have the same overall characteristics as our training images
test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
	validation_dir,
	target_size=(150, 150),
	batch_size=20,
	class_mode='binary')

#Activates our model on manipulated training images and the relevant validation images
history = model.fit_generator(
	train_generator,
	steps_per_epoch=100,
	epochs=30,
	validation_data=validation_generator,
	validation_steps=50)

#We assign the various variables of the model's performance and plotting them
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#Optional: Save your trained model (I use this model in later programs so I keep this in)
model.save('cats_and_dogs_small_2.h5')
