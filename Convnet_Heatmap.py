#In this model, we'll be using the pre-trained VGG16 model to show the areas of a given image that convnets focus on in their
#computations
#We'll be doing this by feeding the model an image, and then keeping track of this activity.
#Finally, we'll superimpose this activity on the original image in a 'heatmap', that should be familiar to anyone who's seen Predator

#We import the model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import cv2

model = VGG16(weights='imagenet')

#Loading and prepping our data
img_path = '/path/to/your/dataset'

img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

#These two lines will run the model on the image and have it print 3 guesses as to what the image is
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

#We then get the index number of our most likely predicition
np.argmax(preds[0])

#The african elephant name in the following variable should be changed to the top predicition of the above prediction
#The model output number, in this case 386, should also be changed to the relevant index number
african_elephant_output = model.output[:, 386]

#We get the output of the last layer in the model
last_conv_layer = model.get_layer('block5_conv3')

#We get the gradient of the image's subject from the output feature map
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

#Create a vector of shape (512,), containg the mean of the gradient over a each channel in the output
pooled_grads = K.mean(grads, axis=(0, 1, 2))

#Prepare our above variables for the loop below
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

#Multiply each channel by the intensity the model assigned it for each seperate window that was processed
for i in range(512):
	conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

#Assign the mean of the the outputs from the loop to the heatamp variable
heatmap = np.mean(conv_layer_output_value, axis=-1)

#Load the heatmap, resize it the original photo's size, 
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#convert the heatmap to RGB format
heatmap = np.uint8(255 * heatmap)
#Finally, apply the heatmap to the original photo
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#0.4 here refers to the intensity of the superimposed image, feel free to tinker with it
superimposed_img = heatmap * 0.4 + img

#Saves the image using CV2. Enjoy!
cv2.imwrite('/home/bzzbzz/Downloads/mini_boss_cam.jpg', superimposed_img)