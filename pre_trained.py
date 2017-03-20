import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import common
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from os.path import join

# Default directory
root_out = '/home/gustavo/src/LibKeras'

# Predictions path
predictions_path = join(root_out, 'predictions/' + 'imagenet_predictions.json');

# Creating and instanciating the chosen CNN
cnet = common.ConvNet()
cnet.build_resnet50(include_top=True, weights='imagenet', classes=1000)
#cnet.build_vgg16(include_top=True, weights='imagenet', classes=1000)
#cnet.build_vgg19(include_top=True, weights='imagenet', classes=1000)

# Loading and pre-processing input image
img_path = 'dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print('Input image shape:', x.shape)

# Predicting input image
preds = cnet.predict(x, batch_size=1)
print('Predicted:', cnet.decode_predictions(preds, path=predictions_path, top=5))

import gc; gc.collect()
