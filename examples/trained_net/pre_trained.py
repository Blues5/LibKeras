import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import common
import numpy as np
from keras.preprocessing import image
from os.path import join

# Input parameters
dataset = 'brain'
grayscale = True
n_predictions = 2

# Default directory
root = common.default_path()

# Model path
model_path = join(root, 'outputs/models/' + dataset + '_model.h5');

# Predictions path
predictions_path = join(root, 'inputs/predictions/' + dataset + '_predictions.json');

# Creating and instanciating the chosen CNN
cnet = common.ConvNet()
cnet.load_model(model_path)

# Loading and pre-processing input image
img_path = ''
img = image.load_img(img_path, grayscale=grayscale)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
print('Input image shape:', x.shape)

# Predicting input image
preds = cnet.predict(x, batch_size=1)
print('Predicted:', cnet.decode_predictions(preds, path=predictions_path, top=n_predictions))

import gc; gc.collect()
