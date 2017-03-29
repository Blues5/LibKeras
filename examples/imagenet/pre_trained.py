import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import argparse
import common
import numpy as np
import sys
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from os.path import join
from PIL import Image, ImageOps

# Parsing input arguments
parser = argparse.ArgumentParser(description='Pre-trained classification with imagenet weights')
parser.add_argument('-i','--image', help='image path to be classified', required=True)
args = vars(parser.parse_args())

# Default directory
root_in = common.default_path() + '/inputs'

# Predictions path
predictions_path = join(root_in, 'predictions/' + 'imagenet_predictions.json');

# Creating and instanciating the chosen CNN
cnet = common.ConvNet()
cnet.build_resnet50(include_top=True, weights='imagenet', classes=1000)
#cnet.build_vgg16(include_top=True, weights='imagenet', classes=1000)
#cnet.build_vgg19(include_top=True, weights='imagenet', classes=1000)

# Loading and pre-processing input image
img_path = sys.argv[2]
img = image.load_img(img_path)
img = ImageOps.fit(img, (224, 224), Image.ANTIALIAS)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print('Input image shape:', x.shape)

# Predicting input image
preds = cnet.predict(x, batch_size=1)
print('Predicted:', cnet.decode_predictions(preds, path=predictions_path, top=5))

import gc; gc.collect()
