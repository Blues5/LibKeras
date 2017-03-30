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
parser = argparse.ArgumentParser(description='Feature extraction with imagenet weights')
parser.add_argument('-i','--image', help='image path to extract features', required=True)
args = vars(parser.parse_args())

# Default directory
root_out = common.default_path() + '/outputs'

# Creating and instanciating the chosen CNN
cnet = common.ConvNet()
cnet.build_resnet50(include_top=False, weights='imagenet', classes=1000)
#cnet.build_vgg16(include_top=False, weights='imagenet', classes=1000)
#cnet.build_vgg19(include_top=False, weights='imagenet', classes=1000)
#cnet.build_inception3(include_top=True, weights='imagenet', classes=1000)

# Loading and pre-processing input image
img_path = sys.argv[2]
img = image.load_img(img_path)

# ResNet50, VGG16 and VGG19 uses (224,224) while InceptionV3 uses (299,299)
img = ImageOps.fit(img, (224, 224), Image.ANTIALIAS)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Use preprocess_input() for ResNet50, VGG16 and VGG19, and preprocess_inception() for InceptionV3
x = preprocess_input(x)
#x = common.preprocess_inception(x)

print('Input image shape:', x.shape)

# Extracting input image features
features = cnet.predict(x, batch_size=1)

# Concatenating to 1-D array
features = np.ndarray.flatten(features)
print("\n[INFO] Output array shape:", features.shape)

# Features output path
features_path = join(root_out, 'features/' + img_path + '_features.txt');

# Saving output
np.savetxt(features_path, features, fmt='%f')

import gc; gc.collect()
