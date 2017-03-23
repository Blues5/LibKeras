"""SampleNet model for Keras.
"""
from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras import backend as K
from keras.models import Model
from keras.layers import Convolution2D, GlobalMaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Input
from keras.layers.core import Dense, Flatten

WEIGHTS_PATH = '/home/gustavo/src/LibKeras/weights/trained.h5'
WEIGHTS_PATH_NO_TOP = '/home/gustavo/src/LibKeras/weights/trained_no_top.h5'

def SampleNet(include_top=True, weights='trained', input_shape=None, pooling=None, classes=2):

	if weights not in {'trained', None}:
		raise ValueError('The `weights` argument should be either `None` (random initialization) or `trained` (pre-trained on own dataset).')

	# Determining input shape
	img_input = Input(shape=input_shape)

	# Main network
	x = Convolution2D(32, (5, 5), activation='relu', padding='same', name='conv1')(img_input)
	x = AveragePooling2D((2, 2), strides=(2, 2), name='pool1')(x)
	x = Flatten(name='flatten')(x)
	x = Dense(100, activation='relu', name='fc1')(x)

	if include_top:
		# Classification block
		x = Dense(classes, activation='softmax', name='predictions')(x)
	else:
		if pooling == 'avg':
			x = GlobalAveragePooling2D()(x)
		elif pooling == 'max':
			x = GlobalMaxPooling2D()(x)

	# Creating model
	inputs = img_input
	model = Model(inputs, x, name='samplenet')

	# Loading weights
	if weights == 'trained':
		if include_top:
			weights_path = WEIGHTS_PATH
		else:
			weights_path = WEIGHTS_PATH_NO_TOP
		model.load_weights(weights_path)

	if K.backend() == 'theano':
		layer_utils.convert_all_kernels_in_model(model)

	if K.image_data_format() == 'channels_first':
		if include_top:
			maxpool = model.get_layer(name='block5_pool')
			shape = maxpool.output_shape[1:]
			dense = model.get_layer(name='fc1')
			layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

			if K.backend() == 'tensorflow':
				warnings.warn('You are using the TensorFlow backend, yet you '
				'are using the Theano '
				'image data format convention '
				'(`image_data_format="channels_first"`). '
				'For best performance, set '
				'`image_data_format="channels_last"` in '
				'your Keras config '
				'at ~/.keras/keras.json.')
	return model
