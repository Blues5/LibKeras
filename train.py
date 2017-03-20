import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import common
import gzip
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import sys
from data import load_data
from itertools import product
from os.path import join
from sklearn.model_selection import train_test_split

# Default directory
root_out = '/home/gustavo/src/LibKeras'

# Type of data entries
n_runs = 1
datasets = [{'name': 'imagenet', 'n_classes': 1000}]
data_types = [{'type': 'image', 'ref': False}]
normalization_methods = ['none']
test_sizes = [0.2]
params = list(product(datasets, data_types, normalization_methods, test_sizes))

# Fixed parameters
learning_rate = 0.01
batch_size = 128
n_epochs = 90
val_size = 0.1
metric = 'accuracy'
loss_func = 'categorical_crossentropy'
results = []

# Loop to hold all the desired configurations
for d, dt, nm, ts in params:
    for i in range(n_runs):
        data, labels = load_data(d['name'], dt['type'], dt['ref'], nm)
        input_shape = data.shape[1:]

        # Splitting data into training and test sets
        data_train, data_test, lab_train, lab_test = train_test_split(data, labels, test_size=ts, random_state=i)

        # Building CNN, note that you can choose the build function according to common.py
        cnet = common.ConvNet()
        cnet.build_resnet50(include_top=True, weights=None, input_shape=input_shape, classes=d['n_classes'])

	# Training current network
        cnet.train(data_train, lab_train, d['n_classes'], learning_rate=learning_rate, batch_size=batch_size, n_epochs=n_epochs, validation_size=val_size, metric=metric, loss_func=loss_func)

        # Evaluating current network
        acc = cnet.evaluate(data_test, lab_test, d['n_classes'], batch_size)

	# Saving network model
        mname = '%s_model.json' % (d['name'])
        cnet.save_model(join(root_out, 'models', mname))

        # Saving trained network weights
        wname = '%s_%s_%s_%.2f_%02i.h5' % (d['name'], dt['type'], nm, ts, i)
        cnet.save_weight(join(root_out, 'weights', wname))

        # Plotting the accuracy history
        history = cnet.get_history()
        fname = '%s_%s_%s_%.2f_%02i' % (d['name'], dt['type'], nm, ts, i)
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='lower right')
        plt.savefig(join(root_out, 'history', 'history_' + fname + '.jpg'))
        plt.close()

        # Dumping history to a .pkl file
        pkl.dump(history, gzip.open(join(root_out, 'history', 'history_' + fname + '.pkl'), 'wb'))

        # Saving results output on a .csv file
        results.append([d['name'], dt['type'], nm, ts, i, acc])
        cnet = None
        df = pd.DataFrame(results, columns=['dataset', 'data_type', 'normalization_method', 'test_size', 'running_num', 'acc'])
        df.to_csv(join(root_out, 'results.csv'))

        # End of current iteration
        print("\n[INFO] Running #{:d} ok!".format(i))

import gc; gc.collect()
