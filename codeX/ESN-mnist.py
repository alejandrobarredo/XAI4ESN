from __future__ import print_function
from DeepESN import DeepESN as DeepESNorig
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from tensorflow.keras.utils import to_categorical
import time
import sys
import multiprocessing
import ntpath
import random
import time
import csv
import copy as cp
import os
import cv2
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import multiprocessing as multi
import time as time

DATASET_PATH = '../datasets/Mnist/'


def initialise_esn(inputs=28*28, units=100, layers=4, DeepIP=0, indexes=[1],
                   eta=0.000001, mu=0, Nepochs=10, sigma=0.1, threshold=0.1,
                   regularizations=np.array([0.001, 0.01, 0.1]),
                   trainMethod='Forest', connectivity=1, iss=0.6,
                   lis=0.01, rhos=0.9, model=False):

    # Parameters
    n_inputs = inputs

    IPconf = {}
    IPconf['DeepIP'] = DeepIP
    IPconf['indexes'] = indexes
    IPconf['eta'] = eta
    IPconf['mu'] = mu
    IPconf['Nepochs'] = Nepochs
    IPconf['sigma'] = sigma
    IPconf['threshold'] = threshold

    readout = {}
    readout['regularizations'] = regularizations
    readout['trainMethod'] = trainMethod  # 'Ridge'  # Lasso # 'Normal'
    # 'SVD' # 'Multiclass' # 'MLP'

    reservoirConf = {}
    reservoirConf['connectivity'] = connectivity

    _configs = {}
    _configs['IPconf'] = IPconf
    _configs['readout'] = readout
    _configs['reservoirConf'] = reservoirConf
    _configs['iss'] = iss
    _configs['lis'] = lis  # 0.2
    _configs['rhos'] = rhos  # 1.3

    if model:
        ESNorig = DeepESNorig(n_inputs, units, layers, _configs)
        return ESNorig, _configs
    else:
        return _configs


def serialize_mnist(_train_x, _train_y, _test_x, _test_y):
    new_train_x = []
    new_test_x = []
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six',
                   'Seven', 'Eight', 'Nine']
    cont = 0
    for image_idx in tqdm(range(_train_x.shape[0])):
        temp_img = _train_x[image_idx] / 255.0
        temp_target = _train_y[image_idx]
        row_wise = temp_img.flatten(order='C').reshape(1, -1)
        # column_wise = temp_img.flatten(order='F').reshape(1, -1)
        # temp_video = np.concatenate([row_wise, column_wise])
        if not os.path.exists(DATASET_PATH + 'Videos-row/' + class_names[
            temp_target]):
            os.mkdir(DATASET_PATH + 'Videos-row/' + class_names[
                temp_target])

        np.save(DATASET_PATH + 'Videos-row/' + class_names[
            temp_target] + '/' + str(cont) + '_' + class_names[
                    temp_target] + '_.npy', row_wise)
        # new_train_x.append(np.concatenate([row_wise, column_wise]))
        cont += 1
    # for image_idx in tqdm(range(_test_x.shape[0])):
    #     temp_img = _test_x[image_idx]
    #     row_wise = temp_img.flatten(order='C').reshape(1, -1)
    #     # column_wise = temp_img.flatten(order='F').reshape(1, -1)
    #     # temp_video = np.concatenate([row_wise, column_wise])
    #     if not os.path.exists(DATASET_PATH + 'Videos-row/' + class_names[
    #         temp_target]):
    #         os.mkdir(DATASET_PATH + 'Videos-row/' + class_names[
    #             temp_target])
    #     np.save(DATASET_PATH + 'Videos-row/' + class_names[
    #         temp_target] + '/' + str(cont) + '_' + class_names[
    #                 temp_target] + '_.npy', row_wise)
    #     # new_test_x.append(np.concatenate([row_wise, column_wise]))
    #     cont += 1

    return new_train_x, list(train_y), new_test_x, list(test_y)


if __name__ == '__main__':
    serialized = True
    if serialized:
        _inputs = 2
    else:
        _inputs = 784
    t1 = time.time()
    # Generate the class dictionary form the folders
    target_dic = {0: 0,
                  1: 1,
                  2: 2,
                  3: 3,
                  4: 4,
                  5: 5,
                  6: 6,
                  7: 7,
                  8: 8,
                  9: 9}

    class_names = list(target_dic.keys())
    ESN, configs = initialise_esn(inputs=_inputs, model=True)
    # We load the whole dataset and calculate its states
    # Initialise the ESN reservoir
    # We check if the base model is already generated and load it if it is so
    # THIS LINES ARE PUT IN THE START OF THE CODE TO ALLOW THE PARALLEL THREADS TO ACCESS IT

    # Generate the train test files for a test set of the group indicate from the arguments

    if not os.path.exists(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    if serialized:
        if not os.path.exists(DATASET_PATH + 'serialized_mnist.pkl'):
            from keras.datasets import mnist
            (train_X, train_y), (test_X, test_y) = mnist.load_data()
            train_X, train_y, test_X, test_y = serialize_mnist(train_X,
                                                               train_y,
                                                               test_X,
                                                               test_y)
            with open(DATASET_PATH + 'serialized_mnist.pkl', 'wb') as f:
                pkl.dump([(train_X, train_y), (test_X, test_y)],
                         f, pkl.HIGHEST_PROTOCOL)
        else:
            with open(DATASET_PATH + 'serialized_mnist.pkl', 'rb') as f:
                [(train_X, train_y), (test_X, test_y)] = pkl.load(f)

        print('Dataset serialized')
    else:
        if not os.path.exists(DATASET_PATH + 'mnist.pkl'):
            from keras.datasets import mnist
            (train_X, train_y), (test_X, test_y) = mnist.load_data()
            train_X = list(train_X.reshape(-1, 28*28, 1))
            test_X = list(test_X.reshape(-1, 28*28, 1))
            with open(DATASET_PATH + 'mnist.pkl', 'wb') as f:
                pkl.dump([(train_X, train_y), (test_X, test_y)],
                         f, pkl.HIGHEST_PROTOCOL)
        else:
            with open(DATASET_PATH + 'mnist.pkl', 'rb') as f:
                [(train_X, train_y), (test_X, test_y)] = pkl.load(f)
# []
    # Get the states for the test files so we dont have to load them for
    # each model
    if not os.path.exists('mnist_global_train_states.npy'):
        global_train_states = ESN.computeState(train_X)
        np.save(DATASET_PATH + 'mnist_global_train_states.npy',
                global_train_states)
    else:
        global_train_states = list(np.load(DATASET_PATH +
                                           'mnist_global_train_states.npy'))

    if not os.path.exists(DATASET_PATH + 'mnist_global_test_states.npy'):
        global_test_states = ESN.computeState(test_X)
        np.save(DATASET_PATH + 'mnist_global_test_states.npy',
                global_test_states)
    else:
        global_test_states = list(np.load(DATASET_PATH +
                                          'mnist_global_test_states.npy'))

    # Get the targets for the test files to create the predictions dataset
    global_train_targets = train_y
    global_test_targets = test_y

    # Iterate through classes and train each readout
    print('Training multiclass')
    print('     Amount of train videos: ' + str(len(global_train_targets)))
    print('     Amount of test videos: ' + str(len(global_test_targets)))

    ESN.trainReadout(global_train_states, global_train_targets)
    t2 = time.time()
    print('Tiempo: ' + str(t2 - t1))
    global_train_predictions = ESN.computeOutput(global_train_states).reshape(len(global_train_states), -1)
    global_test_predictions = ESN.computeOutput(global_test_states).reshape(len(global_test_states), -1)

    global_train_accuracy = accuracy_score(list(global_train_predictions), global_train_targets)
    global_train_recall = recall_score(list(global_train_predictions), global_train_targets, average='macro')
    global_train_precision = precision_score(list(global_train_predictions), global_train_targets, average='macro')

    global_test_accuracy = accuracy_score(list(global_test_predictions), global_test_targets)
    global_test_recall = recall_score(list(global_test_predictions), global_test_targets, average='macro')
    global_test_precision = precision_score(list(global_test_predictions), global_test_targets, average='macro')

    cm = confusion_matrix(global_test_targets, list(global_test_predictions))

    print(cm)

    print('GLOBAL TRAIN:   ')
    print('      Accuracy: ' + str(global_train_accuracy))
    print('        Recall: ' + str(global_train_recall))
    print('     Precision: ' + str(global_train_precision))
    print('GLOBAL TEST:   ')
    print('      Accuracy: ' + str(global_test_accuracy))
    print('        Recall: ' + str(global_test_recall))
    print('     Precision: ' + str(global_test_precision))

    with open(DATASET_PATH + 'mnist_fitted_model.pkl', 'wb') as f:
        pkl.dump(ESN, f, pkl.HIGHEST_PROTOCOL)

