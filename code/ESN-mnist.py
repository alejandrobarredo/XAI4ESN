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
import tqdm
import multiprocessing as multi
import time as time


def initialise_esn(model=False):
    # Parameters

    n_inputs = 28*28
    units = 100
    layers = 4

    IPconf = {}
    IPconf['DeepIP'] = 0
    IPconf['indexes'] = [1]
    IPconf['eta'] = 0.000001
    IPconf['mu'] = 0
    IPconf['Nepochs'] = 10
    IPconf['sigma'] = 0.1
    IPconf['threshold'] = 0.1

    readout = {}
    readout['regularizations'] = np.array([0.001, 0.01, 0.1])  # 0.0001, 0.001, 0.01
    readout['trainMethod'] = 'Forest'  # 'Ridge'  # Lasso # 'Normal'  # 'SVD' # 'Multiclass' # 'MLP'

    reservoirConf = {}
    reservoirConf['connectivity'] = 1

    configs = {}
    configs['IPconf'] = IPconf
    configs['readout'] = readout
    configs['reservoirConf'] = reservoirConf
    configs['iss'] = 0.6
    configs['lis'] = 0.01  # 0.2
    configs['rhos'] = 0.9  # 1.3

    if model:
        ESNorig = DeepESNorig(n_inputs, units, layers, configs)
        return ESNorig, configs
    else:
        return configs


if __name__ == '__main__':
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
    ESN, configs = initialise_esn(model=True)
    # We load the whole dataset and calculate its states
    # Initialise the ESN reservoir
    # We check if the base model is already generated and load it if it is so
    # THIS LINES ARE PUT IN THE START OF THE CODE TO ALLOW THE PARALLEL THREADS TO ACCESS IT

    # Generate the train test files for a test set of the group indicate from the arguments
    with open('mnist.pkl', 'rb') as f:
        (train_X, train_y), (test_X, test_y) = pkl.load(f)

    train_X = list(train_X.reshape(-1, 28*28, 1))
    test_X = list(test_X.reshape(-1, 28*28, 1))

    with open('mnist.pkl', 'wb') as f:
        pkl.dump(((train_X, train_y), (test_X, test_y)), f, \
                 pkl.HIGHEST_PROTOCOL)

    # Get the states for the test files so we dont have to load them for each model
    if not os.path.exists('mnist_global_train_states.npy'):
        global_train_states = ESN.computeState(train_X)
    else:
        global_train_states = list(np.load('mnist_global_train_states.npy'))

    if not os.path.exists('mnist_global_test_states.npy'):
        global_test_states = ESN.computeState(test_X)
    else:
        global_test_states = list(np.load('mnist_global_test_states.npy'))

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

    with open('mnist_fitted_model.pkl', 'wb') as f:
        pkl.dump(ESN, f, pkl.HIGHEST_PROTOCOL)

