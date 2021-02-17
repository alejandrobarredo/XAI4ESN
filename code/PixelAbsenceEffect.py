from __future__ import print_function
from DeepESN import DeepESN as DeepESNorig
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
import matplotlib.pyplot as plt
import time
import sys
import tqdm
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
import multiprocessing as multi

parser = argparse.ArgumentParser(description='Select the dataset size')
parser.add_argument('-frames', default=0, type=int,
                    help='choose the frame to compute')
parser.add_argument('-all', default=1, type=int,
                    help='Choose if pixel effect is grouped or not')


args = parser.parse_args()
frame_to_process = args.frames
all = args.all
'''
En este código tratamos de mostrar la importancia relativa de cada pixel en la salida de la red.
Para ello computamos el resultado de la red para el video al completo retirando un pixel y guardamos
el efecto de este a modo de (diferencia/dirección).
'''
dataset = 'UCF50'
dataset_path = '../datasets/' + dataset + '/'
video_path = dataset_path + 'Videos/'
files_path = dataset_path + 'Splits/' + dataset + '_files.txt'
models_path = dataset_path + 'Models/'
states_path = dataset_path + 'States/'
results_path = dataset_path + 'Results/'


def load_video(_video_path):
    video_arr = []
    cap = cv2.VideoCapture(_video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        else:
            frame_count += 1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.shape[0] != 240 and frame.shape[1] != 320:
            gray_frame = cv2.resize(gray_frame, (240, 320))
        gray_frame = gray_frame.astype('float32')/255
        gray_flat = gray_frame.reshape(-1, 1)
        video_arr.append(gray_flat)
    new_video_arr = video_arr
    _video = np.concatenate(new_video_arr, axis=1)
    return _video


def get_train_test_files(path, test_g=1, val_g=-1, size=-1, allto=True):
    if not allto:
        _train_files = []
        _val_files = []
        _test_files = []
        train_ids = []
        val_ids = []
        test_ids = []

        if dataset == 'UCF50':
            if test_g < 10:
                test_ids = ['_g0' + str(test_g) + '_']
            else:
                test_ids = ['_g' + str(test_g) + '_']
            if val_g != -1:
                if val_g < 10:
                    val_ids = ['_g0' + str(val_g) + '_']
                else:
                    val_ids = ['_g' + str(val_g) + '_']
            train_ids = ['_g0' + str(i) + '_' for i in range(1, 10)] + ['_g' + str(i) + '_' for i in range(10, 26)]
            train_ids.remove(test_ids[0])
            if val_g != -1:
                train_ids.remove(val_ids[0])
        elif dataset == 'KTH':
            test_ids = ['person11', 'person12', 'person13', 'person14', 'person15', 'person16', 'person17', 'person18']
            if val_g == -1:
                train_ids = ['person01', 'person02', 'person03', 'person04', 'person05', 'person06', 'person07', 'person08',
                             'person09', 'person10', 'person19', 'person20', 'person21', 'person22', 'person23', 'person24',
                             'person25']
            else:
                val_ids = ['person19', 'person20', 'person21', 'person23', 'person24', 'person25', 'person01', 'person04']
                train_ids = ['person22', 'person02', 'person03', 'person05', 'person06', 'person07', 'person08', 'person09',
                             'person10']

        if val_g != -1:
            with open(path) as f:
                file_reader = csv.reader(f, delimiter='\n')
                for row in file_reader:
                    for test_id in test_ids:
                        if row[0].find(test_id) != -1:
                            _test_files.append(row[0])
                            break
                    for val_id in val_ids:
                        if row[0].find(val_id) != -1:
                            _val_files.append(row[0])
                            break
                    for train_id in train_ids:
                        if row[0].find(train_id) != -1:
                            _train_files.append(row[0])
                            break
        else:
            with open(path) as f:
                file_reader = csv.reader(f, delimiter='\n')
                for row in file_reader:
                    for test_id in test_ids:
                        if row[0].find(test_id) != -1:
                            _test_files.append(row[0])
                            break
                    for train_id in train_ids:
                        if row[0].find(train_id) != -1:
                            _train_files.append(row[0])
                            break

            if size is not -1:
                _train_files = np.random.choice(_train_files, int(len(_train_files) * (size/100)), replace=False)
                _test_files = np.random.choice(_test_files, int(len(_test_files) * (size/100)), replace=False)
                if val_g != -1:
                    _val_files = np.random.choice(_val_files, int(len(_val_files) * (size/100)), replace=False)

            random.shuffle(_train_files)
            random.shuffle(_test_files)
            if val_g != -1:
                random.shuffle(_val_files)
    else:
        _files = []
        with open(path) as f:
            file_reader = csv.reader(f, delimiter='\n')
            for row in file_reader:
                _files.append(row[0])
        _train_files, _test_files = train_test_split(_files, test_size=0.1, shuffle=True)

    return _train_files, _test_files, None


def get_targets_from_files(_files):
    _targets = []
    for _file in _files:
        for cont, _clase in enumerate(class_names):
            if _file.find('_' + _clase + '_') != -1:
                _targets.append(target_dic[_clase])
    return _targets


def get_square_area(_coordintates, image_width=320, side=25):
    x, y = _coordintates
    _pixels = []
    for i_h in range(side):
        for i_w in range(side):
            _pixels.append(((i_h + x) * image_width) + (i_w + y))
    return _pixels


def get_list_of_coordinates(image_height=240, image_width=320, side=25):
    _coordinates = []
    for i_h in range(0, image_height, side):
        for i_w in range(0, image_width, side):
            _coordinates.append([i_h, i_w])
    return _coordinates


# The traceback calculation for the position of a given pixel is:
#       - The frame is equal
#       - The row is given by: pos_reshaped // 320 (given that there are 320 rows in the frame)
#       - The columns is given by: pos_reshaped % 240 (given that there are 240 columns in the frame)
# We load the model
with open('../datasets/UCF50/Results/fitted_model-A72-R74-P72.pkl', 'rb') as f:
    ESN = pkl.load(f)

# Generate target dic
target_dic = {}
for i, name in enumerate(os.listdir(video_path)):
    target_dic[name] = i
class_names = list(target_dic.keys())

# We set the frame objective
frame_objective = 50

video_files = os.listdir(video_path)
for video_file in video_files:
    # First we calculate the original output of the model
    #   Calculate states
    target = get_targets_from_files(video_file)
    video = load_video(video_path + video_file)
    states = ESN.computeState([video])[0]
    sampled_idx = np.linspace(0, states.shape[1] - 1, frame_objective, dtype=np.int64)
    sampled_states = []
    for frame_sampled in sampled_idx:
        sampled_states.append(states[:, frame_sampled].reshape(-1, 1))
    states = np.concatenate(sampled_states, axis=1)
    #   Calculate probabilities
    original_pred = ESN.computeOutput([states])
    original_probs = ESN.computeOutput([states], probs=True)
    original_probs = np.squeeze(original_probs)
    frame = 0
    if target == original_pred:
        print(video_file)
        print(target)
        print(original_pred)
        break

# We create a matrix to hold the effects (same size as the video)
absence_effect_matrix = []
for i in range(50):
    absence_effect_matrix.append(np.zeros_like(video))

def get_square_area(coordintates, side=8):
    x, y = coordintates
    pixels = []
    for i in range(side):
        for ii in range(side):
            pixels.append(((i + x) * 320) + (ii + y))
    return pixels


def worker(comp):
    if all:
        positions = [comp]
    else:
        positions = get_square_area(comp)

    compute_difference(frame_to_process, positions)


def compute_difference(_frame, positions):
    temp_video = video
    for position in positions:
        temp_video[position, sampled_idx[_frame]] = 0.0
    states = ESN.computeState([temp_video])[0]
    sampled_states = []
    for frame_sampled in sampled_idx:
        sampled_states.append(states[:, frame_sampled].reshape(-1, 1))
    states = np.concatenate(sampled_states, axis=1)
    temp_probs = ESN.computeOutput([states], probs=True)
    temp_probs = np.squeeze(temp_probs)
    difference = temp_probs - original_probs
    if np.max(difference) > 0 or np.min(difference) < 0:
        print('Difference: ' + str(difference))
    np.save(results_path + 'Diffs_1/' + str(_frame) + '-' + str(positions[0]) + 'diff', difference, allow_pickle=False)


if __name__ == '__main__':
    comp = []
    if all:
        for ii in range(76800):
            comp.append(ii)
    else:
        for i in range(0, 240, 4):
            for ii in range(0, 320, 4):
                comp.append([i, ii])
    print('Pixeles para analizar: ' + str(len(comp)))
    pool = multiprocessing.Pool(processes=30)
    for _ in tqdm.tqdm(pool.imap_unordered(worker, comp), total=len(comp)):
        pass

    print('Finished')



