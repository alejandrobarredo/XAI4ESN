from __future__ import print_function
from DeepESN import DeepESN as DeepESNorig
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
import matplotlib.pyplot as plt
import time
import sys
from tqdm import tqdm
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

import warnings

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description='Select the dataset size')
parser.add_argument('-dataset', default='SquaresVsCrosses',
                    help='choose the dataset')
parser.add_argument('-frames', default=-1, type=int,
                    help='choose the frame to compute')
parser.add_argument('-boostrap', default=1, type=int,
                    help='boostrap or not')
parser.add_argument('-height', default=1, type=int,
                    help='choose the frame to compute')
parser.add_argument('-width', default=1, type=int,
                    help='choose the frame to compute')
parser.add_argument('-target', default=-1, type=int,
                    help='choose the target')
parser.add_argument('-all', default=1, type=int,
                    help='Choose if pixel effect is grouped or not')
parser.add_argument('-boostrap_objective', default=100000, type=int,
                    help='boostrap or not')
parser.add_argument('-interpolate', default=0, type=int)
parser.add_argument('-video_name', default=None, type=str)
parser.add_argument('-video_class', default=None, type=str)
parser.add_argument('-recover', default=False, type=bool)
parser.add_argument('-paralel', default=True, type=bool)
parser.add_argument('-effect_time', default=60, type=int)
parser.add_argument('-independent', default=0, type=int)
parser.add_argument('-low_range', default=0, type=int)
parser.add_argument('-high_range', default=50, type=int)
parser.add_argument('-activation', default=0.0, type=float)


args = parser.parse_args()
dataset = str(args.dataset)
frame_objective = args.frames
boostrap = args.boostrap
heigth_objective = args.height
width_objective = args.width
target_objective = args.target
sampled = False
all = args.all
boostrap_objective = args.boostrap_objective
interpolate = args.interpolate
objective_video_name = args.video_name
objective_video_class = args.video_class
recover = args.recover
paralel = args.paralel
effect_time = args.effect_time
independent = args.independent
low_range = args.low_range
high_range = args.high_range
activation = args.activation
sides = [heigth_objective, width_objective]
width = sides[0]
heigth = sides[1]
pixel_size = 1

'''
En este código tratamos de mostrar la importancia relativa de cada pixel en la salida de la red.
Para ello computamos el resultado de la red para el video al completo retirando un pixel y guardamos
el efecto de este a modo de (diferencia/dirección).
'''

print('Dataset: ' + str(dataset))
#dataset = 'SimplifiedVideos'
dataset_path = '../datasets/' + dataset + '/'
video_path = dataset_path + 'Videos/'
files_path = dataset_path + 'Splits/' + dataset + '_files.txt'
models_path = dataset_path + 'Models/'
states_path = dataset_path + 'States/'
results_path = dataset_path + 'Results/'

touched_pixel_image = np.zeros((heigth, width))


def load_video(_video_path, simplified=False):
    video_arr = []
    if simplified:
        _video = np.load(_video_path)
        _video1 = _video.reshape(-1, frame_objective)
        print('Video shape' + str(_video1.shape))
    else:
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
    return _video, _video1


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


def get_square_area(_coordintates, image_width=28, side=3):
    x, y = _coordintates
    _pixels = []
    for i_h in range(side):
        for i_w in range(side):
            pix = ((i_h + x) * image_width) + (i_w + y)
            _pixels.append(pix)
    return _pixels


def calculate_area(_x, _y, _side=5, _image_width=50):
    _new_positions = []
    for i_h in range(_side):
        for i_w in range(_side):
            if (_x + i_h < _image_width) and (_y + i_w < _image_width):
                _new_positions.append([_x + i_h, _y + i_w])
    return _new_positions


def get_list_of_coordinates(image_height=28, image_width=28, side=3):
    _coordinates = []
    for i_h in range(0, image_height, side):
        for i_w in range(0, image_width, side):
            _coordinates.append([i_h, i_w])
    return _coordinates


# The traceback calculation for the position of a given pixel is:
#       - The frame is equal
#       - The row is given by: pos_reshaped // 320 (given that there are 320
#       rows in the frame)
#       - The columns is given by: pos_reshaped % 240 (given that there are
#       240  columns in the frame)
# We load the model

with open('../datasets/' + dataset + \
          '/Results/' + dataset + '_fitted_model_A100.pkl',
          'rb') as f:
    ESN = pkl.load(f)

# Generate target dic
if dataset == 'Mnist':
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six',
                   'Seven', 'Eight', 'Nine']
    target_dic = {'Zero': 0, 'One': 1, 'Two': 2, 'Three': 3, 'Four': 4,
                  'Five': 5, 'Six': 6, 'Seven': 7, 'Eight': 8, 'Nine': 9}
elif dataset == '2ClassSimplifiedVideos':
    class_names = ['Circling', 'Crossing']
    target_dic = {'Circling': 0, 'Crossing': 1}
elif dataset == 'SquaresVsCrosses':
    class_names = ['Crosses', 'Squares']
    target_dic = {'Crosses': 0, 'Squares': 1}
else:
    target_dic = {}
    for i, name in enumerate(os.listdir(video_path)):
        target_dic[name] = i
    class_names = list(target_dic.keys())

print(target_dic)


def worker(comp):
    compute_difference(comp)


def compute_difference(positions):
    # print('Frame: ' + str(_frame) + ' position: ' + str(positions))
    full_positions = []
    if independent == 1:
        temp_video = np.zeros_like(video)
    else:
        temp_video = cp.deepcopy(video)

    if boostrap:
        b_idx = positions[0]
        positions = positions[1:]
    pixel_values = []
    for position in positions:
        _x = position[0]
        _y = position[1]
        touched_pixel_image[_x, _y] = 1
        _frame = position[2]
        new_positions = calculate_area(_x, _y, _side=3, _image_width=50)
        new_positions = [[_x, _y]]
        for new_position in new_positions:
            _x, _y = new_position
            full_positions.append([_x, _y, _frame])
            if sides[0] == 1 or sides[1] == 1:
                pixel_values.append(temp_video[0, _frame])
                if _frame == 0:
                    temp_video[0, _frame] = temp_video[0, _frame + 1]
                elif _frame == frame_objective - 1:
                    temp_video[0, _frame] = temp_video[0, _frame - 1]
                else:
                    v_1 = temp_video[0, _frame - 1]
                    v_2 = temp_video[0, _frame + 1]
                    interp = (v_1 + v_2) / 2
                    temp_video[0, _frame] = interp
            else:
                pixel_values.append(temp_video[_frame, _x, _y])
                if interpolate == 1:
                    if _frame == 0:
                        temp_video[_x, _y, _frame] = temp_video[_x, _y, _frame + 1]
                    elif _frame == frame_objective:
                        temp_video[_x, _y, _frame] = temp_video[_x, _y, _frame - 1]
                    else:
                        v_1 = temp_video[_x, _y, _frame - 1]
                        v_2 = temp_video[_x, _y, _frame + 1]
                        interp = (v_1 + v_2) / 2
                        temp_video[_x, _y, _frame] = interp
                else:
                    if effect_time == -1:
                        temp_video[_frame, _x, _y] = activation
                    else:
                        for each_frame in range(_frame, _frame + effect_time):
                            if each_frame < frame_objective:
                                if activation == -1:
                                    if temp_video[each_frame, _x, _y] < 0.5:
                                        temp_video[each_frame, _x, _y] = 1.0
                                    else:
                                        temp_video[each_frame, _x, _y] = 0.0
                                else:
                                    temp_video[each_frame, _x, _y] = activation

    temp_video = temp_video.reshape(-1, frame_objective)
    states = ESN.computeState([temp_video])[0]

    temp_probs = ESN.computeOutput([states], probs=True)
    temp_probs = np.squeeze(temp_probs)
    difference = temp_probs - original_probs

    for cont, position in enumerate(full_positions):
        _x = position[0]
        _y = position[1]
        _f = position[2]
        pixel_value = pixel_values[cont]
        proportional_difference = difference
        # print('*********************************')
        # print('Pixel value: ' + str(pixel_value))
        # print('Difference: ' + str(difference))
        # print('Proportional differnece: ' + str(proportional_difference))
        # print('*********************************')

        np.save(results_path + 'Diffs_' + video_name + '/' + str(b_idx) +
                '-' + str(_x) + '-' + str(_y) + '-' + str(_f) + '-' + 'diff',
                proportional_difference,
                allow_pickle=False)


def get_testing_video():
    if objective_video_name is None:
        print('Video name is None')
        _, video_files, _ = get_train_test_files(files_path, test_g=1,
                                                 allto=True)
    elif objective_video_name == 'CombinedVideo' or objective_video_name == \
            'NoiseVideo':
        print('Analyzing video combined')
        video_file = '../datasets/' + dataset + '/' + objective_video_name + '.npy'
        video, video_r = load_video(video_file, simplified=True)
        print(video_r.shape)
        states = ESN.computeState([video_r])[0]
        original_pred = ESN.computeOutput([states])[0]
        original_probs = ESN.computeOutput([states], probs=True)
        original_probs = np.squeeze(original_probs)
        video_name = video_file.split(sep='/')[-1][:-4]
        return video, video_r, original_pred, original_probs, \
               video_name

    else:
        video_files = [video_path + objective_video_class +
                       '/' + objective_video_name + '.npy']

    for video_file in tqdm(video_files):
        # First we calculate the original output of the model
        #   Calculate states
        target = get_targets_from_files([video_file])[0]
        if target == target_objective or target_objective == -1:
            video, video_r = load_video(video_file, simplified=True)
            states = ESN.computeState([video_r])[0]
            #   Calculate probabilities
            original_pred = ESN.computeOutput([states])[0]
            original_probs = ESN.computeOutput([states], probs=True)
            original_probs = np.squeeze(original_probs)
            frame = 0

            if target == original_pred:
                print('Found a correct prediction: ')
                print('     ' + video_file)
                print('     ' + str(target))
                print('     ' + str(original_pred))
                print('     ' + str(original_probs))

                video_name = video_file.split(sep='/')[-1][:-4]
                return video, video_r, original_pred, original_probs, \
                       video_name


def get_existing_absence_effect(_path, _absence_matrix):
    list_files = os.listdir(_path)
    for _file in list_files:
        _b = int(_file.split('-')[0])  # boostrap index
        _x = int(_file.split('-')[1])  # x position
        _y = int(_file.split('-')[2])  # y position
        _f = int(_file.split('-')[3])  # frame

        matrix = np.load(results_path + 'Diffs_' + video_name + '/' + _file)
        _absence_matrix[0, _f] = matrix

    return _absence_matrix


if __name__ == '__main__':
    video, video_r, original_pred, original_probs, video_name = \
        get_testing_video()

    if not os.path.exists(results_path + 'Diffs_' + video_name + '/'):
        os.mkdir(results_path + 'Diffs_' + video_name + '/')

    # We create a matrix to hold the effects (same size as the video)
    absence_effect_matrix = []
    for i in range(frame_objective):
        absence_effect_matrix.append(np.zeros_like(video))
    if len(os.listdir(results_path + 'Diffs_' + video_name + '/')) > 0 and \
            recover:
        absence_effect_matrix = get_existing_absence_effect(results_path + 'Diffs_' + video_name + '/')
    comp = []
    available_positions = []
    if sides[0] == 1 and sides[1] == 1:
        for iii in range(0, frame_objective):
            available_positions.append([sides[0], sides[1], iii])
    else:
        for i in range(0, sides[0] - pixel_size, 1):
            for ii in range(0, sides[1] - pixel_size, 1):
                for iii in range(0, frame_objective):
                    available_positions.append([i, ii, iii])
                # available_positions.append([i, ii, 30])

    if boostrap:
        for i in range(boostrap_objective):
            idxs = np.random.choice(range(len(available_positions)), size=10)
            comp.append([i] + [available_positions[idx] for idx in idxs])
        print('Boostrap para analizar: ' + str(len(comp)))
    else:
        comp = available_positions
        print('Pixeles para analizar: ' + str(len(comp)))

    if paralel:
        # compute_difference(comp[0])
        pool = multiprocessing.Pool(processes=30)

        for _ in tqdm(pool.imap_unordered(worker, comp), total=len(comp)):
            pass
    else:
        for cm in comp:
            compute_difference(cm)

    plt.imshow(touched_pixel_image, cmap='binary')
    plt.savefig('TouchedPixelImage.pdf')
    print('Finished')



