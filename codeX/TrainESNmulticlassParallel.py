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


# PARSING ARGUMENTS  *********************************************************
# Elegimos el dataset que queremos entrenar
parser = argparse.ArgumentParser(description='Select the dataset size')
parser.add_argument('-dataset', default='Mnist',
                    help='Choose the dataset to train'
                         '  - Hollywood2'
                         '  - HMDB51'
                         '  - Olympic Sports'
                         '  - KTH'
                         '  - Weizmann'
                         '  - IXMAS'
                         '  - HOHA'
                         '  - UCF11'
                         '  - UCF50'
                         '  - UCF101'
                         '  - SimplifiedVideo')
parser.add_argument('-frames', default=784, type=int,
                    help='Set the length in frames for the videos')
parser.add_argument('-size', default=-1, type=int,
                    help='Set the size in % for the dataset')
parser.add_argument('-frame_width', default=28, type=int)
parser.add_argument('-frame_height', default=28, type=int)
parser.add_argument('-workers', default=2, type=int,
                    help='Choose the amount of workers to parallelize. (default=2)')
parser.add_argument('-makeStates', default=-1, type=int,
                    help='Choose to train codeX the states.')
parser.add_argument('-makeModel', default=-1, type=int,
                    help='Choose to initialise the rservoir')
parser.add_argument('-readOut', default='Forest',
                    help='Choose the readout for the model:'
                         '  - MLP'
                         '  - Forest')
parser.add_argument('-RGB', default='0', type=int,
                    help='Choose between rgb video or gray scale:'
                         'default=0 - gray'
                         '       =1 - rgb')

args = parser.parse_args()
dataset = args.dataset
frame_objective = args.frames
dataset_size = args.size
frame_width = args.frame_width
frame_height = args.frame_height
n_workers = args.workers
make_states = args.makeStates
make_model = args.makeModel
readOut = args.readOut
rgb = args.RGB

# Set the directories for the dataset selected
dataset_path = '../datasets/' + dataset + '/'
video_path = dataset_path + 'Videos/'
files_path = dataset_path + 'Splits/' + dataset + '_files.txt'
models_path = dataset_path + 'Models/'
states_path = dataset_path + 'States/'
results_path = dataset_path + 'Results/'

# PARSING ARGUMENTS  **********************************************************
# *****************************************
def initialise_esn(model=False):
    # Parameters
    if dataset == 'Mnist':
        n_inputs = 1
        units = 20
        layers = 4
    elif dataset == 'SquaresVsCrosses':
        n_inputs = frame_height * frame_width
        units = 50
        layers = 4
    elif rgb:
        n_inputs = frame_height * frame_width * 3
        units = 100
        layers = 4
    else:
        n_inputs = frame_height * frame_width
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
    readout['trainMethod'] = 'Forest'  # 'Ridge'  # Lasso # 'Normal'  #
    # 'SVD' #
    # 'Multiclass' # 'MLP'

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


# Initialise the ESN reservoir
ESN, configs = initialise_esn(model=True)
with open(models_path + 'Base_model.pkl', 'wb') as f:
    pkl.dump([ESN, configs], f, pkl.HIGHEST_PROTOCOL)


def worker(_video_file):
    head, tail = path_leaf(_video_file)

    if os.path.exists(_video_file) or make_states == 1:
        load_video_and_target(_video_file)


def load_video_and_target(_video_file):
    if dataset == 'SquaresVsCrosses':
        _video = np.load(_video_file)
        _video = _video.reshape(-1, frame_objective)
        _states = ESN.computeState([_video])
        head, tail = path_leaf(_video_file)
        np.save(states_path + tail[:-4], _states, allow_pickle=False)
    if dataset == 'Mnist':
        _video = np.load(_video_file)
        _states = ESN.computeState([_video])
        head, tail = path_leaf(_video_file)
        np.save(states_path + tail[:-4], _states, allow_pickle=False)
    else:
        if os.path.exists(_video_file) and _video_file[-4:] == '.avi':
            video_arr = []
            cap = cv2.VideoCapture(_video_file)
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if frame is None:
                    break
                else:
                    frame_count += 1
                if rgb:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if rgb_frame.shape[0] != frame_height and rgb_frame.shape[1]\
                            != frame_width:
                        rgb_frame = cv2.resize(rgb_frame, (frame_height, frame_width))
                    red_frame = rgb_frame[:, :, 0].reshape(-1, 1)
                    greem_frame = rgb_frame[:, :, 1].reshape(-1, 1)
                    blue_frame = rgb_frame[:, :, 2].reshape(-1, 1)

                    rgb_flat = np.concatenate([red_frame, greem_frame, blue_frame])
                    scaler = MaxAbsScaler()
                    scaler.fit(rgb_flat)

                    video_arr.append(rgb_flat)

                else:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if gray_frame.shape[0] != frame_height or gray_frame.shape[
                        1] != frame_width:
                        gray_frame = cv2.resize(gray_frame, (frame_height, frame_width))
                    gray_frame = gray_frame.astype('float32')/255

                    gray_flat = gray_frame.reshape(-1, 1)
                    # scaler = MaxAbsScaler()
                    # scaler.fit(gray_flat)

                    video_arr.append(gray_flat)
            new_video_arr = video_arr

            _video = np.concatenate(new_video_arr, axis=1)
            _states = ESN.computeState([_video])
            head, tail = path_leaf(_video_file)
            np.save(states_path + tail[:-4], _states, allow_pickle=False)


def get_balanced_dataset(_files, _clase, length=300, ratio=1):
    train_idx = []
    test_idx = []
    for cont, _file in enumerate(_files):
        if _file.find(_clase) != -1:
            train_idx.append(cont)
        else:
            test_idx.append(cont)

    if len(train_idx) < length:
        length = len(train_idx)

    train_files_idx = np.random.choice(train_idx, length, replace=False)

    if len(test_idx) < ratio * length:
        test_files_idx = np.random.choice(test_idx, ratio * length, replace=True)
    else:
        test_files_idx = np.random.choice(test_idx, ratio * length, replace=False)

    rest_of_files = []
    for i in range(len(_files)):
        if i not in test_files_idx:
            rest_of_files.append(_files[i])

    _new_train_files = []
    for idx in train_files_idx:
        _new_train_files.append(_files[idx])
    for idx in test_files_idx:
        _new_train_files.append(_files[idx])

    random.shuffle(_new_train_files)

    balance = {1: len(train_files_idx) / (len(train_files_idx) + len(test_files_idx)),
               0: len(test_files_idx) / (len(train_files_idx) + len(test_files_idx))}

    return _new_train_files, balance, rest_of_files


def get_train_test_files(path, test_g=1, val_g=-1, size=-1, allto=True, overfit=False):
    if overfit:
        _files = []
        with open(path) as f:
            file_reader = csv.reader(f, delimiter='\n')
            for row in file_reader:
                _files.append(row[0])
        _train_files = _files
        _test_files = np.random.choice(_files, 500)
    else:
        if not allto:
            print('Real train/test split')
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
                    _train_files = np.random.choice(_train_files, int(len(
                        _train_files) * (size/100)), replace=False)
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
            _train_files, _test_files = train_test_split(_files,
                                                         test_size=0.1,
                                                         shuffle=True)
            print('Files original: ' + str(len(_train_files)))
            # if size != -1:
            #     _train_files = np.random.choice(_train_files, int(len(
            #         _train_files) * (size/100)), replace=False)
            #     print('Files reducido: ' + str(len(_train_files)))

            # # Unbalance the dataset
            # print('Dataset unbalancing')
            # good_indices = []
            # bad_indices = []
            # for i in range(len(_train_files)):
            #     _file = _train_files[i]
            #     if _file.find('_Eight_') != -1:
            #         good_indices.append(i)
            #     else:
            #         bad_indices.append(i)
            # print('Balanced files')
            # print('Eighs: ' + str(len(good_indices)))
            # print('Rest: ' + str(len(bad_indices)))
            # good_indices = np.random.choice(good_indices, int(len(
            #     good_indices) * 0.6), replace=False)
            # bad_indices = np.random.choice(bad_indices, int(len(
            #     bad_indices) * 0.1), replace=False)
            # indices = list(bad_indices) + list(good_indices)
            # _final_train_files = []
            # for i in np.random.choice(indices, len(indices), replace=False):
            #     _final_train_files.append(_train_files[i])
            # _train_files = _final_train_files
            #
            # print('Unbalanced files')
            # print('Eighs: ' + str(len(good_indices)))
            # print('Rest: ' + str(len(bad_indices)))

    return _train_files, _test_files, None


def load_states(_files):
    _states = []
    # found = True
    for cont, _file in enumerate(_files):
        head, tail = path_leaf(_file)
        _state = np.load(states_path + '/' + tail[:-4] + '.npy')
        _state = np.squeeze(_state)
        sampled_idx = np.linspace(0, _state.shape[1] - 1, frame_objective,
                                  dtype=np.int64)
        sampled_states = []
        for frame_sampled in sampled_idx:
            sampled_states.append(_state[:, frame_sampled].reshape(-1, 1))
        _state = np.concatenate(sampled_states, axis=1)
        if _state.shape[0] != 2000 or _state.shape[1] != frame_objective:
            padded_state = np.zeros((_state.shape[0], frame_objective),
                                    dtype=np.float32)
            padded_state[-_state.shape[0]:, -_state.shape[1]:] = _state
            _state = padded_state
        _states.append(_state)
    return _states


def path_leaf(_path):
    head, tail = ntpath.split(_path)
    return head, tail


def get_targets_from_files(_files):
    _targets = []
    for _file in _files:
        past_class = ''
        for cont, _clase in enumerate(class_names):
            if _file.find('_' + _clase + '_') != -1 \
                    and len(_clase) > len(past_class):
                past_class = _clase
        if past_class != '':
            _targets.append(target_dic[past_class])
    return _targets


if __name__ == '__main__':
    t1 = time.time()

    print(' Training ' + dataset)
    print(' Frames set to: ' + str(frame_objective))
    if dataset_size == -1:
        print(' Dataset size set to 100%')
    else:
        print(' Dataset size set to ' + str(dataset_size) + '%')
    print(' Amount of workers selected: ' + str(n_workers) + '\n\n')

    # Create states directory if it does not exist yet
    if not os.path.exists(states_path):
        os.mkdir(states_path)

    # Generate the class dictionary form the folders
    if dataset == 'Mnist':
        class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six',
                       'Seven', 'Eight', 'Nine']
        target_dic = {'Zero': 0, 'One': 1, 'Two': 2, 'Three': 3, 'Four': 4,
                      'Five': 5, 'Six': 6, 'Seven': 7, 'Eight': 8, 'Nine': 9}
    else:
        target_dic = {}
        for i, name in enumerate(os.listdir(video_path)):
            target_dic[name] = i
        class_names = list(target_dic.keys())

    print(target_dic)

    # We load the whole dataset and calculate its states
    # Initialise the ESN reservoir
    # We check if the base model is already generated and load it if it is so
    # THIS LINES ARE PUT IN THE START OF THE CODE TO ALLOW THE PARALLEL THREADS
    # TO ACCESS IT

    # We load every filename into a list
    files = []
    with open(files_path) as f:
        file_reader = csv.reader(f, delimiter='\n')
        for row in file_reader:
            files.append(row[0])

    # Load all the videos to generate the codeX them into state form save at
    # state_path
    if make_states != -1 or make_model != -1:
        print('State codification ...')
        pool = multiprocessing.Pool(processes=n_workers)
        for _ in tqdm.tqdm(pool.imap_unordered(worker, files), total=len(files)):
            pass

    print('State codification finished ...')
    t2 = time.time()
    print('Time elapsed: ' + str(np.round((t2 - t1)/60)) + ' min.')

    # Generate the train test files for a test set of the group indicate from the arguments
    global_train_files, global_test_files, global_val_files = \
        get_train_test_files(files_path, test_g=1, size=dataset_size,
                             allto=True)

    # Get the states for the test files so we dont have to load them for each model
    global_train_states = load_states(global_train_files)
    global_test_states = load_states(global_test_files)

    # Get the targets for the test files to create the predictions dataset
    global_test_targets = get_targets_from_files(global_test_files)
    global_train_targets = get_targets_from_files(global_train_files)

    if global_val_files is not None:
        global_val_states = load_states(global_val_files)
        global_val_targets = get_targets_from_files(global_val_files)
        global_val_targets_cat = to_categorical(np.array(global_val_targets), np.max(global_val_targets) + 1)

    # Create the predictions dataset to hold each models probabilities
    df_predictions = pd.DataFrame(columns=['Target'] + ['FinalPrediction'])
    df_predictions['Target'] = global_test_targets

    # Iterate through classes and train each readout
    print('Training multiclass')
    print('     Amount of train videos: ' + str(len(global_train_targets)))
    if global_val_files is not None:
        print('     Amount of val videos: ' + str(len(global_val_targets)))
    print('     Amount of test videos: ' + str(len(global_test_targets)))
    # For each class we load the base model
    with open(models_path + 'Base_model.pkl', 'rb') as f:
        ESN, configs = pkl.load(f)

    ESN.trainReadout(global_train_states, global_train_targets)

    global_train_predictions = ESN.computeOutput(global_train_states).reshape(len(global_train_states), -1)
    global_test_predictions = ESN.computeOutput(global_test_states).reshape(len(global_test_states), -1)

    df_predictions['FinalPrediction'] = global_test_predictions

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

    print(ESN.Nu)

    with open(results_path + dataset + '_fitted_model_A' + str(
            int(global_train_accuracy*100)) + '.pkl',
              'wb') as f:
        pkl.dump(ESN, f, pkl.HIGHEST_PROTOCOL)


