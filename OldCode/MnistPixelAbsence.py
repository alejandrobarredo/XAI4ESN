from __future__ import print_function
from DeepESN import DeepESN as DeepESNorig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import multiprocessing
import random
import csv
import os
import cv2
import copy as cp
import argparse
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from runstats import Statistics, Regression
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import argparse

parser = argparse.ArgumentParser(description='Select the dataset size')
parser.add_argument('-objective_target', default=-1, type=int,
                    help='Introduce the objective target [0-9] that you want to aim at')
parser.add_argument('-example_position', default=-1, type=int,
                    help='If you already know the position of the example you want to test, introduce it')
parser.add_argument('-boostrap', default=1, type=int,
                    help='If you already know the position of the example you want to test, introduce it')
parser.add_argument('-absence', default=1, type=int,
                    help='Choose if you want to calculate absence or activation')                  
parser.add_argument('-check_saved', default=0, type=int,
                    help='Check the file saved and finishes if there is already one saved.')


DATASET_PATH = '../datasets/MNIST/'

args = parser.parse_args()
objective_target = args.objective_target
example_position = args.example_position
boostrap = args.boostrap == 1
absence = args.absence == 1
check_saved = args.check_saved == 1

print('*************************************************')
print('Input parameters:')
print('     Objective target: ' + str(objective_target))
print('     Example position: ' + str(example_position))
print('     Boostrap: ' + str(boostrap))
print('     Absence: ' + str(absence))
print('     Check saved: ' + str(check_saved))
print('*************************************************')


def get_square_area(_coordintates, image_width=28, side=28):
    x, y = _coordintates
    _pixels = []
    for i_h in range(side):
        for i_w in range(side):
            _pixels.append(((i_h + x) * image_width) + (i_w + y))
    return _pixels


def get_list_of_coordinates(image_height=28, image_width=28, side=1):
    _coordinates = []
    for i_h in range(0, image_height, side):
        for i_w in range(0, image_width, side):
            _coordinates.append([i_h, i_w])
    return _coordinates


def get_correct_prediction(_objective_target=-1, _example_position=-1):
    if _example_position == -1:
        if _objective_target == -1:
            target_idxs = list(range(len(train_X)))
        else:
            target_idxs = []
            for idx in range(len(train_y)):
                if train_y[idx] == _objective_target:
                    target_idxs.append(idx)
        for i in tqdm(target_idxs):
            img = [train_X[i].reshape(-1, 1)]
            target = train_y[i]

            orig_states = ESN.computeState(img)
            orig_probs = np.squeeze(ESN.computeOutput(orig_states, probs=True))
            if np.argmax(np.squeeze(orig_probs)) == target and target == _objective_target:
                print('Found correct prediction: ' + str(i))
                print('     Target: ' + str(target))
                return img, target, orig_probs, i
    else:
        img = [train_X[_example_position].reshape(-1, 1)]
        target = train_y[_example_position]

        orig_states = ESN.computeState(img)
        orig_probs = np.squeeze(ESN.computeOutput(orig_states, probs=True))
        if np.argmax(np.squeeze(orig_probs)) == target and target == objective_target:
            print('Found correct prediction: ' + str(i))
            print('     Target: ' + str(target))
            return img, target, orig_probs, i

    print('No correct prediction')


if __name__ == '__main__':
    print('Code starts')
    if not os.path.exists(DATASET_PATH):
        from keras.datasets import mnist
        os.mkdir(DATASET_PATH)
        data = mnist.load_data()
        with open(DATASET_PATH + 'mnist.pkl', 'wb') as f:
            pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)
        os.system("./ESN-mnist.py")

    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH + 'mnist.pkl', 'rb') as f:
            (train_X, train_y), (test_X, test_y) = pkl.load(f)
        with open(DATASET_PATH + 'mnist_fitted_model.pkl', 'rb') as f:
            ESN = pkl.load(f)
    
    print('Data loaded ...')
    print('Model loaded ...')
    print('Looking for a correct example for target: ' + str(objective_target))
    img, target, orig_probs, correct_example = get_correct_prediction(_objective_target=objective_target, _example_position=example_position)

    fig = plt.figure(figsize=(15, 10))
    if boostrap:
        side_widths = [1]  # [1, 2, 4, 7]
    else:
        side_widths = [1, 2, 4, 7]

    for cont, side_width in enumerate(side_widths):
        gs = fig.add_gridspec(ncols=12, nrows=len(side_widths))
        ax_orig = fig.add_subplot(gs[cont, 0])
        ax_absence = fig.add_subplot(gs[cont, 1:10])
        ax_colorbar = fig.add_subplot(gs[cont, 10:11])

        ax_orig.imshow(img[0].reshape(28, 28), cmap='binary')
        ax_orig.set_title('Target: ' + str(target))
        ax_orig.axis('off')

        absence_effect = []
        stats_dic = {}
        for i in range(10):
            zero_mat = np.zeros_like(img[0], dtype=np.float)
            zero_mat[zero_mat == 0] = np.nan
            absence_effect.append(zero_mat)
            if boostrap:
                for ii in range(10):
                    for pixel in range(784):
                        stats_dic[str(ii) + '_' + str(pixel)] = Statistics()

        coordinates = get_list_of_coordinates(image_height=28, image_width=28,
                                              side=side_width)
        if os.path.exists(DATASET_PATH + 'Explainability/pixel_absence_' + str(correct_example) + '_' + str(target) + '_' + str(absence) + '.pkl') and check_saved:
            with open(DATASET_PATH + 'Explainability/pixel_absence_' + str(correct_example) + '_' + str(target) + '_' + str(absence) + '.pkl',
                      'rb') as f:
                absence_effect = pkl.load(f)
        else:
            if boostrap:
                boostrap_attempt = 0
                while True:
                    # for boostrap_attempt in tqdm(range(1000)):
                    boostrap_attempt += 1
                    sampled_idxs = np.random.choice(range(len(coordinates)),
                                                    size=20)
                    sampled_coordinates = [coordinates[c] for c in sampled_idxs]
                    temp_img = cp.deepcopy(img)
                    boostrapped_pixels = []
                    for coordinate in sampled_coordinates:
                        pixels = get_square_area(coordinate, image_width=28,
                                                 side=side_width)

                        for pixel in pixels:
                            if absence:
                                temp_img[0][pixel] = 0.0
                            else:
                                temp_img[0][pixel] = 1.0
                        boostrapped_pixels = boostrapped_pixels + pixels
                    temp_states = ESN.computeState(temp_img)
                    temp_probs = ESN.computeOutput(temp_states, probs=True)
                    diff = np.squeeze(temp_probs - orig_probs)
                    # print(diff)
                    for ii in range(10):
                        for pixel in boostrapped_pixels:
                            s = stats_dic[str(ii) + '_' + str(pixel)]
                            if absence_effect[ii][pixel] == np.nan and diff[
                                ii] != 0:
                                absence_effect[ii][pixel] = diff[ii]
                                s.push(diff[ii])
                            else:
                                s.push(diff[ii])
                                absence_effect[ii][pixel] = s.mean()
                    if boostrap_attempt % 100 == 0:
                        with open(
                                DATASET_PATH + 'Explainability/pixel_absence_' + str(correct_example) + '_' + str(target) + '_' + str(absence) + '.pkl',
                                'wb') as f:
                            print('Saved iteration ' + str(boostrap_attempt))
                            pkl.dump(absence_effect, f, pkl.HIGHEST_PROTOCOL)
            else:
                for coordinate in coordinates:
                    pixels = get_square_area(coordinate, image_width=28,
                                                 side=side_width)
                    temp_img = cp.deepcopy(img)
                    for pixel in pixels:
                        if absence:
                            temp_img[0][pixel] = 0.0
                        else:
                            temp_img[0][pixel] = 1.0
                    temp_states = ESN.computeState(temp_img)
                    temp_probs = ESN.computeOutput(temp_states, probs=True)
                    diff = np.squeeze(temp_probs - orig_probs)
                    # print(diff)
                    for ii in range(10):
                        for pixel in pixels:
                            absence_effect[ii][pixel] = diff[ii]

            with open(DATASET_PATH + 'Explainability/pixel_absence.pkl',
                      'wb') as f:
                pkl.dump(absence_effect, f, pkl.HIGHEST_PROTOCOL)

        full_absence = []
        for i in range(10):
            full_absence.append(absence_effect[i].reshape(28, 28))
        full_absence = np.concatenate(full_absence, axis=1)
        full_absence[full_absence == 0] = np.nan
        im = ax_absence.imshow(full_absence, cmap='coolwarm')
        ax_absence.axis('off')
        ax_absence.set_title(' '.join(['Target ' + str(t) + '(p:' + str(p) + ') 'for t, p in zip(range(0, 10), np.round(orig_probs, 3))]), fontsize=8)
        plt.colorbar(im, cax=ax_colorbar, )
        plt.tight_layout()
        print('Finished')

    plt.show()
    print()