from __future__ import print_function
from DeepESN import DeepESN as DeepESNorig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tqdm
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
import matplotlib as mpl
import matplotlib.gridspec as gridspec


def get_square_area(_coordintates,image_width=320, side=25):
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


if __name__ == '__main__':
    with open('mnist.pkl', 'rb') as f:
        (train_X, train_y), (test_X, test_y) = pkl.load(f)
    with open('mnist_fitted_model.pkl', 'rb') as f:
        ESN = pkl.load(f)
    # 0 = 5
    # 17 = 8
    for i in range(6000):
        img = [train_X[i, :, :].reshape(-1, 1)]
        # img = [np.ones((784, 1))]
        target = train_y[i]

        orig_states = ESN.computeState(img)
        orig_probs = np.squeeze(ESN.computeOutput(orig_states, probs=True))
        if np.argmax(np.squeeze(orig_probs)) == target \
                and target == 8 \
                and i == 300:
            #  print(i)
            break
            # break
    fig = plt.figure(figsize=(15, 10))
    side_widths = [1, 2, 4, 7]
    for cont, side_width in enumerate(side_widths):
        gs = fig.add_gridspec(ncols=12, nrows=len(side_widths))
        ax_orig = fig.add_subplot(gs[cont, 0])
        ax_absence = fig.add_subplot(gs[cont, 1:10])
        ax_colorbar = fig.add_subplot(gs[cont, 10:11])

        ax_orig.imshow(img[0].reshape(28, 28), cmap='binary')
        ax_orig.set_title('Target: ' + str(target))
        ax_orig.axis('off')
        plt.pause(0.2)

        absence_effect = []
        for i in range(10):
            absence_effect.append(np.zeros_like(img[0], dtype=np.float64))
        coordinates = get_list_of_coordinates(image_height=28, image_width=28,
                                              side=side_width)
        for coordinate in coordinates:
            pixels = get_square_area(coordinate, image_width=28, side=side_width)
            temp_img = cp.deepcopy(img)
            for pixel in pixels:
                temp_img[0][pixel] = 0.0
            temp_states = ESN.computeState(temp_img)
            temp_probs = ESN.computeOutput(temp_states, probs=True)
            diff = np.squeeze(temp_probs - orig_probs)
            # print(diff)
            for ii in range(10):
                for pixel in pixels:
                    absence_effect[ii][pixel] = diff[ii]

        full_absence = []
        for i in range(10):
            full_absence.append(absence_effect[i].reshape(28, 28))
        full_absence = np.concatenate(full_absence, axis=1)
        # full_absence = (full_absence - _min) / (_max - _min)
        im = ax_absence.imshow(full_absence, cmap='coolwarm')
        ax_absence.axis('off')
        ax_absence.set_title(' '.join(['Target ' + str(t) + '(p:' + str(p) + ') 'for t, p in zip(range(0, 10), np.round(orig_probs, 3))]), fontsize=8)
        plt.colorbar(im, cax=ax_colorbar, )
        plt.tight_layout()
        print('Finished')
    plt.show()
    print()