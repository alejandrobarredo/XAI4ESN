from __future__ import print_function
import os
import numpy as np
from tqdm import tqdm
from keras.datasets import mnist

DATASET_PATH = '../datasets/Mnist/'


def serialize_mnist(_train_x, _train_y, _test_x, _test_y):
    new_train_x = []
    new_test_x = []
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six',
                   'Seven', 'Eight', 'Nine']

    for class_name in class_names:
        if not os.path.exists(DATASET_PATH + 'Videos-row-row/' + class_name):
            os.mkdir(DATASET_PATH + 'Videos-row-row/' + class_name)
        if not os.path.exists(DATASET_PATH + 'Videos-row-col/' + class_name):
            os.mkdir(DATASET_PATH + 'Videos-row-col/' + class_name)
        if not os.path.exists(DATASET_PATH + 'Videos-row-full/' + class_name):
            os.mkdir(DATASET_PATH + 'Videos-row-full/' + class_name)

    cont = 0
    for image_idx in tqdm(range(_train_x.shape[0])):
        temp_img = _train_x[image_idx] / 255.0
        temp_target = _train_y[image_idx]

        row_video = temp_img.flatten(order='C').reshape(1, -1)
        col_video = temp_img.flatten(order='F').reshape(1, -1)
        full_video = np.concatenate([row_video, col_video])

        np.save(DATASET_PATH + 'Videos-row-row/' + class_names[
            temp_target] + '/' + str(cont) + '_' + class_names[
                    temp_target] + '_.npy', row_video)
        np.save(DATASET_PATH + 'Videos-row-col/' + class_names[
            temp_target] + '/' + str(cont) + '_' + class_names[
                    temp_target] + '_.npy', col_video)
        np.save(DATASET_PATH + 'Videos-row-full/' + class_names[
            temp_target] + '/' + str(cont) + '_' + class_names[
                    temp_target] + '_.npy', full_video)
        # new_train_x.append(np.concatenate([row_wise, column_wise]))
        cont += 1
    return new_train_x, list(train_y), new_test_x, list(test_y)


if not os.path.exists(DATASET_PATH + 'Videos-row-row/'):
    os.mkdir(DATASET_PATH + 'Videos-row-row/')
if not os.path.exists(DATASET_PATH + 'Videos-row-col/'):
    os.mkdir(DATASET_PATH + 'Videos-row-col/')
if not os.path.exists(DATASET_PATH + 'Videos-row-full/'):
    os.mkdir(DATASET_PATH + 'Videos-row-full/')

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X, train_y, test_X, test_y = serialize_mnist(train_X,
                                                   train_y,
                                                   test_X,
                                                   test_y)