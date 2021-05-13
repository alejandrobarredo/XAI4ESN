'''
    This code generates a simplified video dataset for action recognition video
    Each video will be 60 frames of 28x28 grayscale with representation of 10
    basic movements.
        - Bouncing - 0
        - Vertical Upward - 1
        - Vertical Downward - 2
        - Horizontal Rightward - 3
        - Horizontal Leftward - 4
        - Diagonal Upward - 5
        - Diagonal Downward - 6
        - Circling - 7
        - Crossing - 8
        - Random movement - 9
    Each class will contain 100 different videos
'''

import cv2
import os
import time


import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from numpy.random import randint as r_int
from time import sleep

from math import radians as rad


def movement_function(_clase, _pos, _speed, _dire, _center=None, _radius=None,
                      _t=None, _square_dir=0):
    x = pos[0]
    y = pos[1]
    x_dire = _dire[0]
    y_dire = _dire[1]
    x_speed = _speed[0]
    y_speed = _speed[1]

    if _clase == 0:  # Centered Circling
        x = int(np.cos(t * np.pi / 180) * _radius +
                _center[0])
        y = int(np.sin(t * np.pi / 180) * _radius +
                _center[1])
        _t += 1

    if _clase == 1:  # Centered Squaring
        _t = 1
        if x_dire == 1 and y_dire == 1:
            y += _t
            if y >= center[0] + _radius:
                y_dire = -1

        if x_dire == 1 and y_dire == -1:
            x -= t
            if x <= center[1] - _radius:
                x_dire = -1

        if x_dire == -1 and y_dire == -1:
            y -= t
            if y <= center[1] - _radius:
                y_dire = 1

        if x_dire == -1 and y_dire == 1:
            x += t
            if x >= center[0] + _radius:
                x_dire = 1

    return [x, y], [x_dire, y_dire], _t


n_videos = 600
n_frames = 60
fps = 30

show = False
if show:
    # fig, ax = plt.subplots(1, 1)
    cv2.namedWindow("show", cv2.WINDOW_NORMAL)

VIDEO_PATH = '../datasets/TwoClassVideos/Videos-row/'
class_names = ['CenteredCircling', 'CenteredSquaring']

f_dim = [28, 28]
s_dim = [2, 2]  # subject dimension
empty_frame = np.zeros(f_dim)


for clase in tqdm(list(range(0, 2))):

    for idx_video in range(n_videos):
        video_name = VIDEO_PATH + class_names[clase] + '/' + str(idx_video) \
                     + '_' + class_names[clase] + '_.npy'
        video = []
        if not os.path.exists(VIDEO_PATH + class_names[clase]):
            os.mkdir(VIDEO_PATH + class_names[clase])

        radius = 10
        center = [14, 14]
        pos_x = center[0] + radius
        pos_y = center[1]
        pos = [pos_x, pos_y]
        speed = list(r_int(1, 6, 2))
        dire = [1, 1]
        t = 0

        pos, dire, t = movement_function(clase, pos, speed, dire,
                                         center, radius, t)
        new_frame = np.zeros(f_dim)
        new_frame[pos[0]:pos[0]+s_dim[0], pos[1]:pos[1]+s_dim[1]] = 1

        if show:
            # ax.imshow(new_frame)
            # plt.pause(1/fps)
            imS = cv2.resize(new_frame, (1920, 1920))
            cv2.imshow('show', imS)
            cv2.waitKey(1)

        for idx_frame in range(n_frames):
            pos, dire, t = movement_function(clase, pos, speed, dire,
                                             center, radius, t)
            new_frame = np.zeros(f_dim)
            new_frame[pos[0]:pos[0]+s_dim[0], pos[1]:pos[1]+s_dim[1]] = 1

            if show:
                # ax.imshow(new_frame)
                # plt.pause(0.01)
                imS = cv2.resize(new_frame, (1920, 1920))
                cv2.imshow('show', imS)
                cv2.waitKey(1)

            video.append(new_frame[..., np.newaxis])

        video = np.concatenate(video, axis=2)
        np.save(video_name, video, allow_pickle=False)

