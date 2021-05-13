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
from math import radians as rad


def movement_function(_clase, _pos, _speed, _dire, _center=None, _radius=None,
                      _t=None, _square_dir=0):
    x = pos[0]
    y = pos[1]
    x_dire = _dire[0]
    y_dire = _dire[1]
    x_speed = _speed[0]
    y_speed = _speed[1]

    if _clase == 0:  # Bouncing
        if x + (s_dim[0] * x_dire) > f_dim[0]-1 or \
                x + (s_dim[0] * x_dire) < -1:
            x_dire *= -1
        if y + (s_dim[1] * y_dire) > f_dim[1]-2 or \
                y + (s_dim[1] * y_dire) < 0:
            y_dire *= -1
        x = x + x_speed * x_dire
        y = y + y_speed * y_dire

    if _clase == 1:  # Vertical Upward
        if x + (s_dim[0] * -1) < -1:
            x = x + f_dim[0]
        x = x - x_speed

    if _clase == 2:  # Vertical Downward
        if x + (s_dim[0] * 1) > f_dim[0]-1:
            x = x - f_dim[0]
        x = x + x_speed

    if _clase == 3:  # Horizontal rightward
        if y + (s_dim[1] * 1) > f_dim[1]-1:
            y = y - f_dim[1]
        y = y + y_speed

    if _clase == 4:  # Horizontal lefttward
        if y + (s_dim[1] * -1) < -1:
            y = y + f_dim[1]
        y = y - y_speed

    if _clase == 5:  # Diagonal upward
        if x + (s_dim[0] * -x_dire) > f_dim[0]-1 or \
                x + (s_dim[0] * -x_dire) < -1:
            x = x + f_dim[0]
        if y + (s_dim[1] * -y_dire) > f_dim[1]-2 or \
                y + (s_dim[1] * -y_dire) < 0:
            y = y + f_dim[1]
        x = x + x_speed * -x_dire
        y = y + y_speed * -y_dire

    if _clase == 6:  # Diagonal downward
        if x + (s_dim[0] * x_dire) > f_dim[0]-1 or \
                x + (s_dim[0] * x_dire) < -1:
            x = x - f_dim[0]
        if y + (s_dim[1] * y_dire) > f_dim[1]-2 or \
                y + (s_dim[1] * y_dire) < 0:
            y = y - f_dim[1]
        x = x + x_speed * x_dire
        y = y + y_speed * y_dire

    if _clase == 7:  # Circling
        x = int(_center[0] + np.cos(_t) * 2*_radius)
        y = int(_center[1] + np.sin(_t) * 2*_radius)
        _t += 1

    if _clase == 8:  # Crossing
        if x + (s_dim[0] * -x_dire) > f_dim[0]-1 or \
                x + (s_dim[0] * -x_dire) < -1:
            x = x + f_dim[0]
            y_dire *= -1
        if y + (s_dim[1] * -y_dire) > f_dim[1]-2 or \
                y + (s_dim[1] * -y_dire) < 0:
            y = y + f_dim[1]
            x_dire *= -1
        x = x + x_speed * -x_dire
        y = y + y_speed * -y_dire

    if _clase == 9:  # Centered Squaring
        if x_dire == 1 and y_dire == 0:
            x = x + x_speed
            y = _radius
            if x == f_dim[0]/2 + _radius:
                x_dire = 0
                y_dire = -1
        if x_dire == 0 and y_dire == 1:
            x = x + x_speed
            y = _radius
            if x == f_dim[0]/2 + _radius:
                x_dire = 0
                y_dire = -1

    if _clase == 10:  # Centered Circling
        x = int(f_dim[0]/2 + np.cos(_t) * 2*10)
        y = int(f_dim[1]/2 + np.sin(_t) * 2*10)
        _t += 1

    return [x, y], [x_dire, y_dire], _t


n_videos = 600
n_frames = 60
fps = 2

show = True
if show:
    cv2.namedWindow("show", cv2.WINDOW_NORMAL)
VIDEO_PATH = '../datasets/SimplifiedVideos/Videos-row/'
class_names = ['Bouncing', 'Vertical_Upward', 'Vertical_Downward',
               'Horizontal_Rightward', 'Horizontal_Leftward',
               'Diagonal_Upward', 'Diagonal_Downward', 'Circling', 'Crossing',
               'Random_Movement']

f_dim = [28, 28]
s_dim = [2, 2]  # subject dimension
empty_frame = np.zeros(f_dim)


# np.random.seed(10)

for clase in tqdm(list(range(10, 11))):

    for idx_video in range(n_videos):
        video_name = VIDEO_PATH + class_names[clase] + '/' + str(idx_video) \
                     + '_' + class_names[clase] + '_.npy'
        video = []
        if not os.path.exists(VIDEO_PATH + class_names[clase]):
            os.mkdir(VIDEO_PATH + class_names[clase])

        radius = list(r_int(2, 5, 1))[0]
        center = list(r_int(0 + radius, 28 - radius, 2))
        pos_x = list(r_int(0, 3, 1))[0]
        pos_y = int(np.sqrt(np.power(radius, 2) - np.power(pos_x, 2)))
        pos = [center[0] + pos_x, center[1] + pos_y]
        delta = [center[0] - pos[0], center[1] - pos_y]
        speed = list(r_int(1, 6, 2))
        t = 0

        dire = [1, 1]
        new_frame = np.zeros(f_dim)
        new_frame[pos[0]:pos[0]+s_dim[0], pos[1]:pos[1]+s_dim[1]] = 1

        if show:
            imS = cv2.resize(new_frame, (1920, 1920))
            cv2.imshow('show', imS)
            cv2.waitKey(1)

        for idx_frame in range(n_frames):
            if clase == 9:
                if idx_frame % 5 == 0:
                    r_clase = list(r_int(0, 8, 1))[0]
                pos, dire, t = movement_function(r_clase, pos, speed, dire,
                                                 center, radius, t)
            else:
                pos, dire, t = movement_function(clase, pos, speed, dire,
                                             center, radius, t)
            new_frame = np.zeros(f_dim)
            new_frame[pos[0]:pos[0]+s_dim[0], pos[1]:pos[1]+s_dim[1]] = 1
            if show:
                imS = cv2.resize(new_frame, (1920, 1920))
                cv2.imshow('show', imS)
                cv2.waitKey(1)

            video.append(new_frame[..., np.newaxis])

        video = np.concatenate(video, axis=2)
        np.save(video_name, video, allow_pickle=False)

