import matplotlib.pyplot as plt
import numpy as np
from DeepESN import DeepESN as DeepESNorig
import pickle as pkl
from runstats import Statistics, Regression
from tqdm import tqdm
import os
import cv2
import argparse

# PARSING ARGUMENTS  *********************************************************
# Elegimos el dataset que queremos entrenar
parser = argparse.ArgumentParser(description='Select the dataset size')
parser.add_argument('-dataset', default='', type=str)
parser.add_argument('-target_class', default='', type=str)
parser.add_argument('-video_name', default='', type=str)
parser.add_argument('-model_name', default='', type=str)
parser.add_argument('-frame_size_x', default=0, type=int)
parser.add_argument('-frame_size_y', default=0, type=int)
parser.add_argument('-video_length', default=0, type=int)
parser.add_argument('-min_max', default=1.0, type=float)


args = parser.parse_args()
dataset = args.dataset
target_class = args.target_class
video_name = args.video_name
model_name = args.model_name
frame_size_x = args.frame_size_x
frame_size_y = args.frame_size_y
video_length = args.video_length
min_max = args.min_max


if dataset == '':
    print('No selected dataset: ' + str(dataset))
if target_class == '':
    print('No selected target class: ' + str(target_class))
if video_name == '':
    print('No selected video name: ' + str(video_name))
if model_name == '':
    print('No selected model name: ' + str(model_name))
if frame_size_x == 0 or frame_size_y == 0:
    print('No selected frame size: (' + str(frame_size_x) + 'X' +  str(
        frame_size_y) + ')')
if video_length == 0:
    print('No selected video length: ' + str(video_length))

print('Dataset: ' + dataset)
print('Video: ' + video_name)
print('Frame size: (' + str(frame_size_x) + 'X' +  str(frame_size_y) + ')')
print('Video length: ' + str(video_length))
print('Model: Mnist_fitted_model' + model_name + '.pkl')
print('Ploting vmin-vmax: ' + str(- min_max) + '/' + str(min_max))
print()


def load_video(_video_path):
    _video = np.load(_video_path)
    _video = _video.reshape(-1, video_length)
    _video_1 = _video.reshape(video_length, frame_size_x, frame_size_y)

    return _video, _video_1


def get_targets_from_files(_files):
    _targets = []
    for _file in _files:
        for cont, _clase in enumerate(class_names):
            if _file.find('_' + _clase + '_') != -1:
                _targets.append(target_dic[_clase])
    return _targets


# Set the directories for the dataset selected
dataset_path = '../datasets/' + dataset + '/'
video_path = dataset_path + 'Videos/'
files_path = dataset_path + 'Splits/' + dataset + '_files.txt'
models_path = dataset_path + 'Models/'
states_path = dataset_path + 'States/'
results_path = dataset_path + 'Results/'

frame_objective = video_length

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


with open(results_path + dataset + '_fitted_model' + model_name + '.pkl',
          'rb') as f:
    ESN = pkl.load(f)

video_file = video_path + target_class + '/' + video_name + '.npy'

video, video1 = load_video(video_file)
target = get_targets_from_files([video_file])[0]

states = ESN.computeState([video])[0]
probs = ESN.computeOutput([states], probs=True)
probs = np.squeeze(probs)

# Create the effect matrix equal to the video
absence_effect = []
stats_dic = {}
for i in range(len(class_names)):
    if video.shape[0] == 1:
        zero_mat = np.zeros_like(video, dtype=np.float)
        zero_mat[zero_mat == 0] = np.nan
        absence_effect.append(zero_mat)
        for clase in range(len(class_names)):
                for pixel in range(frame_size_x * frame_size_y):
                    stats_dic[str(clase) + '_' + str(pixel)] = Statistics()
    else:
        zero_mat = np.zeros_like(video1, dtype=np.float)
        zero_mat[zero_mat == 0] = np.nan
        absence_effect.append(zero_mat)
        for clase in range(len(class_names)):
            for frame in range(video_length):
                for pixel_x in range(frame_size_x):
                    for pixel_y in range(frame_size_y):
                        stats_dic[str(clase) + '_' + str(frame) + '_' + str(
                            pixel_x) + '_' + str(pixel_y)] = Statistics()

print('Target class: ' + class_names[target] + '(' + str(target) + ')')
# Iterate over files to populate the absence effect matrix
if not os.path.exists(results_path + video_name + 'absence_effect.npy'):
    # Get the files with the probability difference
    files = os.listdir(results_path + 'Diffs_' + video_name + '/')
    tot_files = len(files)
    try:
        for cont, file in enumerate(files):
            if file[-3:] == 'npy':
                _b = int(file.split('-')[0])  # boostrap index
                _x = int(file.split('-')[1])  # x position
                _y = int(file.split('-')[2])  # y position
                _f = int(file.split('-')[3])  # frame

                matrix = np.load(results_path + 'Diffs_' + video_name + '/' + file)
                # print(file + '  -   ' + str(cont / len(files) * 100))

                pixel_value = video[0, _f]

                if video.shape[0] == 1:
                    for i in range(len(class_names)):
                        s = stats_dic[str(i) + '_' + str(_f)]
                        a = absence_effect[i]
                        if a[0, _f] == np.nan and matrix[i] != 0:
                            a[0, _f] = matrix[i]
                            s.push(matrix[i])
                        else:
                            s.push(matrix[i])
                            a[0, _f] = s.mean()

                        absence_effect[i] = a
                else:
                    for clase in range(len(class_names)):
                        s = stats_dic[str(clase) + '_' + str(_f) + '_' + str(
                            _x) + '_' + str(_y)]
                        a_clase = absence_effect[clase]
                        if a_clase[_f, _x, _y] == np.nan and matrix[i] != 0:
                            a_clase[_f, _x, _y] = matrix[i]
                            s.push(matrix[i])
                        else:
                            s.push(matrix[i])
                            a_clase[_f, _x, _y] = s.mean()
                        absence_effect[clase] = a_clase
            if cont % 100000000 == 0:
                print(str(cont) + ' files processed. (' + str(int((cont /
                                                                     len(
                    files)) * 100)) + '%)')
    except ValueError:
        print(file + '  -   ' + str(cont / len(files) * 100))

    np.save(results_path + video_name + 'absence_effect.npy',
            absence_effect, allow_pickle=False)
else:
    absence_effect = np.load(results_path + video_name + 'absence_effect.npy')

if video.shape[0] == 1:
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(ncols=12, nrows=1)
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_absence = fig.add_subplot(gs[0, 1:10])
    ax_colorbar = fig.add_subplot(gs[0, 10:11])

    ax_orig.imshow(video1, cmap='binary')
    ax_orig.set_title('Target: ' + str(target))
    ax_orig.axis('off')

    full_absence = []
    for i in range(len(class_names)):
        full_absence.append(absence_effect[i].reshape(28, 28))

    full_absence = np.concatenate(full_absence, axis=1)
    im = ax_absence.imshow(full_absence, cmap='seismic',
                           vmin=-min_max, vmax=min_max)
    ax_absence.axis('off')
    ax_absence.set_title(' '.join(['Target ' + str(t) + '(p:' + str(p) + ')     'for t, p in zip(range(0, 10), np.round(probs, 3))]), fontsize=8)
    plt.colorbar(im, cax=ax_colorbar, )
    plt.tight_layout()
    plt.savefig(results_path + dataset + '_' + video_name + 'PixelAbsencePlot.pdf')
    print('Finished')

    plt.show()
else:
    np.save(results_path + video_name + '_video.npy',
            video1, allow_pickle=False)
