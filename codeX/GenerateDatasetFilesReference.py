import os
import argparse


parser = argparse.ArgumentParser(description='Select the dataset size')
parser.add_argument('-dataset', default='SquaresVsCrosses',
                    help='choose the dataset')

args = parser.parse_args()
dataset = str(args.dataset)

dataset_path = '../datasets/' + dataset + '/'
files_path = dataset_path + 'Splits/'
video_path = dataset_path + 'Videos/'
states_path = dataset_path + 'States/'

clases = os.listdir(video_path)

f = open(files_path + dataset + '_files.txt', 'w')
for clase in clases:
    video_files = os.listdir(video_path + clase)
    for file in video_files:
        if file[-4:] == '.npy':
            f.write(video_path + clase + '/' + file + '\n')

f = open(files_path + dataset + '_states_files.txt', 'w')
video_files = os.listdir(states_path)
for file in video_files:
    f.write(states_path + file + '\n')
