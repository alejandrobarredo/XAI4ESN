from tqdm import tqdm
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

dataset = 'SquaresVsCrosses'
dataset_path = '../datasets/' + dataset + '/'
video_path = dataset_path + 'Videos/'
files_path = dataset_path + 'Splits/' + dataset + '_files.txt'
models_path = dataset_path + 'Models/'
states_path = dataset_path + 'States/'
results_path = dataset_path + 'Results/'

# TO VISUALIZE VIDEOS SET show TO TRUE
show = False
if show:
    fig = plt.figure()

# TO GENERATE BOTH VIDEO CLASSES SET classes TO 2
# TO GENERATE JUST SQUARES SET classes TO 0
# TO GENERATE JUST CROSSES SET classes TO 1
classes = 2

n_videos = 600
n_frames = 60
fps = 30


# Make squares between 2-6
if classes == 0 or classes == 2:
    for cont_video in tqdm(range(n_videos)):
        border_distance = np.random.randint(1, 7, 1, dtype=int)
        lower_bound = 0 + border_distance
        higher_bound = 28 - border_distance

        blob = [lower_bound, lower_bound]

        rd = np.random.randn()
        if rd <= 0.25:
            blob = [lower_bound, lower_bound]
        elif 0.25 < rd < 0.5:
            blob = [lower_bound, higher_bound]
        elif 0.5 <= rd < 0.75:
            blob = [higher_bound, lower_bound]
        elif rd >= 0.75:
            blob = [higher_bound, higher_bound]

        video = []
        video_frame = np.zeros((28, 28))
        for frame in range(n_frames):
            new_frame = deepcopy(video_frame)
            new_frame[blob[0], blob[1]] = 1.0
            # To visualize the videos uncomment
            if show and frame % 5 == 0:
                plt.imshow(new_frame, cmap='binary')
                plt.pause(0.01)
            video.append(new_frame)
            if blob[0] == lower_bound and lower_bound <= blob[1] < higher_bound:
                blob[0] = lower_bound
                blob[1] = blob[1] + 1
            elif blob[0] == lower_bound and blob[1] == higher_bound:
                blob[0] = blob[0] + 1
                blob[1] = higher_bound
            elif lower_bound <= blob[0] < higher_bound and blob[1] == higher_bound:
                blob[0] = blob[0] + 1
                blob[1] = higher_bound
            elif blob[0] == higher_bound and blob[1] == higher_bound:
                blob[0] = higher_bound
                blob[1] = blob[1] - 1
            elif blob[0] == higher_bound and lower_bound < blob[1] <= higher_bound:
                blob[0] = higher_bound
                blob[1] = blob[1] - 1
            elif blob[0] == higher_bound and blob[1] == lower_bound:
                blob[0] = blob[0] - 1
                blob[1] = lower_bound
            elif lower_bound < blob[0] <= higher_bound and blob[1] == lower_bound:
                blob[0] = blob[0] - 1
                blob[1] = lower_bound

        np.save(video_path + 'Squares/' + str(cont_video) + '_Squares_.npy',
                video, allow_pickle=False)

# Make crosses within 6
if classes == 1 or classes == 2:
    for cont_video in tqdm(range(n_videos)):
        border_distance = np.random.randint(8, 15, 1, dtype=int)
        lower_bound = 0 + border_distance
        higher_bound = 28 - border_distance
        blob = [lower_bound, lower_bound]

        slope = np.random.randint(1, 3, 1, dtype=int)
        normal = 0
        diagonal = 1
        diag_direction = slope[0]
        normal_direction = 1
        rd = np.random.randn()
        if rd <= 0.25:
            blob = [lower_bound, lower_bound]
            diagonal = 1
            normal = 0
        elif 0.25 < rd < 0.5:
            blob = [lower_bound, higher_bound]
            diagonal = 1
            normal = 0
        elif 0.5 <= rd < 0.75:
            blob = [higher_bound, lower_bound]
            diagonal = 0
            normal = 1
        elif rd >= 0.75:
            blob = [higher_bound, higher_bound]
            diagonal = 0
            normal = 1

        video = []
        video_frame = np.zeros((28, 28))
        for frame in range(n_frames):
            new_frame = deepcopy(video_frame)
            new_frame[blob[0], blob[1]] = 1.0
            # To visualize the videos uncomment
            if show and frame % 5 == 0:
                plt.imshow(new_frame, cmap='binary')
                plt.pause(0.01)
            video.append(new_frame)

            if lower_bound < blob[normal] < higher_bound and lower_bound < \
                    blob[diagonal] < higher_bound:
                blob[normal] = blob[normal] + normal_direction
                blob[diagonal] = blob[diagonal] + diag_direction
            if blob[normal] == lower_bound:
                normal_direction = 1
                blob[normal] = blob[normal] + normal_direction
            if blob[normal] == higher_bound:
                normal_direction = -1
                blob[normal] = blob[normal] + normal_direction
            if blob[diagonal] == lower_bound:
                diag_direction = slope
                blob[diagonal] = blob[diagonal] + diag_direction
            if blob[diagonal] == higher_bound:
                diag_direction = -1 * slope
                blob[diagonal] = blob[diagonal] + diag_direction

        np.save(video_path + 'Crosses/' + str(cont_video) + '_Crosses_.npy',
                video, allow_pickle=False)





