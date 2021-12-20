import matplotlib.pyplot as plt
import numpy as np

dataset = 'SquaresVsCrosses'

# Set the directories for the dataset selected
dataset_path = '../datasets/' + dataset + '/'
video_path = dataset_path + 'Videos/'
files_path = dataset_path + 'Splits/' + dataset + '_files.txt'
models_path = dataset_path + 'Models/'
states_path = dataset_path + 'States/'
results_path = dataset_path + 'Results/'

video1 = np.load(results_path + '590_Crosses_' + 'video.npy')
video1 = np.load(video_path + 'Crosses/414_Crosses_.npy')
absence_effect = np.load(results_path + '590_Crosses_' + 'absence_effect.npy')

class_names = ['Crosses', 'Squares']
target_dic = {'Crosses': 0, 'Squares': 1}

target = 0

fig = plt.figure(figsize=(15, 5))
gs = fig.add_gridspec(ncols=12, nrows=1)
ax_orig = fig.add_subplot(gs[0, 0])
ax_absence = fig.add_subplot(gs[0, 1:10])
ax_colorbar = fig.add_subplot(gs[0, 10:11])


ax_orig.set_title('Target: ' + str(target))
ax_orig.axis('off')

full_absence = []
for i in range(len(class_names)):
    full_absence.append(absence_effect[i])

full_absence = np.concatenate(full_absence, axis=2)


for frame in range(60):
    ax_orig.imshow(video1[frame, :, :], cmap='seismic')
    im = ax_absence.imshow(full_absence[frame, :, :], cmap='seismic',
                           vmin=-0.1, vmax=0.1)
    plt.pause(0.1)

ax_absence.axis('off')
ax_absence.set_title(' '.join(
    ['Target ' + str(t) + '(p:' + str(p) + ')     ' for t, p in
     zip(range(0, 10), np.round(probs, 3))]), fontsize=8)
plt.colorbar(im, cax=ax_colorbar, )
plt.tight_layout()
plt.savefig(results_path + dataset + '_' + video_name + 'PixelAbsencePlot.pdf')
print('Finished')
plt.show()


print('Finished')