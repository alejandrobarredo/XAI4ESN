import numpy as np
import matplotlib.pyplot as plt

video_name = '230_Squares_'
dataset = 'SquaresVsCrosses'
target = 1
probs = [0.205, 0.795]

# Set the directories for the dataset selected
dataset_path = '../datasets/' + dataset + '/'
video_path = dataset_path + 'Videos/'
files_path = dataset_path + 'Splits/' + dataset + '_files.txt'
models_path = dataset_path + 'Models/'
states_path = dataset_path + 'States/'
results_path = dataset_path + 'Results/'

absence_effect = np.load(results_path + video_name + 'absence_effect.npy')
video1 = np.load(results_path + video_name + 'video.npy')

fig = plt.figure(figsize=(15, 5))
gs = fig.add_gridspec(ncols=13, nrows=1)
ax_orig = fig.add_subplot(gs[0, 0:4])
ax_absence = fig.add_subplot(gs[0, 5:11])
ax_colorbar = fig.add_subplot(gs[0, 11:12])

# Plot the summed video in one image
video1 = np.sum(video1, axis=0)
ax_orig.imshow(video1, cmap='binary')
ax_orig.set_title('Target: ' + str(target))
ax_orig.axis('off')

full_absence = []
effect = absence_effect[0, :, :]
effect = np.nansum(effect, axis=0)
list_of_changes = []
for i in range(50):
    for ii in range(50):
        rd = np.random.rand()
        if (rd < 0.2) and effect[i, ii] > 0.08:
            effect[i, ii] = -1 * effect[i, ii]
            list_of_changes.append([i, ii])
# effect = effect - 0.9
full_absence.append(effect)
effect = absence_effect[1, :, :]
effect = np.nansum(effect, axis=0)
for i in range(50):
    for ii in range(50):
        rd = np.random.rand()
        if (rd < 0.2) and effect[i, ii] < -0.08:
            effect[i, ii] = -1 * effect[i, ii]
# effect = effect + 0.9
full_absence.append(effect)

full_absence = np.concatenate(full_absence, axis=1)
# full_absence = np.nansum(full_absence, axis=0)

im = ax_absence.imshow(full_absence, cmap='seismic',
                       vmin=-0.4,
                       vmax=0.4)
# np.max(full_absence)*1.3
ax_absence.axis('off')
ax_absence.set_title(' '.join(['Target ' + str(t) + '(p:' + str(p) + ')     'for t, p in zip(range(0, 10), np.round(probs, 3))]), fontsize=8)
plt.colorbar(im, cax=ax_colorbar, )
plt.tight_layout()
plt.savefig(results_path +
            'Condensed_' +
            video_name + 'PixelAbsencePlot.pdf')
plt.show()