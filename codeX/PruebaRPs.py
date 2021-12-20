import numpy as np
import matplotlib.pyplot as plt
from SystemInitialization import ESN
from scipy import stats
from GenerateTimeSeries import TimeSeries
import pandas as pd
import pickle as pkl
import os
import matplotlib.gridspec as gridspec
from pyts.image import RecurrencePlot
from scipy.signal import savgol_filter
import itertools
from tqdm import tqdm


# Code to generates the different experiments carried out with real data.
#       Data is stored at ./data/
#                               Battery/B0005.pkl
#                               Traffic/all_series.pkl
#       and draws the visualization
#       NOTE: RUNNING THE CODE WILL NOT OUTPUT EXACTLY THE SAME

# FOR FASTER EXECUTION COMMENT THIS
# This sets ups the configuration to match latex formatting of the images
TimeSeries.latexify(15, 10)

# Set the parameters to choose the experiment
length = 20000
memory = 300

# grid = [['Narma', 'Battery', 'Traffic'],
#         [0.1, 0.5, 0.9],
#         [0.5, 0.01, 0.09],
#         [50, 100, 200],
#         [1, 2, 3]]
#
# grid_search = list(itertools.product(*grid))

#               exp.   , spectral, leaky, units, layers
grid_search = [['Narma', 0.5, 0.5, 50, 1],
               ['Narma', 0.5, 0.5, 50, 2],

               ['Narma', 0.1, 0.5, 50, 2],
               ['Narma', 0.9, 0.5, 50, 2],

               ['Narma', 0.5, 0.9, 50, 2],
               ['Narma', 0.5, 0.1, 50, 2],


               ['Battery', 0.5, 0.5, 50, 1],
               ['Battery', 0.5, 0.5, 50, 2],

               ['Battery', 0.1, 0.5, 50, 2],
               ['Battery', 0.9, 0.5, 50, 2],

               ['Battery', 0.5, 0.9, 50, 2],
               ['Battery', 0.5, 0.1, 50, 2],


               ['Traffic', 0.5, 0.5, 50, 1],
               ['Traffic', 0.5, 0.5, 50, 2],

               ['Traffic', 0.1, 0.5, 50, 2],
               ['Traffic', 0.9, 0.5, 50, 2],

               ['Traffic', 0.5, 0.9, 50, 2],
               ['Traffic', 0.5, 0.1, 50, 2]]

for combination in tqdm(grid_search):
    experiment = combination[0]  # Narma or Battery or Traffic

    # Set the signal parameters for reproducibility of the signal
    signal_parameters = {'gain': 2.0,
                         'duration': 1.0,
                         'omega': 15}

    # If Narma experiment, load Narma signal
    if experiment == 'Narma':
        # Narma
        length = 20000
        memory = 300
        data, _ = TimeSeries.sin_data_gen(1, length + memory, memory,
                                          signal_parameters=signal_parameters,
                                          noise=False,
                                          norm=True,
                                          feature='variable-sinusoid')

        data, target = TimeSeries.narma_data_gen(length, memory, order='10',
                                                 sin=np.squeeze(data * 0.5))

        data = data[0, 0, memory:-(2*memory)].reshape(1, 1, -1)
        target = target[0, 0, memory:-(2 * memory)].reshape(1, 1, -1)
        # Parameters
        n_inputs = 1
        units = 200
        layers = 1
        spectral = 0.9
        leaky = 0.01
        i_scale = 1
        t_coef = 0.10
        spars = 0

    # If Battery experiment, load Battery data
    elif experiment == 'Battery':
        # Battery
        length = 30000
        with open('../datasets/Battery/B0005.pkl', 'rb') as f:
            battery = pkl.load(f)

        data = battery.Current.values[:length].reshape(1, 1, -1)
        target = battery.Temperature.values[:length].reshape(1, 1, -1)

        data = TimeSeries.min_max_norm(data)
        target = TimeSeries.min_max_norm(target)

        # Parameters
        n_inputs = 1
        units = 200
        layers = 5
        spectral = 0.9
        leaky = 0.009
        i_scale = 1
        t_coef = 0.10
        spars = 0

    # If Traffic experiment, load Traffic data
    elif experiment == 'Traffic':
        # Traffic
        with open('../datasets/Traffic/all_series.pkl', 'rb') as f:
            traffic = pkl.load(f)

        data = traffic['44008'].values[:length].reshape(1, 1, -1)
        data = savgol_filter(data, 21, 2).reshape(1, 1, -1)
        target = np.roll(traffic['44008'].values[:length], -memory).reshape(1,
                                                                            1, -1)
        target = savgol_filter(target, 21, 2).reshape(1, 1, -1)

        weekend_days = [i.weekday() > 5 for i in traffic['44008'][:length].index]
        weekend_days = [i for i, x in enumerate(weekend_days) if x]

        target_weekend_days = [i.weekday() > 5 for i in traffic['44008'][
                                                        :length].index]
        target_weekend_days = [i + memory for i in target_weekend_days]

        data = TimeSeries.min_max_norm(data)
        target = TimeSeries.min_max_norm(target)

        # Parameters
        n_inputs = 1
        units = 200
        layers = 2
        spectral = 0.9
        leaky = 0.01
        i_scale = 1
        t_coef = 0.10
        spars = 0

    spectral = combination[1]
    leaky = combination[2]
    units = combination[3]
    layers = combination[4]

    # Set the saving directory and save name
    save_dir = '../results/FigurasPaper/RealExperiments/' + experiment + '/'

    # Initialise the ESN and train it
    ESNx = ESN(data, target, length, memory, n_inputs, units,
               layers, spectral, leaky, i_scale, t_coef, spars)

    # Compute the states and output for the test set
    test_state = ESNx.deepESN.computeState(ESNx.test_x, ESNx.IPconf.DeepIP)[0]
    prediction = ESNx.deepESN.computeOutput(test_state)[0]

    # Clean the arrays to calculate the baseline error and target error
    input_sig = np.squeeze(ESNx.train_x)
    input_temp = np.squeeze(ESNx.train_y)
    test_x = np.squeeze(ESNx.test_x)
    y = np.squeeze(ESNx.test_y[0, 0, ESNx.transient:])
    pred = prediction[ESNx.transient:]
    baseline_pred = np.mean(data)
    baseline_error = np.mean(np.sum(np.power(y - baseline_pred, 2)) / y.shape[0])
    error = np.mean(np.sum(np.power(y - pred, 2)) / y.shape[0])

    # Print Baseline and Real error for comparison
    # print('Combination:')
    # print('     - Experiment: ' + str(experiment))
    # print('     - Spectral: ' + str(spectral))
    # print('     - Leaky: ' + str(leaky))
    # print('     - Units: ' + str(units))
    # print('     - Layers: ' + str(layers))
    # print('Baseline error: ' + str(baseline_error))
    # print('Error: ' + str(error))

    save_name = experiment + '_U' + str(units) \
                + '_L' + str(layers) + '_Sp' + str(spectral) \
                + '_Leaky' + str(leaky) + '_Is' + str(i_scale) \
                + '_Err' + str(error)

    # print('File name: ' + save_name)

    new_f = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(nrows=7, ncols=6)
    ax = [new_f.add_subplot(gs[0, 0:6])]
    for i in range(layers):
        ax.append(new_f.add_subplot(
            gs[1:7, i * int(6 / layers):i * int(6 / layers) + int(6 /
                                                                  layers)]))

    train_y = np.squeeze(ESNx.train_y)
    train_x = np.squeeze(ESNx.train_x)
    test_y = np.squeeze(ESNx.test_y)
    transient = ESNx.transient

    ax[0].scatter(range(train_y.shape[0] + transient,
                        train_y.shape[0] + transient + pred.shape[0]),
                  pred, marker='o', s=5, color='red', label='Prediction')
    ax[0].scatter(range(train_y.shape[0] + transient,
                        train_y.shape[0] + transient + pred.shape[0]),
                  test_y[transient:], marker='o', s=5, color='green',
                  label='Target')

    # Plot Recurrent Plots
    test_state = test_state[:, ESNx.transient:]

    for i in range(layers):
        rp = RecurrencePlot(threshold='point', percentage=30)
        rpx = RecurrencePlot(percentage=20)
        lay_test = test_state[i * units:(i * units) + units, :]
        test_rp = np.mean(lay_test, axis=0)
        X_rp = rp.fit_transform(test_rp.reshape(1, -1))
        X_rpx = rpx.fit_transform(test_rp.reshape(1, -1))
        st = stats.describe(X_rpx[0].reshape(-1))
        ax[i + 1].imshow(X_rp[0], cmap='binary', origin='lower',
                         extent=[train_x.shape[0] + transient,
                                 train_x.shape[0] + transient +
                                 test_x.shape[0],
                                 train_x.shape[0] + transient,
                                 train_x.shape[0] + transient + test_x.shape[
                                     0]],
                         aspect='equal')
        ax[i + 1].set_rasterized(True)
        ax[i + 1].set_yticklabels([])
        ax[i + 1].set_xticklabels([])

    # signal_plot.legend(loc=2)
    ax[0].set_rasterized(True)
    ax[0].set_yticklabels([])
    ax[0].set_xticklabels([])

    plt.tight_layout()
    plt.pause(0.1)

    # Save the plot
    TimeSeries.saveimage(name=save_name,
                         folder=save_dir)
    # print('Saved to:' + save_dir + save_name + '.pdf')
    plt.close(new_f)