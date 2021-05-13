from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from DeepESN import DeepESN as DeepESNorig
from GenerateTimeSeries import TimeSeries
import pickle as pkl
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter


def initialise_esn(inputs=28*28, units=100, layers=4, DeepIP=0, indexes=[1],
                   eta=0.000001, mu=0, Nepochs=10, sigma=0.1, threshold=0.1,
                   regularizations=np.array([0.001, 0.01, 0.1]),
                   trainMethod='Forest', connectivity=1, iss=0.6,
                   lis=0.01, rhos=0.9, model=False):

    # Parameters
    n_inputs = inputs

    IPconf = {}
    IPconf['DeepIP'] = DeepIP
    IPconf['indexes'] = indexes
    IPconf['eta'] = eta
    IPconf['mu'] = mu
    IPconf['Nepochs'] = Nepochs
    IPconf['sigma'] = sigma
    IPconf['threshold'] = threshold

    readout = {}
    readout['regularizations'] = regularizations
    readout['trainMethod'] = trainMethod  # 'Ridge'  # Lasso # 'Inverse'
    # 'SVD' # 'Multiclass' # 'MLP'

    reservoirConf = {}
    reservoirConf['connectivity'] = connectivity

    _configs = {}
    _configs['IPconf'] = IPconf
    _configs['readout'] = readout
    _configs['reservoirConf'] = reservoirConf
    _configs['iss'] = iss
    _configs['lis'] = lis  # 0.2
    _configs['rhos'] = rhos  # 1.3

    if model:
        ESNorig = DeepESNorig(n_inputs, units, layers, _configs)
        return ESNorig, _configs
    else:
        return _configs


def split_to_train_val_test(_data, _target):
    if not _data.shape[0] == 1:
        train_cut = int(_data.shape[2] * 0.8)
        train_x = _data[:train_cut, :, :]
        train_y = _target[:train_cut, :, :]
        test_x = _data[train_cut:, :, :]
        test_y = _target[train_cut:, :, :]
    else:
        train_cut = int(_data.shape[2] * 0.7)
        train_x = _data[0, :, :train_cut].reshape(1, 1, -1)
        train_y = _target[0, :, :train_cut].reshape(1, 1, -1)
        test_x = _data[0, :, train_cut:].reshape(1, 1, -1)
        test_y = _target[0, :, train_cut:].reshape(1, 1, -1)

    return train_x, train_y, test_x, test_y


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
transient = 200
experiment = 'Narma'  # or Battery or Traffic

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
    spectral = 0.5
    leaky = 0.01
    i_scale = 1
    t_coef = 0.10
    spars = 0

# If Battery experiment, load Battery data
elif experiment == 'Battery':
    # Battery
    length = 30000
    with open('./data/Batteries/B0005.pkl', 'rb') as f:
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
    with open('./data/Traffic/all_series.pkl', 'rb') as f:
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

# Set the saving directory and save name
save_dir = './FigurasPaper/RealExperiments/' + experiment + '/'
save_name = experiment + '_exp_u' + str(units) \
            + '_l' + str(layers) + '_sp' + str(spectral) \
            + '_leaky' + str(leaky) + '_is' + str(i_scale) \
            + '_spa' + str(spars)

# We split the data into training and test
train_x, train_y, test_x, test_y = split_to_train_val_test(_data=data,
                                                           _target=target)

# Intialize the echo state network with the configuration of the experiment
ESN, configs = initialise_esn(inputs=1, units=units, layers=layers,
                              trainMethod='Inverse',
                              lis=leaky, rhos=spectral, model=True)
# Compute the training states
train_states = ESN.computeState(train_x)
# Compute the readout
ESN.trainReadout(train_states, train_y)

# We calculate the test states and prediction
test_states = ESN.computeState(test_x)
prediction = ESN.computeOutput(test_states)
pred = prediction[0, 0, transient:]
y = test_y[0, 0, transient:]

# We draw the results
train_signal = np.squeeze(train_x)
train_target = np.squeeze(train_y)

test_signal = np.squeeze(test_x)
test_target = np.squeeze(prediction)  # Transient state has to be removed

error = np.mean(np.sum(np.power(y - pred, 2)) / y.shape[0])

# PLOT THE SIGNALS
fig = plt.figure()
if layers < 4:
    lay = 5
else:
    lay = layers
gs = gridspec.GridSpec(nrows=3, ncols=lay)

ax = [fig.add_subplot(gs[0, 0:lay]),
      fig.add_subplot(gs[1, 0:lay - 2]),
      fig.add_subplot(gs[1, lay - 2:lay])]
for i in range(layers):
    ax.append(fig.add_subplot(gs[2, i:i+1]))

signal_plot = ax[0]
relevance_plot = ax[1]
memory_plot = ax[2]

plt.tight_layout()
signal_plot.set_title('Error (MSE): ' + str(error))

# lot train x
signal_plot.scatter(range(train_signal.shape[0]), train_signal, marker='x', s=5,
                    color='black', label='Train_x')

if experiment == 'Narma':
    signal_plot.scatter(range(train_target.shape[0]), train_y, marker='o', \
                                                                       s=5,
                        color='orange', label='Train_y')

    signal_plot.scatter(range(train_target.shape[0],
                              train_target.shape[0] + test_target.shape[0]),
                        test_y, marker='o', s=5, color='green', label='Target')

    signal_plot.scatter(range(train_target.shape[0] + transient,
                              train_target.shape[0] + transient + pred.shape[0]),
                        pred, marker='o', s=5, color='red', label='Prediction')

if experiment == 'Battery':
    y_plot = signal_plot.twinx()
    y_plot.scatter(range(train_target.shape[0]), train_target, marker='o', s=5,
                   color='orange', label='Train_y')

    y_plot.scatter(range(train_target.shape[0],
                         train_target.shape[0] + test_target.shape[0]),
                   test_y, marker='o', s=5, color='green', label='Target')

    y_plot.scatter(range(train_target.shape[0] + transient,
                         train_target.shape[0] + transient + pred.shape[0]),
                   pred, marker='o', s=5, color='red', label='Prediction')

if experiment == 'Traffic':
    signal_plot.scatter(range(train_target.shape[0]), train_target,
                        marker='o',
                        s=5,
                        color='orange', label='Train_y')

    signal_plot.scatter(range(train_target.shape[0],
                              train_target.shape[0] + test_target.shape[0]),
                        test_y, marker='o', s=5, color='green', label='Target')

    signal_plot.scatter(range(train_target.shape[0] + transient,
                              train_target.shape[0] + transient + pred.shape[
                                  0]),
                        pred, marker='o', s=5, color='red', label='Prediction')

signal_plot.scatter(range(train_signal.shape[0],
                          train_signal.shape[0] + test_signal.shape[0]),
                    test_signal, marker='x', s=2, color='grey', label='Test X')

if experiment == 'Traffic':
    signal_plot.bar(weekend_days, 1, color='gray', alpha=0.3)
    signal_plot.bar(target_weekend_days, 1, color='green', alpha=0.3)

# signal_plot.set_xlim(train_x.shape[0] * 0.80, train_y.shape[0] + transient +
#                      pred.shape[0])
# plt.show()

print('SHOW')
plt.show()

