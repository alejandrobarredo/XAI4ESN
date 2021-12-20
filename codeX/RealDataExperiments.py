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
    readout['trainMethod'] = trainMethod  # 'Ridge'  # Lasso # 'Normal'
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
    spectral = 0.9
    leaky = 0.01
    i_scale = 1
    t_coef = 0.10
    spars = 0

# If Battery experiment, load Battery data
elif experiment == 'Battery':
    # Battery
    length = 30000
    with open('./data/Battery/B0005.pkl', 'rb') as f:
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
print('Baseline error: ' + str(baseline_error))
print('Error: ' + str(error))

# Plot the framework
# First plot the signals
# Then plot the potential memory
# Then plot the relevance analysis
# Then plot the recurrence plots
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

train_y = np.squeeze(ESNx.train_y)
train_x = np.squeeze(ESNx.train_x)
test_y = np.squeeze(ESNx.test_y)

signal_plot.scatter(range(train_x.shape[0]), train_x, marker='x', s=5,
                    color='black', label='Train_x')

if experiment == 'Battery':
    y_plot = signal_plot.twinx()
    y_plot.scatter(range(train_y.shape[0]), train_y, marker='o', s=5,
                        color='orange', label='Train_y')

    y_plot.scatter(range(train_y.shape[0],
                              train_y.shape[0] + test_y.shape[0]),
                        test_y, marker='o', s=5, color='green', label='Target')

    y_plot.scatter(range(train_y.shape[0] + ESNx.transient,
                              train_y.shape[0] + ESNx.transient + pred.shape[0]),
                        pred, marker='o', s=5, color='red', label='Prediction')
else:
    signal_plot.scatter(range(train_y.shape[0]), train_y, marker='o', s=5,
                        color='orange', label='Train_y')

    signal_plot.scatter(range(train_y.shape[0],
                              train_y.shape[0] + test_y.shape[0]),
                        test_y, marker='o', s=5, color='green', label='Target')

    signal_plot.scatter(range(train_y.shape[0] + ESNx.transient,
                              train_y.shape[0] + ESNx.transient + pred.shape[0]),
                        pred, marker='o', s=5, color='red', label='Prediction')

signal_plot.scatter(range(input_sig.shape[0],
                          input_sig.shape[0] + test_x.shape[0]),
                    test_x, marker='x', s=2, color='grey', label='Test X')

if experiment == 'Traffic':
    signal_plot.bar(weekend_days, 1, color='gray', alpha=0.3)
    signal_plot.bar(target_weekend_days, 1, color='green', alpha=0.3)

signal_plot.set_xlim(train_x.shape[0] * 0.80, train_y.shape[0] + ESNx.transient +
                     pred.shape[0])
# plt.show()
plt.pause(0.1)

# Potential memory
memories = []
death_signals = []
death_predictions = []
mem_depth = units * layers
for memory_test in np.linspace(ESNx.transient*1.2, ESNx.test_x.shape[2], 10):
    current_input = ESNx.test_x[0, 0, :int(memory_test)].reshape(1, 1, -1)
    death = np.zeros(mem_depth).reshape(1, 1, -1)
    death_signal = np.concatenate([current_input, death], axis=2)
    test_state = ESNx.deepESN.computeState(death_signal, ESNx.IPconf.DeepIP)[0]
    prediction_with_death = ESNx.deepESN.computeOutput(test_state)[0]
    #prediction_with_death = TimeSeries.min_max_norm(prediction_with_death)

    prediction_with_death = prediction_with_death[-(100 + mem_depth):]
    death_signals.append(np.squeeze(death_signal[0, 0, -(100 + mem_depth):]))
    death_predictions.append(prediction_with_death)

    death_val = prediction_with_death[-1]
    mem = 0
    for ii in range(len(prediction_with_death) - 1, 0, -1):
        if not np.round(prediction_with_death[ii], 1) == np.round(death_val,
                                                                  1):
            mem = ii - 100
            break
    memories.append(mem * layers)

memory_plot.scatter(range(death_signals[5].shape[0]), death_signals[5],
                    marker='x',
                    s=5, color='blue', label='Signal death')
memory_plot.scatter(range(death_predictions[5].shape[0]),
                    death_predictions[5], s=3, color='orange',
                    label='Prediction death')
memory_plot.set_title('Memory: n= ' + str(10) + ', mean=' + str(np.mean(mem)
                                                                * layers)
                      + ', std=' + str(np.std(mem)))

plt.pause(0.1)

#  Relevance
test_state = test_state[:, ESNx.transient:]
relevance_point = pred.shape[0]
point = ESNx.transient + relevance_point
current_input = ESNx.test_x[0, 0, :point].reshape(1, 1, -1)

test_state = ESNx.deepESN.computeState(current_input, ESNx.IPconf.DeepIP)[0]

previous_states = [test_state[:, i].reshape(units * layers, 1) for i in
                   range(test_state.shape[1])]
last_state = test_state[:, -1].reshape(units * layers, 1)
relevances = ESNx.state_entropy(last_state, previous_states, layers, units)

relevance = np.array(np.vstack(np.array(relevances)))
relevance = relevance[ESNx.transient:, :]

for i in range(relevance.shape[1]):
    relevance_plot.plot(range(input_sig.shape[0] + ESNx.transient - 1,
                          input_sig.shape[0] + ESNx.transient + relevance_point),
                        relevance[:, i], label='Layer ' + str(i + 1))

relevance_plot.axvline(input_sig.shape[0] + ESNx.transient - 1, color='black')
relevance_plot.axvline(input_sig.shape[0] + ESNx.transient + relevance_point,
                       color='black')
signal_plot.axvline(input_sig.shape[0] + ESNx.transient - 1, color='black',
                    ls='--')
signal_plot.axvline(input_sig.shape[0] + ESNx.transient + relevance_point,
                    color='black', ls='--')

plt.pause(0.1)
test_state = test_state[:, ESNx.transient:]

for i in range(layers):
    print('Layer ' + str(i))
    rp = RecurrencePlot(threshold='point', percentage=30)
    rpx = RecurrencePlot(percentage=20)
    lay_test = test_state[i * units:(i * units) + units, :]
    test_rp = np.mean(lay_test, axis=0)
    X_rp = rp.fit_transform(test_rp.reshape(1, -1))
    X_rpx = rpx.fit_transform(test_rp.reshape(1, -1))
    st = stats.describe(X_rpx[0].reshape(-1))
    ax[i + 3].imshow(X_rp[0], cmap='binary', origin='lower',
                     extent=[train_x.shape[0] + ESNx.transient, train_x.shape[0] + ESNx.transient +
                             test_x.shape[0], train_x.shape[0] + ESNx.transient,
                             train_x.shape[0] + ESNx.transient + test_x.shape[0]]
                     )

    ax[i + 3].set_title('Layer ' + str(i + 1) +' mean:' + str(np.round(
        st.mean, 4)))
    ax[i + 3].set_rasterized(True)

signal_plot.legend(loc=2)
memory_plot.legend(loc=2)
relevance_plot.legend(loc=2)
signal_plot.set_rasterized(True)
memory_plot.set_rasterized(True)
relevance_plot.set_rasterized(True)

plt.pause(0.1)

# Save the plot
TimeSeries.saveimage(name=save_name,
                     folder=save_dir)
print('Saved to:' + save_dir + save_name + '.pdf')
plt.show()