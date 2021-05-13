import numpy as np
from DeepESN import DeepESN
import pandas as pd
from math import sqrt
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# This code is a complementary code that contains many utilities:
#       Generates different signals


class TimeSeries:

    # Function to generate sinusoidal signals
    @staticmethod
    def sin_data_gen(examples, length, _memory=None, blanks=False,
                     signal_parameters=None, gain=None, feature=None,
                     noise=False, norm=False):
        """
        Function to generate sinusoidal waves. Random or by parameter
        :param examples: (integer) Number of sinusoidal waves required
        :param length:   (integer) length of the wave samples requested
        :param _memory:  (integer) time delay forced on the target wave
        :param blanks:   (boolean) If blank are added to the wave
        :param signal_parameters: (dict = {'gain': x ,
                                           'duration':y,
                                           'omega':z}
                                parameters for the wave. If None -> random
        :return: (list, list) _data, _targets
                Returns a series (#examples) of paired signals (_data,
                _targets). _targets consisting on a copy of _data signals
                with a delay o "_memory".
        """
        if examples == -1:
            gain = gain  # amplitude coeficient
            duration = 1.0  # in seconds, may be float
            omega = np.random.randint(4, 10)
            fs = length / duration  # sine frequency, Hz, may be float
            n = int(fs * duration)  # samples for the duration
            t = np.linspace(0, duration, n)
            L = (np.sin(2 * np.pi * omega * t)).astype(np.float32) * gain
            # L1 = list(np.tan(np.arange(0, length, 0.1)))
            _example = np.array(L)
            if noise:
                rd_noise_signal = (np.random.randn(n) * 0.02) * gain
                _example = _example + rd_noise_signal

            return _example
        else:
            data = []
            targets = []

            for cont in range(examples):
                if signal_parameters is None:
                    gain = np.random.randn() + 2  # amplitude coeficient
                    duration = 1.0  # in seconds, may be float
                    omega = np.random.randint(5, 25)  # sampling rate, Hz,
                    # must be integer
                    # omega = 20
                else:
                    gain = signal_parameters['gain']
                    duration = signal_parameters['duration']
                    omega = signal_parameters['omega']

                fs = length / duration  # sine frequency, Hz, may be float

                n = int(fs * duration)  # samples for the duration

                t = np.linspace(0, duration, n)
                L = (np.sin(2 * np.pi * omega * t)).astype(np.float32) * gain

                # rd_noise_signal = (np.random.randn(n) * 0.02) * gain

                # L1 = list(np.tan(np.arange(0, length, 0.1)))
                _example = np.array(L)
                # _example = _example + rd_noise_signal

                if feature == 'non-stationary':
                    for i in range(len(_example)):
                        _example[i] = _example[i] + 0.005 * i
                if feature == 'nested-sinusoid':
                    L1 = (np.sin(2 * np.pi * omega * 10 * t)).astype(
                        np.float32) * gain
                    _example1 = np.array(L1)
                    _example = _example + (_example1 * 0.2)
                    _target = np.roll(_example, _memory)
                    _target[:-_memory]
                if feature == 'variable-sinusoid':
                    change = False
                    L1 = (np.sin(2 * np.pi * omega*4 * t)).astype(
                        np.float32) * gain
                    # rd_noise_signal1 = (np.random.randn(n) * 0.02) * gain
                    _example1 = np.array(L1)
                    # _example1 = _example1 + rd_noise_signal1
                    ii = 0
                    for i in range(len(_example)):
                        if i % 3000 == 0:
                            change = not change
                        if change:
                            _example[i] = _example1[ii]
                            ii += 1
                    window = length * 0.10
                    if window % 2 == 0:
                        window = window + 1
                    window = int(window)
                    _example = savgol_filter(_example, window, 2)
                    _target = np.roll(_example, _memory)
                if feature == 'nested-variable-sinusoid':
                    L2 = (np.sin(2 * np.pi * omega * 4 * t)).astype(
                        np.float32) * gain
                    rd_noise_signal2 = (np.random.randn(n) * 0.01) * gain
                    _example2 = np.array(L2)
                    _example2 = _example2 + rd_noise_signal2
                    ii = 0
                    change = False
                    for i in range(len(_example)):
                        if i % 3000 == 0:
                            change = not change
                        if change:
                            _example[i] = _example2[ii]
                            ii += 1
                    _example = savgol_filter(_example, 151, 3)
                    print()

                    L1 = (np.sin(2 * np.pi * omega * 10 * t)).astype(
                        np.float32) * gain
                    _example1 = np.array(L1)
                    _example = _example + (_example1 * 0.2)
                    _target = np.roll(_example, _memory)
                if feature == 'non-stat-nested':
                    for i in range(len(_example)):
                        _example[i] = _example[i] + 0.005 * i
                    L1 = (np.sin(2 * np.pi * omega * 10 * t)).astype(
                        np.float32) * gain
                    _example1 = np.array(L1)
                    _example = _example + (_example1 * 0.2)
                    _target = np.roll(_example, _memory)
                if feature == 'noisy':
                    rd_noise_signal = (np.random.randn(n) * 0.5) * gain
                    _target = np.roll(_example, _memory)
                    _example = _example + rd_noise_signal
                else:
                    _target = np.roll(_example, _memory)
                    _target[:_memory] = 0

                if norm:
                    _max = np.max(_example)
                    _min = np.min(_example)
                    _example = (_example - _min) / (_max - _min)
                    _target = (_target - _min) / (_max - _min)

                if noise:
                    rd_noise_signal = (np.random.randn(n) * 0.09) * gain
                    _example = _example + rd_noise_signal

                _example = _example[:-_memory]
                _target = _target[:-_memory]

                data.append(_example.reshape(1, -1))
                targets.append(_target.reshape(1, -1))

        return np.array(data), np.array(targets)

    # Function to plot google stock data
    @staticmethod
    def google_data_gen(_memory):
        google = pd.read_csv('../googlestockpricing/Google.csv')
        _data = google.Close.values.reshape(1, -1)
        _target = np.roll(_data[0, :], _memory)
        _target[:_memory] = 0
        _max = np.max(_data)
        _min = np.min(_data)
        _data = (_data - _min) / (_max - _min)
        _target = (_target - _min) / (_max - _min)
        return _data.reshape(1, 1, -1), _target.reshape(1, 1, -1)

    # Function to plot narma signals of order 10 and 20
    @staticmethod
    def narma_data_gen(length, _memory, order='10', sin=None):
        if sin is not None:
            s_t = sin
            # s_t = savgol_filter(sin, 1001, 2) * 0.2
            s_t = s_t + (np.random.randn(sin.shape[0]) * 0.02)
        else:
            s_t = np.random.randn(length) * 0.4

        _data = []
        if order == '10':
            y = np.zeros(10)
            for t in range(9, length):
                term1 = 0.3 * y[t]
                term2 = []
                for i in range(10):
                    term2.append(y[t - i])
                term2 = 0.05 * y[t] * np.sum(term2)
                term3 = 1.5 * s_t[t - 9] * s_t[t] + 0.1
                y_t1 = np.tanh(term1 + term2 + term3)
                y = np.append(y, y_t1)

            # _max = np.max(s_t)
            # _min = np.min(s_t)
            # _data = (s_t - _min) / (_max - _min)
            #
            # _max = np.max(y)
            # _min = np.min(y)
            # y = y[1:]
            # _target = (y - _min) / (_max - _min)
            # _target = np.roll(_target, -_memory)
            _data = np.array(s_t)
            _target = np.array(y[1:])

        if order == '20':
            y = np.zeros(20)
            for t in range(9, length):
                term1 = 0.3 * y[t]
                term2 = []
                for i in range(19):
                    term2.append(y[t - i])
                term2 = 0.05 * y[t] * np.sum(term2)
                term3 = 1.5 * s_t[t - 9] * s_t[t] + 0.1
                y_t1 = np.tanh(term1 + term2 + term3)
                y = np.append(y, y_t1)

            # _max = np.max(s_t)
            # _min = np.min(s_t)
            # _data = (s_t - _min) / (_max - _min)
            # _max = np.max(y)
            # _min = np.min(y)
            # y = y[11:]
            # _target = (y - _min) / (_max - _min)
            _data = np.array(s_t)
            _target = np.array(y[11:])

        if _memory is not None:
            print('memory')
            _target = np.roll(_target, -_memory)
            _target = _target[:-_memory]
            _target = np.concatenate((_target, (np.zeros(_memory*2))))
            _data = np.concatenate((np.zeros(_memory), _data))

        return _data.reshape(1, 1, -1), _target.reshape(1, 1, -1)

    # Code to split a signal into train_x, train_y, test_x and test_y
    def split_data(self, _data, _target):
        if not _data.shape[0] == 1:
            train_cut = int(_data.shape[2] * 0.8)
            train_x = _data[:train_cut, :, :]
            train_y = _target[:train_cut, :, :]
            test_x = _data[train_cut:, :, :]
            test_y = _target[train_cut:, :, :]
        else:
            train_cut = int(_data.shape[2] * 0.8)
            train_x = _data[0, :, :train_cut].reshape(1, 1, -1)
            train_y = _target[0, :, :train_cut].reshape(1, 1, -1)
            test_x = _data[0, :, train_cut:].reshape(1, 1, -1)
            test_y = _target[0, :, train_cut:].reshape(1, 1, -1)

        return train_x, train_y, test_x, test_y

    # Function to carry out min max normalization
    @staticmethod
    def min_max_norm(x):
        _min = np.min(x)
        _max = np.max(x)
        return (x - _min) / (_max - _min)

    # Code to claculate the absolute error and mean absolute error
    def calculate_error(self, _target, _prediction, average=True):
        if average:
            error = np.mean(np.abs(_target - _prediction))
        else:
            error = np.abs(_target - _prediction)

        return error

    # Function to set matplotlib's parameters to fit those of latex
    @staticmethod
    def latexify(fig_width=None, fig_height=None, columns=1):
        """Set up matplotlib's RC params for LaTeX plotting.
        Call this before plotting a figure.
        Parameters
        ----------
        fig_width : float, optional, inches
        fig_height : float,  optional, inches
        columns : {1, 2}
        """

        # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples
        # Width and max height in inches for IEEE journals taken from
        # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

        assert (columns in [1, 2])

        if fig_width is None:
            fig_width = 6.9 if columns == 1 else 13.8  # width in inches #3.39

        if fig_height is None:
            golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
            fig_height = fig_width * golden_mean  # height in inches

        MAX_HEIGHT_INCHES = 16.0
        if fig_height > MAX_HEIGHT_INCHES:
            print(("WARNING: fig_height too large:" + fig_height +
                   "so will reduce to" + MAX_HEIGHT_INCHES + "inches."))
            fig_height = MAX_HEIGHT_INCHES

        params = {
            # 'backend': 'ps',
            'pgf.rcfonts': False,
            'pgf.preamble': ['\\usepackage{gensymb}',
                             '\\usepackage[dvipsnames]{xcolor}'],
            "pgf.texsystem": "pdflatex",
            'text.latex.preamble': ['\\usepackage{gensymb}',
                                    '\\usepackage[dvipsnames]{xcolor}'],

            # values below are useful defaults. individual plot fontsizes are
            # modified as necessary.
            'axes.labelsize': 13,  # fontsize for x and y labels
            'axes.titlesize': 13,
            'font.size': 10,
            'legend.fontsize': 15,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'text.usetex': False,
            'figure.figsize': [fig_width, fig_height],
            'font.family': 'serif',
            'lines.linewidth': 1,
            'lines.markersize': 1,
            'xtick.major.pad': 2,
            'ytick.major.pad': 2
        }

        matplotlib.rcParams.update(params)

    # Function to save an image
    @staticmethod
    def saveimage(name, fig=plt, extension='pdf',
                  folder='../ResultadosPaper/', dpi=300):

        # Minor ticks off by default in matplotlib
        # plt.minorticks_off()

        # grid being off is the default for seaborn white style, so not needed.
        # plt.grid(False, axis = "x")
        # plt.grid(False, axis = "y")

        fig.savefig('{}{}.{}'.format(folder, name, extension),
                    bbox_inches='tight', dpi=dpi)