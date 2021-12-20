import numpy as np
from DeepESNX import DeepESN
from scipy.stats import entropy
class Struct(object): pass


class ESN:

    def __init__(self, data, target, length, memory, n_inputs, units, layers,
                 spectral, leaky, i_scale, t_coef, spars):
        data = data
        target = target
        length = length
        memory = memory
        n_inputs = n_inputs

        parameters = self.choose_parameters(units=units, layers=layers,
                                            spectral=spectral,
                                            leaky=leaky, i_scale=i_scale,
                                            t_coef=t_coef,
                                            spars=spars)

        self.n_rec_units = parameters['n_rec_units']
        self.n_rec_layers = parameters['n_rec_layers']
        self.spectral_radius_list = parameters['spectral_radius']
        self.leaky_rate_list = parameters['leaky_rate']
        self.input_scale_list = parameters['input_scale']
        self.transient_coef = parameters['transient_coef']
        self.sparsity = parameters['sparsity']
        self.transient = int(data.shape[2] * self.transient_coef)
        self.readout_method = 'SVD'  # readout method
        self.readout_reg = 10.0 ** np.array(
            range(-16, -1, 1))  # regularization term for readout
        self.IPconf = Struct()
        self.IPconf.DeepIP = 0

        self.train_x, self.train_y, self.test_x, self.test_y = self.split_to_train_val_test(
            _data=data,
            _target=target)

        if n_inputs > 1:
            self.train_x = self.train_x.reshape(-1, n_inputs,
                                                self.train_x.shape[2])
            self.test_x = self.test_x.reshape(-1, n_inputs,
                                              self.test_x.shape[2])


        self.deepESN = DeepESN(n_inputs,
                               self.n_rec_units,
                               self.n_rec_layers,
                               self.spectral_radius_list,
                               self.leaky_rate_list,
                               self.input_scale_list,
                               self.sparsity,
                               self.readout_method,
                               self.readout_reg,
                               self.transient,
                               self.IPconf,
                               verbose=0,
                               W=None,
                               Win=None,
                               gain=None,
                               bias=None)

        train_states = self.deepESN.computeState(self.train_x, self.IPconf.DeepIP)

        self.deepESN.trainReadout(train_states[0], self.train_y[0], self.readout_reg[6])

        self.train_pred = self.deepESN.computeOutput(train_states[0])
        _max = np.max(self.train_y[0, 0, self.transient:])
        _min = np.min(self.train_y[0, 0, self.transient:])
        self.target_range = np.abs(_max - _min)

        self.train_error = self.calculate_error(_target=self.train_y,
                                                _prediction=self.train_pred)

    @staticmethod
    def choose_parameters(units=None, layers=None,
                          spectral=None, leaky=None,
                          i_scale=None, t_coef=None, spars=None):
        parameter_grid = {'n_rec_units': range(1, 150),
                          'n_rec_layers': range(1, 5),
                          'spectral_radius': np.linspace(0.1, 0.9,
                                                         20).tolist(),
                          'leaky_rate': np.linspace(0.0001, 0.9, 20).tolist(),
                          'input_scale': np.linspace(0.001, 1, 20).tolist(),
                          'transient_coef': np.linspace(0.001, 0.3,
                                                        20).tolist(),
                          'sparsity': np.linspace(0.0001, 0.9, 20).tolist()}

        for key in parameter_grid.keys():
            if key == 'n_rec_units':
                if units is not None:
                    n_rec_units = units
                else:
                    n_rec_units = np.random.choice(parameter_grid[key])

            if key == 'n_rec_layers':
                if layers is not None:
                    n_rec_layers = layers
                else:
                    n_rec_layers = np.random.choice(parameter_grid[key])

            if key == 'spectral_radius':
                if spectral is not None:
                    val = spectral
                else:
                    val = np.random.choice(parameter_grid[key])
                spectral_radius = [val for i in range(n_rec_layers)]

            if key == 'leaky_rate':
                if leaky is not None:
                    val = leaky
                else:
                    val = np.random.choice(parameter_grid[key])
                leaky_rate = [val for i in range(n_rec_layers)]

            if key == 'input_scale':
                if i_scale is not None:
                    val = i_scale
                else:
                    val = np.random.choice(parameter_grid[key])
                input_scale = [val for i in range(n_rec_layers)]

            if key == 'transient_coef':
                if t_coef is not None:
                    transient_coef = t_coef
                else:
                    transient_coef = 0.01

            if key == 'sparsity':
                if spars is not None:
                    sparsity = spars
                else:
                    sparsity = np.random.choice(parameter_grid[key])

        parameters = {'n_rec_units': n_rec_units,
                      'n_rec_layers': n_rec_layers,
                      'spectral_radius': spectral_radius,
                      'leaky_rate': leaky_rate,
                      'input_scale': input_scale,
                      'transient_coef': transient_coef,
                      'sparsity': sparsity}

        return parameters

    @staticmethod
    def split_to_train_val_test(_data, _target):
        if not _data.shape[0] == 1:
            train_cut = int(_data.shape[2] * 0.7)
            train_x = _data[:, :, :train_cut]
            train_y = _target[:, :, :train_cut]
            test_x = _data[:, :, train_cut:]
            test_y = _target[:, :, train_cut:]
        else:
            train_cut = int(_data.shape[2] * 0.7)
            train_x = _data[0, :, :train_cut].reshape(1, 1, -1)
            train_y = _target[0, :, :train_cut].reshape(1, 1, -1)
            test_x = _data[0, :, train_cut:].reshape(1, 1, -1)
            test_y = _target[0, :, train_cut:].reshape(1, 1, -1)

        return train_x, train_y, test_x, test_y

    @staticmethod
    def calculate_error(_target, _prediction, average=True, t_range=0):
        if average:
            error = np.mean(np.abs(_target - _prediction))
        else:
            error = np.abs(_target - _prediction)

        if not t_range == 0:
            error = error / t_range
            if error > 1:
                error = 1
        return error

    @staticmethod
    def state_entropy(_state, _old_states, layers, units):
        _relevances = [list(np.zeros(layers))]
        if len(_old_states) > 0:
            for old_state in _old_states:
                _layer_relevances = []
                old_state = np.abs(old_state)
                _state = np.abs(_state)
                for layer in range(layers):
                    start = layer * units
                    end = start + units
                    os = old_state[start: end, 0]
                    s = _state[start: end, 0]
                    ent = entropy(os[1:], s[1:])
                    _layer_relevances.append(ent)
                _relevances.append(_layer_relevances)

        return _relevances

    @staticmethod
    def to_fuzzy(ranges, val, input=False):
        if input:
            for i in range(len(ranges)):
                if val <= ranges[i]:
                    return i
        else:
            fuzz = np.zeros_like(ranges)
            for i in range(len(ranges)):
                if val <= ranges[i]:
                    fuzz[i] = 1
                    return fuzz.reshape(1, -1)