import numpy as np
import numpy.matlib as npm
import scipy as sc
import random
import sys


class Struct(object): pass


class DeepESN:
    
    '''
    Deep Echo State Network (DeepESN) class:
    this class implement the DeepESN model suitable for 
    time-serie prediction and sequence classification.

    Reference paper for DeepESN model:
    C. Gallicchio, A. Micheli, L. Pedrelli, "Deep Reservoir Computing: A
    Critical Experimental Analysis", Neurocomputing, 2017, vol. 268, pp. 87-99
    
    Reference paper for the design of DeepESN model in multivariate time-series prediction tasks:
    C. Gallicchio, A. Micheli, L. Pedrelli, "Design of deep echo state networks",
    Neural Networks, 2018, vol. 108, pp. 33-47 
    
    ----

    This file is a part of the DeepESN Python Library (DeepESNpy)

    Luca Pedrelli
    luca.pedrelli@di.unipi.it
    lucapedrelli@gmail.com

    Department of Computer Science - University of Pisa (Italy)
    Computational Intelligence & Machine Learning (CIML) Group
    http://www.di.unipi.it/groups/ciml/

    ----
    '''
    
    def __init__(self, n_inputs, n_rec_units, n_rec_layers,
                 spectral_radius_list,
                 leaky_rate_list,
                 input_scale_list,
                 sparsity,
                 readout_method,
                 readout_reg,
                 transient,
                 IPconf,
                 verbose=0,
                 W=None,
                 Win=None,
                 gain=None,
                 bias=None):
        # initialize the DeepESN model
        
        if verbose:
            sys.stdout.write('init DeepESN...')
            sys.stdout.flush()
            
        self.W = {}  # recurrent weights
        self.Win = {}  # recurrent weights
        self.Gain = {}  # activation function gain
        self.Bias = {}  # activation function bias

        self.Nu = n_inputs  # number of inputs
        self.Nr = n_rec_units  # number of units per layer
        self.Nl = n_rec_layers  # number of layers
        self.rhos = spectral_radius_list  # list of spectral radius
        self.lis = leaky_rate_list  # list of leaky rate
        self.iss = input_scale_list  # list of input scale

        self.transient = transient

        self.IPconf = IPconf

        readout = Struct()
        self.trainMethod = readout_method
        readout.regularizations = readout_reg
        self.readout = readout

        # sparse recurrent weights init
        if 1 - sparsity < 1:
            for layer in range(n_rec_layers):
                self.W[layer] = np.zeros((n_rec_units, n_rec_units))
                for row in range(n_rec_units):
                    # number of row elements
                    n_r_e = int((1 - sparsity) * n_rec_units)
                    # row elements
                    r_e = random.sample(range(n_rec_units), n_r_e)
                    self.W[layer][row, r_e] = np.random.uniform(-1,
                                                                +1,
                                                                size=(1, n_r_e))
        # full-connected recurrent weights init      
        else:
            for layer in range(n_rec_layers):
                self.W[layer] = np.random.uniform(-1,
                                                  +1,
                                                  size=(n_rec_units,
                                                        n_rec_units))
        # layers init
        for layer in range(n_rec_layers):

            target_li = leaky_rate_list[layer]
            target_rho = spectral_radius_list[layer]
            input_scale = input_scale_list[layer]

            if layer == 0:
                self.Win[layer] = np.random.uniform(-input_scale,
                                                    input_scale,
                                                    size=(n_rec_units,
                                                          n_inputs + 1))
            else:
                self.Win[layer] = np.random.uniform(-input_scale,
                                                    input_scale,
                                                    size=(n_rec_units,
                                                          n_rec_units + 1))

            Ws = (1 - target_li) * np.eye(self.W[layer].shape[0],
                                          self.W[layer].shape[1]) \
                                        + target_li * self.W[layer]

            eig_value, eig_vector = np.linalg.eig(Ws)
            actual_rho = np.max(np.absolute(eig_value))

            Ws = (Ws * target_rho)/actual_rho

            self.W[layer] = (target_li**-1) * (Ws - (1.-target_li) * np.eye(self.W[layer].shape[0], self.W[layer].shape[1]))
            
            self.Gain[layer] = np.ones((n_rec_units, 1))
            self.Bias[layer] = np.zeros((n_rec_units, 1))

            if Win is not None:
                self.Win = Win
            if W is not None:
                self.W = W
            if gain is not None:
                self.Gain = gain
            if bias is not None:
                self.Bias = bias

        if verbose:
            print('done.')
            sys.stdout.flush()

    def computeLayerState(self, input, layer, initialStatesLayer = None, DeepIP = 0):  
        # compute the state of a layer with pre-training if DeepIP == 1                    
        
        state = np.zeros((self.Nr, input.shape[1]))   
        
        if initialStatesLayer is None:
            initialStatesLayer = np.zeros(state[:, 0:1].shape)
        
        input = self.Win[layer][:, 0:-1].dot(input) \
                + np.expand_dims(self.Win[layer][:, -1], 1)
        
        if DeepIP:
            state_net = np.zeros((self.Nr, input.shape[1]))
            state_net[:, 0:1] = input[:, 0:1]
            state[:, 0:1] = self.lis[layer] * np.tanh(np.multiply(self.Gain[
                                                                     layer], state_net[:, 0:1]) + self.Bias[layer])
        else:
            #state[:,0:1] = self.lis[layer] * np.tanh(np.multiply(self.Gain[layer], input[:,0:1]) + self.Bias[layer])        
            state[:,0:1] = (1-self.lis[layer]) * initialStatesLayer + self.lis[layer] * np.tanh(np.multiply(self.Gain[layer], self.W[layer].dot(initialStatesLayer) + input[:,0:1]) + self.Bias[layer])
 
        for t in range(1,state.shape[1]):
            if DeepIP:
                state_net[:,t:t+1] = self.W[layer].dot(state[:,t-1:t]) + input[:,t:t+1]
                state[:,t:t+1] = (1-self.lis[layer]) * state[:,t-1:t] + self.lis[layer] * np.tanh(np.multiply(self.Gain[layer], state_net[:,t:t+1]) + self.Bias[layer])
                
                eta = self.IPconf.eta
                mu = self.IPconf.mu
                sigma2 = self.IPconf.sigma**2
            
                # IP learning rule
                deltaBias = -eta*((-mu/sigma2)+ np.multiply(state[:,t:t+1], (2*sigma2+1-(state[:,t:t+1]**2)+mu*state[:,t:t+1])/sigma2))
                deltaGain = eta / npm.repmat(self.Gain[layer],1,state_net[:,t:t+1].shape[1]) + deltaBias * state_net[:,t:t+1]
                
                # update gain and bias of activation function
                self.Gain[layer] = self.Gain[layer] + deltaGain
                self.Bias[layer] = self.Bias[layer] + deltaBias
                
            else:
                state[:, t:t+1] = (1-self.lis[layer]) * state[:,t-1:t] + \
                                  self.lis[layer] * np.tanh( np.multiply(self.Gain[layer], self.W[layer].dot(state[:,t-1:t]) + input[:,t:t+1]) + self.Bias[layer])
                
        return state

    def computeDeepIntrinsicPlasticity(self, inputs):
        # we incrementally perform the pre-training (deep intrinsic plasticity) over layers
        
        len_inputs = inputs.shape[0]
        states = []
        
        for i in len_inputs:
            states.append(np.zeros((self.Nr*self.Nl, inputs[i].shape[1])))
        
        for layer in range(self.Nl):

            for epoch in range(self.IPconf.Nepochs):
                Gain_epoch = self.Gain[layer]
                Bias_epoch = self.Bias[layer]


                if len(inputs) == 1:
                    self.computeLayerState(inputs[0][:, self.IPconf.indexes],
                                           layer, DeepIP=1)
                else:
                    for i in self.IPconf.indexes:
                        self.computeLayerState(inputs[i], layer, DeepIP = 1)
                       
                
                if (np.linalg.norm(self.Gain[layer]-Gain_epoch,2) < self.IPconf.threshold) and (np.linalg.norm(self.Bias[layer]-Bias_epoch,2)< self.IPconf.threshold):
                    sys.stdout.write(str(epoch+1))
                    sys.stdout.write('.')
                    sys.stdout.flush()
                    break
                
                if epoch+1 == self.IPconf.Nepochs:
                    sys.stdout.write(str(epoch+1))
                    sys.stdout.write('.')
                    sys.stdout.flush()
            
            inputs2 = []
            for i in range(len(inputs)):
                inputs2.append(self.computeLayerState(inputs[i], layer))
            
            for i in range(len(inputs)):
                states[i][(layer)*self.Nr: (layer+1)*self.Nr,:] = inputs2[i]

            inputs = inputs2                   
            
        return states
    
    def computeState(self, inputs, DeepIP = 0, initialStates = None,
                     verbose=0):
        # compute the global state of DeepESN with pre-training if DeepIP == 1         
        
        if self.IPconf.DeepIP and DeepIP:
            if verbose:
                sys.stdout.write('compute state with DeepIP...')
                sys.stdout.flush()
            states = self.computeDeepIntrinsicPlasticity(inputs)
        else:      
            if verbose:
                sys.stdout.write('compute state...')
                sys.stdout.flush()
            states = []

            for i_seq in range(inputs.shape[0]):
                states.append(self.computeGlobalState(inputs[i_seq], initialStates))
                
        if verbose:        
            print('done.')
            sys.stdout.flush()
        
        return states
    
    def computeGlobalState(self, input, initialStates):
        # compute the global state of DeepESN

        state = np.zeros((self.Nl*self.Nr, input.shape[1]))
        
        initialStatesLayer = None

        for layer in range(self.Nl):
            if initialStates is not None:
                initialStatesLayer = initialStates[(layer)*self.Nr: (layer+1)*self.Nr,:]            
            state[(layer)*self.Nr: (layer+1)*self.Nr,:] = self.computeLayerState(input, layer, initialStatesLayer, 0)    
            input = state[(layer)*self.Nr: (layer+1)*self.Nr,:]   

        return state
        
    def trainReadout(self, trainStates, trainTargets, lb, verbose=0):
        # train the readout of DeepESN

        # add bias
        X = np.ones((trainStates.shape[0]+1, trainStates.shape[1]))
        X[:-1, :] = trainStates
        trainStates = X  
        
        if verbose:
            sys.stdout.write('train readout...')
            sys.stdout.flush()
        
        if self.trainMethod == 'SVD': # SVD, accurate method
            U, s, V = np.linalg.svd(trainStates, full_matrices=False);  
            s = s/(s**2 + lb)
                      
            self.Wout = trainTargets.dot(np.multiply(V.T, np.expand_dims(s,0)).dot(U.T));
            
        else:  # NormalEquation, fast method
            B = trainTargets.dot(trainStates.T)
            A = trainStates.dot(trainStates.T)

            self.Wout = np.linalg.solve((A + np.eye(A.shape[0], A.shape[1]) * lb), B.T).T

        if verbose:
            print('done.')
            sys.stdout.flush()
        
    def computeOutput(self, state):
        # compute a linear combination between the global state and the output weights
        if state.shape[1] < self.transient:
            return np.zeros((self.Wout.shape[0], state.shape[1]))
        else:
            return self.Wout[:, 0:-1].dot(state) \
                   + np.expand_dims(self.Wout[:, -1], 1)  # Wout product + add bias

    def computeLinearActivationTerms(self, state):
        activation_terms = []
        for i in range(state.shape[0]):
            activation_terms.append(self.Wout[0, i + 1] * state[i, 0])
        return activation_terms
