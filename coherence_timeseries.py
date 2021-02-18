import os
import numpy as np
import torch
from torch.utils.data import Dataset


from enum import Enum

class Mode(Enum):
    DEPLOY = 0
    TRAIN = 1
    TEST = 2


class Coherence_Timeseries(Dataset):

    data_dim = 1 # coherence values are scalar (dimension = 1)
    mode = Mode.DEPLOY # default mode

    def __init__(self, data_config):

        # Check fields in data_config
        assert 'path' in data_config and isinstance(data_config['path'], str)
        assert 'shape' in data_config and isinstance(data_config['shape'], list) and len(data_config['shape']) == 2
        assert 'length' in data_config and isinstance(data_config['length'], int)
        assert 'event_index' in data_config and isinstance(data_config['event_index'], int)

        # Load data
        assert data_config['path'][-4:] == '.npy'
        self.data = np.load(data_config['path'])
        assert isinstance(self.data, np.ndarray)

        # Check dataset.shape
        assert len(self.data.shape) == 3
        assert self.data.shape == (data_config['shape'][0], data_config['shape'][1], data_config['length'])
        self.dataset_shape = (self.data.shape[0], self.data.shape[1])
        self.sequence_length = self.data.shape[2]

        # Check event_index
        assert 0 <= data_config['event_index'] < self.sequence_length
        self.event_index = data_config['event_index']

        # Flatten data
        self.data = np.reshape(self.data, (-1, self.data.shape[-1]))
        self.data = np.expand_dims(self.data, axis=2) # last dimension 1 since coherence values are scalars

    def remove_nans(self):
        """Remove sequences with nan values from dataset."""
        nans = np.isnan(self.data)
        nan_count = np.sum(np.sum(nans, axis=-1), axis=-1)
        self.not_nan_inds = np.where(nan_count == 0)[0]
        self.data = self.data[self.not_nan_inds]
    
    def unbound(self,transform):
        """
        Transform coherence values into unbounded range with inverse sigmoid. Can transform coherence or squared coherence
        Transform on squared coherence closely matches cramer-rao bound on phase variance
        """

        if transform == 'logit_squared':
            # Convert to higher precision to avoid divide by zero error in log
            # Don't seem to need this with logit transform
            self.data = np.float64(self.data)
        
        # Make sure all values in [0,1] range first
        eps = 1e-6 # small epsilon value
        self.data[self.data <= 0.0] = eps
        self.data[self.data >= 1.0] = 1.0-eps

        # Apply inverse sigmoid
        print('Using transform: {}'.format(transform))
        if transform == 'logit':
            self.data = np.log(self.data/(1-self.data))
        elif transform == 'logit_squared':
            self.data = np.log(np.square(self.data)/(1.0-np.square(self.data)))
        else:
            raise Exception('Data transform not defined')

    def create_test_set(self, train_split=0.8, seed=128):
        """
        Create test dataset.

        This is memory efficient and doesn't duplicate self.data
        The training set is: self.data[self.shuffle_inds[:self.train_set_size]]
        The test set is:     self.data[self.shuffle_inds[self.train_set_size:]]

        Args:
            train_split: proportion of data to use for training, rest for test.
            seed: seed to fix randomness.
        """

        np.random.seed(seed) # fix randomness
        self.shuffle_inds = np.random.permutation(len(self.data)) # shuffle a random permutation
        self.train_set_size = int(train_split*len(self.data)) # set training set size

    def deploy(self):
        self.mode = Mode.DEPLOY

    def train(self):
        self.mode = Mode.TRAIN

    def test(self):
        self.mode = Mode.TEST

    def __len__(self):
        """
        Length of dataset.
        Must override this method when extending Dataset object.
        """
        if self.mode == Mode.DEPLOY:
            return len(self.data)
        elif self.mode == Mode.TRAIN:
            return self.train_set_size
        elif self.mode == Mode.TEST:
            return len(self.data)-self.train_set_size
        else:
            raise NotImplementedError
        
    def __getitem__(self, index):
        """
        For getting data with indices.
        Must override this method when extending Dataset object.

        Return: 
            (preseismic timeseries, coseismic coherence)
        """
        if self.mode == Mode.DEPLOY:
            batch_preseismic = self.data[index,:self.event_index]
            batch_coseismic = self.data[index,self.event_index] 
        elif self.mode == Mode.TRAIN:
            train_index = self.shuffle_inds[index]
            batch_preseismic = self.data[train_index,:self.event_index]
            batch_coseismic = self.data[train_index,self.event_index]
        elif self.mode == Mode.TEST:
            test_index = self.shuffle_inds[index+self.train_set_size]
            batch_preseismic = self.data[test_index,:self.event_index]
            batch_coseismic = self.data[test_index,self.event_index]
        else:
            raise NotImplementedError

        return torch.tensor(batch_preseismic).float(), torch.tensor(batch_coseismic).float()
