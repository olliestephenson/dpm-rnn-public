import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):

    def __init__(self, model_config):
        super().__init__()

        self.config = model_config
        self._construct_model()

    def _construct_model(self):
        data_dim = self.config['data_dim'] # dimension of data at each timestep (e.g. coherence is scalar, so data_dim=1)
        h_dim = self.config['h_dim'] # size of hidden layers in fully-connected network
        rnn_dim = self.config['rnn_dim'] # size of hidden state for RNN
        rnn_cell = self.config['rnn_cell'] # type of RNN (currently only GRU is implemented)
        num_layers = self.config['num_layers'] # number of RNN layers (default is 1, more may be better but is slower to train)

        # Initialize decoder network that maps hidden state to parameters of Gaussian distribution
        # Currently fixed to be 3-layer network, but can be customizable in the future.
        self.dec_fc = nn.Sequential(
            nn.Linear(rnn_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_mean = nn.Linear(h_dim, data_dim)
        self.dec_logvar = nn.Linear(h_dim, data_dim) # we learn the log-variance instead of standard deviation (numerical stability)

        # Initialize RNN cell
        if rnn_cell == 'gru':
            self.rnn = nn.GRU(data_dim, rnn_dim, num_layers)
        elif rnn_cell == 'lstm':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def num_parameters(self):
        """Count the number of trainable parameters in the model."""
        if not hasattr(self, '_num_parameters'):
            self._num_parameters = 0
            for p in self.parameters():
                count = 1
                for s in p.size():
                    count *= s
                self._num_parameters += count

        return self._num_parameters

    def forward(self, batch, generate_dpm=False):
        """
        Pass a batch of sequences through RNN and compute means and log-variances.
        Assumes batch has shape (batch_size, seq_len, data_dim)
        If generate_dpm is true the function returns all forecasts, including coseismic
        If tgenerate_dpm is false the function returns just preseismic forecasts
        """

        batch = batch.transpose(0,1) # PyTorch method for swapaxes
        seq_len, batch_size, _ = batch.size()

        # Initialize initial hidden state h_0 to be all 0s
        h_0 = torch.zeros(self.config['num_layers'], batch_size, self.config['rnn_dim']).to(batch.device)

        # Run batch sequences through RNN to compute all hidden states
        # By default, PyTorch assumes first dimension of batch is time, which is why we need to tranpose batch above
        # hiddens has shape (seq_len, batch_size, rnn_dim), which corresponds to h_1 to h_T
        self.rnn.flatten_parameters()
        hiddens, _ = self.rnn(batch, h_0)

        # h_(T-1) is used to predict x_T which is the final preseismic coherence measurement 
        # We want to use h_(t-1) to predict x_t, so we want hidden states h_0 to h_(T-1) when traning on preseismic data
        # Only want h_0 that corresponds to the output hidden state when using a stacked RNN
        if generate_dpm==False:
            hiddens = torch.cat([h_0[-1,:,:].unsqueeze(dim=0), hiddens], dim=0)[:-1] # move h_0 to front, truncate h_T
        elif generate_dpm==True:
            hiddens = torch.cat([h_0[-1,:,:].unsqueeze(dim=0), hiddens], dim=0) # move h_0 to front, keep h_T
            seq_len = seq_len + 1 # Increase sequence length to take account of additional element

        # We use decoder to map each hidden state to parameters of a Gaussian distribution
        # To forgo looping, we will reshape hiddens into one giant batch, then reshape back after
        hiddens = hiddens.view(seq_len*batch_size, -1)
        dec_h = self.dec_fc(hiddens) # passing through fully connected layers
        dec_mean = self.dec_mean(dec_h) # means of Gaussian
        dec_logvar = self.dec_logvar(dec_h) # log-variances of Gaussian

        # Reshape the means and log-variances from (seq_len*batch_size, data_dim) back to (seq_len, batch_size, data_dim)
        dec_mean = dec_mean.view(seq_len, batch_size, -1)
        dec_logvar = dec_logvar.view(seq_len, batch_size, -1)

        # Swap back to (batch_size, seq_len, data_dim) to match original input batch shape
        dec_mean = dec_mean.transpose(0,1)
        dec_logvar = dec_logvar.transpose(0,1)

        return dec_mean, dec_logvar
