# dpm-coherence-rnn

Deep learning code implementing the satellite-based damage mapping method from Stephenson et al. 2021, IEEE TGRS. 

Written in PyTorch v1.0.1

## Notes 

These scripts are used to create damage proxy maps from InSAR coherence time series using machine learning. The input data is a sequential series of preseismic InSAR coherence images (i.e. if you have SAR acquisitions A,B,C then we want the coherences for AB and BC) and one coseismic coherence image. Best performance will be obtained when the temporal baseline is constant between acquisitsions.

This code assumes that you already have a stack of coherence images. These images can be produced using Sentiel-1 data (https://asf.alaska.edu/data-sets/sar-data-sets/sentinel-1/) processed using the InSAR Scientific Computing Environment (https://github.com/isce-framework/isce2).

THIS IS RESEARCH CODE PROVIDED TO YOU "AS IS" WITH NO WARRANTIES OF CORRECTNESS. USE AT YOUR OWN RISK.

## Code structure

### Python files

`generate_dpm.py` is the main script that will generate a DPM. You can follow the code starting from here.

`train.py` contains all the code for training the model (i.e. batching data, computing objective, performing gradient descent, etc.).

`coherence_timeseries.py` contains the Coherence_Timeseries object for manipulating data. Currently expects `*.npy` files, but other format can be easily added. 

`rnn_model.py` contains the RNN model, implemented with PyTorch.

`scripts` directory contains simple scripts for exploring the code output

### JSON files

`config_jsons/` contains configuration JSONs for `generate_dpm.py`:
1. `train_dataset` (required, str) - coherence dataset used to train model (usually over large region)
2. `deploy_dataset` (required, str) - coherence dataset used to generate DPM (usually over smaller region)
3. `model_hyperparamters` (optional, dict) - see `config_jsons/example.json`, default parameters in code
4. `training_hyperparameters` (optional, dict) - see `config_jsons/example.json`, default parameters in code
5. `transform` (optional, str) - transform applied to map the coherence from [0,1] to an unbounded space before training. Either logit_squared` (the logit transform of the coherence squared, used in the paper) or `logit` (logit transform without squaring the coherence). Other tranforms can easily be added. 

`dataset.json` contains information about available coherence time series datasets. The dataset names are keys in this file and also used for `train_dataset` and `deploy_dataset` fields in config files.  
1. `path` (required, str) - path to data file
2. `shape` (required, list) - shape of data as list with 2 integers
3. `length` (required, int) - length of coherence timeseries (for sequential coherence and N SAR images this will be N-1)
4. `event_index` (required, int) - index of event in timeseries (using zero indexing). Only data before this will be used in training
5. `pre_num` (optional, int) - number of pre-event coherence images to use in training. Must be >= 2 and <= event_index 

## Usage

### General usage

`python generate_dpm.py -d <device_id> --config_path <path to config JSON directory> --config <config JSON filename> --save_dir <save directory> --return_ts <bool>`

### Example
We provide some randomly generated data (test_dataset.npy) on which to test the code. You can test the code by running: 

`python generate_dpm.py -d 0 --config_path config_jsons --config test --save_dir saved --return_ts True`

This will train a model with configuration in `config_jsons/test.json` on GPU device 0 (use -1 for CPU training). Results will be saved in `saved/test/` and will include:
1. `best_model.pth` - the model that achieved the best test loss during training
2. `final_model.pth` - the final training model 
3. `log.json` - log file of training
4. `summary.json` - various summary statistics computed during training
5. `config.json` - duplicate of the config file (for reproducability)
6. `coseismic_dpm/` - folder that contains the mean and standard deviation and z-scores of coseismic coherence under distribution predicted by the model. All outputs are in the transformed space
7. `full_ts/` - optional folder that contains the full time series of the coherence, forecast means, forcast standard deviations and calculated z-scores (controlled by `return_ts` boolean). All outputs are in the transformed space 

## Hyperparameter search

Default hyperparameters included in the code may not always be the best. We have not yet systematically explored the optimal hyperparameters. General guidelines for tuning hyperparameters are below:

### Model parameters

1. `rnn_dim`: [64, 128, 256, 512] should often be sufficient.
2. `num_layers`: almost always 1, but can increase if large `rnn_dim` are not working well (expect significant increase in training time)
3. `h_dim`: usually no larger than `rnn_dim`.
4. `rnn_cell`: currently only gated recurrent unit (GRU) is implemented.

In general, you want as small a model as possible without affecting performance.

### Training parameters

1. `learning_rate`: usually the most easily tune-able hyperparameter, but also most dataset-dependent. Recommended range is [0.01, 0.00001]. Smaller is better, but would also require more `num_epochs` to converge.
2. `batch_size`: smaller is better, but larger can decrease training time. No larger than 512 is recommended.
3. `num_epochs`: should increase as `batch_size` increases, or as `learning_rate` decreases. 
4. `seed`: seed for the random number generators 

In general, you want small `learning_rate` and `batch_size` as long as it doesn't take too many `num_epochs` to converge.

## Future improvements

[ ] Randomness seeding is not working as intended. Relevant for reproducing results.

[ ] Add L2-weight normalization (`weight_decay` parameter in optimizer, see [here](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam)). Can sometimes help.

[ ] Adaptive `learning_rate` that decreases as learning plateaus. 


## Credit 

Citation: Stephenson et al. 2021, IEEE TGRS (In Revision)
Code written by Eric Zhan, with contributions by Oliver Stephenson 
Contact: oliver.stephenson@caltech.edu

## Dislaimer
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


