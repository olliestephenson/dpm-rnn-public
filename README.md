# dpm-coherence-rnn

Deep learning code implementing the satellite-based damage mapping method from Stephenson et al. 2021, IEEE TGRS (in revision). 

Contact: oliver.stephenson@caltech.edu

Written in PyTorch v1.0.1, tested for v1.7.0

THIS IS RESEARCH CODE PROVIDED TO YOU "AS IS" WITH NO WARRANTIES OF CORRECTNESS. USE AT YOUR OWN RISK.

## Notes 

This readme assumes you already have familiarity with [SAR](https://en.wikipedia.org/wiki/Synthetic-aperture_radar), [InSAR](https://en.wikipedia.org/wiki/Interferometric_synthetic-aperture_radar), [Python](https://www.python.org/), and [deep learning](https://en.wikipedia.org/wiki/Deep_learning) using [PyTorch](https://pytorch.org/). All software and data used are open source, but the processing can be involved for people unfamiliar with the details. Please check out our paper for a more detailed presentation of the method. If you're interested in satellite-based damage mapping but any of these terms are unfamiliar to you, please get in touch.  

These scripts are used to create damage proxy maps from sequential InSAR coherence time series using machine learning. The input data are a sequential series of pre-event InSAR coherence images (i.e. if you have SAR acquisitions A,B,C then we want the coherences for A-B and B-C) and one co-event coherence image (i.e. the coherence between the final pre-event SAR acquisition and the first post-event SAR acquisition). Best performance will be obtained when the temporal baseline is constant between acquisitsions.

This code assumes that you already have a stack of coherence images. These images can be produced using freely available [Sentiel-1 data](https://asf.alaska.edu/data-sets/sar-data-sets/sentinel-1/), which can be processed using the [InSAR Scientific Computing Environment (ISCE)](https://github.com/isce-framework/isce2). The method has not been tested with data from other SAR satellites, but will presumably work similarly assuming there are regular acqusitions before the natural hazard.

When creating damage maps you will need to think about your coordinate system. We do all of our processing in 'radar' coordinates, then map the final damage map to geographic coordinates for plotting/analysis. 

This method assumes that your natural hazard occured between two satellite acqusitions, with no anomalous behavior beforehand. Results may be worse for seasonal hazards, or hazards that occured over a longer period of time. We welcome discussions about potential improvements/modifications. Please get in touch! 


## Code structure

### Python files

`generate_dpm.py` is the main script that will generate a DPM. You can follow the code starting from here.

`train.py` contains all the code for training the model (i.e. batching data, computing objective, performing gradient descent, etc.).

`coherence_timeseries.py` contains the Coherence_Timeseries object for manipulating data. Currently expects `*.npy` files, but other formats can be easily added. 

`rnn_model.py` contains the RNN model, implemented with PyTorch.

`scripts` directory contains simple scripts for exploring the code output.

### JSON files

`config_jsons/` contains configuration JSONs for `generate_dpm.py`:
1. `train_dataset` (required, str) - coherence dataset used to train model (usually over a large region, around 100 km by 100 km).
2. `deploy_dataset` (required, str) - coherence dataset used to generate DPM (usually over a smaller area in the same geographic region, e.g. a town or a city).
3. `model_hyperparamters` (optional, dict) - see `config_jsons/example.json`, default parameters in code.
4. `training_hyperparameters` (optional, dict) - see `config_jsons/example.json`, default parameters in code.
5. `transform` (optional, str) - transform applied to map the coherence from [0,1] to an unbounded space before training. Either `logit_squared` (the logit transform of the coherence squared, used in the paper, default) or `logit` (logit transform without squaring the coherence). Other tranforms can easily be added. 

`dataset.json` contains information about available coherence time series datasets. The dataset names are keys in this file and also used for `train_dataset` and `deploy_dataset` fields in config files.  
1. `path` (required, str) - path to data file.
2. `shape` (required, list) - shape of data as list with 2 integers, same as returned by numpy.shape(). 
3. `length` (required, int) - length of coherence timeseries (for sequential coherence from N SAR images this will be N-1).
4. `event_index` (required, int) - index of event in timeseries (using zero indexing). Only data before this will be used in training. Anomaly detection will be performed on the coherence image at the event_index. Data after this image will not be used at all.
5. `pre_num` (optional, int) - number of pre-event coherence images to use in training. Must be >= 2 and <= event_index.

## Usage

### General usage

`python generate_dpm.py -d <device_id> --config_path <path to config JSON directory> --config <config JSON filename> --save_dir <save directory> --return_ts <bool>`

### Command line variables 

`--config` - Name of the JSON configuration file for this specific run.

`--config_path` - Path to directory containing the JSON configuration files. 

`--dataset_json` - JSON file containing details on all training and deployment datasets 

`--save_dir` - Directory in which to save outputs. If there is already a trained model in the relevant sub-directory, the code will just deploy the model on the data.

`--return_ts` - If true, code returns the mean and standard deviation of the forecast for every timestep, rather than just the final damage proxy map.

`--best_model` - Path to PyTorch `state_dict` saved from previous training (optional). Takes precedence over any previously saved models in `save_dir`.  

`-d` - Device id. Controls the GPU used for training. Set to -1 to train using a CPU. Defaults to CPU if GPU is not available. `print(torch.has_cuda)` should return `True` for GPU training. 


### Example
We provide some randomly generated data (test_dataset.npy) on which to test the code. As the data is randomly generated the results will not be physically meaningful. You can test the code by running: 

`python generate_dpm.py -d 0 --config_path config_jsons --config test --save_dir saved --return_ts True`

This will train a model with configuration in `config_jsons/test.json` on GPU device 0. Results will be saved in `saved/test/` and will include:
1. `best_model.pth` - the model that achieved the best test loss during training.
2. `final_model.pth` - the final training model.
3. `log.json` - log file of training.
4. `summary.json` - various summary statistics computed during training.
5. `config.json` - duplicate of the config file (for reproducability).
6. `coseismic_dpm/` - folder that contains the mean and standard deviation and z-scores of coseismic (or co-event, for non-earthquake nautral hazards) coherence under distribution predicted by the model. All outputs are in the transformed space (i.e. the coherence has been mapped to an unbounded space).
7. `full_ts/` - optional folder that contains the full time series of the coherence, forecast means, forcast standard deviations and calculated z-scores (controlled by `return_ts` boolean). All outputs are in the transformed space.

## Hyperparameter search

Default hyperparameters included in the code may not always be the best. We have not yet systematically explored the optimal hyperparameters. General guidelines for tuning hyperparameters are below:

### Model parameters

1. `rnn_dim`: [64, 128, 256, 512] should often be sufficient.
2. `num_layers`: set to 1 in our experiments, but can increase if large `rnn_dim` are not working well (expect significant increase in training time).
3. `h_dim`: usually no larger than `rnn_dim`.
4. `rnn_cell`: currently only gated recurrent unit (GRU) is implemented.

In general, you want as small a model as possible (which will be less prone to overfitting) without affecting performance. Note that we have not performed a systematic hyperparameter search for our work. 

### Training parameters

1. `learning_rate`: usually the most easily tune-able hyperparameter, but also most dataset-dependent. Recommended range is [0.01, 0.00001]. Smaller is better, but would also require more `num_epochs` to converge.
2. `batch_size`: smaller is better, but larger can decrease training time. No larger than 512 is recommended.
3. `num_epochs`: should increase as `batch_size` increases, or as `learning_rate` decreases. 
4. `seed`: seed for the random number generators.

In general, you want small `learning_rate` and `batch_size` as long as it doesn't take too many `num_epochs` to converge.

### Installation

In order to run this code, you need to install PyTorch and several dependencies. We recommend using a package management system such as [Conda](https://docs.conda.io/en/latest/) to install these packages into a dedicated environment.  

If you have GPUs available and want to make use of them during training (which is substantially faster), you will need to install the relevant version of the cudatoolkit package, or potentially build from source. This will depend on your machine and CUDA version. See [here](https://pytorch.org/get-started/locally/) for more information. 

To check if you have access to GPU training, after installation open a python terminal and do 'import torch; torch.cuda.is_available()'. This should be true.


## Credit 

Citation: Stephenson et al. 2021, IEEE TGRS (in revision), and see this [AGU abstract](https://ui.adsabs.harvard.edu/abs/2019AGUFM.G13C0567S/abstract). 

Code written by Eric Zhan and Oliver Stephenson 

Contact: oliver.stephenson@caltech.edu

## Dislaimer
This software is distributed under an MIT License. 

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


