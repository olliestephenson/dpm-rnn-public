import argparse
import json
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import random

from coherence_timeseries import Coherence_Timeseries
from rnn_model import RNN
from train import train_model


DEFAULT_MODEL_PARAMS = {
    "rnn_dim": 256,
    "rnn_cell" : "gru",
    "h_dim": 128,
    "num_layers": 1
}


DEFAULT_TRAINING_PARAMS = {
    "batch_size": 256,
    "num_epochs": 20,
    "learning_rate": 0.0005,
    "seed": 128
}


def load_json(json_path):
    if json_path[-5:] != '.json':
        json_path += '.json'
    assert os.path.isfile(json_path)

    with open(json_path, 'r') as f:
        json_file = json.load(f)

    return json_file

def get_device(device_id):
    if torch.cuda.is_available() and args.device_id >= 0:
        print('Using GPU')
        assert device_id < torch.cuda.device_count()
        return torch.device('cuda', device_id)
    else:
        print('Using CPU')
        return torch.device('cpu')


def compute_scores(model, dataset, device, return_ts=False):
    '''  
    Compute DPM scores with a given dataset and model
    Can chose to return the forecast means and standard deviations for every time step using return_ts bool
    '''
    
    print('Computing coseismic scores ...')
    # Initialize dataloader
    # Each batch is a column in space and all steps in time
    dataloader = DataLoader(deploy_dataset, batch_size=deploy_dataset.dataset_shape[1], shuffle=False)
    dataloader.dataset.deploy()

    dataset_shape = deploy_dataset.dataset_shape
    # DPM np.arrays to be saved
    dpm_means = np.zeros(dataset_shape, dtype=np.half)
    dpm_stds = np.zeros(dataset_shape, dtype=np.half)
    dpm_scores = np.zeros(dataset_shape, dtype=np.half)

    if return_ts:
        # Full prediction arrays - can be large 
        print('Outputting a prediction for every time step. This can be a lot of data for large coherence timeseries.')
        # Just save pre- and co-event data/predicitions
        sequence_length = deploy_dataset.event_index + 1
        all_means = np.zeros((*dataset_shape,sequence_length), dtype=np.half)
        all_stds = np.zeros((*dataset_shape,sequence_length), dtype=np.half)
        all_scores = np.zeros((*dataset_shape,sequence_length), dtype=np.half)
        # Duplicating the coherence for convenience in plotting 
        all_coherence_pred = np.zeros((*dataset_shape,sequence_length), dtype=np.half)

    # Iterating through deploy dataset in order (dataloader has shuffle=False)
    for batch_idx, (batch_preseismic, batch_coseismic) in enumerate(dataloader):
        assert isinstance(batch_coseismic, torch.Tensor)
        assert isinstance(batch_preseismic, torch.Tensor)
        assert batch_preseismic.size(1) == deploy_dataset.event_index
       
        batch_preseismic = batch_preseismic.to(device) # Preseismic coherence values
        batch_coseismic = batch_coseismic.to(device) # Coseismic coherence values 

        # Compute without keeping track of gradient information (uses less memory)
        with torch.no_grad():
            pred_means, pred_logvars = model(batch_preseismic,generate_dpm=True) # this calls model.forward()

            # Extract the full time series of forecasts
            if return_ts:
                pred_stds = torch.sqrt(torch.exp(pred_logvars))
                # Compute z-score for all timesteps  
                # Construct the coherence time series that we're actually trying to predict
                batch_pred = torch.cat((batch_preseismic,batch_coseismic.unsqueeze(-1)),dim=1) 
                score_all = (pred_means-batch_pred) / pred_stds

            # Extract predicted coseismic mean and standard deviation (std)
            mean_coseismic = pred_means[:,-1]
            logvar_coseismic = pred_logvars[:,-1]
            std_coseismic = torch.sqrt(torch.exp(logvar_coseismic))

            # Compute coseismic z-score
            score_coseismic = (mean_coseismic-batch_coseismic) / std_coseismic
        
        # Store the DPM values 
        dpm_means[batch_idx] = mean_coseismic.squeeze().cpu().numpy()
        dpm_stds[batch_idx] = std_coseismic.squeeze().cpu().numpy()
        dpm_scores[batch_idx] = score_coseismic.squeeze().cpu().numpy()

        if return_ts:
            # Store all values 
            all_stds[batch_idx] = pred_stds.squeeze().cpu().numpy() 
            all_means[batch_idx] = pred_means.squeeze().cpu().numpy() 
            all_scores[batch_idx] = score_all.squeeze().cpu().numpy() 
            all_coherence_pred[batch_idx] = batch_pred.squeeze().cpu().numpy() 

    result_dic = dict()
    result_dic['dpm_means'] = dpm_means # mean for co-event forecast
    result_dic['dpm_stds'] = dpm_stds # std. dev. for co-event forecast
    result_dic['dpm_scores'] = dpm_scores # z-score for co-event damage proxy map

    # Output values for every timestep 
    if return_ts:
        result_dic['all_means'] = all_means # mean of forecast
        result_dic['all_stds'] = all_stds # std. dev. of forecast
        result_dic['all_scores'] = all_scores # z score 
        result_dic['all_coherence_pred'] = all_coherence_pred # coherence values that we're trying to predict
        # This duplicates the coherence for convenience. Can remove if coherence datasets are large

    return result_dic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        required=True,
                        help='config JSON file')
    parser.add_argument('--config_path', type=str,
                        required=False, default='config_jsons',
                        help='path to config json')
    parser.add_argument('--dataset_json', type=str,
                        required=False, default='dataset.json',
                        help='location of dataset json')
    parser.add_argument('--save_dir', type=str,
                        required=False, default='saved',
                        help='save directory')
    parser.add_argument('--return_ts',type=bool,
                        required=False, default=False,
                        help='If true, saves mean, std and z-score for every timestep. Can rerun after training to output full ts with best model')
    parser.add_argument('--best_model', default=None, type=str,
                        required=False,
                        help='path to PyTorch state_dict from previous training to deploy on the data (skips training, optional)')
    parser.add_argument('-d', '--device_id', type=int,
                        required=False, default=-1,
                        help='device to use (cpu or gpu)')
    args = parser.parse_args()

    # Load config JSON and check for required fields
    config = load_json(os.path.join(os.getcwd(), args.config_path, args.config))
    assert 'train_dataset' in config
    assert 'deploy_dataset' in config
    if 'transform' not in config:
        print('Defaulting to logit squared transform on coherence')
        config['transform'] = 'logit_squared'
    assert config['transform'] in ['logit','logit_squared']

    # Create save directory
    trial_name = args.config[:-5] if args.config[-5:] == '.json' else args.config
    save_dir = os.path.join(os.getcwd(), args.save_dir, trial_name)
    coseismic_dpm_dir = os.path.join(save_dir,'coseismic_dpm')
    full_ts_dir = os.path.join(save_dir,'full_ts')
    
    # Check if we have output directories, create them if not
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))
    if not os.path.exists(coseismic_dpm_dir):
        os.makedirs(coseismic_dpm_dir)
    if args.return_ts:
        if not os.path.exists(full_ts_dir):
            os.makedirs(full_ts_dir)
        
    print('Save directory:\t {}'.format(save_dir))

    # Get device (i.e. GPU or CPU)
    device = get_device(args.device_id)

    # Load datasets
    dataset_json = load_json(os.path.join(os.getcwd(), args.dataset_json))
    train_dataset = Coherence_Timeseries(dataset_json[config['train_dataset']]) # for training RNN model
    deploy_dataset = Coherence_Timeseries(dataset_json[config['deploy_dataset']]) # for generating DPM 
    
    training_params = config['training_hyperparameters'] if 'training_hyperparameters' in config else DEFAULT_TRAINING_PARAMS
    
    # Sample and fix a random seed if not set in train config
    # See https://pytorch.org/docs/stable/notes/randomness.html
    if 'seed' not in training_params:
        print('Seeding randomly')
        training_params['seed'] = random.randint(0, 9999)
    else:
        print('Using supplied seed')
    seed = training_params['seed']
    
    # Preprocess training dataset
    train_dataset.remove_nans() # remove nans from data
    train_dataset.unbound(config['transform']) # apply transform to coherence values
    train_dataset.create_test_set(seed=seed) # create test set

    # Load model
    model_params = config['model_hyperparameters'] if 'model_hyperparameters' in config else DEFAULT_MODEL_PARAMS
    model_params['data_dim'] = train_dataset.data_dim
    model = RNN(model_params).to(device) # move model onto device

    # Check if best_model.pth already exists from previous training, or if we're passing a model via the command line

    # If we're passing a previous model from the command line, use that 
    if args.best_model is not None: 
        best_model_path = args.best_model
        print("Loading a pre-existing model from {}".format(args.best_model))

    # If we can't find a model from the command line, look for one saved in save_dir
    elif os.path.isfile(os.path.join(save_dir,'best_model.pth')):
        best_model_path = os.path.join(save_dir,'best_model.pth')
        print('Found a best model from previous training: {}'.format(best_model_path))

    # If we don't have a model yet, we'll have to train one
    elif not os.path.isfile(os.path.join(save_dir, 'best_model.pth')):
        print("No model supplied or found, starting training")
        # Train model
        # training_params = config['training_hyperparameters'] if 'training_hyperparameters' in config else DEFAULT_TRAINING_PARAMS
        training_params, summary, log = train_model(training_params, model, train_dataset, device, save_dir)

        # Save summary file
        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

        # Save log file
        with open(os.path.join(save_dir, 'log.json'), 'w') as f:
            json.dump(log, f, indent=4)

        # Save config JSON (for reproducability)
        config['model_hyperparameters'] = model_params
        config['training_hyperparameters'] = training_params
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        best_model_path = os.path.join(save_dir,'best_model.pth')

    else:
        # Shouldn't end up here!
        raise Exception('Problem loading or training a model')

    # Load best model for computing dpm
    # Lambda function to deal with GPU vs CPU storage
    state_dict = torch.load(best_model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model = model.eval()

    # Prepare deploy dataset
    # deploy_dataset.remove_nans() # Removing nans causes problems with the shape of the output data  
    # Not necessary for forecasting, although should mask them out if deploying in response scenario
    deploy_dataset.unbound(config['transform'])

    dpm_output_dic = compute_scores(model, deploy_dataset, device, args.return_ts)
    
    dpm_means = dpm_output_dic['dpm_means'] # mean for co-event forecast
    dpm_stds = dpm_output_dic['dpm_stds'] # std. dev. for co-event forecast
    dpm_scores = dpm_output_dic['dpm_scores'] # z-score for co-event damage proxy map

    # Output values for every timestep 
    if args.return_ts:
        all_means = dpm_output_dic['all_means'] # mean of forecast
        all_stds = dpm_output_dic['all_stds'] # std. dev. of forecast
        all_scores = dpm_output_dic['all_scores'] # z score 
        all_coherence_pred = dpm_output_dic['all_coherence_pred'] # coherence values that we're trying to predict

        # full_ts.npz duplicates the coherence time series
        np.savez(os.path.join(full_ts_dir,'full_ts.npz'),pred_means=all_means,pred_stds=all_stds,z_scores=all_scores,coherence=all_coherence_pred)

    # Save DPMS
    np.save(os.path.join(coseismic_dpm_dir, 'means.npy'), dpm_means)
    np.save(os.path.join(coseismic_dpm_dir, 'stds.npy'), dpm_stds)
    np.save(os.path.join(coseismic_dpm_dir, 'scores.npy'), dpm_scores)
