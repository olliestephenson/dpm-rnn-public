import os
import numpy as np
import random
import time
import torch
from torch.utils.data import DataLoader


def nll_gaussian(mean, logvar, value):
    """Compute negative log-likelihood of Gaussian."""
    assert mean.size() == logvar.size() == value.size()
    pi = torch.FloatTensor([np.pi]).to(value.device)
    nll_element = (value - mean).pow(2) / torch.exp(logvar) + logvar + torch.log(2*pi)
    return torch.sum(0.5*nll_element)


def run_epoch(dataloader, model, optimizer, device, train=True):
    """Perform one epoch of training by looping through the dataset once."""

    # Setting models and datasets into train/test mode
    if train:
        model = model.train()
        dataloader.dataset.train()
    else:
        model = model.eval()
        dataloader.dataset.test()

    nll_total = 0.0

    for batch_idx, (batch,_) in enumerate(dataloader):
        assert isinstance(batch, torch.Tensor)
        batch = batch.to(device) # batch is preseismic timeseries

        # Compute loss (negative log-likelihood)
        pred_means, pred_logvars = model(batch,generate_dpm=False) # this calls model.forward()
        nll_batch = nll_gaussian(pred_means, pred_logvars, batch) # compute negative log-likelihood of batch under predicted Gaussians

        if train:
            optimizer.zero_grad()
            nll_batch.backward() # compute gradients, stored in 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10) # clips norm of gradients to 10
            optimizer.step() # one step of gradient descent

        nll_total += nll_batch.item() # .item() remove gradient information, which is more memory efficient

    nll_average = nll_total / len(dataloader.dataset) # average NLL per sequence
    print('{}\t| nll: {:.6f}'.format('TRAIN' if train else 'TEST', nll_average))

    return nll_average


def train_model(train_config, model, dataset, device, save_dir):
    assert 'batch_size' in train_config and isinstance(train_config['batch_size'], int) and train_config['batch_size'] > 0
    assert 'num_epochs' in train_config and isinstance(train_config['num_epochs'], int) and train_config['num_epochs'] > 0
    assert 'learning_rate' in train_config and train_config['learning_rate'] > 0.0

    # Sample and fix a random seed if not set in train config (for reproducability)
    # See https://pytorch.org/docs/stable/notes/randomness.html
    if 'seed' not in train_config:
        train_config['seed'] = random.randint(0, 9999)
        print('Seeding randomly')
    else:
        print('Using supplied seed')

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) 
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    
    deterministic=True
    # Some issues with determinism 
    if deterministic: 
        print('Some issues with deterministic training, please be careful')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_deterministic(True) 
        torch.backends.cudnn.enabled = False
    
    # Also may need to consider the number of workers in dataloader

    # Initialize dataloaders
    # See documentation at https://pytorch.org/docs/stable/data.html
    dataloader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=True,worker_init_fn=np.random.seed(seed),num_workers=0) # set batch_size here

    # Initialize optimizer (default using ADAM optimizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate']) # set learning_rate here

    # Initialize bookkeeping variables
    log = []
    best_test_epoch = 0
    best_test_loss = float('inf')
    start_time = time.time()

    for epoch in range(train_config['num_epochs']):
        print('--- EPOCH [{}/{}] ---'.format(epoch+1, train_config['num_epochs']))

        epoch_start_time = time.time()
        train_loss = run_epoch(dataloader, model, optimizer, device, train=True)
        test_loss = run_epoch(dataloader, model, optimizer, device, train=False)
        epoch_time = time.time() - epoch_start_time
        print('{:.3f} seconds'.format(epoch_time))
        
        log.append({
            'epoch' : epoch+1,
            'train_loss' : train_loss,
            'test_loss' : test_loss,
            'time' : epoch_time
            })

        # Save model with best test loss
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_test_epoch = epoch+1
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print('BEST test loss')

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    print('--- DONE training model ---')

    # Compute summary statistics
    summary = {
        'total_time': round(time.time()-start_time, 3),
        'average_epoch_time': round((time.time()-start_time)/train_config['num_epochs'], 3),
        'best_test_loss': best_test_loss,
        'best_test_epoch': best_test_epoch,
        'num_trainable_params': model.num_parameters()
    }

    return train_config, summary, log
