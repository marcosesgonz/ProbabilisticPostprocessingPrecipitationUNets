import os 
import sys
main_path = os.path.abspath(os.path.join(__file__,'..','..'))
sys.path.append(main_path)
import torch
#import wandb
import random
import numpy as np
import os
from models.UNet import UNet
import utils as u
from Datasets import WRFdataset, desired_configs
import argparse


def execute_model(project_ref, name_experim,  epochs = 35, optimizer = 'sgd', use_weights = False, normalize_data = True, pca_data = True, feat_selec_data = False,
                     down_selec_data = False, batch_size = 8, lr = 0.01, run_id = None, reduct_factor = 1):
    
    wandb.login() #Login in wandb 
    u.set_seed() #set the seed for reproducibility

    # Necessary to resume past training
    if run_id is not None:
        name_experim = None
        checkpoint_file = input('Checkpoint file(.kpath) with desired trained model:')
        resume_ = 'must'
    else:
        resume_ = None
        
    with wandb.init(project = project_ref, name = name_experim, dir = os.path.dirname(__file__),
                    id = run_id,resume = resume_): 
        device = ("cpu")

        #Load train, validation and test sets
        train_set = WRFdataset(data_subset = 'train', group_configs = desired_configs)
        val_set = WRFdataset(data_subset = 'val', group_configs =  desired_configs)
        test_set = WRFdataset(data_subset = 'test', group_configs =  desired_configs)

        # Normalize data (subtract mean and divide by standard deviation)
        if normalize_data:
            train_set.normalize_data()
            val_set.normalize_data(train_set.mean_og_data, train_set.std_og_data)
            test_set.normalize_data(train_set.mean_og_data, train_set.std_og_data)
        # Apply PCA 
        if pca_data:
            print('Applying PCA')
            train_set.apply_PCA(ncomponents=10) # keeping the 10 PCA components with the highest weight.
            val_set.apply_PCA(eigvec_transp = train_set.W_t_PCA)
            test_set.apply_PCA(eigvec_transp = train_set.W_t_PCA)
            n_inp_channels = len(train_set.eigv_PCA) + 1 # It is added the height terrain (HGT) map.
        # Apply least correlated feature selection 
        elif feat_selec_data:
            print('Applying Feat.Selec')
            train_set.apply_feature_selec(10) # keeping 10 least correlated variables + HGT.
            val_set.apply_feature_selec(feat_selec_mask = train_set.feat_selec_mask)  
            test_set.apply_feature_selec(feat_selec_mask = train_set.feat_selec_mask)  
            n_inp_channels = np.sum(train_set.feat_selec_mask)
        # Apply down-selection (DS) feature selection strategy 
        elif down_selec_data:
            print('Applying Down Selection (check preloaded mask being used, if necessary)')
            #Edit this mask with ur own down selection mask (or your own feature selection mask). This default mask is obtained as described in the down selection method of the study.
            feat_selec_mask_ds_XLowerCRPSVal = [False, True, True, False, False, True, False, True, False, False, False, False, True, False, False, True, False, False, False, False, True, False, True, True, True]#, True]
            train_set.apply_feature_selec(feat_selec_mask = feat_selec_mask_ds_XLowerCRPSVal)
            val_set.apply_feature_selec(feat_selec_mask = feat_selec_mask_ds_XLowerCRPSVal)  
            test_set.apply_feature_selec(feat_selec_mask = feat_selec_mask_ds_XLowerCRPSVal)  
            n_inp_channels = np.sum(train_set.feat_selec_mask)
        else:
            n_inp_channels = train_set.num_ensembles + 1 # All ensemble members plus HGT map
            
        train_size_,val_size_, test_size_ = len(train_set),len(val_set), len(test_set)
        # Load UNet arquitecture
        net = UNet(n_inp_channels = n_inp_channels, n_outp_channels = 3, red_factor = reduct_factor)
        n_params = u.num_trainable_params(net)

        if run_id is None:
            hyperparameters = dict(epochs = epochs,
                train_size = train_size_,
                n_inp_channels = n_inp_channels,
                val_size = val_size_,
                test_size = test_size_,
                batch_size = batch_size,
                learning_rate = lr,
                optimizer = optimizer,
                device = device,
                architecture='Unet',
                n_trainable_params = '%.6e'%n_params)
            wandb.config.update(hyperparameters)

        # Weigthed labels trial (it isnt used in paper)
        if use_weights:
            y_weights, y_bin_edges = train_set.compute_frequency_weights()
            print('Weights of “labels” according to recorded rainfall (and their bins):\n',y_weights, y_bin_edges)
        else:
            y_weights, y_bin_edges = None, None
            print('NO weights are used for “labels”.')

        print(net)
        net.to(device) # Warning: Loss function isn't implemented in CUDA and MPS devices

        # Dataloader pytorch objects for training and testing 
        train_data_loader = torch.utils.data.DataLoader(
            dataset = train_set, batch_size = batch_size,
            shuffle = True, drop_last = True
            #pin_memory=True,
            #num_workers = 2
        )
        val_data_loader = torch.utils.data.DataLoader(
            dataset=val_set, batch_size=batch_size,
            shuffle=False, drop_last=False
            #pin_memory=True,
            #num_workers = 2
        )
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=batch_size,
            shuffle=False, drop_last=False
            #pin_memory=True,
            #num_workers = 2
        )
        print('Using %s as device'% device)
        print('Number of trainable paramenters:  %.6e'%n_params)
        print('\nNº instancias train/val/test:', train_size_,'/',val_size_,'/', test_size_)

        # Optimizer used to refresh UNet weights during training
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9) 
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min= 1e-5)
            #lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=lr,
            #                                        step_size_up=5,step_size_down = 5, mode='triangular2')
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr = lr)
            lr_scheduler = None
        elif optimizer == 'adamw':
            optimizer = torch.optim.AdamW(net.parameters(), lr = lr)
            lr_scheduler = None
        else:
            raise NotImplementedError
        
        min_val_loss = 1e4
        start_epoch = 0
        # Loading data if a run being resumed
        if run_id is not None:
            dicts = torch.load(checkpoint_file)
            #lr_scheduler.load_state_dict(dicts['lr_scheduler'])
            optimizer.load_state_dict(dicts['optimizer'])
            net.load_state_dict(dicts['net'])
            min_val_loss = dicts['min_val_loss']
            start_epoch = dicts['epoch'] + 1
            print('Trained model succesfully loaded')
        
        # Folder to save logs
        root_file = os.path.dirname(__file__)
        out_dir = os.path.join(root_file,'result_logs', f'b{batch_size}_lr{lr}_{name_experim}')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            print(f'Logs saved in {out_dir}.')
        
        patience = 10 # Number of waiting epochs with no improvement in results until execution stops
        patience_counter = 0
        for epoch in range(start_epoch, epochs):
            train_loss = u.train_model_all_stations(net = net, tr_loader = train_data_loader, data = train_set, weights = y_weights, bin_edges = y_bin_edges,
                                                optimizer = optimizer, device = device, lr_scheduler = lr_scheduler)
            val_loss, val_mae_loss,val_mse_loss, val_bs_loss = u.test_model(net = net,tst_loader = val_data_loader, data = val_set, device = device)

            # Registering values in wandb
            wandb.log({
                'train_crps_loss': train_loss,
                'val_crps_loss': val_loss,
                'val_bs_loss': val_bs_loss,
                'val_mse_loss': val_mse_loss,
                'val_mae_loss': val_mae_loss,
            }, step=epoch)

            checkpoint = {
                'net': net.state_dict(),'optimizer': optimizer.state_dict(),
                'epoch': epoch, 'min_val_loss': min_val_loss if (min_val_loss < val_loss) else val_loss
            }
            if lr_scheduler is not None:
                checkpoint['lr_scheduler'] = lr_scheduler.state_dict()

            if (val_loss < min_val_loss):
                min_val_loss = val_loss
                wandb.run.summary['min_val_loss'] = min_val_loss
                patience_counter = 0
                torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pt'))
            else:
                patience_counter += 1

            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pt'))

            print(f'epoch = {epoch}, train_loss ={train_loss: .5f}, val_loss ={val_loss: .5f}')    
            
            if (patience_counter >= patience) and epoch > 19:
                print('Early stopping')
                break
            
        #Loading the model with best results in validation set
        max_checkpoint = torch.load(os.path.join(out_dir, 'checkpoint_max.pt'))
        net.load_state_dict(max_checkpoint['net'])

        print('Checking:')
        val_loss, val_mae_loss,val_mse_loss, val_bs_loss = u.test_model(net = net,tst_loader = val_data_loader, data = val_set, device = device)
        print(f'Val loss reloading best model is: {val_loss}. The value saved is: {min_val_loss}')
        
        #Showing results in test sets and registering it in wandba
        test_loss,test_mae_loss, test_mse_loss, test_bs_loss = u.test_model_all_stations(net = net,tst_loader = test_data_loader, data = test_set, device = device)
        print('------------------------------')
        print(f'Final test results. loss(crps): {test_loss} brier_score: {test_bs_loss} MAE: {test_mae_loss}')
        print('------------------------------')
        wandb.run.summary['test_loss'] = test_loss
        wandb.run.summary['test_bs_loss'] = test_bs_loss
        wandb.run.summary['test_mse_loss'] = test_mse_loss
        wandb.run.summary['test_mae_loss'] = test_mae_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Execute WRF model training')
    parser.add_argument('--project_ref', type=str, default='WRF-IA', help='Project reference for wandb')
    parser.add_argument('--name_experim', type=str, required = True, help='Experiment name for wandb')
    parser.add_argument('--epochs', type=int, default = 40, help='Number of epochs (default: 40)')
    parser.add_argument('--batch_size', type=int, default = 32, help='Batch size (default: 32)')
    parser.add_argument('--optimizer', type=str, default = 'adamw', help='Optimizer used during training (default: adamW)')
    parser.add_argument('--lr', type=float, default = 0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--run_id', type=str, default=None, help='Wandb run ID to resume')
    parser.add_argument('--use_weights', action = 'store_true', help = 'Use "label" weighting (default: false)')
    parser.add_argument('--normalize', action = 'store_true', help = 'Normalize data')
    parser.add_argument('--PCA', action = 'store_true', help = 'Apply PCA along ensemble precipitations')
    parser.add_argument('--FeatSelec', action = 'store_true', help = 'Apply Feature selection')
    parser.add_argument('--DownSelec', action = 'store_true', help = 'Apply Down selection')
    parser.add_argument('--RedFact', type=int, default = 1, help = 'Reduction factor(int) to Unet number of filters (default: 1, no reduction)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    execute_model(project_ref = args.project_ref, name_experim = args.name_experim, epochs = args.epochs, normalize_data = args.normalize, down_selec_data = args.DownSelec,
                  feat_selec_data = args.FeatSelec, pca_data = args.PCA, optimizer = args.optimizer, batch_size = args.batch_size, lr = args.lr,
                  use_weights= args.use_weights,run_id = args.run_id, reduct_factor = args.RedFact)