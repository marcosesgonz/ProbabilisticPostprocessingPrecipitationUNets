import os 
import sys
main_path = os.path.abspath(os.path.join(__file__,'..','..'))
sys.path.append(main_path)
import torch 
import wandb
import numpy as np
from models.UNet import UNet
import utils as u
from Datasets import WRFdataset,MeteoCentersSplitter
import argparse

def fold_validation(name_experim,project_ref = 'StationsFoldValidation', epochs = 40, optimizer_name = 'adamw', normalize_data = True, pca_data = True, feat_selec_data = False,
                     batch_size = 32, lr = 1e-3, reduct_factor = 1):
    
    wandb.login() #Log in in wandb
    u.set_seed() #set the seed for reproducibility

    with wandb.init(project = project_ref, name = name_experim, dir = os.path.dirname(__file__)): 
        device = 'cpu'
        
        train_set = WRFdataset(data_subset = 'train')
        val_set = WRFdataset(data_subset = 'val')
        test_set = WRFdataset(data_subset = 'test')
        if normalize_data:
            train_set.normalize_data()
            val_set.normalize_data(train_set.mean_og_data, train_set.std_og_data)
            test_set.normalize_data(train_set.mean_og_data, train_set.std_og_data)
        if pca_data:
            print('Applying PCA')
            train_set.apply_PCA(ncomponents=10)
            val_set.apply_PCA(eigvec_transp = train_set.W_t_PCA)
            test_set.apply_PCA(eigvec_transp = train_set.W_t_PCA)
            n_inp_channels = len(train_set.eigv_PCA) + 1
        elif feat_selec_data:
            print('Applying Feat.Selec')
            train_set.apply_feature_selec(11)
            val_set.apply_feature_selec(feat_selec_mask = train_set.feat_selec_mask)   #train_set.feat_selec_mask)
            test_set.apply_feature_selec(feat_selec_mask = train_set.feat_selec_mask)  #train_set.feat_selec_mask)
            n_inp_channels = np.sum(train_set.feat_selec_mask)
        else:
            n_inp_channels = train_set.num_ensembles + 1

        splitter = MeteoCentersSplitter(train_set, nfolds=5, stratify = False)
        final_test_losses_list = [] 
        for fold_id in range(5):
            print(f'Starting fold {fold_id +1}:')
            mask_ctrs_tr, mask_ctrs_tst = splitter[fold_id]

            net = UNet(n_inp_channels = n_inp_channels, n_outp_channels = 3, red_factor = reduct_factor, bottleneck = False)

            if fold_id == 0:
                train_size_ = len(train_set)*np.sum(mask_ctrs_tr)
                val_size_ = len(val_set)*np.sum(mask_ctrs_tst)
                test_size_ = len(test_set) * np.sum(mask_ctrs_tst)
                n_params = u.num_trainable_params(net)
                hyperparameters = dict(epochs = epochs,
                        train0_size = train_size_,
                        n_inp_channels = n_inp_channels,
                        val0_size = val_size_,
                        test0_size = test_size_,
                        batch_size = batch_size,
                        learning_rate = lr,
                        optimizer = optimizer_name,
                        device = device,
                        architecture='Unet',
                        n_trainable_params = '%.6e'%n_params)
                wandb.config.update(hyperparameters)
                print(net)
                print('Number of trainable paramenters:  %.6e'%n_params)
                print('\nNº instancias train/val/test fold 0:', train_size_,'/',val_size_,'/',test_size_)


            net.to(device)

            train_data_loader = torch.utils.data.DataLoader(
                    dataset = train_set, batch_size = batch_size,
                    shuffle = True, drop_last = True)
            
            val_data_loader = torch.utils.data.DataLoader(
                    dataset = val_set, batch_size = batch_size,
                    shuffle = False, drop_last = False)
            
            test_data_loader = torch.utils.data.DataLoader(
                    dataset = test_set, batch_size = batch_size,
                    shuffle = False, drop_last = False)
            
            if optimizer_name == 'sgd':
                optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9) 
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min= 1e-5)
            elif optimizer_name == 'adam':
                optimizer = torch.optim.Adam(net.parameters(), lr = lr)
                lr_scheduler = None
            elif optimizer_name == 'adamw':
                optimizer = torch.optim.AdamW(net.parameters(), lr = lr)
                lr_scheduler = None
            else:
                raise NotImplementedError
            
            min_val_loss = 1e4
            start_epoch = 0
            root_file = os.path.dirname(__file__)
            out_dir = os.path.join(root_file,'result_logs_5folds', f'b{batch_size}_lr{lr}_{name_experim}')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                print(f'Logs saved in {out_dir}.')
            patience = 10
            patience_counter = 0
            for epoch in range(start_epoch, epochs):
                train_ctrs_loss = u.train_model(net = net, tr_loader = train_data_loader, data = train_set,mask_centers_tr = mask_ctrs_tr, weights = None, bin_edges = None,
                                    optimizer = optimizer, device = device, lr_scheduler = lr_scheduler)
                val_ctrs_loss, val_ctrs_val_bias, val_ctrs_val_mse, val_ctrs_val_bs = u.test_model(net = net,tst_loader = val_data_loader, data = val_set, mask_centers_tst = mask_ctrs_tst, device = device)

                wandb.log({
                    'train_ctrs_tr_loss': train_ctrs_loss,
                    'val_ctrs_val_loss': val_ctrs_loss,
                    'val_ctrs_val_bias': val_ctrs_val_bias,
                    'val_ctrs_val_mse': val_ctrs_val_mse,
                    'val_ctrs_val_bs': val_ctrs_val_bs,
                    f'fold {fold_id + 1} epoch': epoch,
                    'fold': fold_id + 1
                })

                if val_ctrs_loss < min_val_loss:
                    min_val_loss = val_ctrs_loss
                    wandb.run.summary[f'min_val_loss_fold{fold_id + 1}'] = min_val_loss
                    patience_counter = 0
                    checkpoint = {
                        'net': net.state_dict(),'optimizer': optimizer.state_dict(),
                        'epoch': epoch, 'min_val_loss': min_val_loss 
                        }
                    torch.save(checkpoint, os.path.join(out_dir, f'fold{fold_id + 1}_checkpoint_max.pt'))
                else:
                    patience_counter += 1
                
                print(f'epoch = {epoch}, train_loss ={train_ctrs_loss: .5f}, val_loss = {val_ctrs_loss: .5f}')    
                if (patience_counter >= patience) and epoch > 19:
                    print('Early stopping')
                    break

            max_checkpoint = torch.load(os.path.join(out_dir, f'fold{fold_id + 1}_checkpoint_max.pt'))
            net.load_state_dict(max_checkpoint['net'])

            print('Comprobación:')
            val_ctrs_loss, val_bias_loss, val_mse, val_bs_loss = u.test_model(net = net,tst_loader = val_data_loader, data = val_set, mask_centers_tst = mask_ctrs_tst, device = device)
            print(f'El val loss recargando el mejor modelo es {val_ctrs_loss} y se tiene guardado el valor {min_val_loss}')
            #Viendo el rendimiento del modelo en el conjunto de test y guardándolo en wandb
            test_df  = u.test_model_df(net = net,tst_loader = test_data_loader, data = test_set, mask_centers_tst = mask_ctrs_tst, device = device)
            test_df.to_csv(os.path.join(out_dir,f'test_results_fold{fold_id + 1}.csv'))
            test_loss = test_df['crps'].mean()
            test_bs_loss = test_df['bs0.1'].mean()
            test_mse_loss = test_df['mse'].mean()
            test_bias_loss = test_df['bias'].mean()
            print('------------------------------')
            print(f'Final test results fold {fold_id + 1}. loss(crps): {test_loss} brier_score: {test_bs_loss} bias: {test_bias_loss}')
            print('------------------------------')
            wandb.run.summary[f'test_loss_fold{fold_id + 1}'] = test_loss
            wandb.run.summary[f'test_bs_loss_fold{fold_id + 1}'] = test_bs_loss
            wandb.run.summary[f'test_bias_loss_fold{fold_id + 1}'] = test_bias_loss
            wandb.run.summary[f'test_mse_loss_fold{fold_id + 1}'] = test_mse_loss
            final_test_losses_list.append(test_loss)

        mean_test_loss = np.mean(final_test_losses_list)
        wandb.run.summary['Mean_Test_Loss'] = mean_test_loss
        print('-----------------------------------------')
        print(f'Test losses per fold: {final_test_losses_list}')
        print(f'Final test loss (average of all folds): {mean_test_loss}')    

def parse_args():
    parser = argparse.ArgumentParser(description="Execute 5 fold stations UNet experiment")
    parser.add_argument('--project_ref', type=str, default='StationsFoldValidation', help='Project reference for wandb')
    parser.add_argument('--name_experim', type=str, required = True, help='Experiment name for wandb')
    parser.add_argument('--epochs', type=int, default = 40, help='Number of epochs (default: 40)')
    parser.add_argument('--batch_size', type=int, default = 32, help='Batch size (default: 32)')
    parser.add_argument('--optimizer', type=str, default = 'adamw', help='Optimizer used during training (default: adamW)')
    parser.add_argument('--lr', type=float, default = 0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--normalize', action = 'store_true', help = 'Normalize data')
    parser.add_argument('--PCA', action = 'store_true', help = 'Apply PCA along ensemble precipitations')
    parser.add_argument('--FeatSelec', action = 'store_true', help = 'Apply Feature selection')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    fold_validation(project_ref = args.project_ref, name_experim = args.name_experim, epochs = args.epochs,
                    batch_size = args.batch_size, optimizer_name = args.optimizer, lr = args.lr,
                    normalize_data = args.normalize, pca_data = args.PCA, feat_selec_data = args.FeatSelec)