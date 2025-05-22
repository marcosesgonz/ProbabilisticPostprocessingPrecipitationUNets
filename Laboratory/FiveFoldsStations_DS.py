import os 
import sys
main_path = os.path.abspath(os.path.join(__file__,'..','..'))
sys.path.append(main_path)
import torch 
import wandb
import numpy as np
from captum.attr import IntegratedGradients
from models.UNet import UNet
from utils import CRPS_mine
from collections import defaultdict
import utils as u
import pandas as pd
from Datasets import WRFdataset,MeteoCentersSplitter
import argparse


def fold_validation_ds(name_experim,project_ref = 'StationsFoldValidation', epochs = 40, optimizer_name = 'adamw',
                     batch_size = 32, lr = 1e-3,reduct_factor = 1):
    
    wandb.login() #Log in in wandb
    u.set_seed() #set the seed for reproducibility

    with wandb.init(project = project_ref, name = name_experim, dir = os.path.dirname(__file__)): 
        device = 'cpu'

        train_set = WRFdataset(data_subset = 'train')
        val_set = WRFdataset(data_subset = 'val')
        test_set = WRFdataset(data_subset = 'test')

        train_set.normalize_data()
        val_set.normalize_data(train_set.mean_og_data, train_set.std_og_data)
        test_set.normalize_data(train_set.mean_og_data, train_set.std_og_data)

        
        splitter = MeteoCentersSplitter(train_set, nfolds=5, stratify = False)
        #final_test_losses_fullch_list = [] 
        final_test_losses_list = [] 
        final_test_bias_losses_list = []
        final_test_mse_losses_list = []
        final_test_bs_losses_list = []
        patience = 10 #Early stopping patience in training process
        for fold_id in range(5):
            print(f'Starting fold {fold_id +1}:')
            mask_ctrs_tr, mask_ctrs_tst = splitter[fold_id]
            n_inp_channels = train_set.num_ensembles + 1
            net = UNet(n_inp_channels = n_inp_channels, n_outp_channels = 3, red_factor = reduct_factor, bottleneck = False)

            if fold_id == 0:
                train_size_ = len(train_set)*np.sum(mask_ctrs_tr)
                val_size_ = len(val_set)*np.sum(mask_ctrs_tst)
                test_size_ = len(test_set) * np.sum(mask_ctrs_tst)
                hyperparameters = dict(epochs = epochs,
                        train0_size = train_size_,
                        n_inp_channels = n_inp_channels,
                        val0_size = val_size_,
                        test0_size = test_size_,
                        batch_size = batch_size,
                        learning_rate = lr,
                        optimizer = optimizer_name,
                        device = device,
                        architecture='Unet')
                wandb.config.update(hyperparameters)
                print(net)
                print('\nNº instancias train/val/test fold 0:', train_size_,'/',val_size_,'/',test_size_)

            root_file = os.path.dirname(__file__)
            out_dir = os.path.join(root_file,'result_logs_5folds', f'{name_experim}')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                print(f'Logs saved in {out_dir}.')

            net.to(device)

            test_data_loader = torch.utils.data.DataLoader(
                    dataset = test_set, batch_size = batch_size,
                    shuffle = False, drop_last = False)

            train_data_loader = torch.utils.data.DataLoader(
                    dataset = train_set, batch_size = batch_size,
                    shuffle = True, drop_last = True)
            
            val_data_loader = torch.utils.data.DataLoader(
                    dataset = val_set, batch_size = batch_size,
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
            patience = 10
            patience_counter = 0
            for epoch in range(start_epoch, epochs):
                train_ctrs_loss = u.train_model(net = net, tr_loader = train_data_loader, data = train_set,mask_centers_tr = mask_ctrs_tr, weights = None, bin_edges = None,
                                    optimizer = optimizer, device = device, lr_scheduler = lr_scheduler)
                val_ctrs_loss, val_ctrs_val_bias, val_ctrs_val_mse, val_ctrs_val_bs = u.test_model(net = net,tst_loader = val_data_loader, data = val_set, mask_centers_tst = mask_ctrs_tst, device = device)

                wandb.log({
                    'train_ctrs_tr_fullch_loss': train_ctrs_loss,
                    'val_ctrs_val_fullch_loss': val_ctrs_loss,
                    'val_ctrs_val_fullch_bias': val_ctrs_val_bias,
                    'val_ctrs_val_fullch_mse': val_ctrs_val_mse,
                    'val_ctrs_val_fullch_bs': val_ctrs_val_bs,
                    f'fold {fold_id + 1} epoch': epoch,
                    'fold': fold_id + 1
                })

                if val_ctrs_loss < min_val_loss:
                    min_val_loss = val_ctrs_loss
                    wandb.run.summary[f'min_val_fullch_loss_fold{fold_id + 1}'] = min_val_loss
                    patience_counter = 0
                    checkpoint = {
                        'net': net.state_dict(),'optimizer': optimizer.state_dict(),
                        'epoch': epoch, 'min_val_fullch_loss': min_val_loss 
                        }
                    max_fullch_root = os.path.join(out_dir, f'fold{fold_id + 1}_fullch_checkpoint_max.pt')
                    torch.save(checkpoint, max_fullch_root)
                else:
                    patience_counter += 1
                
                print(f'epoch = {epoch}, train_loss ={train_ctrs_loss: .5f}, val_loss = {val_ctrs_loss: .5f}')    
                if (patience_counter >= patience) and epoch > 19:
                    print('Early stopping')
                    break

            max_checkpoint = torch.load(max_fullch_root, weights_only=True)
            net.load_state_dict(max_checkpoint['net'])
            
            print('Comprobación:')
            #Viendo el rendimiento del modelo en el conjunto de test y guardándolo en wandb
            test_df  = u.test_model_df(net = net,tst_loader = test_data_loader, data = test_set, mask_centers_tst = mask_ctrs_tst, device = device)
            test_df.to_csv(os.path.join(out_dir,f'test_results_fullch_fold{fold_id + 1}.csv'))
            test_loss = test_df['crps'].mean()
            test_bs_loss = test_df['bs0.1'].mean()
            test_mse_loss = test_df['mse'].mean()
            test_bias_loss = test_df['bias'].mean()
            print('------------------------------')
            print(f'Test metrics of FullCh UNet: fold {fold_id + 1}. loss(crps): {test_loss} brier_score: {test_bs_loss} bias: {test_bias_loss}')
            print('------------------------------')
            
            print('Calculation of attributions:')
            val_data = WRFdataset(data_subset='val')
            assert len(val_set) == len(val_data) #Normalized vs NotNormalized
            dataset = defaultdict(list)
            crps_loss = u.CRPS_CSGDloss()
            for idx in range(len(val_data)):
                X,y = val_data[idx]
                crps_ensemble = CRPS_mine()
                no_na_mask = ~np.isnan(y) & mask_ctrs_tr
                x_y_posit = val_data.meteo_centers_info['x_y_d04'][no_na_mask]
                x_positions, y_postions = x_y_posit[...,0], x_y_posit[...,1]
                ensemble = X[:-1, y_postions - val_data.crop_y[0], x_positions - val_data.crop_x[0]]
                targets = y[no_na_mask]
                crps_ensemb = crps_ensemble.compute(X_f = ensemble, X_o = targets)
                dataset['CRPS_Ensemble'].append(crps_ensemb)
                
                X_norm, _ = val_set[idx]
                with torch.no_grad():
                    outp = net(torch.tensor(X_norm).unsqueeze(0))
                    outp_centers = outp[:,:, y_postions - val_data.crop_y[0], x_positions - val_data.crop_x[0]]
                    targets_batched = torch.tensor(targets).unsqueeze(0)
                    crps_unet = crps_loss(outp_centers, targets_batched )

                dataset['CRPS_UnetAll'].append(crps_unet)
                dataset['SumPrecWeatStat'].append(np.sum(targets))
                dataset['SumPrecWRF'].append(np.sum(np.mean(ensemble, axis = 0)))

            crps_dataset = pd.DataFrame(data = dataset)
            print(crps_dataset.columns)
            crps_dif =  crps_dataset['CRPS_Ensemble'] - crps_dataset['CRPS_UnetAll']
            crps_dif_sorted = crps_dif.sort_values(ascending=False)

            selected_indices = []
            # Iterar sobre los índices de los valores ordenados
            for idx in crps_dif_sorted.index:
                # Verificar que la distancia con todos los índices seleccionados es mayor a 23
                if all(abs(idx - prev_idx) > 23 for prev_idx in selected_indices):
                    selected_indices.append(idx)
                    
                # Parar el bucle cuando tengamos 3 índices seleccionados
                if len(selected_indices) == 3:
                    break
            
            print(f'Selected indices: {selected_indices}')
            print(crps_dataset.iloc[selected_indices])
            print('Starting sensitivities calculation with batch of rainy data:')
            input_maps = torch.tensor([val_set[idx][0] for idx in selected_indices])
            H, W = input_maps.shape[2], input_maps.shape[3]
            global_attr_channel_0 = torch.zeros(input_maps.shape[1])
            global_attr_channel_1 = torch.zeros(input_maps.shape[1])
            global_attr_channel_2 = torch.zeros(input_maps.shape[1])
            
            x_y_posit = train_set.meteo_centers_info['x_y_d04'][mask_ctrs_tr]
            positions = [(x - train_set.crop_x[0],y - train_set.crop_y[0]) for (x,y) in x_y_posit]
            net.eval()
            ig = IntegratedGradients(net)
            for idx,(x,y) in enumerate(positions):
                target_0 = (0, y, x) 
                attr_channel_0_batch = ig.attribute(input_maps, target= target_0, return_convergence_delta=False)
                global_attr_channel_0 += attr_channel_0_batch.sum(dim= (0,2,3), keepdim=False) #Solo se queda la dimensión de nºcanales de entrada
                # Calcular atribuciones para el canal de salida 1
                target_1 = (1, y, x) 
                attr_channel_1_batch = ig.attribute(input_maps, target=target_1, return_convergence_delta=False)
                global_attr_channel_1 += attr_channel_1_batch.sum(dim= (0,2,3), keepdim=False)
                # Calcular atribuciones para el canal de salida 2
                target_2 = (2, y, x) 
                attr_channel_2_batch = ig.attribute(input_maps, target=target_2, return_convergence_delta=False)
                global_attr_channel_2 += attr_channel_2_batch.sum(dim= (0,2,3), keepdim=False)
                print(f'{idx}/{len(positions)}.',x,y,' Done')

            np.savez(os.path.join(out_dir,f'fold{fold_id + 1}_sensitivities_batch_LowerCRPSVal'), ch0 = global_attr_channel_0.detach().numpy(),
                ch1 = global_attr_channel_1.detach().numpy(), ch2 = global_attr_channel_2.detach().numpy())
            attr_total = np.abs(global_attr_channel_0.detach().numpy()) + np.abs(global_attr_channel_1.detach().numpy()) + np.abs(global_attr_channel_2.detach().numpy())
            #attr_total_norm = attr_total / np.sum(attr_total)
            top_11_indices = np.argsort(attr_total)[-11:]
            mask_ds = [False if i not in top_11_indices else True for i in range(len(attr_total))]

            train_set_ds = WRFdataset(data_subset = 'train')
            val_set_ds = WRFdataset(data_subset = 'val')
            test_set_ds = WRFdataset(data_subset = 'test')

            train_set_ds.normalize_data(train_set.mean_og_data, train_set.std_og_data)
            val_set_ds.normalize_data(train_set.mean_og_data, train_set.std_og_data)
            test_set_ds.normalize_data(train_set.mean_og_data, train_set.std_og_data)

            train_set_ds.apply_feature_selec(feat_selec_mask = mask_ds)
            val_set_ds.apply_feature_selec(feat_selec_mask = mask_ds) 
            test_set_ds.apply_feature_selec(feat_selec_mask = mask_ds)
            n_inp_channels_ds = 11

            net = UNet(n_inp_channels = n_inp_channels_ds, n_outp_channels = 3, red_factor = reduct_factor, bottleneck = False)

            train_ds_data_loader = torch.utils.data.DataLoader(
                    dataset = train_set_ds, batch_size = batch_size,
                    shuffle = True, drop_last = True)
            
            val_ds_data_loader = torch.utils.data.DataLoader(
                    dataset = val_set_ds, batch_size = batch_size,
                    shuffle = False, drop_last = False)
            
            test_ds_data_loader = torch.utils.data.DataLoader(
                    dataset = test_set_ds, batch_size = batch_size,
                    shuffle = False, drop_last = False)
            
            optimizer = torch.optim.AdamW(net.parameters(), lr = lr)
            lr_scheduler = None
            min_val_loss = 1e4
            start_epoch = 0
            patience_counter = 0
            for epoch in range(start_epoch, epochs):
                train_ctrs_loss = u.train_model(net = net, tr_loader = train_ds_data_loader, data = train_set_ds,mask_centers_tr = mask_ctrs_tr, weights = None, bin_edges = None,
                                    optimizer = optimizer, device = device, lr_scheduler = lr_scheduler)
                val_ctrs_loss, val_ctrs_val_bias, val_ctrs_val_mse, val_ctrs_val_bs = u.test_model(net = net,tst_loader = val_ds_data_loader, data = val_set_ds, mask_centers_tst = mask_ctrs_tst, device = device)

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

            max_checkpoint = torch.load(os.path.join(out_dir, f'fold{fold_id + 1}_checkpoint_max.pt'), weights_only=True)
            net.load_state_dict(max_checkpoint['net'])

            test_df = u.test_model_df(net = net,tst_loader = test_ds_data_loader, data = test_set_ds, mask_centers_tst = mask_ctrs_tst, device = device)
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
            final_test_bias_losses_list.append(test_bias_loss)
            final_test_bs_losses_list.append(test_bs_loss)
            final_test_mse_losses_list.append(test_mse_loss)
        
        mean_test_crps_loss = np.mean(final_test_losses_list)
        mean_test_bias_loss = np.mean(final_test_bias_losses_list)
        mean_test_mse_loss = np.mean(final_test_mse_losses_list)
        mean_test_bs_loss = np.mean(final_test_bs_losses_list)
        wandb.run.summary['Mean_CRPS_Loss'] = mean_test_crps_loss
        wandb.run.summary['Mean_BS_Loss'] = mean_test_bs_loss
        wandb.run.summary['Mean_BIAS_Loss'] = mean_test_bias_loss
        wandb.run.summary['Mean_MSE_Loss'] = mean_test_mse_loss
        print('-----------------------------------------')
        print(f'Test losses per fold: {final_test_losses_list}')

def parse_args():
    parser = argparse.ArgumentParser(description="Execute 5 fold stations UNet experiment using DownSelection")
    parser.add_argument('--project_ref', type=str, default='StationsFoldValidation', help='Project reference for wandb')
    parser.add_argument('--name_experim', type=str, required = True, help='Experiment name for wandb')
    parser.add_argument('--epochs', type=int, default = 40, help='Number of epochs (default: 40)')
    parser.add_argument('--batch_size', type=int, default = 32, help='Batch size (default: 32)')
    parser.add_argument('--optimizer', type=str, default = 'adamw', help='Optimizer used during training (default: adamW)')
    parser.add_argument('--lr', type=float, default = 0.01, help='Learning rate (default: 0.01)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    fold_validation_ds(project_ref = args.project_ref, name_experim = args.name_experim, epochs = args.epochs,
                    batch_size = args.batch_size, optimizer_name = args.optimizer, lr = args.lr)