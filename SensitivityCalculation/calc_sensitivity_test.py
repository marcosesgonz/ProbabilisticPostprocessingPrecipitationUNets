"""
This script is used to obtain sensitivity results for all models
"""

import sys
import os
main_path = os.path.abspath(os.path.join(__file__,'..','..'))
sys.path.append(main_path)
import torch
from collections import defaultdict
from captum.attr import IntegratedGradients
from Datasets import WRFdataset, desired_configs
import numpy as np
from models.UNet import UNet
from prob_scores import CRPS_mine
from utils import test_model_all_stations, CRPS_CSGDloss
import matplotlib.pyplot as plt
import pandas as pd

train_set = WRFdataset(data_subset = 'train', group_configs = desired_configs , station_split = False)
train_set.normalize_data()
mean_og_data, std_og_data = train_set.mean_og_data, train_set.std_og_data
data_normalized = WRFdataset(data_subset='test',station_split=False, group_configs = desired_configs)
data_normalized.normalize_data(train_set.mean_og_data, train_set.std_og_data)

unet_model = UNet(n_inp_channels=26,n_outp_channels=3,bottleneck = False)
model_data = torch.load('/disk/barbusano/barbusano3/Laboratory/result_logs/b32_lr0.001_b32_lr1e-3_AdamW_Norm_AllChannels_SqrtStdReg/checkpoint_max.pt')
unet_model.load_state_dict(model_data['net'])
crps_loss = CRPS_CSGDloss()
unet_model.eval()

data = WRFdataset(data_subset='test',station_split=False, group_configs = desired_configs)
dataset = defaultdict(list)
for idx in range(len(data)):
    X,y = data[idx]
    crps_ensemble = CRPS_mine()
    no_na_mask = ~np.isnan(y)
    x_y_posit = data.meteo_centers_info['x_y_d04'][no_na_mask]
    x_positions, y_postions = x_y_posit[...,0], x_y_posit[...,1]
    dataset['SumPrecWRF'].append(np.sum(X))
    ensemble = X[:-1, y_postions - data.crop_y[0], x_positions - data.crop_x[0]]
    targets = y[no_na_mask]
    crps_ensemb = crps_ensemble.compute(X_f = ensemble, X_o = targets)
    dataset['CRPS_Ensemble'].append(crps_ensemb)
    
    X_norm, y = data_normalized[idx]
    with torch.no_grad():
        outp = unet_model(torch.tensor(X_norm).unsqueeze(0))
        outp_centers = outp[:,:, y_postions - data.crop_y[0], x_positions - data.crop_x[0]]
        targets_batched = torch.tensor(targets).unsqueeze(0)
        crps_unet = crps_loss(outp_centers, targets_batched )

    dataset['CRPS_UnetAll'].append(crps_unet)
    dataset['SumPrecWeatStat'].append(np.sum(targets))
    dataset['SumPrecWRFStat'].append(np.sum(np.mean(ensemble, axis = 0)))

crps_dataset = pd.DataFrame(data = dataset)
print(crps_dataset.columns)

ds_sorted = crps_dataset.sort_values(by='SumPrecWRF',ascending=False)
print(ds_sorted)
selected_instances = ds_sorted.iloc[0:3,:]
selected_indices = list(ds_sorted.index[0:3])
print(f' Instance selected:{selected_instances}\n Instance idx: {selected_indices} ')

#------------------------------------------UNet-All-------------------------------------------
print(np.nansum(data[selected_indices[0]][1]),crps_dataset[['SumPrecWeatStat']].iloc[selected_indices[0]].values)

unet_model = UNet(n_inp_channels=26,n_outp_channels=3,bottleneck = False)
model_data = torch.load('/disk/barbusano/barbusano3/Laboratory/result_logs/b32_lr0.001_b32_lr1e-3_AdamW_Norm_AllChannels_SqrtStdReg/checkpoint_max.pt')
unet_model.load_state_dict(model_data['net'])

crps_loss = CRPS_CSGDloss()
unet_model.eval()
batch_X = []
batch_y = []
with torch.no_grad():
    for idx in selected_indices:
        X, y = data_normalized[idx]
        no_na_mask = ~np.isnan(y)
        x_y_posit = data_normalized.meteo_centers_info['x_y_d04'][no_na_mask]
        x_positions, y_postions = x_y_posit[...,0], x_y_posit[...,1]
        targets = y[no_na_mask]

        outp = unet_model(torch.tensor(X).unsqueeze(0))
        outp_centers = outp[:,:, y_postions - data_normalized.crop_y[0], x_positions - data_normalized.crop_x[0]]
        targets = torch.tensor(targets).unsqueeze(0)
        print(outp_centers.shape, targets.shape)
        crps = crps_loss(outp_centers, targets )
        print(f'crps for unet with all channels model in input {idx}: {crps}')
        batch_X.append(X)
        batch_y.append(y)

print('Attribution calculation for Unet-All:')
test_dataloader = torch.utils.data.DataLoader(data, batch_size = 32, shuffle = False, num_workers = 2)
test_loss, rmse_loss, bs_loss = test_model_all_stations(unet_model, tst_loader=test_dataloader, data = data, device='cpu')
print(f'Checking loss in test set for UNet-All: {test_loss}')

ig = IntegratedGradients(unet_model)
input_maps = torch.tensor(batch_X)
H, W = input_maps.shape[2], input_maps.shape[3]
print(input_maps.shape)
global_attr_channel_0 = torch.zeros(input_maps.shape[1])
global_attr_channel_1 = torch.zeros(input_maps.shape[1])
global_attr_channel_2 = torch.zeros(input_maps.shape[1])

x_y_posit = data.meteo_centers_info['x_y_d04']

print(f'Para el batch de datos con índices {selected_indices}, calculamos IG:')
positions = [(x - data.crop_x[0],y - data.crop_y[0]) for (x,y) in x_y_posit]
print(f'Posiciones del output donde se calcula (Nº: {len(positions)}) es en la posición de las estaciones:')
print(positions)

for idx,(x,y) in enumerate(positions):
        print(idx,'.',x,y)
        target_0 = (0, y, x) 
        attr_channel_0_batch = ig.attribute(input_maps, target= target_0, return_convergence_delta=False)
        global_attr_channel_0 += attr_channel_0_batch.sum(dim= (0,2,3), keepdim=False) #Solo se queda la dimensión de nºcanales de entrada
        print(' channel 0 done')
        # Calcular atribuciones para el canal de salida 1
        target_1 = (1, y, x) 
        attr_channel_1_batch = ig.attribute(input_maps, target=target_1, return_convergence_delta=False)
        global_attr_channel_1 += attr_channel_1_batch.sum(dim= (0,2,3), keepdim=False)
        print(' channel 1 done')
        # Calcular atribuciones para el canal de salida 2
        target_2 = (2, y, x) 
        attr_channel_2_batch = ig.attribute(input_maps, target=target_2, return_convergence_delta=False)
        global_attr_channel_2 += attr_channel_2_batch.sum(dim= (0,2,3), keepdim=False)
        print(' channel 2 done')

#np.savez(os.path.join(main_path,'data','Sensitivity_UnetAllChSab5_XMaxSumBatch3'), ch0 = global_attr_channel_0.detach().numpy(),
#         ch1 = global_attr_channel_1.detach().numpy(), ch2 = global_attr_channel_2.detach().numpy())


#------------------------------------------------------PCA--------------------------------------

print('Attribution calculation for Unet-PCA:')
unet_model = UNet(n_inp_channels=11,n_outp_channels=3,bottleneck = False)
model_data = torch.load('/disk/barbusano/barbusano3/Laboratory/result_logs/b32_lr0.001_b32_lr1e-3_AdamW_Norm_11PCA_SqrtStdReg_NoBottleNeck/checkpoint_max.pt')
unet_model.load_state_dict(model_data['net'])
unet_model.eval()

train_set = WRFdataset(data_subset = 'train', group_configs = desired_configs , station_split = False)
train_set.normalize_data(mean_og_data, std_og_data)
train_set.apply_PCA(ncomponents=10)

data = WRFdataset(data_subset='test',station_split=False, group_configs = desired_configs)
data.normalize_data(mean_og_data, std_og_data)
data.apply_PCA(eigvec_transp = train_set.W_t_PCA)

test_dataloader = torch.utils.data.DataLoader(data, batch_size = 32, shuffle = False, num_workers = 2)
test_loss, test_rmse, tess_bs = test_model_all_stations(unet_model, tst_loader=test_dataloader, data = data, device='cpu')
print(f'Checking test set loss for Unet-PCA: {test_loss}')

ig = IntegratedGradients(unet_model)

input_maps = torch.tensor([data[idx][0] for idx in selected_indices])
H, W = input_maps.shape[2], input_maps.shape[3]
print(input_maps.shape)
global_attr_channel_0 = torch.zeros(input_maps.shape[1])
global_attr_channel_1 = torch.zeros(input_maps.shape[1])
global_attr_channel_2 = torch.zeros(input_maps.shape[1])

x_y_posit = data.meteo_centers_info['x_y_d04']
positions = [(x - data.crop_x[0],y - data.crop_y[0]) for (x,y) in x_y_posit]
print(positions)
for idx,(x,y) in enumerate(positions):
        print(idx,'.',x,y)
        target_0 = (0, y, x) 
        attr_channel_0_batch, _ = ig.attribute(input_maps, target= target_0, return_convergence_delta=True)
        global_attr_channel_0 += attr_channel_0_batch.sum(dim= (0,2,3), keepdim=False)
        print(' channel 0 done')
        # Calcular atribuciones para el canal de salida 1
        target_1 = (1, y, x) 
        attr_channel_1_batch, _ = ig.attribute(input_maps, target=target_1, return_convergence_delta=True)
        global_attr_channel_1 += attr_channel_1_batch.sum(dim= (0,2,3), keepdim=False)
        print(' channel 1 done')
        # Calcular atribuciones para el canal de salida 1
        target_2 = (2, y, x) 
        attr_channel_2_batch, _ = ig.attribute(input_maps, target=target_2, return_convergence_delta=True)
        global_attr_channel_2 += attr_channel_2_batch.sum(dim= (0,2,3), keepdim=False)
        print(' channel 2 done')

#np.savez(os.path.join(main_path,'data','Sensitivity_Unet11PCA_XMaxSumBatch3'), ch0 = global_attr_channel_0.detach().numpy(),
#         ch1 = global_attr_channel_1.detach().numpy(), ch2 = global_attr_channel_2.detach().numpy())

#--------------------------------------------------------------------FeatSelec-----------------------------------------------

print('Attribution calculation for Unet-FS:')
unet_model = UNet(n_inp_channels=11,n_outp_channels=3,bottleneck = False)
model_data = torch.load('/disk/barbusano/barbusano3/Laboratory/result_logs/b32_lr0.001_b32_lr1e-3_AdamW_Norm_11FeatSelec_SqrtStdReg_NoBottleNeck/checkpoint_max.pt')
unet_model.load_state_dict(model_data['net'])
unet_model.eval()

train_set = WRFdataset(data_subset = 'train', group_configs = desired_configs , station_split = False)
train_set.normalize_data(mean_og_data, std_og_data)
train_set.apply_feature_selec(11)

data = WRFdataset(data_subset='test',station_split=False, group_configs = desired_configs)
data.normalize_data(mean_og_data, std_og_data)
data.apply_feature_selec(feat_selec_mask = train_set.feat_selec_mask)

test_dataloader = torch.utils.data.DataLoader(data, batch_size = 32, shuffle = False, num_workers = 2)
test_loss = test_model_all_stations(unet_model, tst_loader=test_dataloader, data = data, device='cpu')
print(f'Checking test set loss for Unet-FS: {test_loss}')

ig = IntegratedGradients(unet_model)
input_maps = torch.tensor([data[idx][0] for idx in selected_indices])
H, W = input_maps.shape[2], input_maps.shape[3]
print(input_maps.shape)
global_attr_channel_0 = torch.zeros(input_maps.shape[1])
global_attr_channel_1 = torch.zeros(input_maps.shape[1])
global_attr_channel_2 = torch.zeros(input_maps.shape[1])

x_y_posit = data.meteo_centers_info['x_y_d04']
positions = [(x - data.crop_x[0],y - data.crop_y[0]) for (x,y) in x_y_posit]

for idx,(x,y) in enumerate(positions):
        print(idx,'.',x,y)
        target_0 = (0, y, x) 
        attr_channel_0_batch, _ = ig.attribute(input_maps, target= target_0, return_convergence_delta=True)
        global_attr_channel_0 += attr_channel_0_batch.sum(dim= (0,2,3), keepdim=False) #Solo se queda la dimensión de nºcanales de entrada
        print(' channel 0 done')
        # Calcular atribuciones para el canal de salida 1
        target_1 = (1, y, x) 
        attr_channel_1_batch, _ = ig.attribute(input_maps, target=target_1, return_convergence_delta=True)
        global_attr_channel_1 += attr_channel_1_batch.sum(dim= (0,2,3), keepdim=False)
        print(' channel 1 done')
        # Calcular atribuciones para el canal de salida 2
        target_2 = (2, y, x) 
        attr_channel_2_batch, _ = ig.attribute(input_maps, target=target_2, return_convergence_delta=True)
        global_attr_channel_2 += attr_channel_2_batch.sum(dim= (0,2,3), keepdim=False)
        print(' channel 2 done')

#np.savez(os.path.join(main_path,'data','Sensitivity_Unet11FS_XMaxSumBatch3'), ch0 = global_attr_channel_0.detach().numpy(),
#         ch1 = global_attr_channel_1.detach().numpy(), ch2 = global_attr_channel_2.detach().numpy())

#-------------------------------------------------DS-----------------------------------
print('Calculando atribuciones Unet-DS:')
unet_model = UNet(n_inp_channels=11,n_outp_channels=3,bottleneck = False)
model_data = torch.load('/disk/barbusano/barbusano3/Laboratory/result_logs/b32_lr0.001_b32_lr1e-3_AdamW_Norm_11DownSelectXLowerCRPSVal_SqrtStdReg/checkpoint_max.pt')
unet_model.load_state_dict(model_data['net'])
unet_model.eval()

#Mask extracted from: /disk/barbusano/barbusano3/data/Sensitivity_UnetAllChSab5_XLowerCRPS_Val.npz . Obtained from calc_sensitivity_DS.py
feat_selec_mask_ds_XLowerCRPSVal = [False, True, True, False, False, True, False, True, False, False, False, False, True, False, False, True, False, False, False, False, True, False, True, True, True, True]

data = WRFdataset(data_subset='test',station_split=False, group_configs = desired_configs)
data.normalize_data(mean_og_data, std_og_data)
data.apply_feature_selec(feat_selec_mask = feat_selec_mask_ds_XLowerCRPSVal)

test_dataloader = torch.utils.data.DataLoader(data, batch_size = 32, shuffle = False, num_workers = 2)
test_loss = test_model_all_stations(unet_model, tst_loader=test_dataloader, data = data, device='cpu')
print(f'Checking test set loss for Unet-DS: {test_loss}')

ig = IntegratedGradients(unet_model)
input_maps = torch.tensor([data[idx][0] for idx in selected_indices])
H, W = input_maps.shape[2], input_maps.shape[3]
print(input_maps.shape)
global_attr_channel_0 = torch.zeros(input_maps.shape[1])
global_attr_channel_1 = torch.zeros(input_maps.shape[1])
global_attr_channel_2 = torch.zeros(input_maps.shape[1])

x_y_posit = data.meteo_centers_info['x_y_d04']
positions = [(x - data.crop_x[0],y - data.crop_y[0]) for (x,y) in x_y_posit]

for idx,(x,y) in enumerate(positions):
        print(idx,'.',x,y)
        target_0 = (0, y, x) 
        attr_channel_0_batch, _ = ig.attribute(input_maps, target= target_0, return_convergence_delta=True)
        global_attr_channel_0 += attr_channel_0_batch.sum(dim= (0,2,3), keepdim=False) #Solo se queda la dimensión de nºcanales de entrada
        print(' channel 0 done')
        # Calcular atribuciones para el canal de salida 1
        target_1 = (1, y, x) 
        attr_channel_1_batch, _ = ig.attribute(input_maps, target=target_1, return_convergence_delta=True)
        global_attr_channel_1 += attr_channel_1_batch.sum(dim= (0,2,3), keepdim=False)
        print(' channel 1 done')
        # Calcular atribuciones para el canal de salida 2
        target_2 = (2, y, x) 
        attr_channel_2_batch, _ = ig.attribute(input_maps, target=target_2, return_convergence_delta=True)
        global_attr_channel_2 += attr_channel_2_batch.sum(dim= (0,2,3), keepdim=False)
        print(' channel 2 done')

#np.savez(os.path.join(main_path,'data','Sensitivity_Unet11DS_XMaxSumBatch3'), ch0 = global_attr_channel_0.detach().numpy(),
#         ch1 = global_attr_channel_1.detach().numpy(), ch2 = global_attr_channel_2.detach().numpy())
