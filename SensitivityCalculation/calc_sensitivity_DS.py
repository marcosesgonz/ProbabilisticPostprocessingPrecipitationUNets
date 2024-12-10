"""
This script is used to obtain down-selected features for UNet-DS
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

data = WRFdataset(data_subset='val')

train_set = WRFdataset(data_subset = 'train', group_configs = desired_configs , station_split = False)
train_set.normalize_data()
mean_og_data, std_og_data = train_set.mean_og_data, train_set.std_og_data
data_normalized = WRFdataset(data_subset='val',station_split=False, group_configs = desired_configs)
data_normalized.normalize_data(train_set.mean_og_data, train_set.std_og_data)

unet_model = UNet(n_inp_channels=26,n_outp_channels=3,bottleneck = False)
model_data = torch.load(os.path.join(main_path,'Laboratory','result_logs','b32_lr0.001_b32_lr1e-3_AdamW_Norm_AllChannels_SqrtStdReg','checkpoint_max.pt'))
unet_model.load_state_dict(model_data['net'])
crps_loss = CRPS_CSGDloss()
unet_model.eval()

dataset = defaultdict(list)
for idx in range(len(data)):
    X,y = data[idx]
    crps_ensemble = CRPS_mine()
    no_na_mask = ~np.isnan(y)
    x_y_posit = data.meteo_centers_info['x_y_d04'][no_na_mask]
    x_positions, y_postions = x_y_posit[...,0], x_y_posit[...,1]
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
    dataset['SumPrecWRF'].append(np.sum(np.mean(ensemble, axis = 0)))

crps_dataset = pd.DataFrame(data = dataset)
crps_dif =  crps_dataset['CRPS_Ensemble'] - crps_dataset['CRPS_UnetAll']
crps_dif_sorted = crps_dif.sort_values(ascending=False)

print(crps_dif_sorted.head(40))

selected_indices = []
# Iterate over ordered indices
for idx in crps_dif_sorted.index:
    # Verify that corresponds to another day than previously selected indices
    if all(abs(idx - prev_idx) > 23 for prev_idx in selected_indices):
        selected_indices.append(idx)
        
    # Stop when we have 3 indicess
    if len(selected_indices) == 3:
        break

print(f'Selected indices: {selected_indices}')
crps_dataset['CRPS_dif'] = crps_dif
print(crps_dataset.iloc[selected_indices])

print('For first index selected:')
print(np.nansum(data[selected_indices[0]][1]),crps_dataset[['SumPrecWeatStat']].iloc[selected_indices[0]].values)

#Creating batch for being used for sensitivity calculation
batch_X = []
batch_y = []
for idx in selected_indices:
    X, y = data_normalized[idx]
    batch_X.append(X)
    batch_y.append(y)

print('Sensitivity calculation of UNet-ALL in validation set cases:')
test_dataloader = torch.utils.data.DataLoader(data, batch_size = 32, shuffle = False, num_workers = 0)
test_loss, test_rmse, test_bs = test_model_all_stations(unet_model, tst_loader=test_dataloader, data = data, device='cpu')
print(f'Checking error of UNet-All in validation set. CRPS: {test_loss}')


ig = IntegratedGradients(unet_model)

input_maps = torch.tensor(batch_X)
H, W = input_maps.shape[2], input_maps.shape[3]
print(input_maps.shape)
global_attr_channel_0 = torch.zeros(input_maps.shape[1])
global_attr_channel_1 = torch.zeros(input_maps.shape[1])
global_attr_channel_2 = torch.zeros(input_maps.shape[1])

print(f'Para el batch de datos con índices {selected_indices}, calculamos IG:')
x_y_posit = data.meteo_centers_info['x_y_d04']
positions = [(x - data.crop_x[0],y - data.crop_y[0]) for (x,y) in x_y_posit]
print(f'Posiciones del output donde se calcula (Total: {len(positions)}) es en la posición de las estaciones:')
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


plt.figure(figsize=(10, 5))
plt.bar(np.arange(input_maps.shape[1]), global_attr_channel_0)
plt.xlabel('Canales de Entrada')
plt.ylabel('Sensibilidad')
plt.title('Sensibiliad de los canales de entrada para el primer canal de salida de Unet 26 Ch (sab5) en Val. Set')
#plt.savefig('')

plt.figure(figsize=(10, 5))
plt.bar(np.arange(input_maps.shape[1]), global_attr_channel_1)
plt.xlabel('Canales de Entrada')
plt.ylabel('Sensibilidad')
plt.title('Sensibiliad de los canales de entrada para el segundo canal de salida de Unet 26 Ch (sab5) en Val. Set')
#plt.savefig('')

#np.savez(os.path.join(main_path,'data','Sensitivity_UnetAllChSab5_XLowerCRPS_Val'), ch0 = global_attr_channel_0.detach().numpy(),
#         ch1 = global_attr_channel_1.detach().numpy(), ch2 = global_attr_channel_2.detach().numpy())


attr_total_ds = np.abs(global_attr_channel_0.detach().numpy()) + np.abs(global_attr_channel_1.detach().numpy()) + np.abs(global_attr_channel_2.detach().numpy())
top_11_indices = np.argsort(attr_total_ds)[-11:]

print('Features obtained from down selection to be used in UNet-DS:')
mask_ds = [False if i not in top_11_indices else True for i in range(len(attr_total_ds))]