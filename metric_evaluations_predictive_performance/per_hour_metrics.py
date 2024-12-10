import os 
import sys
main_path = os.path.abspath(os.path.join(__file__,'..','..'))
sys.path.append(main_path)
import torch
from Datasets import WRFdataset
import numpy as np
from models.UNet import UNet
from prob_scores import CRPS_mine
from utils import test_model_all_stations
from properscoring import threshold_brier_score
from collections import defaultdict


train_set = WRFdataset(data_subset = 'train')
test_set = WRFdataset(data_subset='test', return_hour=True)
train_set.normalize_data()

print('----------------------------------------------------------')
print('Ensemble model:')
crps_ensemble = CRPS_mine()
crps_ensemble_list = defaultdict(list)
rmse_ensemble_list = defaultdict(list)
brier_score_ensemble_list = defaultdict(list)
prec_list = defaultdict(list)

for X,y, hour in test_set:
    no_na_mask = ~np.isnan(y)
    x_y_posit = test_set.meteo_centers_info['x_y_d04'][no_na_mask]
    x_positions, y_postions = x_y_posit[...,0], x_y_posit[...,1]
    ensemble = X[:-1, y_postions - test_set.crop_y[0], x_positions - test_set.crop_x[0]]
    targets = y[no_na_mask]
    for i in range(len(targets)):
        mean_ensemble = np.mean(ensemble[:,i])
        ensemble_sample = np.expand_dims(ensemble[:,i],axis = -1)
        rmse_ensemble_list[hour].append(np.sqrt((mean_ensemble - targets[i])**2))
        brier_score_ensemble_list[hour].append(threshold_brier_score(targets[i], ensemble[:,i], threshold = 1))
        crps_ensemble_list[hour].append(crps_ensemble.compute(X_f = ensemble_sample, X_o = targets[i]))
    prec_list[hour].extend(list(targets))

print('')
crps_ensemble_phour = {i:np.mean(crps_ensemble_list[i]) for i in crps_ensemble_list.keys()}
brier_score_ensemble_phour = {i:np.mean(brier_score_ensemble_list[i]) for i in brier_score_ensemble_list.keys()}
rmse_ensemble_phour = {i:np.mean(rmse_ensemble_list[i]) for i in rmse_ensemble_list.keys()}
len_prec_phour = {i:len(prec_list[i]) for i in prec_list.keys()}

print('Mean CRPS per hour:')
print(crps_ensemble_phour)
print('Mean Brier Score per hour:')
print(brier_score_ensemble_phour)
print('Mean RMSE per hour:')
print(rmse_ensemble_phour)
print('Length data per hour:')
print(len_prec_phour)

print('----------------------------------------------------------')
print('All channels model:')
train_set.normalize_data()
test_set.return_hour = False
test_set.normalize_data(train_set.mean_og_data, train_set.std_og_data)

unet_model = UNet(n_inp_channels=26,n_outp_channels=3, bottleneck = False)
fullch_w_path = os.path.join(main_path, 'Laboratory','result_logs','b32_lr0.001_b32_lr1e-3_AdamW_Norm_AllChannels_SqrtStdReg','checkpoint_max.pt')

model_data = torch.load(fullch_w_path, map_location=torch.device('cpu'))
print(model_data['min_val_loss'])
unet_model.load_state_dict(model_data['net'])

test_set.return_hour = True
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = False, drop_last = False)
test_loss, test_rmse_loss, test_bs_loss, nsamples_loss = test_model_all_stations(unet_model, tst_loader=test_dataloader, data = test_set, device='cpu', per_hour=True)
mean_crps_loss_phour = {i:test_loss[i]/nsamples_loss[i] for i in test_loss.keys()}
mean_bs_loss_phour = {i:test_bs_loss[i]/nsamples_loss[i] for i in test_loss.keys()}
mean_rmse_loss_phour = {i:test_rmse_loss[i]/nsamples_loss[i] for i in test_loss.keys()}
print('Mean CRPS per hour:')
print(mean_crps_loss_phour)
print('Mean Brier Score per hour:')
print(mean_bs_loss_phour)
print('Mean RMSE per hour:')
print(mean_rmse_loss_phour)
print('Length data per hour:')
print(nsamples_loss)
#np.savez('metrics_per_hour_full_channels', crps = mean_crps_loss_phour, bs = mean_bs_loss_phour, rmse = mean_rmse_loss_phour)

print('----------------------------------------------------------')
print('11DownSelec Model:')
mean_og_data, std_og_data = train_set.mean_og_data, train_set.std_og_data

#Mask extracted from "/disk/barbusano/barbusano3/data/Sensitivity_UnetAllCh_XMaxLenCorrectedNewDataSplits.npz"
#Mask extracted from XLowerCRPSVal:
DS_mask  = [False, True, True, False, False, True, False, True, False, False, False, False, True, False, False, True, False, False, False, False, True, False, True, True, True, True]
train_set = WRFdataset(data_subset = 'train')
test_set = WRFdataset(data_subset='test')

train_set.normalize_data(mean_og_data, std_og_data)
test_set.normalize_data(mean_og_data, std_og_data)

train_set.apply_feature_selec(feat_selec_mask= DS_mask)
test_set.apply_feature_selec(feat_selec_mask = DS_mask)

unet_model = UNet(n_inp_channels=11,n_outp_channels=3, bottleneck = False)
ds_w_path = os.path.join(main_path, 'Laboratory','result_logs','b32_lr0.001_b32_lr1e-3_AdamW_Norm_11DownSelectXLowerCRPSVal_SqrtStdReg','checkpoint_max.pt')

model_data = torch.load(ds_w_path, map_location=torch.device('cpu'),weights_only=True)
print(f"Min val loss saved: {model_data['min_val_loss']}")
unet_model.load_state_dict(model_data['net'])

test_set.return_hour = True
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = False, drop_last = False)
test_loss, test_rmse_loss, test_bs_loss, nsamples_loss = test_model_all_stations(unet_model, tst_loader=test_dataloader, data = test_set, device='cpu', per_hour=True)
mean_crps_loss_phour = {i:test_loss[i]/nsamples_loss[i] for i in test_loss.keys()}
mean_bs_loss_phour = {i:test_bs_loss[i]/nsamples_loss[i] for i in test_loss.keys()}
mean_rmse_loss_phour = {i:test_rmse_loss[i]/nsamples_loss[i] for i in test_loss.keys()}
print('Mean CRPS per hour:')
print(mean_crps_loss_phour)
print('Mean Brier Score per hour:')
print(mean_bs_loss_phour)
print('Mean RMSE per hour:')
print(mean_rmse_loss_phour)
print('Length data per hour:')
print(nsamples_loss)

#np.savez('metrics_per_hour_11DownSelecLowerCRPSVal', crps = mean_crps_loss_phour, bs = mean_bs_loss_phour, rmse = mean_rmse_loss_phour)

print('----------------------------------------------------------')
print('FeatSelec Model:')
mean_og_data, std_og_data = train_set.mean_og_data, train_set.std_og_data

train_set = WRFdataset(data_subset = 'train')
test_set = WRFdataset(data_subset='test')

train_set.normalize_data(mean_og_data, std_og_data)
test_set.normalize_data(mean_og_data, std_og_data)

train_set.apply_feature_selec(num_components=11)
test_set.apply_feature_selec(feat_selec_mask = train_set.feat_selec_mask)

unet_model = UNet(n_inp_channels=11,n_outp_channels=3, bottleneck = False)
fs_w_path = os.path.join(main_path, 'Laboratory','result_logs','b32_lr0.001_b32_lr1e-3_AdamW_Norm_11FeatSelec_SqrtStdReg_NoBottleNeck','checkpoint_max.pt')

model_data = torch.load(fs_w_path, map_location=torch.device('cpu'))
print(model_data['min_val_loss'])
unet_model.load_state_dict(model_data['net'])

test_set.return_hour = True
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = False, drop_last = False)
test_loss, test_rmse_loss, test_bs_loss, nsamples_loss = test_model_all_stations(unet_model, tst_loader=test_dataloader, data = test_set, device='cpu', per_hour=True)
mean_crps_loss_phour = {i:test_loss[i]/nsamples_loss[i] for i in test_loss.keys()}
mean_bs_loss_phour = {i:test_bs_loss[i]/nsamples_loss[i] for i in test_loss.keys()}
mean_rmse_loss_phour = {i:test_rmse_loss[i]/nsamples_loss[i] for i in test_loss.keys()}
print('Mean CRPS per hour:')
print(mean_crps_loss_phour)
print('Mean Brier Score per hour:')
print(mean_bs_loss_phour)
print('Mean RMSE per hour:')
print(mean_rmse_loss_phour)
print('Length data per hour:')
print(nsamples_loss)

#np.savez('metrics_per_hour_11FeatSelec', crps = mean_crps_loss_phour, bs = mean_bs_loss_phour, rmse = mean_rmse_loss_phour)
print('----------------------------------------------------------')
print('PCA model:')
train_set = WRFdataset(data_subset = 'train')
test_set = WRFdataset(data_subset='test')

train_set.normalize_data(mean_og_data, std_og_data)
test_set.normalize_data(mean_og_data, std_og_data)

train_set.apply_PCA(ncomponents=10)
test_set.apply_PCA(eigvec_transp = train_set.W_t_PCA)

pca_w_path = os.path.join(main_path, 'Laboratory','result_logs','b32_lr0.001_b32_lr1e-3_AdamW_Norm_11PCA_SqrtStdReg_NoBottleNeck','checkpoint_max.pt')
model_data = torch.load(pca_w_path, map_location=torch.device('cpu'))
print(model_data['min_val_loss'])
unet_model.load_state_dict(model_data['net'])

test_set.return_hour = True
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = False, drop_last = False)
test_loss, test_rmse_loss, test_bs_loss, nsamples_loss = test_model_all_stations(unet_model, tst_loader=test_dataloader, data = test_set, device='cpu', per_hour=True)
mean_crps_loss_phour = {i:test_loss[i]/nsamples_loss[i] for i in test_loss.keys()}
mean_bs_loss_phour = {i:test_bs_loss[i]/nsamples_loss[i] for i in test_loss.keys()}
mean_rmse_loss_phour = {i:test_rmse_loss[i]/nsamples_loss[i] for i in test_loss.keys()}
print('Mean CRPS per hour:')
print(mean_crps_loss_phour)
print('Mean Brier Score per hour:')
print(mean_bs_loss_phour)
print('Mean RMSE per hour:')
print(mean_rmse_loss_phour)

#np.savez('metrics_per_hour_11PCA', crps = mean_crps_loss_phour, bs = mean_bs_loss_phour, rmse = mean_rmse_loss_phour)