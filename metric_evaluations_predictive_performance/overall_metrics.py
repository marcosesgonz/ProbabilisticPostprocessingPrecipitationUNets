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

train_set = WRFdataset(data_subset = 'train')
test_set = WRFdataset(data_subset='test')


print('----------------------------------------------------------')
print('Ensemble model:')
crps_ensemble = CRPS_mine()
crps_ensemble_list = []
rmse_ensemble_list = []
brier_score_ensemble_list = []
prec_list = []
#length_centers_list = []
for X,y in test_set:
    no_na_mask = ~np.isnan(y)
    x_y_posit = test_set.meteo_centers_info['x_y_d04'][no_na_mask]
    x_positions, y_postions = x_y_posit[...,0], x_y_posit[...,1]
    ensemble = X[:-1, y_postions - test_set.crop_y[0], x_positions - test_set.crop_x[0]]
    targets = y[no_na_mask]
    for i in range(len(targets)):
        mean_ensemble = np.mean(ensemble[:,i])
        ensemble_sample = np.expand_dims(ensemble[:,i],axis = -1)
        rmse = np.sqrt((mean_ensemble - targets[i])**2)
        rmse_ensemble_list.append(rmse)
        brier_score_ensemble_list.append(threshold_brier_score(targets[i], ensemble[:,i], threshold = 1))
        crps_ensemble_list.append(crps_ensemble.compute(X_f = ensemble_sample, X_o = targets[i]))
    prec_list.extend(list(targets))
    #length_centers_list.append(np.sum(~np.isnan(y)))
print('')
print(f'NºDatos:{len(crps_ensemble_list)}\n CRPS:{np.mean(crps_ensemble_list)} RMSE:{np.mean(rmse_ensemble_list)} BS:{np.mean(brier_score_ensemble_list)}')
print(len(prec_list))


print('----------------------------------------------------------')
print('FeatSelec Model:')
train_set.normalize_data()
test_set.normalize_data(train_set.mean_og_data, train_set.std_og_data)

train_set.apply_feature_selec(num_components=11)
test_set.apply_feature_selec(feat_selec_mask = train_set.feat_selec_mask)

unet_model = UNet(n_inp_channels=11,n_outp_channels=3, bottleneck=False)
fs_w_path = os.path.join(main_path, 'Laboratory','result_logs','b32_lr0.001_b32_lr1e-3_AdamW_Norm_11FeatSelec_SqrtStdReg_NoBottleNeck','checkpoint_max.pt')
model_data = torch.load(fs_w_path, map_location=torch.device('cpu'))
print(model_data['min_val_loss'])
unet_model.load_state_dict(model_data['net'])

test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = 32, shuffle = False, drop_last = False)
test_loss, test_rmse_loss,  test_bs_loss = test_model_all_stations(unet_model, tst_loader=test_dataloader, data = test_set, device='cpu', per_hour=False)
print('CRPS:',test_loss)
print('RMSE:',test_rmse_loss)
print('BS:',test_bs_loss)


print('----------------------------------------------------------')

print('----------------------------------------------------------')
print('DownSelec Model:')
mean_og_data, std_og_data = train_set.mean_og_data, train_set.std_og_data

train_set = WRFdataset(data_subset = 'train')
test_set = WRFdataset(data_subset='test')

train_set.normalize_data(mean_og_data, std_og_data)
test_set.normalize_data(mean_og_data, std_og_data)

train_set.apply_feature_selec(num_components=11)
DS_mask = [False, True, True, False, False, True, False, True, False, False, False, False, True, False, False, True, False, False, False, False, True, False, True, True, True, True]
test_set.apply_feature_selec(feat_selec_mask = DS_mask)

unet_model = UNet(n_inp_channels=11,n_outp_channels=3, bottleneck=False)
fs_w_path = os.path.join(main_path, 'Laboratory','result_logs','b32_lr0.001_b32_lr1e-3_AdamW_Norm_11DownSelectXLowerCRPSVal_SqrtStdReg','checkpoint_max.pt')
model_data = torch.load(fs_w_path, map_location=torch.device('cpu'))
print(model_data['min_val_loss'])
unet_model.load_state_dict(model_data['net'])

test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = 32, shuffle = False, drop_last = False)
test_loss, test_rmse_loss, test_bs_loss = test_model_all_stations(unet_model, tst_loader=test_dataloader, data = test_set, device='cpu', per_hour=False)
print('CRPS:',test_loss)
print('RMSE:',test_rmse_loss)
print('BS:',test_bs_loss)

print('----------------------------------------------------------')
print('PCA model:')
mean_og_data, std_og_data = train_set.mean_og_data, train_set.std_og_data

train_set = WRFdataset(data_subset = 'train')
test_set = WRFdataset(data_subset='test')

train_set.normalize_data(mean_og_data, std_og_data)
test_set.normalize_data(mean_og_data, std_og_data)

train_set.apply_PCA(ncomponents=10)
test_set.apply_PCA(eigvec_transp = train_set.W_t_PCA)

pca_w_path = os.path.join(main_path, 'Laboratory','result_logs','b32_lr0.001_b32_lr1e-3_AdamW_Norm_11PCA_SqrtStdReg_NoBottleNeck','checkpoint_max.pt')
model_data = torch.load(pca_w_path, map_location=torch.device('cpu'))
print(model_data['min_val_loss'])
unet_model = UNet(n_inp_channels=11,n_outp_channels=3, bottleneck=False)
unet_model.load_state_dict(model_data['net'])
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = 32, shuffle = False, drop_last = False)
test_loss, test_rmse_loss, test_bs_loss = test_model_all_stations(unet_model, tst_loader=test_dataloader, data = test_set, device='cpu', per_hour=False)
print('CRPS:',test_loss)
print('RMSE:',test_rmse_loss)
print('BS:',test_bs_loss)
