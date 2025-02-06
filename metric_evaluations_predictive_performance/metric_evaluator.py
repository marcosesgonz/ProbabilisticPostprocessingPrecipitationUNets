import os 
import sys
main_path = os.path.abspath(os.path.join(__file__,'..','..'))
sys.path.append(main_path)
import torch
from Datasets import WRFdataset
import numpy as np
from models.UNet import UNet
import pandas as pd
from utils import test_model, obtain_params_results, pit_histogram, CRPS_mine
from properscoring import threshold_brier_score
import matplotlib.pyplot as plt
from collections import defaultdict

class evaluator:
    def __init__(self):
        self.mean_og_data = None
        self.std_og_data = None

    def obtain_metrics_ensemble(self, return_metrics_mean = True):
        test_set = WRFdataset(data_subset='test',station_split=False, wrf_variables=['prec'])
        crps_ensemble = CRPS_mine()
        crps_ensemble_list = []
        mse_ensemble_list = []
        results_ensemble_list = []
        brier_score_ensemble_list = []
        prec_list = []
        centers_list = []
        for X,y in test_set:
            no_na_mask = ~np.isnan(y)
            x_y_posit = test_set.meteo_centers_info['x_y_d04'][no_na_mask]
            name_centers = test_set.meteo_centers_info['name'][no_na_mask]
            x_positions, y_postions = x_y_posit[...,0], x_y_posit[...,1]
            ensemble = X[:-1, y_postions - test_set.crop_y[0], x_positions - test_set.crop_x[0]]
            targets = y[no_na_mask]
            mean_ensemble_targets = ensemble.mean(axis = 0) 
            for i in range(len(targets)):
                mean_ensemble = mean_ensemble_targets[i]
                ensemble_sample = np.expand_dims(ensemble[:,i],axis = -1)
                mse_ensemble_list.append((mean_ensemble - targets[i])**2)
                brier_score_ensemble_list.append(threshold_brier_score(targets[i], ensemble[:,i], threshold = 1))
                crps_ensemble_list.append(crps_ensemble.compute(X_f = ensemble_sample, X_o = targets[i]))
            prec_list.extend(list(targets))
            centers_list.extend(name_centers)
            results_ensemble_list.extend(mean_ensemble_targets)
        
        #print(len(crps_ensemble_list), np.mean(crps_ensemble_list), np.mean(mse_ensemble_list), np.mean(brier_score_ensemble_list))
        if return_metrics_mean:
            return {'crps':np.mean(crps_ensemble_list), 'mse':np.mean(mse_ensemble_list),
                'bs': np.mean(brier_score_ensemble_list), 'prec(mm)':np.array(prec_list),
                'center':np.array(centers_list), 'ensemble_mean': np.array(results_ensemble_list)}
        else:
            return {'crps':np.array(crps_ensemble_list), 'mse':np.array(mse_ensemble_list),
                'bs': np.array(brier_score_ensemble_list), 'prec(mm)':np.array(prec_list),
                'center':np.array(centers_list), 'ensemble_mean': np.array(results_ensemble_list)}


    def obtain_results_ensemble(self, mean_ensemble = True):
        #Ensemble model
        print("Warning: Ensemble results don't follow the same ordering than other results since it uses csvs files to compute calculations")
        csvs_path = os.path.join(main_path,'data','csvs_for_classic_methods')
        assert os.path.exists(csvs_path)
        prec_ensemb = []
        results_ensemb = []
        center = []
        for path in os.listdir(csvs_path):
            if path.endswith('test.csv'):
                ensemb_test_df = pd.read_csv(os.path.join(csvs_path,path), parse_dates=['date']).dropna()
                prec_ensemb.extend(ensemb_test_df.iloc[:,1].values)
                if mean_ensemble:
                    results_ensemb.extend(ensemb_test_df.iloc[:,2:].values.mean(axis=1))
                    name_center = path.split('_')[0]
                    center.extend([name_center for n in range(len(ensemb_test_df))])

        return np.array(results_ensemb), np.array(prec_ensemb), np.array(center)

    def prepare_data_and_net(self,model_name):
        train_set = WRFdataset(data_subset = 'train')
        test_set = WRFdataset(data_subset = 'test')
        
        train_set.normalize_data(self.mean_og_data,self.std_og_data)
        self.mean_og_data, self.std_og_data = train_set.mean_og_data, train_set.std_og_data
        test_set.normalize_data(train_set.mean_og_data, train_set.std_og_data)
        init_timedim = 1
        inp_channels = 11
        
        if model_name == 'unet_ds':
            DS_mask = [False, True, True, False, False, True, False, True, False, False, False, False, True, False, False, True, False, False, False, False, True, False, True, True, True]#, True]
            test_set.apply_feature_selec(feat_selec_mask = DS_mask)
            w_path = os.path.join(main_path, 'Laboratory','result_logs','b32_lr0.001_b32_lr1e-3_AdamW_Norm_11DownSelectXLowerCRPSVal_SqrtStdReg','checkpoint_max.pt')
        elif model_name == 'unet_fs':
            train_set.apply_feature_selec(num_components=10)
            test_set.apply_feature_selec(feat_selec_mask = train_set.feat_selec_mask)
            w_path = os.path.join(main_path, 'Laboratory','result_logs','b32_lr0.001_b32_lr1e-3_AdamW_Norm_11FeatSelec_SqrtStdReg_NoBottleNeck','checkpoint_max.pt')
        elif model_name == 'unet_pca':
            train_set.apply_PCA(ncomponents=10)
            test_set.apply_PCA(eigvec_transp = train_set.W_t_PCA)
            w_path = os.path.join(main_path, 'Laboratory','result_logs','b32_lr0.001_b32_lr1e-3_AdamW_Norm_11PCA_SqrtStdReg_NoBottleNeck','checkpoint_max.pt')
        
        elif model_name == 'unet_all':
            w_path = os.path.join(main_path, 'Laboratory','result_logs','b32_lr0.001_b32_lr1e-3_AdamW_Norm_AllChannels_SqrtStdReg','checkpoint_max.pt')
            inp_channels = 26

        net_model = UNet(n_inp_channels = inp_channels,n_outp_channels=3, bottleneck=False, initial_time_dim = init_timedim)
        model_data = torch.load(w_path, map_location=torch.device('cpu'), weights_only=True)
        try:
            print('Min val loss model:',model_data['min_val_loss'])
        except:
            print('Min val loss model:',model_data['val_loss'])
        
        net_model.load_state_dict(model_data['net'])

        return net_model, test_set

    def obtain_results_net(self,model_name,params = False, results = True, return_metrics_mean = True):
        net_model, test_set = self.prepare_data_and_net(model_name)

        if results == True:
            test_loss, test_mae_loss, test_mse_loss, test_bs_loss = test_model(net_model, data = test_set, device='cpu', per_hour=False, return_mean = return_metrics_mean)
            if params == True:
                dict_params = obtain_params_results(net_model,test_set,'cpu') #['mean','std_dev_no_reg','shift', 'center', 'prec(mm)']
                return [test_loss, test_mae_loss, test_mse_loss, test_bs_loss], dict_params
            else:
                return test_loss, test_mae_loss, test_mse_loss, test_bs_loss
        if params == True:
            dict_params = obtain_params_results(net_model,test_set,'cpu')
            return dict_params
    
    def eval(self,model_name,include_ensemble_metrics = True,return_params = False, results = True, return_metrics_mean = True, mean_ensemble = True):
        if model_name == 'ensemble':
            if include_ensemble_metrics:
                return self.obtain_metrics_ensemble(return_metrics_mean = return_metrics_mean)
            else:
                return self.obtain_results_ensemble(mean_ensemble = mean_ensemble)
        elif model_name.startswith('unet') or model_name.startswith('utransf'):
            return self.obtain_results_net(model_name = model_name, params = return_params, results = results,
                                             return_metrics_mean= return_metrics_mean)
