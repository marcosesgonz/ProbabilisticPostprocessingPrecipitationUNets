import os 
import sys
main_path = os.path.abspath(os.path.join(__file__,'..','..'))
sys.path.append(main_path)
import pickle
import torch
from Datasets import WRFdataset, MeteoCentersSplitter
import numpy as np
from models.UNet import UNet
import pandas as pd
from utils import test_model, obtain_params_results, pit_histogram, CRPS_mine
from properscoring import threshold_brier_score
import matplotlib.pyplot as plt
from models.classic_models import PredictiveCSGD,AnalogEnsemble,ANNCSGD
from collections import defaultdict

class evaluator:
    def __init__(self):
        self.mean_og_data = None
        self.std_og_data = None
        self.csvs_dir_ensemble = os.path.join(main_path,'data','csvs_for_classic_methods')

    def obtain_metrics_ensemble(self, return_metrics_mean = True, use_fs_columns = False, meteoc_folds = False):
        fs_columns = [True, True, True, False, True, False, False, True, False, False, False, False, False, False, False, True, True, True, False, False, True, False, False, False, True]
        test_set = WRFdataset(data_subset='test',station_split=False, wrf_variables=['prec'])
        crps_ensemble = CRPS_mine()
        crps_ensemble_list = []
        mse_ensemble_list = []
        results_ensemble_list = []
        brier_score_ensemble_list = []
        prec_list = []
        centers_list = []
        if meteoc_folds:
            splitter = MeteoCentersSplitter(test_set, nfolds=5, stratify = False)
            folds = range(5)
        else:   
            folds = range(1)
            splitter = None
        for fold_id in folds:
            if splitter is not None:
                _ , mask_ctrs_tst = splitter[fold_id]
            else:
                mask_ctrs_tst = None
            for X,y in test_set:
                no_na_mask = ~np.isnan(y)
                if mask_ctrs_tst is not None:
                    no_na_mask = np.logical_and(mask_ctrs_tst, no_na_mask)
                x_y_posit = test_set.meteo_centers_info['x_y_d04'][no_na_mask]
                name_centers = test_set.meteo_centers_info['name'][no_na_mask]
                x_positions, y_postions = x_y_posit[...,0], x_y_posit[...,1]
                ensemble = X[:-1, y_postions - test_set.crop_y[0], x_positions - test_set.crop_x[0]]
                if use_fs_columns:
                    ensemble = ensemble[fs_columns]
                targets = y[no_na_mask]
                mean_ensemble_targets = ensemble.mean(axis = 0) 
                for i in range(len(targets)):
                    mean_ensemble = mean_ensemble_targets[i]
                    ensemble_sample = np.expand_dims(ensemble[:,i],axis = -1)
                    mse_ensemble_list.append((mean_ensemble - targets[i])**2)
                    brier_score_ensemble_list.append(threshold_brier_score(targets[i], ensemble[:,i], threshold = 0.1))
                    crps_ensemble_list.append(crps_ensemble.compute(X_f = ensemble_sample, X_o = targets[i]))
                prec_list.extend(list(targets))
                centers_list.extend(name_centers)
                results_ensemble_list.extend(mean_ensemble_targets)
        
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

    def prepare_data_and_net(self,model_name, meteoc_folds = False):
        train_set = WRFdataset(data_subset = 'train')
        test_set = WRFdataset(data_subset = 'test')
        
        train_set.normalize_data(self.mean_og_data,self.std_og_data)
        self.mean_og_data, self.std_og_data = train_set.mean_og_data, train_set.std_og_data
        test_set.normalize_data(train_set.mean_og_data, train_set.std_og_data)
        init_timedim = 1
        inp_channels = 11
        
        if model_name == 'unet_ds':
            if meteoc_folds:
                path = os.path.join(main_path, 'results','unet_weights','5folds','b32_lr1e-3_11DownSelecLowerCRPSVal')
                w_path = [os.path.join(path,f'fold{i}_checkpoint_max.pt') for i in range(1,6)]
            else:
                DS_mask = [False, True, True, False, False, True, False, True, False, False, False, False, True, False, False, True, False, False, False, False, True, False, True, True, True]#, True]
                test_set.apply_feature_selec(feat_selec_mask = DS_mask)
                w_path = os.path.join(main_path,'results','unet_weights','overall','b32_lr0.001_b32_lr1e-3_AdamW_Norm_11DownSelectXLowerCRPSVal_SqrtStdReg','checkpoint_max.pt')
        elif model_name == 'unet_fs':
            train_set.apply_feature_selec(num_components=10)
            test_set.apply_feature_selec(feat_selec_mask = train_set.feat_selec_mask)
            if meteoc_folds:
                path = os.path.join(main_path,'results','unet_weights','5folds','b32_lr0.001_b32_lr1e-3_11FeatSelec')
                w_path = [os.path.join(path,f'fold{i}_checkpoint_max.pt') for i in range(1,6)]
            else:
                w_path = os.path.join(main_path,'results','unet_weights','overall','b32_lr0.001_b32_lr1e-3_AdamW_Norm_11FeatSelec_SqrtStdReg_NoBottleNeck','checkpoint_max.pt')
        elif model_name == 'unet_pca':
            train_set.apply_PCA(ncomponents=10)
            test_set.apply_PCA(eigvec_transp = train_set.W_t_PCA)
            if meteoc_folds:
                path = os.path.join(main_path,'results','unet_weights','5folds','b32_lr0.001_b32_lr1e-3_11PCA')
                w_path = [os.path.join(path,f'fold{i}_checkpoint_max.pt') for i in range(1,6)]
            else:
                w_path = os.path.join(main_path,'results','unet_weights','overall','b32_lr0.001_b32_lr1e-3_AdamW_Norm_11PCA_SqrtStdReg_NoBottleNeck','checkpoint_max.pt')
        
        elif model_name == 'unet_all':
            if meteoc_folds:
                path = os.path.join(main_path,'results','unet_weights','5folds','b32_lr1e-3_11DownSelecLowerCRPSVal')
                w_path = [os.path.join(path,f'fold{i}_fullch_checkpoint_max.pt') for i in range(1,6)]
            else:
                w_path = os.path.join(main_path,'results','unet_weights','overall','b32_lr0.001_b32_lr1e-3_AdamW_Norm_AllChannels_SqrtStdReg','checkpoint_max.pt')
            inp_channels = 26

        net_model = UNet(n_inp_channels = inp_channels,n_outp_channels=3, bottleneck=False, initial_time_dim = init_timedim)
        """model_data = torch.load(w_path, map_location=torch.device('cpu'), weights_only=True)
        try:
            print('Min val loss model:',model_data['min_val_loss'])
        except:
            print('Min val loss model:',model_data['val_loss'])
        
        net_model.load_state_dict(model_data['net'])"""

        return net_model, test_set, w_path

    def obtain_results_net(self,model_name,params = False, results = True, return_metrics_mean = True, meteoc_folds = True):
        net_model, test_set, w_path = self.prepare_data_and_net(model_name = model_name, meteoc_folds = meteoc_folds)

        if meteoc_folds:
            splitter = MeteoCentersSplitter(test_set, nfolds=5, stratify = False)
        else:   
            w_path = [w_path]
            splitter = None
            
        results_dict = defaultdict(list)
        for fold_id, w_p in enumerate(w_path):
            model_data = torch.load(w_p, map_location=torch.device('cpu'), weights_only=True)
            net_model.load_state_dict(model_data['net'])
            if splitter is not None:
                _ , mask_ctrs_tst = splitter[fold_id]
                if 'ds' in model_name:
                    attr_path = os.path.join(main_path,'results','unet_weights','overall','b32_lr1e-3_11DownSelecLowerCRPSVal',f'fold{fold_id + 1}_sensitivities_batch_LowerCRPSVal.npz')
                    global_attr = np.load(attr_path)
                    attr_total = np.abs(global_attr['ch0']) + np.abs(global_attr['ch1']) + np.abs(global_attr['ch2'])
                    #attr_total_norm = attr_total / np.sum(attr_total)
                    top_11_indices = np.argsort(attr_total)[-11:]
                    mask_wrf_simul_ds = [False if i not in top_11_indices else True for i in range(len(attr_total))]
                    test_set.apply_feature_selec(feat_selec_mask = mask_wrf_simul_ds[:-1])#[:-1] because of HGT map data was put in the past as another value in the mask
            else: 
                mask_ctrs_tst = None

            if results == True:
                test_loss, test_mae_loss, test_mse_loss, test_bs_loss = test_model(net = net_model, data = test_set, device='cpu',mask_centers_tst = mask_ctrs_tst,
                                                                                    per_hour=False, return_mean = return_metrics_mean)
                for metric_, loss_ in zip(['crps','mae','mse','bs'],[test_loss,test_mae_loss,test_mse_loss, test_bs_loss]):
                    if type(loss_) == list:
                        results_dict[metric_].extend(loss_)
                    else:
                        results_dict[metric_].extend([loss_])
            if params == True:
                dict_params = obtain_params_results(net_model,test_set,'cpu', mask_centers_tst = mask_ctrs_tst)
                for key,value in dict_params.items():
                    results_dict[key].extend(value)

        return results_dict   
    
    def eval(self,model_name,return_params = False, results = True, return_metrics_mean = True, meteo_center_folds = False):
        """
        Evaluates different models based on the specified model name.
        
        Parameters:
        - model_name (str): The name of the model to evaluate. Supports 'ensemble','unet_pca','unet_ds','unet_fs','unet_all'.
        - include_ensemble_metrics (bool): If True, calculates CRPS, MSE, Brier Score, and ensemble mean.
        - return_params (bool): If True, returns model parameters.
        - results (bool): If True, returns model results.
        - return_metrics_mean (bool): If True, returns mean values of the computed metrics.
        - meteo_center_folds (bool): If True, returns results of the five-folds station experiments. Else, it corresponds to overall experiments.
        
        Returns:
        - A dictionary containing the computed evaluation metrics.
        
        Notes:
        - If 'ensemble' is selected, either full ensemble results or specific ensemble metrics are returned.
        - If 'unet' models are selected, evaluation is based on pre-trained UNet models.
        """
        if model_name == 'ensemble':
            return self.obtain_metrics_ensemble(return_metrics_mean = return_metrics_mean, meteoc_folds = meteo_center_folds)
        elif model_name == 'ensemble_fs':
            return self.obtain_metrics_ensemble(return_metrics_mean = return_metrics_mean, use_fs_columns=True, meteoc_folds = meteo_center_folds)
        elif model_name.startswith('unet') or model_name.startswith('utransf'):
            return self.obtain_results_net(model_name = model_name, params = return_params, results = results,
                                             return_metrics_mean= return_metrics_mean, meteoc_folds= meteo_center_folds)
        
