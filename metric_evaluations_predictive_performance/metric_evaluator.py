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
from utils import ReliabilityDiagram, CRPS_mine, test_model_df
from properscoring import threshold_brier_score
import matplotlib.pyplot as plt
from models.classic_models import PredictiveCSGD,AnalogEnsemble,ANNCSGD

np.random.seed(0)
class evaluator:
    def __init__(self):
        self.mean_og_data = None
        self.std_og_data = None
        self.csvs_path = os.path.join(main_path,'data','csvs_for_classic_methods')
        self.fs_columns = ['GFS_ysutr', 'ARPEGE_ysumc', 'GEM_uwtc', 'GEM_myjtr', 'GFS_uwmc', 
              'GFS_uwtr', 'ARPEGE_uwmc', 'ARPEGE_uwtc', 'GFS_uwtc', 'GEM_mynn2mc']

    
    def obtain_metrics_ensemble(
        self,return_metrics_mean=True,use_fs_columns=False,meteoc_folds=False,threshold=0.1):
        fs_columns = [True, True, True, False, True, False, False, True, False, False, False, False, False, False, False, True, True, True, False, False, True, False, False, False, True]

        test_set = WRFdataset(data_subset='test', wrf_variables=['prec'])
        crps_ensemble = CRPS_mine()

        # Determinar número de folds y splitter si aplica
        if meteoc_folds:
            splitter = MeteoCentersSplitter(test_set, nfolds=5, stratify=False)
            folds = range(5)
        else:
            splitter = None
            folds = range(1)

        # Normalizamos threshold a lista para simplificar columnas
        if isinstance(threshold, (int, float)):
            thresholds = [threshold]
        else:
            thresholds = list(threshold)

        # Vamos acumulando filas en una lista de dicts
        rows = []

        for fold_id in folds:
            if splitter is not None:
                _, mask_ctrs_tst = splitter[fold_id]
            else:
                mask_ctrs_tst = None

            for time_idx,(X, y) in enumerate(test_set):
                # máscara de valores válidos
                no_na_mask = ~np.isnan(y)
                if mask_ctrs_tst is not None:
                    no_na_mask &= mask_ctrs_tst

                # información de centros
                coords = test_set.meteo_centers_info['x_y_d04'][no_na_mask]
                names = test_set.meteo_centers_info['name'][no_na_mask]
                names = [n.replace('/', '_') for n in names]
                x_pos, y_pos = coords[..., 0], coords[..., 1]

                # extraer ensemble y targets
                ensemble = X[:-1,
                            y_pos - test_set.crop_y[0],
                            x_pos - test_set.crop_x[0]]
                if use_fs_columns:
                    ensemble = ensemble[fs_columns]

                targets = y[no_na_mask]
                mean_ens = ensemble.mean(axis=0)

                # para cada punto temporal/ubicación
                for i, target in enumerate(targets):
                    row = {
                        'center': names[i],
                        'date': test_set.meteo_data.iloc[time_idx]['date'],
                        'prec(mm)': float(target),
                        'ensemble_mean': float(mean_ens[i]),
                        'bias': float(mean_ens[i] - target),
                        'mse': float((mean_ens[i] - target) ** 2),
                        'crps': float(
                            crps_ensemble.compute(
                                X_f=np.expand_dims(ensemble[:, i], -1),
                                X_o=target
                            )
                        )
                    }
                    # Brier scores para cada threshold
                    for th in thresholds:
                        bs_val = threshold_brier_score(target, ensemble[:, i], threshold=th)
                        row[f'bs{th}'] = float(bs_val)

                    rows.append(row)
        # Construir DataFrame
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        # Ordenar por estación y luego por tiempo para legibilidad y escalabilidad
        df = df.sort_values(by=['center', 'date'], kind='mergesort')

        # Si piden medias por estación, agrupar y calcular
        if return_metrics_mean:
            agg_dict = {
                'mse': 'mean',
                'crps': 'mean',
                'ensemble_mean': 'mean'
            }
            # añadir bs para cada umbral
            for th in thresholds:
                agg_dict[f'bs{th}'] = 'mean'

            df_mean = df.groupby('center').agg(agg_dict).reset_index(drop=True)
            return df_mean

        # Devolver DataFrame completo ordenado
        return df.reset_index(drop=True)
    


    def obtain_results_ensemble(
            self,
            mean_ensemble=True
        ):
            """
            Lee CSVs de resultados de ensemble, consolida en DataFrame y ordena cronológicamente.
            Si mean_ensemble=True, combina columnas de ensemble en su media.
            """
            assert os.path.exists(self.csvs_path), f"No existe ruta {self.csvs_path}"

            dfs = []
            for fname in os.listdir(self.csvs_path):
                if not fname.endswith('_test.csv'):
                    continue
                path = os.path.join(self.csvs_path, fname)
                df = pd.read_csv(path, parse_dates=['date']).dropna()
                center = fname.replace('_test.csv', '')
                df['center'] = center

                # Columnas de ensemble
                ens_cols = [c for c in df.columns if c not in ['date', 'center', df.columns[1]]]
                # La segunda columna es prec(mm)
                df.rename(columns={df.columns[1]: 'prec(mm)'}, inplace=True)

                if mean_ensemble:
                    df['ensemble_mean'] = df[ens_cols].mean(axis=1)
                else:
                    # opcional mantener lista de valores
                    df['ensemble'] = df[ens_cols].values.tolist()

                dfs.append(df[['center', 'date', 'prec(mm)'] + (['ensemble_mean'] if mean_ensemble else ['ensemble'])])

            all_df = pd.concat(dfs, ignore_index=True)
            # Ordenar por estación y fecha ascendente
            all_df = all_df.sort_values(by=['center', 'date'], kind='mergesort').reset_index(drop=True)
            return all_df

    def assert_csvs_and_wrfmaps(self):
        results_dict_csvs = self.obtain_results_benchmark_df('ANNCSGD',True,True)
        #results_dict_csvs.to_csv('resultados_ensemble_desde_CSVs.csv')
        unique_stations_csv = np.unique(results_dict_csvs['center'])
        #results_dict_wrf_maps = self.obtain_metrics_ensemble(return_metrics_mean=False)
        results_dict_wrf_maps = self.obtain_results_net('unet_all',params=False,results=False,meteoc_folds=False)
        #results_dict_wrf_maps.to_csv('resultados_ensemble_desde_WRF_DS.csv')
        unique_stations_wrf = np.unique(results_dict_wrf_maps['center'])
        are_equal = np.all(unique_stations_csv == unique_stations_wrf)
        print(f'Are the unique stations the same? {are_equal}')
        if not are_equal:
            for station in unique_stations_csv:
                if station not in unique_stations_wrf:
                    print(f'Station {station} not in wrf set')
            for station in unique_stations_wrf:
                if station not in unique_stations_csv:
                    print(f'Station {station} not in csv set')
        for key in results_dict_csvs.columns:
            print(key)
            if key in ['ensemble_mean','prec(mm)']:
                print(np.max((results_dict_csvs[key]-results_dict_wrf_maps[key])))
                if key == 'prec(mm)':
                    for sample_center in unique_stations_csv:
                        print('Center: ',sample_center)
                        one_station_sample_csvs = results_dict_csvs[key][results_dict_csvs['center'] == sample_center]
                        one_station_sample_wrf = results_dict_wrf_maps[key][results_dict_wrf_maps['center'] == sample_center]
                        print(np.max(one_station_sample_wrf - one_station_sample_csvs))
                        diffs = (one_station_sample_csvs != one_station_sample_wrf)
                        if diffs.any():
                            print(one_station_sample_csvs[diffs])
                            print(one_station_sample_wrf[diffs])
            elif key =='center':
                print(results_dict_csvs[key].values)
                print(np.all(results_dict_csvs[key].values == results_dict_wrf_maps[key].values))
            else: 
                print(f'Saltándose la clave {key}')


    def prepare_data_and_net(self,model_name, meteoc_folds = False):
        train_set = WRFdataset(data_subset = 'train')
        test_set = WRFdataset(data_subset = 'test')
        
        train_set.normalize_data(self.mean_og_data,self.std_og_data)
        self.mean_og_data, self.std_og_data = train_set.mean_og_data, train_set.std_og_data
        test_set.normalize_data(train_set.mean_og_data, train_set.std_og_data)
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

        net_model = UNet(n_inp_channels = inp_channels,n_outp_channels=3, bottleneck=False)

        return net_model, test_set, w_path
    
    def obtain_dataframe(self,file,use_fs_columns):
        df = pd.read_csv(file, parse_dates=['date']).dropna()
        dates = df['date'].values
        ground_truth = df.iloc[:, 1]
        wrf_ensemble = df.loc[:, self.fs_columns] if use_fs_columns else df.iloc[:, 2:]
        return dates, ground_truth, wrf_ensemble
    
    def prepare_csvs(self):
        assert os.path.exists(self.csvs_path)
        list_station_names = set(name[:-len(name.split('_')[-1]) -1] for name in os.listdir(self.csvs_path) if name.endswith('.csv')) #Avoiding .gitkeep if exists
        assert len(list_station_names) == 130, "Number of stations don't match with 130"
        station_files = dict()
        for station_name in list_station_names:
            ensemb_train_file = os.path.join(self.csvs_path ,station_name + '_train.csv')
            ensemb_val_file =  os.path.join(self.csvs_path ,station_name + '_val.csv')
            ensemb_test_file = os.path.join(self.csvs_path ,station_name + '_test.csv')
            station_files[station_name] = [ensemb_train_file,ensemb_val_file,ensemb_test_file]
        return station_files
    
    def obtain_weights(self,model_name,use_fs_columns):
        if model_name == 'PredCSGD':
            name_weigths = 'PredCSGD_pesos.pkl' if not use_fs_columns else 'PredCSGD_10FS_pesos.pkl'
        elif model_name == 'AnEn':
            name_weigths = 'AnEn_weights_dict.pkl' if not use_fs_columns else 'AnEn_10FS_weights_dict.pkl'
        elif model_name == 'ANNCSGD':
            name_weigths = 'weights_anncsgd.pt' if not use_fs_columns else  'weights_10FS_anncsgd.pt'
        elif model_name == 'ensemble':
            return None

        full_path = os.path.join(main_path, 'results',f'{model_name}_weights',name_weigths)
        if name_weigths.endswith('.pkl'):
            with open(full_path, 'rb') as f:
                weights = pickle.load(f)
        elif name_weigths.endswith('.pt'):
            weights = torch.load(full_path)

        return weights
 
    def obtain_results_benchmark_df(self,model_name,results=True,use_fs_columns=False,return_prec=True,
                                    return_stations=True,return_ensemble_mean=True, bs_threshold:list = [1]):
        station_files = self.prepare_csvs()
        weights_dict = self.obtain_weights(model_name, use_fs_columns)


        all_rows = []
        for station_name, (ensemb_train_file, _, ensemb_test_file) in station_files.items():
            _, ground_truth_train, wrf_ensemble_train  = self.obtain_dataframe(ensemb_train_file, use_fs_columns)
            dates, ground_truth, wrf_ensemble  = self.obtain_dataframe(ensemb_test_file, use_fs_columns)
            ensemble_values = wrf_ensemble.values

            if model_name == 'PredCSGD':
                pred_csgd = PredictiveCSGD(
                    params=weights_dict[station_name]['pred'],
                    forecast_climat_mean=np.mean(wrf_ensemble_train.values),
                    climat_params=weights_dict[station_name]['climat'],
                    verbose=False
                )
                mean, std, shift = pred_csgd.obtain_defining_parameters(
                    ground_truth.values,
                    ensemble_values,
                    pred_csgd.forecast_climat_mean,
                    pred_csgd.fitted_params,
                    pred_csgd.fitted_climat_params
                )
                if results:
                    crps_test, bias_test, mse_test, bs_test = pred_csgd.predict(
                        ground_truth.values, ensemble_values, metrics='all', return_mean=False, bs_threshold = bs_threshold
                    )
                

            elif model_name == 'ANNCSGD':
                anncsgd = ANNCSGD(verbose=False, input_dim=ensemble_values.shape[1])
                anncsgd.load_state_dict(weights_dict[station_name])
                anncsgd.train_mean = wrf_ensemble_train.values.mean(axis=0)
                anncsgd.train_std = wrf_ensemble_train.values.std(axis=0)
                anncsgd.eval()
                with torch.no_grad():
                    output = anncsgd(torch.from_numpy(ensemble_values).float())
                mean, std, shift = output[:, 0].numpy(), output[:, 1].numpy(), output[:, 2].numpy()
                if results:
                    crps_test, bias_test, mse_test, bs_test = anncsgd.predict(
                        ensemble_values, ground_truth.values, return_mean=False, bs_threshold = bs_threshold
                    )
            elif model_name == 'AnEn':
                best_weights = weights_dict[station_name]["best_weights"]
                best_n_members = weights_dict[station_name]["best_n_members"]
                AnEn = AnalogEnsemble(weights=best_weights, n_members=best_n_members, t_window=3)
                AnEn.hist_predicts = np.array([
                    AnEn._obtain_neighbourhood(wrf_ensemble_train, j) for j in range(len(wrf_ensemble_train))
                ])
                AnEn.hist_observs = ground_truth_train.values
                pred_analogues = AnEn.obtain_analogues(X_test=wrf_ensemble, y_test=ground_truth)
                mean = pred_analogues.mean(axis=1)
                if results:
                    crps_test, bias_test, mse_test, bs_test = AnEn.predict(
                        X_test=wrf_ensemble, y_test=ground_truth, return_mean=False, bs_threshold = bs_threshold
                    )

            elif model_name == 'ensemble':
                crps_ensemble = CRPS_mine()
                mean_ens = ensemble_values.mean(axis=1)
                crps_test =  [crps_ensemble.compute(np.expand_dims(ensemble_values[i,:],-1),obs) for i,obs in enumerate(ground_truth.values) ]
                bias_test = (mean_ens - ground_truth.values)
                mse_test = bias_test**2
                bs_test = threshold_brier_score(ground_truth.values, ensemble_values, threshold = bs_threshold)
                
            # Recolectar resultados por fila
            for i in range(len(dates)):
                row = {
                    'center': station_name,
                    'date': dates[i]
                }
                if model_name == 'ensemble':
                    row['ensemble_mean'] = float(mean_ens[i])
                    row[model_name] = ensemble_values[i,:].tolist()
                    #row['crps'] =  float(crps_ensemble.compute(
                                #X_f=np.expand_dims(wrf_ensemble[i, :], -1),
                                #X_o=ground_truth.values[i]
                                #))
                    #row['mse']

                elif model_name in ['PredCSGD','ANNCSGD']:
                    row[f'{model_name}_mean'] = float(mean[i])
                    row[f'{model_name}_std'] = float(std[i])
                    row[f'{model_name}_shift'] = float(shift) if model_name == 'PredCSGD' else float(shift[i])


                elif model_name == 'AnEn':
                    row[f'{model_name}'] = pred_analogues[i, :].tolist()

                if return_prec:
                    row['prec(mm)'] = float(ground_truth.iloc[i])
                if return_stations:
                    row['center'] = station_name
                if results:
                    row['crps'] = float(crps_test[i])
                    row['bias'] = float(bias_test[i])
                    row['mse'] = float(mse_test[i])
                    for j,th in enumerate(bs_threshold):    
                        #print('Forma de bs_test, i , j:',bs_test.shape, i , j)
                        #print('bs thresholds:',bs_threshold)
                        row[f'bs{th}'] = float(bs_test[j,i])
                all_rows.append(row)

        # Convertimos a DataFrame y ordenamos
        df_result = pd.DataFrame(all_rows)
        df_result = df_result.sort_values(by=['center', 'date'], kind='mergesort').reset_index(drop=True)
        return df_result
    
    def obtain_results_net(self,model_name,params = False, results = True, meteoc_folds = True, bs_threshold:list = [1]):
        net_model, test_set, w_path = self.prepare_data_and_net(model_name = model_name, meteoc_folds = meteoc_folds)

        if meteoc_folds:
            splitter = MeteoCentersSplitter(test_set, nfolds=5, stratify = False)
        else:   
            w_path = [w_path]
            splitter = None
            
        results_list = [] 
        for fold_id, w_p in enumerate(w_path):
            model_data = torch.load(w_p, map_location=torch.device('cpu'), weights_only=True)
            net_model.load_state_dict(model_data['net'])
            if splitter is not None:
                _ , mask_ctrs_tst = splitter[fold_id]
                if 'ds' in model_name:
                    attr_path = os.path.join(main_path,'results','unet_weights','5folds','b32_lr1e-3_11DownSelecLowerCRPSVal',f'fold{fold_id + 1}_sensitivities_batch_LowerCRPSVal.npz')
                    global_attr = np.load(attr_path)
                    attr_total = np.abs(global_attr['ch0']) + np.abs(global_attr['ch1']) + np.abs(global_attr['ch2'])
                    #attr_total_norm = attr_total / np.sum(attr_total)
                    top_11_indices = np.argsort(attr_total)[-11:]
                    mask_wrf_simul_ds = [False if i not in top_11_indices else True for i in range(len(attr_total))]
                    test_set.apply_feature_selec(feat_selec_mask = mask_wrf_simul_ds[:-1])#[:-1] because of HGT map data was put in the past as another value in the mask
            else: 
                mask_ctrs_tst = None

            test_df = test_model_df(net = net_model, data = test_set, device='cpu',mask_centers_tst = mask_ctrs_tst, bs_thresholds = bs_threshold,
                                    return_results=results, return_params=params)
            
            if meteoc_folds:
                #results_dict[f'fold_{fold_id}'] = test_df
                test_df['fold'] = fold_id
                results_list.append(test_df)
            else:
                return test_df

        return pd.concat(results_list).sort_values(by=['center', 'date'], kind='mergesort').reset_index(drop=True)


    def eval(self,model_name,return_params = False, results = True, return_metrics_mean = True, meteo_center_folds = False, **kwargs):
        """
        Evaluates different models based on the specified model name.
        
        Parameters:
        - model_name (str): The name of the model to evaluate. Supports 'ensemble','unet_pca','unet_ds','unet_fs','unet_all'.
        - include_ensemble_metrics (bool): If True, calculates CRPS, MSE, Brier Score, and ensemble mean.
        - return_params (bool): If True, returns model parameters.
        - results (bool): If True, returns model result metrics.
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
        else:
            use_fs_columns = True if model_name.endswith('_fs') else False
            return self.obtain_results_benchmark_df(model_name=model_name,results=results,use_fs_columns=use_fs_columns,**kwargs)



    def obtain_reliability_diagram_old(self,model_name, use_fs_in_benchm, thresholds, nbins = 10):
        if model_name in ['ANNCSGD','AnEn','PredCSGD','ensemble']:
            results_dict = self.obtain_results_benchmark_df(model_name,results=False,use_fs_columns=use_fs_in_benchm)
            print(results_dict.columns)
        else:
            use_fs_in_benchm =False #It is not a benchmark model
            results_dict = self.obtain_results_net(model_name,params=True,results=False,meteoc_folds=False)

        for threshold in thresholds:
            RD = ReliabilityDiagram(threshold,n_bins=nbins)
            threshold_string = f'{threshold}'.replace('.','_')
            plots_path = os.path.join(main_path,'plots','ReliabilityDiagrams',f'threshold_{threshold}')
            if not os.path.exists(plots_path):
                print(f'Creating path {plots_path}')
                os.makedirs(plots_path)

            plot_output_file = os.path.join(plots_path,f'{model_name}_' + ('fs_' if use_fs_in_benchm else '') + threshold_string  + f'nb{nbins}.png')
            if  model_name.startswith('unet'):
                RD.evaluate_csgd(mean = np.array(results_dict['mean']), std = np.array(results_dict['std_dev_no_reg']) + np.sqrt(results_dict['mean']),
                shift = np.array(results_dict['shift']), observed = np.array(results_dict['prec(mm)']), path = plot_output_file)

            elif model_name in ['AnEn','ensemble']:
                RD.evaluate_ensemble(ensemble_values = results_dict[model_name].tolist(),
                                      observed = results_dict['prec(mm)'].values, path = plot_output_file)
            
            elif model_name in ['PredCSGD','ANNCSGD']:
                RD.evaluate_csgd(mean = results_dict[model_name + '_mean'].values, std = results_dict[model_name + '_std'].values,
                shift = results_dict[model_name + '_shift'].values, observed = results_dict['prec(mm)'].values, path = plot_output_file)

    def obtain_reliability_diagram(self, model_names, thresholds, nbins=10, ncols=3,hist_max = None):
        """
        Creates gridded reliability diagrams for a list of models at each threshold.
        Model names ending in '_fs' will use feature-selection columns.

        :param model_names: List[str] model identifiers (append '_fs' for feature-selected benchmark models)
        :param thresholds: List[float] thresholds to plot
        :param nbins: int number of probability bins
        :param ncols: int number of columns in the subplot grid
        """
        # Precompute DataFrames for each model
        results = {}
        for name in model_names:
            use_fs = name.endswith('_fs')
            base_name = name[:-3] if use_fs else name
            if base_name in ['ANNCSGD', 'AnEn', 'PredCSGD', 'ensemble']:
                df = self.obtain_results_benchmark_df(
                    base_name,
                    results=False,
                    use_fs_columns=use_fs
                )
                results[name] = ('benchmark', df)
            else:
                df = self.obtain_results_net(
                    name,
                    params=True,
                    results=False,
                    meteoc_folds=False
                )
                results[name] = ('net', df)

        # Loop thresholds and plot grid
        for threshold in thresholds:
            n_models = len(model_names)
            cols = min(ncols, n_models)
            rows = int(np.ceil(n_models / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
            axes = np.array(axes).reshape(-1)

            for idx, name in enumerate(model_names):
                model_type, df = results[name]
                observed = df['prec(mm)'].values
                RD = ReliabilityDiagram(threshold, n_bins=nbins, hist_max=hist_max)

                if model_type == 'benchmark':
                    base_name = name[:-3] if name.endswith('_fs') else name
                    if base_name in ['AnEn', 'ensemble']:
                        ensemble_vals = np.vstack(df[base_name].tolist()) if base_name == 'ensemble' else df[base_name].tolist()
                        probs = RD.compute_probabilities_ensemb(ensemble_vals)
                    else:
                        mean = df[f'{base_name}_mean'].values
                        std = df[f'{base_name}_std'].values
                        shift = df[f'{base_name}_shift'].values
                        probs = RD.compute_probabilities_csgd(mean, std, shift)
                    title = name[0].upper() + (name[1:-3]+ '-FS' if name.endswith('_fs') else name[1:])
                else:
                    mean = df['mean'].values
                    std = df['std_dev_no_reg'].values + np.sqrt(mean)
                    shift = df['shift'].values
                    probs = RD.compute_probabilities_csgd(mean, std, shift)
                    title = 'UNet-' + {'unet_all':'All','unet_pca':'PCA','unet_ds':'DS','unet_fs':'FS'}[name]

                RD.plot(probs, observed, ax=axes[idx])
                # clean title without suffix
                #title = name[:-3] if name.endswith('_fs') else name
                axes[idx].set_title(title)

            # remove unused axes
            for j in range(n_models, len(axes)):
                fig.delaxes(axes[j])

            fig.suptitle(f"Reliability Diagrams (Threshold = {threshold})")
            fig.tight_layout(rect=[0, 0, 1, 0.96])

            out_dir = os.path.join(main_path, 'plots', 'ReliabilityDiagrams')
            os.makedirs(out_dir, exist_ok=True)
            output_file = os.path.join(out_dir, f'grid_threshold_{threshold}_v3.png')
            fig.savefig(output_file)
            plt.close(fig)

#----------------------------Different usage examples----------------------------------------
#Remember to place UNet weights in the corresponding folders in order to evaluate these models.
""" Obtain reliability diagrams for different thresholds
eval = evaluator()
model_names = ['ensemble','ensemble_fs','unet_all','unet_ds','unet_fs','unet_pca','ANNCSGD','ANNCSGD_fs','PredCSGD','PredCSGD_fs','AnEn','AnEn_fs']
eval.obtain_reliability_diagram(model_names,[0.1],nbins=10,hist_max=3*10**5)
"""


""" Obtain metric results of different models
bs = [0.1,0.3,0.5,1]
df_ensemble =eval.obtain_metrics_ensemble(return_metrics_mean=False,use_fs_columns=False,meteoc_folds=False,threshold=bs)
df_ensemble.to_csv('results_ensemble.csv')

df_ensemble_fs =eval.obtain_metrics_ensemble(return_metrics_mean=False,use_fs_columns=True,meteoc_folds=False,threshold=bs)
df_ensemble_fs.to_csv('results_ensemble_fs.csv')

#eval.assert_csvs_and_wrfmaps()
df_all = eval.obtain_results_net('unet_all',params=True,meteoc_folds=False,bs_threshold=bs)
print(np.mean(df_all['crps']))
df_all.to_csv('results_unet_all.csv')

df_ds = eval.obtain_results_net('unet_ds',params=True,meteoc_folds=False,bs_threshold=bs)
print(np.mean(df_ds['crps']))
df_ds.to_csv('results_unet_ds.csv')

df_fs = eval.obtain_results_net('unet_fs',params=True,meteoc_folds=False,bs_threshold=bs)
print(np.mean(df_fs['crps']))
df_fs.to_csv('results_unet_fs.csv')

df_pca = eval.obtain_results_net('unet_pca',params=True,meteoc_folds=False,bs_threshold=bs)
print(np.mean(df_pca['crps']))
df_pca.to_csv('results_unet_pca.csv')

df_ANN_fs = eval.obtain_results_benchmark_df('ANNCSGD', use_fs_columns=True,bs_threshold=bs)
print(np.mean(df_ANN_fs['crps']))
df_ANN_fs.to_csv('results_ANNCSGD_fs.csv')

df_ANN = eval.obtain_results_benchmark_df('ANNCSGD', use_fs_columns=False,bs_threshold=bs)
print(np.mean(df_ANN['crps']))
df_ANN.to_csv('results_ANNCSGD.csv')

df_PredCSGD_fs = eval.obtain_results_benchmark_df('PredCSGD', use_fs_columns=True,bs_threshold=bs)
print(np.mean(df_PredCSGD_fs['crps']))
df_PredCSGD_fs.to_csv('results_PredCSGD_fs.csv')

df_PredCSGD = eval.obtain_results_benchmark_df('PredCSGD', use_fs_columns=False,bs_threshold=bs)
print(np.mean(df_PredCSGD['crps']))
df_PredCSGD.to_csv('results_PredCSGD.csv')

df_AnEn_fs = eval.obtain_results_benchmark_df('AnEn', use_fs_columns=True,bs_threshold=bs)
print(np.mean(df_AnEn_fs['crps']))
df_AnEn_fs.to_csv('results_AnEn_fs.csv')

df_AnEn = eval.obtain_results_benchmark_df('AnEn', use_fs_columns=False,bs_threshold=bs)
print(np.mean(df_AnEn['crps']))
df_AnEn.to_csv('results_AnEn.csv')"""

#df_all = eval.obtain_results_net('unet_all',params=True,meteoc_folds=True,bs_threshold=bs)
#print(np.mean(df_all['crps']))
#df_all.to_csv('results_unet_all_5folds.csv')

"""df_ds = eval.obtain_results_net('unet_ds',params=True,meteoc_folds=True,bs_threshold=bs)
print(np.mean(df_ds['crps']))
df_ds.to_csv('results_unet_ds_5folds.csv')

df_fs = eval.obtain_results_net('unet_fs',params=True,meteoc_folds=True,bs_threshold=bs)
print(np.mean(df_fs['crps']))
df_fs.to_csv('results_unet_fs_5folds.csv')

df_pca = eval.obtain_results_net('unet_pca',params=True,meteoc_folds=True,bs_threshold=bs)
print(np.mean(df_pca['crps']))
df_pca.to_csv('results_unet_pca_5folds.csv')"""