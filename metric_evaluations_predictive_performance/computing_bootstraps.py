import pandas as pd
import numpy as np
import os
from arch.bootstrap import CircularBlockBootstrap
import argparse

class Bootstrap:
    def __init__(self, data):
        self.data = np.array(data)

    """def compute_bootstraps(self, num_bootstraps:int =1000, sample_size:int = None):
        if sample_size is None:
            sample_size = len(self.data)
        boot_means = np.array([
            np.mean(np.random.choice(self.data, size=sample_size, replace=True))
            for _ in range(num_bootstraps)
        ])
        return boot_means"""
    
    def compute_block_bootstraps(self, num_bootstraps=1000, block_size=24):
        mbb = CircularBlockBootstrap(block_size, self.data)
        results = [np.mean(data[0]) for data in mbb.bootstrap(num_bootstraps)]
        return np.array(results)

def compute_bootstrap_from_csv(csv_paths, model_names, reference_name="Ensemble",block_size = 10, num_bootstraps=1000):
    assert len(csv_paths) == len(model_names), "CSV paths and model names must align"

    #metrics = ['crps', 'bias', 'mae', 'mse', 'bs0.1', 'bs0.3', 'bs0.5', 'bs1', 'bs2', 'bs3', 'bs4']
    metrics = ['crps', 'bias', 'mae', 'mse', 'bs0.1','bs0.3', 'bs0.5']
    eps = 1e-3
    all_data = {}

    # Load all CSVs into memory
    for path, name in zip(csv_paths, model_names):
        df = pd.read_csv(path)
        df['Model'] = name
        all_data[name] = df
        if name != reference_name:
            stations = df['center'].unique()

    # Use 'Ensemble' as reference
    reference_df = all_data[reference_name]
    print(reference_df.columns)
    #stations = reference_df['center'].unique()

    results = []

    for metric in metrics:
        for model, df_model in all_data.items():
            if model == reference_name and metric != 'bias':
                continue  # Skip comparison to self
            print(f"Bootstrapping {metric} - {model}")
            for station in stations:
                if metric != 'mae':
                    ref_values = reference_df[reference_df['center'] == station][metric].dropna().values
                    model_values = df_model[df_model['center'] == station][metric].dropna().values
                else:
                    # MAE = abs(Bias)
                    ref_values = np.abs(reference_df[reference_df['center'] == station]['bias'].dropna().values)
                    model_values = np.abs(df_model[df_model['center'] == station]['bias'].dropna().values)
                assert len(ref_values) == len(model_values)

                if len(ref_values) == 0:
                    print(f"Warning: Station {station} doesn't have values")
                    continue  # Skip if no valid data
                
                if metric == 'bias':
                    #ref_values = reference_df[(reference_df['center'] == station) & (reference_df['prec(mm)'] > 0.1)][metric].dropna().values
                    #model_values = df_model[(df_model['center'] == station) & (df_model['prec(mm)'] > 0.1) ][metric].dropna().values
                    #metric_name = 'thr_' + metric
                    #Calculate relative bias
                    model_values = model_values/(df_model[df_model['center'] == station]['prec(mm)'].dropna().values + 0.1)
                    metric_name = 'rel_' + metric

 
                #if (np.mean(ref_values) <= eps) and metric != 'bias':
                #In order to avoid numerical inestabilities
                ref_values += eps
                model_values += eps

                bootstrap = Bootstrap(model_values)
                boot_means = bootstrap.compute_block_bootstraps(num_bootstraps,block_size)

                #0. brier score con percentil 99 para cada estación
                if metric == 'mse':
                    bootstraped_result = 1 - np.sqrt(boot_means) / np.sqrt(np.mean(ref_values))
                    raw_result = 1 - np.sqrt(np.mean(model_values)) / np.sqrt(np.mean(ref_values))
                    metric_name = 'rmsess'
                elif metric == 'bias':
                    #1. (observ > 0.1)
                    #2. (pred - observ)/(observ + 0.1)  rel_bias
                    bootstraped_result = boot_means 
                    raw_result = np.mean(model_values)
                    #metric_name = metric
                else:
                    bootstraped_result = 1 - boot_means / np.mean(ref_values)
                    raw_result = 1 - np.mean(model_values) / np.mean(ref_values)
                    metric_name = metric + 'ss'

                results.append({
                    'Metric': metric_name,
                    'Model': model,
                    'Station': station,
                    'Mean': bootstraped_result.mean(),
                    'Var': bootstraped_result.var(),
                    'RawMean': raw_result
                })

    return pd.DataFrame(results)

def final_results_from_bootstraps(bootstrap_results_df):
    final_results = []
    for (metric, model), group in bootstrap_results_df.groupby(['Metric', 'Model']):
        station_means = group['RawMean']
        station_variances = group['Var']
        aggregated_mean = station_means.mean()
        aggregated_std = np.sqrt(station_variances.sum()) / len(station_variances)
        lower_ci = aggregated_mean - 1.96 * aggregated_std
        upper_ci = aggregated_mean + 1.96 * aggregated_std
        final_results.append({
            'Metric': metric,
            'Model': model,
            'Mean': aggregated_mean,
            'Lower CI': lower_ci,
            'Upper CI': upper_ci
        })
    return pd.DataFrame(final_results)

#
csv_names_overall = [
    'results_unet_all.csv',
    'results_unet_ds.csv',
    'results_unet_fs.csv',
    'results_unet_pca.csv',
    'results_ANNCSGD.csv',
    'results_ANNCSGD_fs.csv',
    'results_PredCSGD.csv',
    'results_PredCSGD_fs.csv',
    'results_AnEn.csv',
    'results_AnEn_fs.csv',
    'results_ensemble.csv',
    'results_ensemble_fs.csv'
]

csv_names_5folds = [
    'results_unet_all_5folds.csv',
    'results_unet_ds_5folds.csv',
    'results_unet_fs_5folds.csv',
    'results_unet_pca_5folds.csv',
    'results_ensemble.csv',
    'results_ensemble_fs.csv'
]

csv_names_StatRed_fs = [os.path.join('models_results','station_reduction_results_11FS',result_file) for result_file in ['test_results_80.csv',
                                                                                                                        'test_results_60.csv',
                                                                                                                        'test_results_40.csv',
                                                                                                                        'test_results_20.csv']]                 
csv_names_StatRed_fs = csv_names_StatRed_fs + [os.path.join('models_results','overall','results_ensemble.csv')]


model_names_overall = [
    'UNet-All',
    'UNet-DS',
    'UNet-FS',
    'UNet-PCA',
    'ANN-CSGD',
    'ANN-CSGD-FS',
    'PredCSGD',
    'PredCSGD-FS',
    'AnEn',
    'AnEn-FS',
    'Ensemble',
    'Ensemble-FS'
]

model_names_5folds = [
    'UNet-All',
    'UNet-DS',
    'UNet-FS',
    'UNet-PCA',
    'Ensemble',
    'Ensemble-FS'
]

model_names_StatRed_fs = [
    'UNet-FS-80',
    'UNet-FS-20',
    'UNet-FS-40',
    'UNet-FS-60',
    'Ensemble'

]

folder_5folds = os.path.join(os.path.dirname(__file__),'models_results','5folds')
folder_overall = os.path.join(os.path.dirname(__file__),'models_results','overall')

def parse_args():
    parser = argparse.ArgumentParser(description='Execute Bootstrapping as it was done in our work.' \
                                    'You will need to place the csv result files in the list of paths presented in the code (or create your own paths).')
    parser.add_argument('--type_of_experiment', type=str,default = 'StatRed',
                        choices=['Overall','FiveFolds','StatRed'], help='Which experiment you want to boot strap')
    parser.add_argument('--block_size', type=int, default = 10, help='Block size used in bootstrapping')
    parser.add_argument('--num_bootstraps', type=int, default = 1000, help='Number of boot straps done.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.type_of_experiment == 'FiveFolds':
        csv_paths = [os.path.join(folder_5folds,csv_name) if 'ensemble' not in csv_name else os.path.join(folder_overall,csv_name) for csv_name in csv_names_5folds]
        model_names = model_names_5folds
    elif args.type_of_experiment == 'Overall':
        csv_paths = [os.path.join(folder_overall, csv_name) for csv_name in csv_names_overall]
        model_names = model_names_overall
    elif args.type_of_experiment == 'StatRed':
        csv_paths = [os.path.join(os.path.dirname(__file__), csv_name) for csv_name in csv_names_StatRed_fs]
        model_names = model_names_StatRed_fs

    bootstrap_df = compute_bootstrap_from_csv(csv_paths, model_names, reference_name="Ensemble", num_bootstraps=args.num_bootstraps,block_size= args.block_size)
    final_df = final_results_from_bootstraps(bootstrap_df)

    # Save results
    bootstrap_df.to_csv(f"bootstrap_per_station_blocks{args.bloq_size}_{args.type_of_experiment}", index=False)
    final_df.to_csv(f"bootstrap_aggregated_blocks{args.bloq_size}_{args.type_of_experiment}", index=False)

#Bootstrapping reducing stations experiment
"""
bootstrap_df = compute_bootstrap_from_csv(csv_names_StatRed_fs, model_names_StatRed_fs, reference_name="Ensemble", num_bootstraps=500,block_size = bloqsize)
final_df = final_results_from_bootstraps(bootstrap_df)
bootstrap_df.to_csv(f'bootstrap_per_station_blocks{bloqsize}_StatRed.csv')
final_df.to_csv(f'bootstrap_aggregated_blocks{bloqsize}_StatRed.csv')"""