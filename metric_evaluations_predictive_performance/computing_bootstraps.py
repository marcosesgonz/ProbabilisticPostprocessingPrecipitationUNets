import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

#Change this boolean variable to switch between generalization results and overall forecasting results.
COMPUTE_FIVE_FOLDS = True


class Bootstrap:
    def __init__(self, data, labels=None):
        self.data = np.array(data)
        self.labels = np.array(labels) if labels is not None else None

    def compute_bootstraps(self, num_bootstraps, sample_size=None):
        if sample_size is None:
            sample_size = len(self.data)
        bootstrap_means = np.array([
            np.mean(np.random.choice(self.data, size=sample_size, replace=True)) 
            for _ in range(num_bootstraps)
        ])
        return bootstrap_means

    def confidence_interval(self, bootstrap_means, confidence_level=0.95):
        mean_value = np.mean(bootstrap_means)
        lower_bound = np.percentile(bootstrap_means, (1 - confidence_level) / 2 * 100)
        upper_bound = np.percentile(bootstrap_means, (1 + confidence_level) / 2 * 100)
        return mean_value, lower_bound, upper_bound

def obtain_results(compute_five_folds_results = False):
    path_results = os.path.join(os.path.abspath(os.path.join(__file__,'..','..')),'results')
    if compute_five_folds_results:
        five_folds_results = np.load(os.path.join(path_results,'results_5folds.npz'))
        overall_results = np.load(os.path.join(path_results,'results_overall.npz'))#Previously named 'New_results_overall.npz'
        ensemble_fs_results = {key:value for key,value in overall_results.items() if 'ensemble_fs' in key}

        ordered_indices_five_folds = np.argsort(five_folds_results['center'])
        ordered_indices_overall = np.argsort(overall_results['center'])
        print(np.all(overall_results['center'][ordered_indices_overall] == five_folds_results['center'][ordered_indices_five_folds]))

        results = dict(five_folds_results) | dict(ensemble_fs_results)
        final_results = {key:np.array(value)[ordered_indices_overall] if ('ensemble_fs' in key) else np.array(value)[ordered_indices_five_folds] for key,value in results.items()}
        ordered_centers = final_results.pop('center')
    else:
        results = np.load(os.path.join(path_results,'results_overall.npz'))#Previously named 'New_results_overall.npz'

        print(len(results['center']),len(results['center_emos'])) #The first one is for unet/ensemble models. The second one for PredCSGD and AnEn.
        print('Name differences:')
        centers_unique = np.unique(results['center'])
        emos_centers = results['center_emos']
        emos_centers_unique = np.unique(emos_centers)
        for center in centers_unique: #IF i put unique here, it goes toooo slow (why?)
            if center not in emos_centers_unique:
                print(center)
        print('--------------------------'*3)
        for emos_center in emos_centers_unique:
            if emos_center not in centers_unique:
                print(emos_center)

        #Fixing differences
        unet_ensemb_centers = [name.replace('/','_') for name in results['center']]
        print('Corrected:',set(unet_ensemb_centers) == set(emos_centers))


        ordered_indices_unet_ensemb = np.argsort(unet_ensemb_centers)
        ordered_indices_emos = np.argsort(emos_centers)
        final_results = {key:np.array(value)[ordered_indices_emos] if ('PredCSGD' in key or 'AnEn' in key or 'center_emos' == key or 'AnnCSGD' in key) else np.array(value)[ordered_indices_unet_ensemb] for key,value in results.items()}
        ordered_centers = final_results.pop('center_emos')
        ordered_unet_ensemb_centers = final_results.pop('center')
        print(np.all([center_.replace('/','_') for center_ in ordered_unet_ensemb_centers] == ordered_centers))

    final_results =  dict(final_results)
    return final_results, ordered_centers


def compute_bootstrap_results(final_results,compute_five_folds_results = True,
                              num_bootstraps = 1000):
    bootstrap_results = []
    metrics = ['mae', 'mse', 'crps', 'bs']
    if not compute_five_folds_results:
        models = ['fullch', 'ds', 'pca', 'fs', 'AnEn','AnEn_fs', 'PredCSGD','PredCSGD_fs','AnnCSGD','AnnCSGD_fs','ensemble_fs']
    else:
        models  = ['fullch', 'ds', 'pca', 'fs','ensemble_fs']

    for metric in metrics:
        reference_col = f"{metric}_ensemble"
    
        for model in models:
            model_col = f"{metric}_{model}"
            print(f'Computing bootstrap for {model_col}')
            # Extract station-level values for the model and reference
            reference_values = final_results[reference_col]
            model_values = final_results[model_col]

            for center in np.unique(ordered_centers):
                mask_station = (center == ordered_centers)

                # Apply bootstrapping for each model/metric
                one_station_model_values = model_values[mask_station]
                one_station_reference_values = reference_values[mask_station]
                if (np.mean(one_station_reference_values) <= 1e-3):
                    #print(metric, reference_result.index[reference_result == 0])
                    one_station_model_values += 1e-3
                    one_station_reference_values += 1e-3
                #one_station_skill_scores = 1 - (one_station_model_values)/(np.mean(one_station_reference_values))

                bootstrap = Bootstrap(data=one_station_model_values)
                boot_means = bootstrap.compute_bootstraps(num_bootstraps=num_bootstraps)
                if metric == 'mse': #Compute RMSE instead of MSE
                    boot_skill_scores = 1 - np.sqrt(boot_means)/np.sqrt(np.mean(one_station_reference_values))
                else:
                    boot_skill_scores = 1 - boot_means/np.mean(one_station_reference_values)
                # Compute mean and confidence intervals from bootstrap samples
                mean_value = boot_skill_scores.mean()
                var_value = boot_skill_scores.var()
                #mean_value, lower_ci, upper_ci = bootstrap.confidence_interval(boot_means)
                raw_mean = 1 - (np.mean(one_station_model_values))/(np.mean(one_station_reference_values))
                # Append bootstrap results for plotting
                bootstrap_results.append({
                    'Metric': metric if metric != 'mse' else 'rmse',
                    'Model': model,
                    'Station':center,
                    'Mean': mean_value,
                    'Var': var_value,
                    'RawMean': raw_mean
                    #'Lower CI': lower_ci,
                    #'Upper CI': upper_ci
                })

    #  Convert to DataFrame for easier analysis and visualization
    bootstrap_results_df =  pd.DataFrame(bootstrap_results)
    if compute_five_folds_results:
        bootstrap_results_df['Model'].replace({'fs':'UNet-FS','ds':'UNet-DS','pca':'UNet-PCA','fullch':'UNet-All','ensemble_fs':'Ensemble-FS'},
                                           inplace=True)
    else:
        bootstrap_results_df['Model'].replace({'AnEn_fs':'AnEn-FS','AnnCSGD_fs':'ANN-CSGD-FS','AnnCSGD':'ANN-CSGD','PredCSGD_fs':'PredCSGD-FS','fs':'UNet-FS','ds':'UNet-DS','pca':'UNet-PCA','fullch':'UNet-All','ensemble_fs':'Ensemble-FS'}, inplace=True)
    
    return bootstrap_results_df

def final_results_from_bootstraps(bootstrap_results_df):
    final_results = []
    for (metric, model), group in bootstrap_results_df.groupby(['Metric', 'Model']):
        station_means = group['Mean']
        station_variances = group['Var']#(group['Upper CI'] - group['Lower CI'])**2 / 4  # Approximation from CI width

        # Compute weighted average and final confidence intervals
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

final_results, ordered_centers = obtain_results(compute_five_folds_results=COMPUTE_FIVE_FOLDS)
bootstrap_results_df = compute_bootstrap_results(final_results = final_results,
                                                compute_five_folds_results=COMPUTE_FIVE_FOLDS,
                                                num_bootstraps=2000)
final_results_df = final_results_from_bootstraps(bootstrap_results_df = bootstrap_results_df)


if COMPUTE_FIVE_FOLDS:
    # Plotting the aggregated results
    metrics = ['mae', 'mse', 'crps', 'bs']
    os.mkdir('plots/SpatialGeneralization') if not os.path.exists('plots/SpatialGeneralization') else None
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        # Overlay error bars for aggregated bootstrap mean and 95% CI
        br_filtered = bootstrap_results_df[bootstrap_results_df['Metric'] == metric]
        print(len(br_filtered['Mean']))
        ci_filtered = final_results_df[final_results_df['Metric'] == metric]
        model_order = br_filtered['Model'].unique()
        sns.boxplot(
            data=br_filtered, order= model_order,
            x='Model', y='Mean', showfliers=False, whis=(10,90), width=0.4
        )
        # Overlay error bars for 95% confidence intervals
        for x_pos, model in enumerate(model_order):  # Ensure same order
            row = ci_filtered[ci_filtered['Model'] == model]
            if not row.empty:  # Check if model exists in final_results_df
                mean_val = row['Mean'].values[0]
                lower_ci = row['Lower CI'].values[0]
                upper_ci = row['Upper CI'].values[0]

                plt.errorbar(
                    x_pos, mean_val,
                    yerr=[[mean_val - lower_ci], [upper_ci - mean_val]],
                    fmt='o', color='red', capsize=5, markersize=5
                )
        plt.grid(alpha = 0.5)
        plt.hlines(0,-0.3, len(ci_filtered) - 0.5,'r',linestyles='dashed')
        plt.xlim(-0.3, len(ci_filtered) - 0.7)
        plt.ylabel('')
        plt.yticks(fontsize = 13)
        plt.xticks(ticks=range(len(model_order)), labels=model_order, rotation=0, fontsize=14)
        plt.xlabel('')
        plt.tight_layout()
        plt.savefig(f'plots/SpatialGeneralization/BootStrap{metric.upper()}SkillScore_95CI_SpatGen_OtherAspRatio')
        plt.title(f"Bootstrap Error Bars for {metric.upper()} Skill Scores (95% CI)")
        plt.show() 
else:
    # Plotting the aggregated results
    metrics = ['mae', 'mse', 'crps', 'bs']
    os.mkdir('plots/Overall') if not os.path.exists('plots/Overall') else None
    for metric in metrics:
        plt.figure(figsize=(10, 7))
        # Overlay error bars for aggregated bootstrap mean and 95% CI
        br_filtered = bootstrap_results_df[bootstrap_results_df['Metric'] == metric]
        print(len(br_filtered['Mean']))
        ci_filtered = final_results_df[final_results_df['Metric'] == metric]
        model_order = br_filtered['Model'].unique()
        sns.boxplot(
            data=br_filtered, order= model_order,
            x='Model', y='RawMean', showfliers=False, whis=(5,95), width=0.4
        )
        # Overlay error bars for 95% confidence intervals
        for x_pos, model in enumerate(model_order):  # Ensure same order
            row = ci_filtered[ci_filtered['Model'] == model]
            if not row.empty:  # Check if model exists in final_results_df
                mean_val = row['Mean'].values[0]
                lower_ci = row['Lower CI'].values[0]
                upper_ci = row['Upper CI'].values[0]

                plt.errorbar(
                    x_pos, mean_val,
                    yerr=[[mean_val - lower_ci], [upper_ci - mean_val]],
                    fmt='o', color='red', capsize=5, markersize=5
                )
        
        plt.grid(alpha = 0.5)
        plt.hlines(0,-0.3, len(ci_filtered) - 0.5,'r',linestyles='dashed')
        plt.xlim(-0.3, len(ci_filtered) - 0.7)
        plt.ylabel(metric.upper() + ('S' if metric.endswith('s') else 'SS'))
        plt.xlabel('')
        plt.xticks(ticks=range(len(model_order)), labels=model_order, rotation=20, fontsize=11)
        plt.tight_layout()
        if metric =='rmse':
            plt.savefig(f'plots/Overall/BootStrap{metric.upper()}SkillScore_95CI_OverAll')
        plt.title(f"Bootstrap Error Bars for {metric.upper()} Skill Scores (95% CI)")
        plt.show()