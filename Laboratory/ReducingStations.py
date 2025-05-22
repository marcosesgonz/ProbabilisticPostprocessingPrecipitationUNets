import os 
import sys
main_path = os.path.abspath(os.path.join(__file__,'..','..'))
sys.path.append(main_path)
import torch 
import wandb
import numpy as np
import random
import models
import utils as u
from Datasets import WRFdataset,MeteoCentersSplitter, desired_configs
import argparse

def reduction_trainstations_experiment(name_experim, project_ref='ReducedTrainStations', epochs=40, optimizer_name='adamw',
                                    normalize_data=True, pca_data=True, feat_selec_data=False,
                                       batch_size=32, lr=1e-3, gpu=False, reduct_factor=1):
    #set the seed for reproducibility
    #Registro en wandb para la monitorización
    wandb.login()
    u.set_seed(24)

    with wandb.init(project=project_ref, name=name_experim, dir=os.path.dirname(__file__)):
        device = ("cuda" if (torch.cuda.is_available() and gpu) else 'mps' if gpu else 'cpu')

        # Cargar datasets
        train_set = WRFdataset(data_subset='train', group_configs=desired_configs, station_split=False, wrf_variables=['prec'])
        val_set = WRFdataset(data_subset='val', group_configs=desired_configs, station_split=False, wrf_variables=['prec'])
        test_set = WRFdataset(data_subset='test', group_configs=desired_configs, station_split=False, wrf_variables=['prec'])

        # Normalización y reducción dimensional
        if normalize_data:
            train_set.normalize_data()
            val_set.normalize_data(train_set.mean_og_data, train_set.std_og_data)
            test_set.normalize_data(train_set.mean_og_data, train_set.std_og_data)

        if pca_data:
            train_set.apply_PCA(ncomponents=10)
            val_set.apply_PCA(eigvec_transp=train_set.W_t_PCA)
            test_set.apply_PCA(eigvec_transp=train_set.W_t_PCA)
            n_inp_channels = len(train_set.eigv_PCA) + 1
        elif feat_selec_data:
            train_set.apply_feature_selec(10)
            val_set.apply_feature_selec(feat_selec_mask=train_set.feat_selec_mask)
            test_set.apply_feature_selec(feat_selec_mask=train_set.feat_selec_mask)
            n_inp_channels = np.sum(train_set.feat_selec_mask) + 1 #Plus HGT map
        else:
            n_inp_channels = train_set.num_ensembles + 1

        # Obtener máscaras del primer fold
        splitter = MeteoCentersSplitter(train_set, nfolds=5, stratify=False)
        mask_ctrs_tr_full, mask_ctrs_tst = splitter[0]
        tst_ctrs_names = train_set.meteo_centers_info['name'][mask_ctrs_tst]
        print(f"Centers test: {tst_ctrs_names}")
        #total_train_stations = np.where(mask_ctrs_tr_full)[0]
        all_train_indices = np.where(mask_ctrs_tr_full)[0]
        np.random.shuffle(all_train_indices)

        reductions = [0.75, 0.5,0.25]  # Proporciones del conjunto de entrenamiento (60%, 40% y 20% del total)
        out_dir = os.path.join(os.path.dirname(__file__), f'result_logs_reduction/b{batch_size}_lr{lr}_{name_experim}')
        os.makedirs(out_dir, exist_ok=True)

        for red in reductions:
            u.set_seed(24)
            selected_indices = all_train_indices[:int(len(all_train_indices) * red)]
            #selected_train_stations = np.array(random.sample(list(total_train_stations), int(len(total_train_stations) * red)))
            mask_ctrs_tr = np.zeros_like(mask_ctrs_tr_full, dtype=bool)
            mask_ctrs_tr[selected_indices] = True
            assert not np.any(mask_ctrs_tr & mask_ctrs_tst)

            tr_ctrs_names = train_set.meteo_centers_info['name'][mask_ctrs_tr]
            print(f'Using {np.sum(mask_ctrs_tr)} training centers: {tr_ctrs_names}')
            # Crear red y optimizador
            net = models.UNet(n_inp_channels=n_inp_channels, n_outp_channels=3, red_factor=reduct_factor, bottleneck=False)
            net.to(device)

            optimizer = {
                'adam': torch.optim.Adam,
                'adamw': torch.optim.AdamW,
                'sgd': lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.9)
            }.get(optimizer_name, None)

            if optimizer is None:
                raise NotImplementedError

            optimizer = optimizer(net.parameters(), lr=lr)
            lr_scheduler = None  # Puedes ajustar esto si quieres

            # Dataloaders
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

            # Entrenamiento
            min_val_loss = 1e4
            patience = 10
            patience_counter = 0

            for epoch in range(epochs):
                train_loss = u.train_model(net, train_loader, train_set, mask_ctrs_tr,optimizer, device, False, lr_scheduler)
                val_loss, val_bias, val_mse, val_bs = u.test_model(net = net,tst_loader = val_loader,
                                                                    data = val_set,mask_centers_tst = mask_ctrs_tst, device = device)

                wandb.log({
                    f'train_loss_{int(80 * red)}%': train_loss,
                    f'val_loss_{int(80 * red)}%': val_loss,
                    f'val_bias_{int(80 * red)}%': val_bias,
                    f'val_mse_{int(80 * red)}%': val_mse,
                    f'val_bs(0.1mm)_{int(80 * red)}%': val_bs,
                    'epoch': epoch,
                    'reduction': f'{int(80 * red)}%'
                })

                if val_loss < min_val_loss:
                    wandb.run.summary[f'min_val_loss_{int(80 * red)}%'] = min_val_loss
                    min_val_loss = val_loss
                    patience_counter = 0
                    torch.save(net.state_dict(), os.path.join(out_dir, f'best_model_{int(80 * red)}.pt'))
                else:
                    patience_counter += 1

                if patience_counter >= patience and epoch > 24:
                    print('Early stopping')
                    break
            
            net.load_state_dict(torch.load(os.path.join(out_dir, f'best_model_{int(80 * red)}.pt')))

            print('Comprobación:')
            val_df = u.test_model_df(net = net,tst_loader = val_loader, data = val_set, mask_centers_tst = mask_ctrs_tst, device = device, bs_thresholds=0.1)
            recovered_value = val_df['crps'].mean()
            print(f'El val loss recargando el mejor modelo es {recovered_value} y se tiene guardado el valor {min_val_loss}')
            # Evaluación en test con el mejor modelo
            test_df = u.test_model_df(net = net,tst_loader = test_loader, data = test_set, mask_centers_tst = mask_ctrs_tst, device = device, bs_thresholds=[0.1,0.3,0.5,1,4])
            test_df.to_csv(os.path.join(out_dir,f'test_results_{int(80 * red)}.csv'))
            wandb.run.summary[f'test_loss_{int(80 * red)}%'] = test_df['crps'].mean()
            wandb.run.summary[f'test_bias_{int(80 * red)}%'] = test_df['bias'].mean()
            wandb.run.summary[f'test_mse_{int(80 * red)}%'] = test_df['mse'].mean()
            wandb.run.summary[f'test_bs(0.1mm)_{int(80 * red)}%'] = test_df['bs0.1'].mean()

def parse_args():
    parser = argparse.ArgumentParser(description="Execute progressive station-reduction UNet experiment")
    parser.add_argument('--project_ref', type=str, default='ReducedTrainStations', help='Project reference for wandb')
    parser.add_argument('--name_experim', type=str, required = True, help='Experiment name for wandb')
    parser.add_argument('--epochs', type=int, default = 40, help='Number of epochs (default: 40)')
    parser.add_argument('--batch_size', type=int, default = 32, help='Batch size (default: 32)')
    parser.add_argument('--optimizer', type=str, default = 'adamw', help='Optimizer used during training (default: adamW)')
    parser.add_argument('--lr', type=float, default = 0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--normalize', action = 'store_true', help = 'Normalize data')
    parser.add_argument('--PCA', action = 'store_true', help = 'Apply PCA along ensemble precipitations')
    parser.add_argument('--FeatSelec', action = 'store_true', help = 'Apply Feature selection')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    reduction_trainstations_experiment(project_ref = args.project_ref, name_experim = args.name_experim, epochs = args.epochs,
                    batch_size = args.batch_size, optimizer_name = args.optimizer, lr = args.lr,
                    normalize_data = args.normalize, pca_data = args.PCA, feat_selec_data = args.FeatSelec)
    #In order to reproduce our work results: 
    # 'reduction_trainstations_experiment(name_experim='TrainStationsReduction_FS',pca_data=False, feat_selec_data=True,normalize_data = True)'

