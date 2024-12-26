from Datasets import WRFdataset, desired_configs
import numpy as np
import os
import pandas as pd

train_set = WRFdataset(data_subset = 'train')
val_set = WRFdataset(data_subset = 'val')
test_set = WRFdataset(data_subset = 'test')
all_stations = train_set.meteo_data.columns[1:]
print(len(all_stations))

for idx,station in enumerate(all_stations):
    print(idx, station, train_set.meteo_centers_info['name'][idx])
    station_meteo_data_train_dates = train_set.meteo_data.loc[:,['date', station]]
    station_meteo_data_val_dates = val_set.meteo_data.loc[:,['date', station]]
    station_meteo_data_test_dates = test_set.meteo_data.loc[:,['date', station]]
    x,y = train_set.meteo_centers_info['x_y_d04'][idx]

    wrf_data_train_dates = []
    for day in train_set.dates:
        rain_data = np.load(os.path.join('data','temp_wrf_data_for_model',f'{day}_PrHourlyCropped.npy')) #Dims: [N_ensembles, times_hourly, height , width]
        wrf_data_train_dates.append(rain_data[:,:, y - train_set.crop_y[0], x - train_set.crop_x[0] ])
    wrf_data_train_dates = np.concatenate(wrf_data_train_dates,axis=1) #Dims: [N_ensembles, times_hourly * N_days_train]
    wrf_data_train_dates = np.moveaxis(wrf_data_train_dates,0,1)
    meteo_plus_wrf_data_train_dates = np.concatenate([station_meteo_data_train_dates.values, wrf_data_train_dates],axis = 1)

    wrf_data_val_dates = []
    for day in val_set.dates:
        rain_data = np.load(os.path.join('data','temp_wrf_data_for_model',f'{day}_PrHourlyCropped.npy')) #Dims: [N_ensembles, times_hourly, height , width]
        wrf_data_val_dates.append(rain_data[:,:, y - val_set.crop_y[0], x - val_set.crop_x[0] ])
    wrf_data_val_dates = np.concatenate(wrf_data_val_dates,axis=1) #Dims: [N_ensembles, times_hourly * N_days_test]
    wrf_data_val_dates = np.moveaxis(wrf_data_val_dates,0,1)
    meteo_plus_wrf_data_val_dates = np.concatenate([station_meteo_data_val_dates.values, wrf_data_val_dates],axis = 1)

    wrf_data_test_dates = []
    for day in test_set.dates:
        rain_data = np.load(os.path.join('data','temp_wrf_data_for_model',f'{day}_PrHourlyCropped.npy')) #Dims: [N_ensembles, times_hourly, height , width]
        wrf_data_test_dates.append(rain_data[:,:, y - test_set.crop_y[0], x - test_set.crop_x[0] ])
    wrf_data_test_dates = np.concatenate(wrf_data_test_dates,axis=1) #Dims: [N_ensembles, times_hourly * N_days_test]
    wrf_data_test_dates = np.moveaxis(wrf_data_test_dates,0,1)
    meteo_plus_wrf_data_test_dates = np.concatenate([station_meteo_data_test_dates.values, wrf_data_test_dates],axis = 1)

    onestation_data_train_final = pd.DataFrame(meteo_plus_wrf_data_train_dates,columns = ['date',station] + desired_configs)
    onestation_data_val_final = pd.DataFrame(meteo_plus_wrf_data_val_dates,columns = ['date',station] + desired_configs)
    onestation_data_test_final = pd.DataFrame(meteo_plus_wrf_data_test_dates,columns = ['date',station] + desired_configs)
    
    csvs_path = os.path.join('data','csvs_for_classic_methods')
    if not os.path.exists(csvs_path):
        print('Creating directory for csvs:',csvs_path)
        os.makedirs(csvs_path)

    onestation_data_train_final.to_csv(os.path.join(csvs_path,f'{station.replace("/","_")}_train.csv'),index=False)
    onestation_data_val_final.to_csv(os.path.join(csvs_path,f'{station.replace("/","_")}_val.csv'),index=False)
    onestation_data_test_final.to_csv(os.path.join(csvs_path,f'{station.replace("/","_")}_test.csv'),index=False)