import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def stratified_split_dates(dataset):
    #dataset = WRFdataset(data_subset='all',group_configs=desired_configs, station_split = False)
    data = dataset.meteo_data
    data['mean_prec'] = data.iloc[:,1:].mean(axis=1, skipna=True)
    # Ajustar las horas para la agrupación para que coincidan con los días de inicialización de los modelos WRF
    def adjust_dates(row):
        date = row['date']
        if date.hour < 6:
            # Para las horas de 00:00 a 05:59, restar un día
            return date - pd.Timedelta(days=1)
        else:
            # Para las horas de 06:00 a 23:59, mantener la misma fecha
            return date
    # Aplicar la función a cada fila para obtener la fecha ajustada
    data['WRF_date'] = data.apply(adjust_dates, axis=1).dt.date

    daily_precipitation = data.groupby('WRF_date')['mean_prec'].sum().reset_index()

    limits = [-np.inf,0.1,1,4,np.inf]
    # Crear una nueva columna para las clases de precipitación
    daily_precipitation['precip_class'] = pd.cut(daily_precipitation['mean_prec'], bins=limits,right = True, labels=['No LLuvia', 'Baja','Media', 'Alta'])
    # Verificar la distribución de las clases
    print(daily_precipitation['precip_class'].value_counts())
    #Configurar StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=46)
    train_idx, temp_idx = next(split.split(daily_precipitation, daily_precipitation['precip_class']))

    # Dividir entre entrenamiento y conjunto temporal (validación + test)
    train_set = daily_precipitation.iloc[train_idx]
    temp_set = daily_precipitation.iloc[temp_idx]

    # Dividir el conjunto temporal en validación y test
    split_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=49)
    val_idx, test_idx = next(split_val_test.split(temp_set, temp_set['precip_class']))

    val_set = temp_set.iloc[val_idx]
    test_set = temp_set.iloc[test_idx]

    # Convertir las fechas a formato crudo 'YYYYMMDD'
    train_dates = sorted(train_set['WRF_date'].apply(lambda x: int(x.strftime('%Y%m%d'))).tolist())
    val_dates = sorted(val_set['WRF_date'].apply(lambda x: int(x.strftime('%Y%m%d'))).tolist())
    test_dates = sorted(test_set['WRF_date'].apply(lambda x: int(x.strftime('%Y%m%d'))).tolist())

    split_dates = {
        'train_dates': train_dates,
        'val_dates': val_dates,
        'test_dates': test_dates
    }

    #with open('/disk/barbusano/barbusano3/data/split_dates.json', 'w') as f:
    #    json.dump(split_dates, f)

