import json
import os 
import numpy as np
import torch
import pandas as pd
from prob_scores import CRPS_mine
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data import DataLoader

#Config con 25 modelos y 306 datos
desired_configs = ['GFS_ysutr', 'ARPEGE_ysumc', 'GEM_uwtc', 'GFS_myjtr', 'GEM_myjtr', 'GFS_myjmc', 'ARPEGE_ysutc', 'GFS_uwmc', 'GEM_mynn2tr', 'ARPEGE_myjtr', 'ARPEGE_mynn2mc', 'GFS_ysutc', 'GEM_ysutr', 'GFS_mynn2tc', 'ARPEGE_myjtc', 'GFS_uwtr', 'ARPEGE_uwmc', 'ARPEGE_uwtc', 'GEM_myjtc', 'GFS_myjtc', 'GFS_uwtc', 'GFS_mynn2mc', 'GFS_mynn2tr', 'ARPEGE_mynn2tc', 'GEM_mynn2mc']

class WRFdataset(torch.utils.data.Dataset):
    def __init__(self,data_subset:str = 'all', station_split:bool = False,return_hour:bool = False, default_num_workers = 0):
        super(WRFdataset,self).__init__()
        assert data_subset in ['train','val','test','all']
        root_file = os.path.dirname(__file__)
        meteo_centers_info = np.load(os.path.join(root_file,'data/meteo_centers_d04_info.npz')) #Keys: lat_lon, x_y_d04, name
        meteo_data = pd.read_csv(os.path.join(root_file,'data/MixtureNewAemetAgrocab_PrHourly20190715_20200821.csv'), parse_dates=['date']) #Contiene todas las estaciones de aemet y de agrocabildo, para toda canarias. 
        #Number of pixels already cropped for matrices
        self.crop_y = 20 , 20  #Arriba y abajo
        self.crop_x = 5 ,0      #Izquierda y derecha
        self.num_ensembles = len(desired_configs) #25
        self.configs_names = desired_configs
        self.return_hour = return_hour
        self.num_workers = default_num_workers

        
        with open(os.path.join(root_file,'data/split_dates.json'), 'r') as f:
            loaded_split_dates = json.load(f)       
            if data_subset in ['train','val','test']:     #70,15,15 
                self.dates = loaded_split_dates[data_subset + '_dates']
            elif data_subset == 'all':
                dates = loaded_split_dates['train_dates'] + loaded_split_dates['val_dates'] + loaded_split_dates['test_dates']
                idx_temp_ordered = np.argsort(dates)
                self.dates = np.array(dates)[idx_temp_ordered]

            self.num_days = len(self.dates)
        
        #Me quedo solo con estaciones de d04. Además con esto, nos aseguramos que las columnas del csv sigan el mismo orden que las del diccionario 'meteo_centers_info'
        desired_centers = list(meteo_centers_info['name']) 
        cols_desired = ['date'] + desired_centers
        meteo_data = meteo_data[cols_desired]  
        data_hours = [i for i in range(6,30,1)] #El último índice del range se puede variar para cambiar el acumulado
        years_days_hours_desired = []
        for day in self.dates:
            day_at00h = pd.to_datetime(str(day),format='%Y%m%d') 
            for hour in data_hours:
                years_days_hours_desired.append(day_at00h + pd.DateOffset(hours=hour))
        
        meteo_data = meteo_data[meteo_data['date'].isin(years_days_hours_desired)]
        #Is len meteo data equal to wrf data(len(self.dates))?
        assert len(meteo_data) == len(years_days_hours_desired)
        self.meteo_centers_info = meteo_centers_info
        self.meteo_data = meteo_data
        self.data_hours = data_hours 
        self.num_data_per_day = len(self.data_hours) #El número de horas que van desde 6 hasta 30 horas.OJO: CAMBIA SI COJO UN ACUMULADO DE MÁS HORAS.
        
        hgt_data = np.load(os.path.join('data','temp_wrf_data_for_model','HGTCropped.npy'))  
        self.hgt_data_norm = np.expand_dims(hgt_data , axis = 0)  #Shape after operation: [1,size_y,size_x]
        self.data_subset = data_subset
        self.station_split = station_split
        self.data_normalization = False
        self.data_applyPCA = False
        self.data_applyFeatSelec = False


    def normalize_data(self,mean = None, std = None, num_workers = None):
        num_workers = self.num_workers if num_workers is None else num_workers
        if (mean is None) and (std is None):
            print('Como no se dió "mean" y "std", normalizando a partir de la media y desv. de los propios datos.')
            mean = self.mean(num_workers = num_workers)
            std = self.std(mean = mean,num_workers = num_workers)
        else:
            print('Normalizando a partir de los valores mean y std dados.')
        self.data_normalization = True
        self.mean_og_data = mean#.astype(np.float32)
        self.std_og_data = std#.astype(np.float32)

    def apply_PCA(self,eigvec_transp = None, num_workers = None, columns = 25, min_expl_var = None, ncomponents = None):
        num_workers = self.num_workers if num_workers is None else num_workers
        if not self.data_normalization:
            print('OJO, datos no normalizados antes de aplicar PCA')
        if (eigvec_transp is None):
            print('Como no se dió los vectores propios se calculan.')
            eigval, eigvec = self.PCA(columns = columns, num_workers = num_workers)
            cumsum_eigval = np.cumsum(eigval)
            
            if ncomponents: 
                print(f'Varianza explicada: {cumsum_eigval[ncomponents - 1]/cumsum_eigval[-1]}')
            elif min_expl_var:
                ncomponents = np.argmax((cumsum_eigval/cumsum_eigval[-1]) > min_expl_var) + 1 #Hay que sumar uno al índice
                print(f'Número de componentes PCA es {ncomponents}')
            
            self.W_t_PCA = eigvec[:,:ncomponents].T  #Una vez traspuesta: (MxD) con (D:Nºcaract.totales) y (M:Nºcaract.reducidas)
            self.eigv_PCA = eigval[:ncomponents]
        else:
            print('Aplicando PCA a partir de los autovectores dados.')
            self.W_t_PCA = eigvec_transp
        self.data_applyPCA = True

    def mean(self, num_workers = None):
        num_workers = self.num_workers if num_workers is None else num_workers
        print('Calculando media para cada canal')
        dataloader = DataLoader(self, batch_size = 30, shuffle=False,num_workers = num_workers)
        sum = 0.0
        nb_samples = 0
        for batch_data, _ in dataloader:
            batch_data = batch_data.double()
            batch_sum = batch_data.sum(dim=(0, 2, 3), keepdim=True)
            batch_samples = batch_data.size(0) * batch_data.size(2) * batch_data.size(3)
            sum += batch_sum
            nb_samples += batch_samples

        mean = (sum / nb_samples).squeeze(0).float().numpy()
        return mean
    
    def std(self,mean = None, num_workers = None):
        num_workers = self.num_workers if num_workers is None else num_workers
        print('Calculando std_dev para cada canal')
        dataloader = DataLoader(self, batch_size = 30, shuffle=False,num_workers = num_workers)

        if mean is None:
            mean = torch.tensor(self.mean()).unsqueeze(0)
        else:
            mean = torch.tensor(mean).unsqueeze(0)
        std_sum = 0.0
        nb_samples = 0
        for batch_data, _ in dataloader:
            batch_data = batch_data.double()
            batch_std_sum = ((batch_data - mean)**2).sum(dim=(0, 2, 3), keepdim=True)
            batch_samples = batch_data.size(0) * batch_data.size(2) * batch_data.size(3)
            std_sum += batch_std_sum
            nb_samples += batch_samples

        std = torch.sqrt(std_sum / nb_samples).squeeze(0).float().numpy()
        return std
    
    def cov_matrix(self, columns = 26, num_workers = None):
        num_workers = self.num_workers if num_workers is None else num_workers
        dataloader = DataLoader(self, batch_size = 30, shuffle=False,num_workers = num_workers)  
        
        if type(columns) == int:
            num_channels = columns
            print(f'OJO asumiendo análisis para {columns} primeros canales de las imágenes')
        elif type(columns) == list:
            num_channels = len(columns)

        if self.data_normalization == False: #Si fuera true, no hace falta esta operación ya que los datos tendrían media 0.
            mean = self.mean(num_workers=num_workers)
            mean = np.expand_dims(mean[:num_channels,:,:],axis=0) #Adding batch dimension and only desired columns

        sum_sq = np.zeros((num_channels, num_channels))
        n_samples = 0
        for batch,_ in dataloader:
            batch = batch[:,:num_channels,:,:]
            if self.data_normalization == False: #Si fuera true, no hace falta esta operación ya que los datos tendrían media 0.
                batch -= mean  #centered
            batch = batch.view(batch.size(0), num_channels, -1)  # (batch_size, num_channels, 110*202)
            batch = batch.numpy()
            batch = batch.transpose((0, 2, 1))  # (batch_size, 110*202, num_channels)

            n = batch.shape[0] * batch.shape[1]
            n_samples += n
            sum_sq += np.einsum('ijk,ijl->kl', batch, batch)

        covariance = sum_sq / n_samples
        return covariance.astype(np.float32)
    
    def corr_matrix(self, num_workers = None, columns = 26):
        num_workers = self.num_workers if num_workers is None else num_workers
        covariance = self.cov_matrix(num_workers=num_workers, columns = columns)
        if self.data_normalization:
            return covariance
        else:
            std_dev = np.sqrt(np.diag(covariance))
            correlation_matrix = covariance / np.outer(std_dev, std_dev)
            correlation_matrix[covariance == 0] = 0
            return correlation_matrix
    
    def PCA(self,columns = 25, num_workers = None):
        num_workers = self.num_workers if num_workers is None else num_workers
        #Si los datos se centran en cero y se les pone desv.est. 1, la matriz de covarianza creo que coincidirá con la matriz de correlación.
        cov_matrix = self.cov_matrix(num_workers = num_workers, columns = columns)
        eigval, eigvec = np.linalg.eig(cov_matrix)
        sort_idxs = np.argsort(eigval)[::-1]
        eigvec = eigvec[:,sort_idxs]
        eigval = eigval[sort_idxs]
        return eigval, eigvec
    
    def apply_feature_selec(self, num_components = 16, below_max_corr = None, feat_selec_mask = None):
        if feat_selec_mask:
            print('Aplicando selección de variables con las variables introducidas')
            self.feat_selec_mask = feat_selec_mask
        else:
            sorted_variables = self.sorted_least_corr_var()
            if num_components:
                # Seleccionar las num_components variables con las correlaciones máximas más bajas
                least_correlated_variables = sorted_variables.index[:num_components].tolist()
                print(f'Mayor correlación entre datos después de feat.selecc: {sorted_variables.values[num_components - 1]}')
            elif below_max_corr:
                least_correlated_variables = (least_correlated_variables[least_correlated_variables < below_max_corr]).index.tolist()
            
            var_names = self.configs_names + ['HGT']
            mask_vars = []
            for name in var_names:
                mask_vars.append(True) if name in least_correlated_variables else mask_vars.append(False)

            self.feat_selec_mask = mask_vars

        self.data_applyFeatSelec = True
            
    def sorted_least_corr_var(self):
        corr_mat = self.corr_matrix()
        #LLeno de zeros a la parte superior del triángulo(encima de la diagonal)
        corr_mat = np.tril(corr_mat)
        variable_names = self.configs_names + ['HGT']  # Tu lista con los nombres de las variables
        correlation_df = pd.DataFrame(corr_mat, index=variable_names, columns=variable_names)
        # Primero, necesitamos reemplazar la diagonal con un valor bajo para ignorarla en la búsqueda de la máxima correlación
        np.fill_diagonal(correlation_df.values, 0)
        # Luego, calculamos la correlación máxima para cada variable
        max_correlations = correlation_df.abs().max(axis=1)
        # Ordenar las variables por su correlación máxima
        sorted_variables = max_correlations.sort_values()
        return sorted_variables
    
    def __len__(self):
        return self.num_days * self.num_data_per_day #Número de días por el número de horas de cada día (de 06 a 24). OJO: CAMBIA SI COJO UN ACUMULADO DE MÁS DÍAS.

    @staticmethod #Se usa junto a reduce() para combinar multiples dataframes por fecha.
    def combinar_df(left, right):
        return pd.merge(left, right, on='DatetimeBMA', how='outer')
    
    def compute_frequency_weights(self):
        #https://medium.com/@ravi.abhinav4/improving-class-imbalance-with-class-weights-in-machine-learning-af072fdd4aa4
        labels = self.meteo_data.iloc[:,1:].to_numpy(dtype=np.float32)
        labels_mask = np.isnan(labels)
        labels = labels[~labels_mask]
        size_labels = len(labels)
        # Calcular histograma de frecuencias
        hist, bin_edges = np.histogram(labels, bins=[0,1e-2,1,np.inf], density = False) #[0,1e-2), [1e-2,1), [1, max]
        # Calcular peso inverso de cada bin. Se divide entre tres, por el número de bins.
        weights =  size_labels / (3 * hist)
        return weights, bin_edges
    
    def __getitem__(self,idx):
        idx_day = idx // self.num_data_per_day
        day = self.dates[idx_day]
        idx_hour = idx % self.num_data_per_day
        rain_data = np.load(os.path.join('data','temp_wrf_data_for_model',f'{day}_PrHourlyCropped.npy'))
        rain_data = rain_data[:,idx_hour,:,:]  #Shape after operation: [num_ensembles, size_y, size_x]
        X = np.concatenate([rain_data, self.hgt_data_norm], axis = 0)
        if self.data_normalization:
            X = (X - self.mean_og_data) / self.std_og_data
        if self.data_applyPCA:
            X_noHGT = X[:self.num_ensembles,:,:] 
            Z = np.einsum('md,dij->mij',self.W_t_PCA,X_noHGT) #Z_t = (W_t * X_t) con Z_t con forma (M, size_y, size_x)
            X = np.concatenate([Z,X[self.num_ensembles:,:,:]], axis = 0)
        elif self.data_applyFeatSelec:
            X = X[self.feat_selec_mask,:,:]
        meteo_data_filtered = self.meteo_data.iloc[idx,1:].to_numpy(dtype=np.float32)
        if self.return_hour:
            return X, meteo_data_filtered, self.data_hours[idx_hour] #x_y_d04_filtered
        else:
            return X, meteo_data_filtered
    def calc_ensemble_crps(self, columns= None):
        """
        Columns: An int, a mask or a list specifing columns(stations) indices.
        """
        if self.data_subset == 'all' or (columns is not None) or (self.station_split == False):
            crps = CRPS_mine()
        else:
            crps_tr = CRPS_mine()
            crps_tst = CRPS_mine()
        for day in self.dates:
            rain_data = np.load(os.path.join('data','temp_wrf_data_for_model',f'{day}_PrHourlyCropped.npy'))
            hours_day = []
            for hour in self.data_hours:
                hours_day.append(pd.to_datetime(str(day) + '00',format='%Y%m%d%H') + pd.DateOffset(hours=hour))

            meteo_data_filtered = self.meteo_data[self.meteo_data['date'].isin(hours_day)].drop(columns='date')
            
            if columns is not None:
                meteo_data_filtered_column_centers = meteo_data_filtered[meteo_data_filtered.columns[columns]]
                xy = self.meteo_centers_info['x_y_d04'][columns]
                wrf_data_in_stations = rain_data[:,:,xy[:,1] - self.crop_y[0], xy[:,0] - self.crop_x[0]] 
                crps.CRPS_accum(X_f = wrf_data_in_stations, X_o = meteo_data_filtered_column_centers.values)
        
            elif (self.data_subset == 'all') or (self.station_split == False):
                xy = self.meteo_centers_info['x_y_d04']
                wrf_data_in_stations = rain_data[:,:,xy[:,1] - self.crop_y[0], xy[:,0] - self.crop_x[0]] 
                crps.CRPS_accum(X_f = wrf_data_in_stations, X_o = meteo_data_filtered.values)
            else:
                meteo_data_filtered_tr_centers = meteo_data_filtered[meteo_data_filtered.columns[self.mask_centers_tr]]
                xy_tr_centers = self.meteo_centers_info['x_y_d04'][self.mask_centers_tr]
                wrf_data_in_stations = rain_data[:,:,xy_tr_centers[:,1] - self.crop_y[0], xy_tr_centers[:,0] - self.crop_x[0]] 
                crps_tr.CRPS_accum(X_f = wrf_data_in_stations, X_o = meteo_data_filtered_tr_centers.values)


                meteo_data_filtered_tst_centers = meteo_data_filtered[meteo_data_filtered.columns[self.mask_centers_tst]]
                xy_tst_centers = self.meteo_centers_info['x_y_d04'][self.mask_centers_tst]
                wrf_data_in_stations = rain_data[:,:,xy_tst_centers[:,1] - self.crop_y[0], xy_tst_centers[:,0] - self.crop_x[0]] 
                crps_tst.CRPS_accum(X_f = wrf_data_in_stations, X_o = meteo_data_filtered_tst_centers.values)

        if self.data_subset == 'all' or (columns is not None) or (self.station_split == False):
            return crps._compute()
        else:
            return crps_tr._compute(), crps_tst._compute()


class MeteoCentersSplitter:
    def __init__(self, WRFDataset, nfolds = 5, stratify = False):
        self.meteo_centers_info = WRFDataset.meteo_centers_info
        self.meteo_data = WRFDataset.meteo_data.iloc[:,1:]
        self.nfolds = nfolds
        self.stratify = stratify
        fila_name = f'{nfolds}folds_stations_splits_stratified.json' if stratify else  f'{nfolds}folds_stations_splits.json'
        self.splits_file = os.path.join(os.path.dirname(__file__),'data',fila_name)
        self._load_or_create_splits()

    def _load_or_create_splits(self):
        if os.path.exists(self.splits_file):
            # Leer los splits guardados
            print('Reading saved json with station splits.')
            with open(self.splits_file, 'r') as f:
                splits = json.load(f)
        else:
            # Crear los splits y guardarlos
            print('Creating station splits.')
            names = self.meteo_centers_info['name']
            if self.stratify:
                mean_prec_values = self.meteo_data.mean(axis = 0)
                station_labels, bins = pd.qcut(x = mean_prec_values,q = 3,labels = False, retbins=True)
                station_labels = station_labels.tolist()
                skf = StratifiedKFold(n_splits = self.nfolds, shuffle=True, random_state=19)#21)
                splits = [(train.tolist(), test.tolist()) for train, test in skf.split(X = names, y = station_labels)]
            else:
                kf = KFold(n_splits=5, shuffle=True, random_state=13)
                splits = [(train.tolist(), test.tolist()) for train, test in kf.split(names)]

            with open(self.splits_file, 'w') as f:
                json.dump(splits, f)

        self.splits = splits

    def __len__(self):
        return self.nfolds

    def __getitem__(self,idx):
        if idx < 0 or idx >= self.nfolds:
            raise ValueError(f"El índice del fold debe estar entre 0 y {self.nfolds - 1}.")

        train_indices, test_indices = self.splits[idx]

        len_centers = len(self.meteo_centers_info['name'])
        mask_centers_train = np.zeros(len_centers, dtype=bool)
        mask_centers_test = np.zeros(len_centers, dtype=bool)

        mask_centers_train[train_indices] = True
        mask_centers_test[test_indices] = True

        return mask_centers_train, mask_centers_test
    
"""
#Calculation of ensemble crps in train,val,test sets and in 5-folds station splits.
train_data = WRFdataset(data_subset='train', station_split=False)
val_data = WRFdataset(data_subset='val', station_split=False)
test_data = WRFdataset(data_subset='test', station_split=False)
splitter = MeteoCentersSplitter(train_data, nfolds=5, stratify=False)

for i in range(len(splitter)):
    mask_tr,mask_tst = splitter[i]
    print(f'Fold {i +1}')
    tr_ctrs_tr_loss = train_data.calc_ensemble_crps(columns=mask_tr)
    val_ctrs_tr_loss = train_data.calc_ensemble_crps(columns=mask_tst)
    print(f' Train set:  (Train ctrs:{tr_ctrs_tr_loss}, Val ctrs: {val_ctrs_tr_loss})')
    tr_ctrs_val_loss = val_data.calc_ensemble_crps(columns=mask_tr)
    val_ctrs_val_loss = val_data.calc_ensemble_crps(columns=mask_tst)
    print(f' Val set:  (Train ctrs:{tr_ctrs_val_loss}, Val ctrs: {val_ctrs_val_loss})')
    tr_ctrs_test_loss = test_data.calc_ensemble_crps(columns=mask_tr)
    val_ctrs_test_loss = test_data.calc_ensemble_crps(columns=mask_tst)
    print(f' Test set:  (Train ctrs:{tr_ctrs_test_loss}, Val ctrs: {val_ctrs_test_loss})')
"""