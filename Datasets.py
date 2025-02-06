"""
This script contains the dataset class. It is necessary to processs data through the different models in an structured way.
"""

import json
import os 
import numpy as np
import torch
import pandas as pd
from utils import CRPS_mine
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data import DataLoader

#The 25 set of WRF simulation configurations.
desired_configs = ['GFS_ysutr', 'ARPEGE_ysumc', 'GEM_uwtc', 'GFS_myjtr', 'GEM_myjtr', 'GFS_myjmc', 'ARPEGE_ysutc', 'GFS_uwmc', 'GEM_mynn2tr', 'ARPEGE_myjtr', 'ARPEGE_mynn2mc', 'GFS_ysutc', 'GEM_ysutr', 'GFS_mynn2tc', 'ARPEGE_myjtc', 'GFS_uwtr', 'ARPEGE_uwmc', 'ARPEGE_uwtc', 'GEM_myjtc', 'GFS_myjtc', 'GFS_uwtc', 'GFS_mynn2mc', 'GFS_mynn2tr', 'ARPEGE_mynn2tc', 'GEM_mynn2mc']

class WRFdataset(torch.utils.data.Dataset):
    """
    WRFdataset is a custom PyTorch Dataset class designed to manage meteorological  data. 
    This dataset supports various functionalities such as data normalization, principal component analysis (PCA),
    and feature selection to facilitate efficient data preparation and analysis.

    Key Features:
    - Supports data subsets (train, val, test, or all) with a defined date split.
    - Integrates meteorological center data and allows for station-specific data handling.
    - Includes methods for data normalization, PCA, and feature selection for dimensionality reduction.
    - Provides functionality to compute statistical properties (mean, std, covariance, correlation).
    - Handles multivariate time series data and facilitates ensemble CRPS (Continuous Ranked Probability Score) computation.

    Attributes:
    - dates: It includes the dates where there are the 25 WRF simulations data. Internally, it was computed scanning folders of internal paths. 
    However, in this github repo it is simply obtained reading 'split_dates.json'.
    - crop_x, crop_y: Tuple values defining the cropping dimensions for WRF inner domain matrices.
    - num_ensembles: Number of ensemble members in the dataset.
    - meteo_centers_info: Metadata about meteorological centers.
    - meteo_data: Preprocessed meteorological data to only include dates where there is WRF simulations data.
    - data_hours: Hours of data to be included per forecasting "rollout".
    - num_data_per_day: Number of data points per day based on the selected hours.

    This class is designed for flexibility and extensibility, allowing users to preprocess and analyze 
    meteorological datasets efficiently for machine learning applications.
    """
    def __init__(self,data_subset:str = 'all',return_hour:bool = False, default_num_workers = 0):
        """
        Initialize the WRFdataset instance.
        
        Args:
            data_subset (str): The subset of data to use ('train', 'val', 'test', or 'all').
            return_hour (bool): Whether to include the hour in the returned data.
            default_num_workers (int): The number of workers for data loading operations.
        """
        super(WRFdataset,self).__init__()
        # Ensure the data_subset value is valid
        assert data_subset in ['train','val','test','all']
        # Paths and data loading
        root_file = os.path.dirname(__file__)
        meteo_centers_info = np.load(os.path.join(root_file,'data/meteo_centers_d04_info.npz')) #Keys: lat_lon, x_y_d04, name
        meteo_data = pd.read_csv(os.path.join(root_file,'data/MixtureNewAemetAgrocab_PrHourly20190715_20200821.csv'), parse_dates=['date']) #Contiene todas las estaciones de aemet y de agrocabildo, para toda canarias. 
        
        # Define crop sizes and other initialization
        self.crop_y = 20 , 20  # Cropping top and bottom
        self.crop_x = 5 ,0     # Cropping left and right
        self.num_ensembles = len(desired_configs) #25
        self.configs_names = desired_configs
        self.return_hour = return_hour
        self.num_workers = default_num_workers

        #Load split_dates
        with open(os.path.join(root_file,'data/split_dates.json'), 'r') as f:
            loaded_split_dates = json.load(f)       
            if data_subset in ['train','val','test']:     #70,15,15 
                self.dates = loaded_split_dates[data_subset + '_dates']
            elif data_subset == 'all':
                dates = loaded_split_dates['train_dates'] + loaded_split_dates['val_dates'] + loaded_split_dates['test_dates']
                idx_temp_ordered = np.argsort(dates)
                self.dates = np.array(dates)[idx_temp_ordered]

        self.num_days = len(self.dates)
        
        # Filter data for desired meteorological centers
        desired_centers = list(meteo_centers_info['name']) 
        cols_desired = ['date'] + desired_centers
        meteo_data = meteo_data[cols_desired]  

        # Create a list of desired timestamps for filtering
        data_hours = [i for i in range(6,30,1)] # Hours from 6 to 29 inclusive
        years_days_hours_desired = []
        for day in self.dates:
            day_at00h = pd.to_datetime(str(day),format='%Y%m%d') 
            for hour in data_hours:
                years_days_hours_desired.append(day_at00h + pd.DateOffset(hours=hour))
        
        meteo_data = meteo_data[meteo_data['date'].isin(years_days_hours_desired)]
        # Verify alignment between meteorological and WRF data
        assert len(meteo_data) == len(years_days_hours_desired)

        # Store processed data
        self.meteo_centers_info = meteo_centers_info
        self.meteo_data = meteo_data
        self.data_hours = data_hours 
        self.num_data_per_day = len(self.data_hours) #El número de horas que van desde 6 hasta 30 horas.OJO: CAMBIA SI COJO UN ACUMULADO DE MÁS HORAS.
        
        # Load height data
        hgt_data = np.load(os.path.join('data','temp_wrf_data_for_model','HGTCropped.npy'))  
        self.hgt_data_norm = np.expand_dims((hgt_data - np.mean(hgt_data))/np.std(hgt_data) , axis = 0)  #Shape after operation: [1,size_y,size_x] #Add a batch dimension

        # Additional dataset properties
        self.data_subset = data_subset
        self.data_normalization = False
        self.data_applyPCA = False
        self.data_applyFeatSelec = False

    # Data normalization method. The mean of WRF simulations (for each configuration) is substracted and then divided by the standard deviation.
    def normalize_data(self,mean = None, std = None, num_workers = None):
        num_workers = self.num_workers if num_workers is None else num_workers
        if (mean is None) and (std is None):
            print('Calculating mean and std from the dataset.')
            mean = self.mean(num_workers = num_workers)
            std = self.std(mean = mean,num_workers = num_workers)
        else:
            print('Using provided mean and std for normalization.')
        self.data_normalization = True
        self.mean_og_data = mean
        self.std_og_data = std

    # PCA transformation method
    def apply_PCA(self,eigvec_transp = None, num_workers = None, columns = 25, min_expl_var = None, ncomponents = None):
        num_workers = self.num_workers if num_workers is None else num_workers
        if not self.data_normalization:
            print('Warning: Data is not normalized before PCA.')
        if (eigvec_transp is None):
            print('Computing eigenvectors and eigenvalues.')
            eigval, eigvec = self.PCA(columns = columns, num_workers = num_workers)
            cumsum_eigval = np.cumsum(eigval)
            
            if ncomponents: 
                print(f'Explained variance: {cumsum_eigval[ncomponents - 1]/cumsum_eigval[-1]}')
            elif min_expl_var:
                ncomponents = np.argmax((cumsum_eigval/cumsum_eigval[-1]) > min_expl_var) + 1 #Hay que sumar uno al índice
                print(f'Number of PCA components: {ncomponents}')
            
            self.W_t_PCA = eigvec[:,:ncomponents].T  #Una vez traspuesta: (MxD) con (D:Nºcaract.totales) y (M:Nºcaract.reducidas)
            self.eigv_PCA = eigval[:ncomponents]
        else:
            print('Applying PCA using provided eigenvectors.')
            self.W_t_PCA = eigvec_transp
        self.data_applyPCA = True

    # Compute mean for normalization (batched computation)
    def mean(self, num_workers = None):
        num_workers = self.num_workers if num_workers is None else num_workers
        print('Computing mean for each channel.')
        dataloader = DataLoader(self, batch_size = 30, shuffle=False,num_workers = num_workers)
        sum = 0.0
        nb_samples = 0
        for batch_data, _ in dataloader:
            batch_data = batch_data[:,:-1].double()
            batch_sum = batch_data.sum(dim=(0, 2, 3), keepdim=True)
            batch_samples = batch_data.size(0) * batch_data.size(2) * batch_data.size(3)
            sum += batch_sum
            nb_samples += batch_samples

        mean = (sum / nb_samples).squeeze(0).float().numpy()
        return mean
    
    # Compute standard deviation for normalization (batched computation)
    def std(self,mean = None, num_workers = None):
        num_workers = self.num_workers if num_workers is None else num_workers
        print('Computing standard deviation for each channel')
        dataloader = DataLoader(self, batch_size = 30, shuffle=False,num_workers = num_workers)

        if mean is None:
            mean = torch.tensor(self.mean()).unsqueeze(0)
        else:
            mean = torch.tensor(mean).unsqueeze(0)
        std_sum = 0.0
        nb_samples = 0
        for batch_data, _ in dataloader:
            batch_data = batch_data[:,:-1].double()
            batch_std_sum = ((batch_data - mean)**2).sum(dim=(0, 2, 3), keepdim=True)
            batch_samples = batch_data.size(0) * batch_data.size(2) * batch_data.size(3)
            std_sum += batch_std_sum
            nb_samples += batch_samples

        std = torch.sqrt(std_sum / nb_samples).squeeze(0).float().numpy()
        return std
    
    # Compute the covariance matrix (batched computation)
    def cov_matrix(self, columns = 26, num_workers = None):
        num_workers = self.num_workers if num_workers is None else num_workers
        dataloader = DataLoader(self, batch_size = 30, shuffle=False,num_workers = num_workers)  
        
        if type(columns) == int:
            num_channels = columns
            print(f'Warning: Assuming analysis for the first {columns} channels of the input data.')
        elif type(columns) == list:
            num_channels = len(columns)

        if self.data_normalization == False: #If normalization this operation is not necessary (mean = 0)
            mean = self.mean(num_workers=num_workers)
            mean = np.expand_dims(mean[:num_channels,:,:],axis=0) #Adding batch dimension and only desired columns

        sum_sq = np.zeros((num_channels, num_channels))
        n_samples = 0
        for batch,_ in dataloader:
            batch = batch[:,:num_channels,:,:]
            if self.data_normalization == False: 
                batch -= mean  #centered
            batch = batch.view(batch.size(0), num_channels, -1)  # (batch_size, num_channels, 110*202)
            batch = batch.numpy()
            batch = batch.transpose((0, 2, 1))  # (batch_size, 110*202, num_channels)

            n = batch.shape[0] * batch.shape[1]
            n_samples += n
            sum_sq += np.einsum('ijk,ijl->kl', batch, batch)

        covariance = sum_sq / n_samples
        return covariance.astype(np.float32)
    
    # Compute correlation matrix
    def corr_matrix(self, num_workers = None, columns = 26):
        num_workers = self.num_workers if num_workers is None else num_workers
        covariance = self.cov_matrix(num_workers=num_workers, columns = columns)
        if self.data_normalization: #If normalization, correlation_matrix == covariance_matrix
            return covariance
        else:
            std_dev = np.sqrt(np.diag(covariance))
            correlation_matrix = covariance / np.outer(std_dev, std_dev)
            correlation_matrix[covariance == 0] = 0
            return correlation_matrix
    
    # Compute principal components analysis
    def PCA(self,columns = 25, num_workers = None):
        num_workers = self.num_workers if num_workers is None else num_workers
        #Si los datos se centran en cero y se les pone desv.est. 1, la matriz de covarianza creo que coincidirá con la matriz de correlación.
        cov_matrix = self.cov_matrix(num_workers = num_workers, columns = columns)
        eigval, eigvec = np.linalg.eig(cov_matrix)
        sort_idxs = np.argsort(eigval)[::-1]
        eigvec = eigvec[:,sort_idxs]
        eigval = eigval[sort_idxs]
        return eigval, eigvec
    
    # Apply feature selection selecting least correlated variables.
    def apply_feature_selec(self, num_components = 16, below_max_corr = None, feat_selec_mask = None):
        if feat_selec_mask:
            print('Applying feature selection with the feature mask introduced.')
            self.feat_selec_mask = feat_selec_mask
        else:
            sorted_variables = self.sorted_least_corr_var()
            if num_components:
                # Select the num_components variables with the lowest maximum correlations. 
                least_correlated_variables = sorted_variables.index[:num_components].tolist()
                print(f'Biggest correlation between variables after feature selection: {sorted_variables.values[num_components - 1]}')
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
        # Filling with zeros the upper triangle (above the diagonal)
        corr_mat = np.tril(corr_mat)
        variable_names = self.configs_names + ['HGT']  
        correlation_df = pd.DataFrame(corr_mat, index=variable_names, columns=variable_names)
        # The diagonal is also filled with zeros, since we want to analyze correlations and no standard deviations.
        np.fill_diagonal(correlation_df.values, 0)
        # Check the maximum correlation for each variable
        max_correlations = correlation_df.abs().max(axis=1)
        # Sort variables by its maximum correlation
        sorted_variables = max_correlations.sort_values()
        return sorted_variables
    
    def __len__(self):
        return self.num_days * self.num_data_per_day #Number of days times the number of hours per forecast rollout.

    @staticmethod 
    def combinar_df(left, right):
        return pd.merge(left, right, on='DatetimeBMA', how='outer')
    
    def compute_frequency_weights(self): 
        #https://medium.com/@ravi.abhinav4/improving-class-imbalance-with-class-weights-in-machine-learning-af072fdd4aa4
        labels = self.meteo_data.iloc[:,1:].to_numpy(dtype=np.float32)
        labels_mask = np.isnan(labels)
        labels = labels[~labels_mask]
        size_labels = len(labels)
        # Frequency histogram
        bins_ = [0,1e-2,1,np.inf]
        hist, bin_edges = np.histogram(labels, bins=bins_, density = False) #[0,1e-2), [1e-2,1), [1, max]
        weights =  size_labels / (len(bins_) * hist)
        return weights, bin_edges
    
    def __getitem__(self,idx):
        """
        Retrieve a specific data sample by index.

        This method returns processed data for a specific hour of a given day, along with the corresponding meteorological observations.
        The data can be optionally normalized, reduced via PCA, or subjected to feature selection.

        Args:
            idx (int): The index of the data sample to retrieve.

        Returns:
            tuple: Contains the following:
                - X (numpy.ndarray): The processed WRF data (rain and height data).
                - meteo_data_filtered (numpy.ndarray): The meteorological observations for the specific hour.
                - self.data_hours[idx_hour] (optional): The hour of the day for the sample (if return_hour is True).
        """
        idx_day = idx // self.num_data_per_day
        day = self.dates[idx_day]
        idx_hour = idx % self.num_data_per_day
        rain_data = np.load(os.path.join('data','temp_wrf_data_for_model',f'{day}_PrHourlyCropped.npy'))
        rain_data = rain_data[:,idx_hour,:,:]  #Shape after operation: [num_ensembles, size_y, size_x]
        X = rain_data
        #X = np.concatenate([rain_data, self.hgt_data], axis = 0)
        if self.data_normalization:
            X = (X - self.mean_og_data) / self.std_og_data

        if self.data_applyPCA:
            X = np.einsum('md,dij->mij',self.W_t_PCA,X)

        elif self.data_applyFeatSelec:
            X = X[self.feat_selec_mask,:,:]

        X = np.concatenate([X, self.hgt_data_norm], axis = 0)
        
        meteo_data_filtered = self.meteo_data.iloc[idx,1:].to_numpy(dtype=np.float32)  #The first column of the csv is the date.
        if self.return_hour:
            return X, meteo_data_filtered, self.data_hours[idx_hour] #x_y_d04_filtered
        else:
            return X, meteo_data_filtered
    def calc_ensemble_crps(self, columns= None):
        """
        Compute the ensemble CRPS (Continuous Ranked Probability Score) of the data 
        loaded in the WRFdataset class.

        Parameters:
        - columns: An int, a mask, or a list specifying columns (stations) indices. 
        If None, all stations are considered.

        Returns:
        - The computed CRPS for the specified columns-subset of data or the entire dataset.
        """
        # Initialize CRPS computation object.
        crps = CRPS_mine()
        
        # Iterate over all available dates in the dataset
        for day in self.dates:
            # Load daily rainfall data from preprocessed WRF files.
            rain_data = np.load(os.path.join('data','temp_wrf_data_for_model',f'{day}_PrHourlyCropped.npy'))
            hours_day = []
            # Generate a list of hourly timestamps for the current day based on data_hours.
            for hour in self.data_hours:
                hours_day.append(pd.to_datetime(str(day) + '00',format='%Y%m%d%H') + pd.DateOffset(hours=hour))

            # Filter meteorological data to match the generated hourly timestamps.
            meteo_data_filtered = self.meteo_data[self.meteo_data['date'].isin(hours_day)].drop(columns='date')
            
            if columns is not None:
                # Filter meteorological data for specific columns (stations) and compute CRPS.
                meteo_data_filtered_column_centers = meteo_data_filtered[meteo_data_filtered.columns[columns]]
                xy = self.meteo_centers_info['x_y_d04'][columns] # Retrieve station coordinates.
                # Extract WRF data for the specified stations based on their coordinates
                wrf_data_in_stations = rain_data[:,:,xy[:,1] - self.crop_y[0], xy[:,0] - self.crop_x[0]] 
                # Accumulate CRPS values for the filtered WRF and observed data.
                crps.CRPS_accum(X_f = wrf_data_in_stations, X_o = meteo_data_filtered_column_centers.values)
        
            else:
                # Use all meteorological centers to compute CRPS if no subset is specified.
                xy = self.meteo_centers_info['x_y_d04']
                wrf_data_in_stations = rain_data[:,:,xy[:,1] - self.crop_y[0], xy[:,0] - self.crop_x[0]] 
                crps.CRPS_accum(X_f = wrf_data_in_stations, X_o = meteo_data_filtered.values)

        #Return the mean CRPS value
        return crps._compute()



class MeteoCentersSplitter:
    def __init__(self, WRFDataset, nfolds = 5, stratify = False):
        """
        Load/Create the station splits used in the generalization performance study of the UNets.

        Parameters:
        - WRFDataset: Dataset containing meteorological information. Normally use the training subset of the data if you are going to create the centers splits.
        - nfolds: Number of folds for cross-validation.
        - stratify: Whether to stratify the splits based on rainfall distribution of meteocenters. 
        """
        self.meteo_centers_info = WRFDataset.meteo_centers_info # Metadata for meteorological centers
        self.meteo_data = WRFDataset.meteo_data.iloc[:,1:] # Meteorological data excluding date column
        self.nfolds = nfolds # Number of folds for cross-validation
        self.stratify = stratify #Boolean indicating stratification
        
        # Filename for saving/loading splits
        fila_name = f'{nfolds}folds_stations_splits_stratified.json' if stratify else f'{nfolds}folds_stations_splits.json'
        self.splits_file = os.path.join(os.path.dirname(__file__),'data',fila_name)
        self._load_or_create_splits()

    def _load_or_create_splits(self):
        # Load saved splits if it exists
        if os.path.exists(self.splits_file):
            print('Reading saved json with station splits.')
            with open(self.splits_file, 'r') as f:
                splits = json.load(f)
        else:
            # Create new splits and save them
            print('Creating station splits.')
            names = self.meteo_centers_info['name']
            if self.stratify:
                mean_prec_values = self.meteo_data.mean(axis = 0) #Calculate the mean precipitation .
                station_labels, bins = pd.qcut(x = mean_prec_values,q = 3,labels = False, retbins=True) # Bin the data into 3 categories.
                station_labels = station_labels.tolist()
                # Perform stratified k-fold splitting.
                skf = StratifiedKFold(n_splits = self.nfolds, shuffle=True, random_state=19)#21)
                splits = [(train.tolist(), test.tolist()) for train, test in skf.split(X = names, y = station_labels)]
            else:
                # Perform simple k-fold splitting.
                kf = KFold(n_splits=5, shuffle=True, random_state=13)
                splits = [(train.tolist(), test.tolist()) for train, test in kf.split(names)]
            
            #Save the splits in a JSON file
            with open(self.splits_file, 'w') as f:
                json.dump(splits, f)

        self.splits = splits

    def __len__(self):
        return self.nfolds

    def __getitem__(self,idx):
        """
        Retrieves the train and test masks for a given fold index.

        Parameters:
        - idx: Index of the fold.

        Returns:
        - mask_centers_train: Boolean mask for training centers.
        - mask_centers_test: Boolean mask for testing centers.

        Raises:
        - ValueError: If the index is out of range.
        """
        if idx < 0 or idx >= self.nfolds:
            raise ValueError(f"El índice del fold debe estar entre 0 y {self.nfolds - 1}.")

        train_indices, test_indices = self.splits[idx]

        len_centers = len(self.meteo_centers_info['name'])
        mask_centers_train = np.zeros(len_centers, dtype=bool)
        mask_centers_test = np.zeros(len_centers, dtype=bool)

        mask_centers_train[train_indices] = True
        mask_centers_test[test_indices] = True

        return mask_centers_train, mask_centers_test