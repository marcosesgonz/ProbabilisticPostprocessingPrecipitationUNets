"""
This is an example script showing the preprocessing step of ncfiles in order to obtain numpy array files. 

"""

import os 
import sys
main_path = os.path.abspath(os.path.join(__file__,'..','..'))
sys.path.append(main_path)
from Datasets import desired_configs
import numpy as np
from netCDF4 import Dataset

# This is the example nc file available in drive folder.
file_example = '20191004_GFS_ysutr.nc'
path_example = os.path.join(main_path,'data','example_netcdf_data',file_example)

# Load the ncfile
ncfile = Dataset(path_example)

rainc = np.array(ncfile.variables['RAINC']) # Convective precipitation
rainnc = np.array(ncfile.variables['RAINNC']) # Non-convective precipitation
assert rainc.shape == (49,150,207)
assert rainnc.shape == (49,150,207)
assert np.all(np.array(ncfile.variables['XTIME'][5:30])/60 == np.arange(5,30,dtype=float))

# crop the hours from 5 to 29, crop in 'y' 20 pixels up and down, crop in 'x' 5 pixels on the left and 0 on the right.
rainc_filtered = rainc[5:30,20:-20,5:]
rainnc_filtered = rainnc[5:30,20:-20,5:]


rain_total_one_ensemble = rainc_filtered + rainnc_filtered # Total precipitation = convective + non-convective
# rain_total_one_ensemble is the accumulated rainfall, subtract the accumulated rainfall of the previous hour to obtain the hourly rainfall.
rain_total_ph_one_ensemble = np.diff(rain_total_one_ensemble,axis=0) #First hour (5:00 am) is removed with this operation. So data is from 6 a.m. to 5 a.m. of the next day.

assert np.all(ncfile.variables['XTIME'][:] == (np.arange(0,49)*60)) 
assert rain_total_ph_one_ensemble.shape == (24,110,202)

channel_example_config = desired_configs.index('GFS_ysutr')

data_saved = np.load(f"data/temp_wrf_data_for_model/{file_example.split('_')[0]}_PrHourlyCropped.npy")

assert np.all(data_saved[channel_example_config] == rain_total_ph_one_ensemble)
