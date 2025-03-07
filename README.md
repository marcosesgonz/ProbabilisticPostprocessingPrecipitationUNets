# Probabilistic Postprocessing of Precipitation Using UNets

Implementation of the paper **"Postprocessing of Convection-Permitting Precipitation Using UNets"**.

---

## Data Availability

To reproduce the experiments, you need to download the required data and models:

1. **Download Data**: All required data can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1FEzji7PpGvqXzRPBh961NDqujRgPf7JZ?usp=sharing).
   - Place the station information CSV file, `MixtureNewAemetAgrocab_PrHourly20190715_20200821.csv`, inside the `data/` folder.
   - Place the unzipped `.rar` data from the ensemble WRF set inside the folder `data/temp_wrf_data_for_model`. This data corresponds to a numpy (`.npy`) file for each day of data. These numpy files were preprocessed from netcdf archives. The script `data_manipulation_scripts/example_numpy_from_netcdf.py` shows an example of preprocessing one day data of a WRF simulation from netcdf to a numpy file. To run this script, place the file `2019104_GFS_ysutr.nc` inside `data/example_netcdf_data/` folder.
   - Place the results files `results_5folds.npz` and `results_overall.npz` inside the `results/` folder in order to run bootstraped results using `metric_evaluations_predictive_performance/computing_bootstraps.py` numpy file.

2. **Pretrained Models**: To use pretrained models (`.pt` files), place them in their corresponding folder under `results/unet_weights`.

3. **Classical Methods Data**: To generate data for classical methods, run the script `data_manipulation_scripts/create_csv_for_classic_methods.py`. The scripts for evaluating classical methods are located in `Laboratory/Calculation_classical_methods.ipynb`.

---

## Results Reproducibility

### Dependency Guidance

A `requirements.txt` file is provided as a guideline for the dependencies needed to run the scripts in this repository. While it captures the main libraries and versions used during development, some adjustments might be required depending on your environment. Ensure compatibility with your Python version (e.g., Python 3.8 or later) and test the setup in a clean environment to avoid conflicts.

### Replicating paper results

- Users can now reproduce the results using the "metric_evaluator" class located at `metric_evaluations_predictive_performance/metric_evaluator.py`. To use this class, refer to the comments inside the eval() method. Below is an example of how to evaluate generalization results for the UNet-FS model:
  
  ```
  from metric_evaluator import evaluator
  eval = evaluator()
  #Generalization results for UNet-FS model
  results_UNet_fs  = eval.eval(model_name='unet_fs',return_params=False,results = True,return_metrics_mean=False, meteo_centers_five_folds=True)
  ```
- In order to replicate bootstrapped results, run `metric_evaluations_predictive_performance/computing_bootstraps.py`. This script reads result files stored in the `results` folder.
- Ensure you download the pretrained models and data from the provided Google Drive link before running the script.

### Training Your Own Model:
- **Forecasting Performance Experiments**:
  - UNet models:
    - Use the script `Laboratory/TrainValTest.py` with appropriate arguments to train your own UNet model. Use `-h` argument to show a help message.
    - For the UNet-DS approach:
      1. Train the UNet with all input channels (or use the pretrained UNet-All weights from Google Drive).
      2. Run the script `SensitivityCalculation/calc_sensitivity_DS.py`, editing the path to the UNet-All weights.
      3. This script will generate the final mask, which is required to train the Down Selection (DS) model.
      4. Use the generated mask as input in `Laboratory/TrainValTest.py` with the DownSelec option enabled.

  - Classical Methods:
    - Training classical methods can be reproduced by running the cells in `Laboratory/Calculation_classical_methods.ipynb`.
    - Note: Classical model weights are not available yet (unless requested), so they must be trained and then evaluated manually.

- **Spatial Generalization Performance Experiments**:
  - Use the script `Laboratory/FiveFoldsStations_DS.py` for evaluating the Down Selection (DS) approach.
  - Use the script `Laboratory/FiveFoldsStations.py` for evaluating other UNet methods.

---

## Note on Reproducibility

While every effort has been made to ensure reproducibility of the results presented in the paper, exact replication may be challenging. Factors such as differences in random seed initialization, hardware variations, and library versioning can lead to slight deviations in results. However, the methodology and scripts provided should allow you to achieve comparable performance.

---

## Summary

This repository provides scripts and pretrained models to replicate the predictive and generalization performance of UNet-based postprocessing methods for precipitation. Ensure all dependencies are met and data is placed in the correct directories before executing the scripts.

For questions or contributions, please feel free to open an issue or submit a pull request.
