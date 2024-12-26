import os 
import sys
main_path = os.path.abspath(os.path.join(__file__,'..','..'))
sys.path.append(main_path)
from Datasets import WRFdataset,MeteoCentersSplitter

#Calculation of ensemble crps in train,val,test sets and in the 5-folds station splits.

train_data = WRFdataset(data_subset='train')
val_data = WRFdataset(data_subset='val')
test_data = WRFdataset(data_subset='test')
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