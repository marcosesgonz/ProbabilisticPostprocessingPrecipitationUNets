import sys
import os
main_path = os.path.abspath(os.path.join(__file__,'..','..'))
sys.path.append(main_path)
import numpy as np
from Datasets import WRFdataset,desired_configs
import matplotlib.pyplot as plt

data_path = os.path.join(main_path,'data')
data_26 = np.load(os.path.join(data_path,'Sensitivity_UnetAllChSab5_XMaxSumBatch3.npz'))
data_11pca = np.load(os.path.join(data_path,'Sensitivity_Unet11PCA_XMaxSumBatch3.npz'))
data_11fs = np.load(os.path.join(data_path,'Sensitivity_Unet11FS_XMaxSumBatch3.npz'))
data_11ds = np.load(os.path.join(data_path,'Sensitivity_Unet11DS_XMaxSumBatch3.npz'))

def normalize(data_npz):
    value = np.abs(data_npz['ch0']) + np.abs(data_npz['ch1']) + np.abs(data_npz['ch2'])
    value_norm =  value /np.sum(value)
    return value_norm

attr_26 = normalize(data_26)
attr_11pca = normalize(data_11pca)
attr_11fs = normalize(data_11fs)
attr_11ds = normalize(data_11ds)

#For PCA decomposition:
train_set = WRFdataset(data_subset = 'train', group_configs = desired_configs , station_split = False)
train_set.normalize_data()
train_set.apply_PCA(ncomponents=10)
W_t = train_set.W_t_PCA

attr_PCA = attr_11pca[:-1]
attr_HGT = attr_11pca[-1]
attr_originals = np.absolute(attr_PCA @ W_t) #X = ZW_t
attr_11pca_total = np.append(attr_originals, attr_HGT)
attr_11pca_total = attr_11pca_total/np.sum(attr_11pca_total) # Since I obtain absolute values, I have to normalize again.

#Using FS mask to expand
train_set_fs = WRFdataset(data_subset = 'train', group_configs = desired_configs , station_split = False)
train_set_fs.normalize_data(train_set.mean_og_data,train_set.std_og_data)
train_set_fs.apply_feature_selec(11)
attr_11fs_total = np.zeros(26)
attr_11fs_total[train_set_fs.feat_selec_mask] = attr_11fs

#Using DS mask to expand
attr_11ds_total = np.zeros(26)
#Mask extracted from: main_path/data/Sensitivity_UnetAllChSab5_XLowerCRPS_Val.npz
feat_selec_mask_ds_XLowerCRPSVal = [False, True, True, False, False, True, False, True, False, False, False, False, True, False, False, True, False, False, False, False, True, False, True, True, True, True]
attr_11ds_total[feat_selec_mask_ds_XLowerCRPSVal] = attr_11ds

# Plotting
tick_labels = list(train_set.data.keys()) + ['HGT']
fig, axs = plt.subplots(4, 1, sharey=True, figsize=(10, 8), gridspec_kw={'hspace': 0.25})  # Menos espacio entre subplots
name_subplot = ['UNet-All', 'UNet-PCA', 'UNet-FS', 'UNet-DS']

for i, (ax, attr, name_sub) in enumerate(zip(axs, [attr_26, attr_11pca_total, attr_11fs_total, attr_11ds_total], name_subplot)):
    # Indices of 11 biggest bars
    if name_sub in [name_subplot[0], name_subplot[1]]:
        top_11_indices = np.argsort(attr)[-11:]
        colors = ['#FF9999' if i not in top_11_indices else 'red' for i in range(len(attr))]
    else:
        colors = ['red' for i in range(len(attr))]

    ax.bar(np.arange(len(attr)), attr, color=colors)
    ax.set_xticks(np.arange(len(attr)))
    if i == len(axs) - 1:
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    else:
        ax.set_xticklabels([])  

    ax.set_title(name_sub)
    ax.set_xlim(-0.2, len(attr) - 0.8)
    ax.set_ylim(0, 0.5) 
    ax.set_yticks(np.arange(0,0.6,0.1))
    ax.grid(alpha=0.7)

fig.text(0.06, 0.5, 'Sensitivity', va='center', rotation='vertical')
axs[-1].set_xlabel('Input channels')
plt.tight_layout()  

#plt.savefig('Sensitivities_AllUnetModels_BatchRainyTest.png')
