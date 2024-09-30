#from batch_dataset import batch_dataset
from Net import Net
from train_model import train_model

import pandas as pd
import numpy as np
import os
import numpy as np
import random
import shutil
import joblib

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score, roc_auc_score, r2_score,explained_variance_score

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.nn import Linear, Dropout, BatchNorm1d
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler, DataLoader, TensorDataset
from torch.utils.data.dataset import ConcatDataset
import torch.optim as optim
from torch.backends import cudnn

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def main(test_file, scaler_model, descr_names, model_dir, output_file_dir, device):
    
    os.chdir(model_dir)
    
    import warnings
    warnings.filterwarnings("ignore")
    
    # load test descriptor file 
    test_set = pd.read_csv(test_file, header = 0, index_col = 0)
    
    # lets go through each folder in 
    
    # lets convert Y in range of 0 and 1
    Y_scaler = MinMaxScaler(feature_range=(0, 1)).fit(mc_set_values.values)
    #mc_set_values['VALUES'] = Y_scaler.fit_transform(mc_set_values['VALUES'])
    #mc_set_values['VALUES'] = Y_scaler.inverse_transform(mc_set_values['VALUES'])
    mc_set_values = pd.DataFrame(Y_scaler.transform(mc_set_values.values), columns=mc_set_values.columns, index=None)