'''
#pip install torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score
from npt.column_encoding_dataset import ColumnEncodingDataset
from npt.model import NPTModel
import wandb
wandb.login()

#Values are taked from npt/configs.py
wandb.init(project='npt_test', config={
   # 'data_path': '/content/drive/MyDrive/Software-Defect-Detection-Phase4/',
    #'data_path': '/content/',
    #'data_path': '/Users/OpenU/Desktop/non-parametric-transformers',
    'data_path': '/mnt/c/Users/OpenU/Desktop/non-parametric-transformers',
    'custom_data_path': '/mnt/c/Users/OpenU/Desktop/Tomcat Project/Test data/data_tom.csv', #test
    'exp_lr': 5e-3,
    'architecture': "NPT",
    'model_stacking_depth' : 8,
    'model_dim_hidden' : 64,
    'model_num_heads' : 8,
    'model_feature_type_embedding': True,
    'model_feature_index_embedding': True,
    'model_rff_depth': 1,
    'model_hidden_dropout_prob': 0.1,
    'exp_gradient_clipping': 1.,
    'mp_distributed' : False,
    'model_dtype' : 'float32',
    'exp_device': 'cuda',
    'model_class':'NPT',
    'data_set': 'cross-project-defect-df',
    'label_column': 'label',  # Label column
    'test_size': 0.2,  # test set
    'validation_size': 0.1,  # validation set
})

# Access the config object
c = wandb.config

DEFAULT_AUGMENTATION_BERT_MASK_PROB = {
    'train': 0.15,
    'val': 0.,
    'test': 0.
}
DEFAULT_LABEL_BERT_MASK_PROB = {
    'train': 1,
    'val': 1,
    'test': 1
}

#Values are taked from npt/configs.py
wandb.config['exp_scheduler']='flat_and_anneal'
wandb.config['model_checkpoint_key']= 'cross_project_defect_prediction'
#wandb.config['exp_num_total_steps']= 100e3
wandb.config['exp_num_total_steps']= 1000
wandb.config['debug_eval_row_interactions']= False #row corruption
wandb.config['exp_batch_size']=1800#batch size
#wandb.config['exp_batch_size']= 1
wandb.config['data_loader_nprocs']= 0
wandb.config['exp_checkpoint_setting']='best_model'
wandb.config['exp_n_runs']= 1
wandb.config['exp_cache_cadence']= 1
wandb.config['exp_patience']= -1
wandb.config['exp_load_from_checkpoint']=False
wandb.config['viz_att_maps']= False
wandb.config['exp_tradeoff']= 0.5
#wandb.config['exp_tradeoff']= 1
wandb.config['exp_tradeoff_annealing']= 'cosine'
wandb.config['exp_tradeoff_annealing_proportion']= 1
wandb.config['exp_minibatch_sgd']= True
wandb.config['exp_eval_every_epoch_or_steps']= 'epochs'
wandb.config['model_att_score_norm'] = 'softmax'
wandb.config['exp_eval_test_at_end_only']= False
wandb.config['mp_nodes']= 1
#evaluation
wandb.config['verbose']= True
wandb.config['exp_eval_every_n']= 5
wandb.config['exp_print_every_nth_forward']= 1
wandb.config['data_set_on_cuda']= False
wandb.config['debug_row_interactions']= False
wandb.config['debug_label_leakage']= False
wandb.config['data_dtype']='float32'
wandb.config['model_amp']= False
#test
wandb.config['exp_test_perc']= 0.2
wandb.config['model_is_semi_supervised']= True
wandb.config['np_seed']= 42
wandb.config['torch_seed']= 42
wandb.config['exp_name']= None
wandb.config['exp_group']= None

wandb.config['data_force_reload']= True
wandb.config['metrics_auroc']= True
wandb.config['data_clear_tmp_files']= False
#npt/utils/train_utils.py
wandb.config['exp_optimizer']= 'lookahead_lamb'
wandb.config['debug_no_stratify']= False
wandb.config['exp_val_perc']= 0.1
wandb.config['exp_show_empirical_label_dist']= False
wandb.config['debug_leakage']= False
wandb.config['model_bert_augmentation']= True
wandb.config['data_log_mem_usage']= False
wandb.config['exp_batch_mode_balancing']= True
wandb.config['model_label_bert_mask_prob']= DEFAULT_LABEL_BERT_MASK_PROB
wandb.config['model_augmentation_bert_mask_prob']= DEFAULT_AUGMENTATION_BERT_MASK_PROB
wandb.config['model_bert_mask_percentage']= 0.9
wandb.config['exp_azure_sweep']= False
#wandb.config['exp_azure_sweep']= True
wandb.config['exp_optimizer_warmup_proportion']= 0.01
#from scripts/row_corruption_tests.sh --protein dataset
#wandb.config['debug_eval_row_interactions_timer']= 0.5
wandb.config['debug_eval_row_interactions_timer']= None
wandb.config['exp_weight_decay']= 0
wandb.config['debug_corrupt_standard_dataset_ablate_shuffle'] = False

wandb.config['exp_lookahead_update_cadence']= 6
wandb.config['exp_batch_class_balancing']= True
wandb.config['model_sep_res_embed']= True
wandb.config['exp_eval_every_n']= 5

c = wandb.config

c
'''
'''
# Loading the model from checkpoint
def load_model(checkpoint_path):
    test_dataset = ColumnEncodingDataset(c)
    print(test_dataset.metadata)
    model = NPTModel(c=c,metadata= test_dataset.metadata, device=c.exp_device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # Load model parameters with ignoring missing and unexpected keys
    model.load_state_dict(state_dict, strict=False)
    return model

def evaluate_model(model, dataloader):
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            #predictions.extend(outputs.cpu().numpy())
            predictions = torch.sigmoid(outputs).cpu().numpy()
            #targets.extend(target.cpu().numpy())
            targets = targets.cpu().numpy()

            all_predictions.extend(predictions)
            all_targets.extend(targets)

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Convert predictions to binary values for accuracy calculation (error)
    binary_predictions = (all_predictions > 0.5).astype(int)

   # auroc = roc_auc_score(all_targets, all_predictions)
    auroc = roc_auc_score(all_targets, binary_predictions)
    accuracy = accuracy_score(all_targets, binary_predictions)

    return auroc, accuracy


# Example usage
if __name__ == "__main__":
    checkpoint_path = '/mnt/c/Users/OpenU/Desktop/non-parametric-transformers/cross-project-defect-df/ssl__True/np_seed=42__n_cv_splits=5__exp_num_runs=1/cross_project_defect_prediction/model_checkpoints/model_15.pt'
    model = load_model(checkpoint_path)

    # Test
    test_data_path = '/mnt/c/Users/OpenU/Desktop/Tomcat Project/Test data/data_tom.csv'
    test_data = pd.read_csv(test_data_path)
    test_data= test_data.iloc[:, 4:]
    print(test_data)
    X_test = test_data.iloc[:, :-1].values  # Features
    y_test = test_data.iloc[:, -1].values  # Target labels

    X_test = X_test.astype(float)  # Convert features to float bcs of error
    y_test = y_test.astype(int)
    print(y_test)

    # PyTorch DataLoader
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    auroc, accuracy = evaluate_model(model, test_loader)

    print("AUROC on test data:", auroc)
    print("Accuracy on test data:", accuracy)
else:
    print(f"Test data file not found: {test_data_path}")
'''

# -*- coding: utf-8 -*-
#git clone https://github.com/Huda-AlGhussein/non-parametric-transformers #forked repository

"""# Installing Packages

Packages and requirement from file "environment_no_gpu.yml"
"""


"""# Importing Libraries"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/non-parametric-transformers

import json
import os
import pickle

#import torch
from npt.datasets.base import BaseDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut

from pathlib import Path

from sklearn.datasets import fetch_openml

from npt.utils.cv_utils import (get_class_reg_train_val_test_splits, get_n_cv_splits)
from npt.utils.encode_utils import encode_data_dict
from npt.utils.memory_utils import get_size
from npt.utils.preprocess_utils import (get_matrix_from_rows)
from npt.column_encoding_dataset import ColumnEncodingDataset

from npt.utils.cv_utils import (
    get_class_reg_train_val_test_splits, get_n_cv_splits)
from npt.utils.encode_utils import encode_data_dict
from npt.utils.memory_utils import get_size
from npt.utils.preprocess_utils import (
    get_matrix_from_rows)
from npt.datasets.base import BaseDataset
from npt.column_encoding_dataset import ColumnEncodingDataset
from npt.configs import build_parser
import torchmetrics


"""## Initialize the Model

The model handle the input dimensions internally based on the provided metadata and configuration.

###wandb configuration
"""

import wandb
wandb.login()

#Values are taked from npt/configs.py
wandb.init(project='npt_project', config={
   # 'data_path': '/content/drive/MyDrive/Software-Defect-Detection-Phase4/',
    #'data_path': '/content/',
    #'data_path': '/Users/OpenU/Desktop/non-parametric-transformers','
    'data_path': '/mnt/c/Users/OpenU/Desktop/non-parametric-transformers',
    'custom_data_path': '/mnt/c/Users/OpenU/Desktop/Tomcat Project/Test data/data_tom.csv', #test
    'exp_lr': 5e-3,
    'architecture': "NPT",
    'model_stacking_depth' : 8,
    'model_dim_hidden' : 64,
    'model_num_heads' : 8,
    'model_feature_type_embedding': True,
    'model_feature_index_embedding': True,
    'model_rff_depth': 1,
    'model_hidden_dropout_prob': 0.1,
    'exp_gradient_clipping': 1.,
    'mp_distributed' : False,
    'model_dtype' : 'float32',
    'exp_device': 'cuda',
    'model_class':'NPT',
    'data_set': 'cross-project-defect-df',
    'label_column': 'label',  # Label column
    'test_size':0.2,  # test set
    'validation_size': 0.1,  # validation set
})

# Access the config object
c = wandb.config

DEFAULT_AUGMENTATION_BERT_MASK_PROB = {
    'train': 0.15,
    'val': 0.,
    'test': 0.
}
DEFAULT_LABEL_BERT_MASK_PROB = {
    'train': 1,
    'val': 1,
    'test': 1
}

#Values are taked from npt/configs.py
wandb.config['exp_scheduler']='flat_and_anneal'
wandb.config['model_checkpoint_key']= 'cross_project_defect_prediction'
#wandb.config['exp_num_total_steps']= 100e3
wandb.config['exp_num_total_steps']= 1000
wandb.config['debug_eval_row_interactions']= False #row corruption
wandb.config['exp_batch_size']=1800#batch size
wandb.config['data_loader_nprocs']= 0
wandb.config['exp_checkpoint_setting']='best_model'
wandb.config['exp_n_runs']= 1
wandb.config['exp_cache_cadence']= 1
wandb.config['exp_patience']= -1
wandb.config['exp_load_from_checkpoint']=False
wandb.config['viz_att_maps']= False
wandb.config['exp_tradeoff']= 0.5
#wandb.config['exp_tradeoff']= 1
wandb.config['exp_tradeoff_annealing']= 'cosine'
wandb.config['exp_tradeoff_annealing_proportion']= 1
wandb.config['exp_minibatch_sgd']= True
wandb.config['exp_device']= 'cuda'
wandb.config['exp_eval_every_epoch_or_steps']= 'epochs'
wandb.config['model_att_score_norm'] = 'softmax'
wandb.config['exp_eval_test_at_end_only']= False
wandb.config['mp_nodes']= 1
#evaluation
wandb.config['verbose']= True
wandb.config['exp_eval_every_n']= 5
wandb.config['exp_print_every_nth_forward']= 1
wandb.config['data_set_on_cuda']= False
wandb.config['debug_row_interactions']= False
wandb.config['debug_label_leakage']= False
wandb.config['data_dtype']='float32'
wandb.config['model_amp']= False
#test
wandb.config['exp_test_perc']= 0.2
wandb.config['model_is_semi_supervised']= True
wandb.config['np_seed']= 42
wandb.config['torch_seed']= 42
wandb.config['exp_name']= None
wandb.config['exp_group']= None

wandb.config['data_force_reload']= True
wandb.config['metrics_auroc']= True
wandb.config['data_clear_tmp_files']= False
#npt/utils/train_utils.py
wandb.config['exp_optimizer']= 'lookahead_lamb'
wandb.config['debug_no_stratify']= False
wandb.config['exp_val_perc']= 0.1
wandb.config['exp_show_empirical_label_dist']= False
wandb.config['debug_leakage']= False
wandb.config['model_bert_augmentation']= True
wandb.config['data_log_mem_usage']= False
wandb.config['exp_batch_mode_balancing']= True
wandb.config['model_label_bert_mask_prob']= DEFAULT_LABEL_BERT_MASK_PROB
wandb.config['model_augmentation_bert_mask_prob']= DEFAULT_AUGMENTATION_BERT_MASK_PROB
wandb.config['model_bert_mask_percentage']= 0.9
wandb.config['exp_azure_sweep']= False
#wandb.config['exp_azure_sweep']= True
wandb.config['exp_optimizer_warmup_proportion']= 0.01
#from scripts/row_corruption_tests.sh --protein dataset
#wandb.config['debug_eval_row_interactions_timer']= 0.5
wandb.config['debug_eval_row_interactions_timer']= None
wandb.config['exp_weight_decay']= 0
wandb.config['debug_corrupt_standard_dataset_ablate_shuffle'] = False

wandb.config['exp_lookahead_update_cadence']= 6
wandb.config['exp_batch_class_balancing']= True
wandb.config['model_sep_res_embed']= True
wandb.config['exp_eval_every_n']= 5

c = wandb.config

c

"""### Train Dataset"""

from torch.utils.data import DataLoader
from npt.column_encoding_dataset import ColumnEncodingDataset

train_dataset= ColumnEncodingDataset(c)

#train_loader = DataLoader(train_dataset, batch_size=c.exp_batch_size, shuffle=True, num_workers=2)

print(train_dataset.metadata)

"""###Model Initializtion"""

from npt.model import NPTModel

# Device can be 'cuda' if GPU is used or 'cpu' if not
device = c.exp_device

# Initialize the NPT model
npt_model = NPTModel(c= c, metadata= train_dataset.metadata, device=device)

from npt.utils.encode_utils import get_torch_dtype

npt_model_torch_dtype = get_torch_dtype(dtype_name=c.model_dtype)
npt_model = npt_model.to(device=device).type(npt_model_torch_dtype)

"""### Optimizer and Scaler"""

from npt.utils.train_utils import count_parameters, init_optimizer
from torch.cuda.amp import GradScaler

#default value for optimizer from train_utils.py
optimizer = init_optimizer(
        c= c, model_parameters= npt_model.parameters(), device=device)

scaler = GradScaler(enabled=c.model_amp)

"""###Model"""

from npt.train import Trainer

#from train.py
trainer = Trainer(
    model= npt_model,
    optimizer= optimizer,
    scaler= scaler,
    c= c,
    cv_index=0, #from npt/utils/analysis.py
    dataset= train_dataset,  #Dataset
    distributed_args=None,
    wandb_run= wandb.run
)

trainer.dataset.load_next_cv_split()

"""## Training and Evaluation"""
print(c.exp_num_total_steps)

trainer.train_and_eval()

wandb.finish()
