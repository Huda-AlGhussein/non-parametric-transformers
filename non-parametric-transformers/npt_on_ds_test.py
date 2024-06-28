import json
import torch
import os
import pickle
import wandb
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from pathlib import Path
from sklearn.datasets import fetch_openml
from npt.utils.cv_utils import (get_class_reg_train_val_test_splits, get_n_cv_splits)
from npt.utils.encode_utils import encode_data_dict
from npt.utils.memory_utils import get_size
from npt.utils.preprocess_utils import (get_matrix_from_rows)
from npt.column_encoding_dataset import ColumnEncodingDataset
from npt.configs import build_parser
import torchmetrics
from torch.utils.data import DataLoader
from npt.model import NPTModel
from npt.utils.encode_utils import get_torch_dtype
from npt.train import Trainer
from npt.utils.model_init_utils import (init_model_opt_scaler, setup_ddp_model)
from npt.utils.eval_checkpoint_utils import EarlyStopCounter


# Wandb login and initialization
wandb.login()
# Load the configuration
wandb.init(project='npt_test', config={
    'data_path': '/mnt/c/Users/OpenU/Desktop/non-parametric-transformers',
    'custom_data_path': '/mnt/c/Users/OpenU/Desktop/Tomcat Project/Test data/data_tom.csv',
    'exp_lr': 5e-3,
    'architecture': "NPT",
    'model_stacking_depth': 8,
    'model_dim_hidden': 64,
    'model_num_heads': 8,
    'model_feature_type_embedding': True,
    'model_feature_index_embedding': True,
    'model_rff_depth': 1,
    'model_hidden_dropout_prob': 0.1,
    'exp_gradient_clipping': 1.,
    'mp_distributed': False,
    'model_dtype': 'float32',
    'exp_device': 'cuda',
    'model_class': 'NPT',
    'data_set': 'cross-project-defect-df',
    'label_column': 'label',
    'test_size': 0.2,
    'validation_size': 0.1,
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

# Values are taken from npt/configs.py
wandb.config['exp_scheduler'] = 'flat_and_anneal'
wandb.config['model_checkpoint_key'] = 'cross_project_defect_prediction'
wandb.config['exp_num_total_steps'] = 1000
wandb.config['debug_eval_row_interactions'] = False
wandb.config['exp_batch_size'] = 32 #for test, for train 1800
wandb.config['data_loader_nprocs'] = 0
wandb.config['exp_checkpoint_setting'] = 'best_model'
wandb.config['exp_n_runs'] = 1
wandb.config['exp_cache_cadence'] = 1
wandb.config['exp_patience'] = -1
wandb.config['exp_load_from_checkpoint'] = True
wandb.config['viz_att_maps'] = False
wandb.config['exp_tradeoff'] = 0.5
wandb.config['exp_tradeoff_annealing'] = 'cosine'
wandb.config['exp_tradeoff_annealing_proportion'] = 1
wandb.config['exp_minibatch_sgd'] = True
wandb.config['exp_device'] = 'cuda'
wandb.config['exp_eval_every_epoch_or_steps'] = 'epochs'
wandb.config['model_att_score_norm'] = 'softmax'
wandb.config['exp_eval_test_at_end_only'] = True #change to test only
wandb.config['mp_nodes'] = 1
wandb.config['verbose'] = True
wandb.config['exp_eval_every_n'] = 5
wandb.config['exp_print_every_nth_forward'] = 1
wandb.config['data_set_on_cuda'] = False
wandb.config['debug_row_interactions'] = False
wandb.config['debug_label_leakage'] = False
wandb.config['data_dtype'] = 'float32'
wandb.config['model_amp'] = False
wandb.config['exp_test_perc'] = 0.2
wandb.config['model_is_semi_supervised'] = True
wandb.config['np_seed'] = 42
wandb.config['torch_seed'] = 42
wandb.config['exp_name'] = None
wandb.config['exp_group'] = None
wandb.config['data_force_reload'] = True
wandb.config['metrics_auroc'] = True
wandb.config['data_clear_tmp_files'] = False
wandb.config['exp_optimizer'] = 'lookahead_lamb'
wandb.config['debug_no_stratify'] = False
wandb.config['exp_val_perc'] = 0.1
wandb.config['exp_show_empirical_label_dist'] = False
wandb.config['debug_leakage'] = False
wandb.config['model_bert_augmentation'] = True
wandb.config['data_log_mem_usage'] = False
wandb.config['exp_batch_mode_balancing'] = True
wandb.config['model_label_bert_mask_prob'] = DEFAULT_LABEL_BERT_MASK_PROB
wandb.config['model_augmentation_bert_mask_prob'] = DEFAULT_AUGMENTATION_BERT_MASK_PROB
wandb.config['model_bert_mask_percentage'] = 0.9
wandb.config['exp_azure_sweep'] = False
wandb.config['exp_optimizer_warmup_proportion'] = 0.01
wandb.config['debug_eval_row_interactions_timer'] = None
wandb.config['exp_weight_decay'] = 0
wandb.config['debug_corrupt_standard_dataset_ablate_shuffle'] = False
wandb.config['exp_lookahead_update_cadence'] = 6
wandb.config['exp_batch_class_balancing'] = True
wandb.config['model_sep_res_embed'] = True
wandb.config['exp_eval_every_n'] = 5
#added this config to test
wandb.config['exp_checkpoint_path']= '/mnt/c/Users/OpenU/Desktop/non-parametric-transformers/cross-project-defect-df/ssl__True/np_seed=42__n_cv_splits=5__exp_num_runs=1/cross_project_defect_prediction/model_checkpoints/model_15.pt'

c = wandb.config

# Load dataset
test_dataset = ColumnEncodingDataset(c)
print(test_dataset.metadata)
# Initialize model
device = c.exp_device
npt_model = NPTModel(c=c, metadata=test_dataset.metadata, device=device)
# Load model checkpoint
npt_model, optimizer, scaler = init_model_opt_scaler(c, metadata=test_dataset.metadata, device=device)

# Load from checkpoint, populate state dicts
checkpoint = torch.load(c.exp_checkpoint_path, map_location= device)
# Strict setting -- allows us to load saved attention maps
# when we wish to visualize them
npt_model.load_state_dict(checkpoint['model_state_dict'], strict=(not c.viz_att_maps))

if c.viz_att_maps:
    optimizer = None
    scaler = None
else:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

print(
f'Successfully loaded cached model from best performing epoch '
f'{checkpoint["epoch"]}.')

from npt.utils.train_utils import count_parameters, init_optimizer
from torch.cuda.amp import GradScaler

test = Trainer(
    model= npt_model,
    optimizer= optimizer,
    scaler= scaler,
    c= c,
    cv_index=0, #from npt/utils/analysis.py
    dataset= test_dataset,  #Dataset
    distributed_args=None,
    wandb_run= wandb.run
)
from npt.utils.batch_utils import collate_with_pre_batching
from npt.utils.encode_utils import torch_cast_to_dtype
from torchmetrics.functional import auroc as lightning_auroc

test.dataset.load_next_cv_split()
test.dataset.set_mode(mode='test', epoch=0)
test_loader = DataLoader(test.dataset.cv_dataset, batch_size=1,
                          shuffle=False, num_workers=c.mp_nodes, collate_fn=collate_with_pre_batching)
print(test_loader)

# Define the evaluate_model function
eval_model=True
npt_model.eval()
total_auc=0
batch_counter=0
#npt_model.dataset.load_next_cv_split()
with torch.no_grad():
    for batch_index, batch_dict in enumerate(test_loader):
        #print(batch_dict)
        masked_tensors, label_mask_matrix, augmentation_mask_matrix = (
            batch_dict['masked_tensors'],
            batch_dict['label_mask_matrix'],
            batch_dict['augmentation_mask_matrix'])
        ground_truth_tensors = batch_dict['data_arrs']
        # Cast tensors to appropriate data type
        ground_truth_tensors = [
            torch_cast_to_dtype(obj=data, dtype_name=c.data_dtype)
            for data in ground_truth_tensors]
        ground_truth_tensors = [
            data.to(device=device, non_blocking=True)
            for data in ground_truth_tensors]
        masked_tensors = [data.to(device=device, non_blocking=True) for data in masked_tensors]
        if augmentation_mask_matrix is not None:
            augmentation_mask_matrix = augmentation_mask_matrix.to(
                device=device, non_blocking=True)

        # Need label_mask_matrix for stochastic label masking
        if label_mask_matrix is not None:
            label_mask_matrix = label_mask_matrix.to(
                device=device, non_blocking=True)
        outputs = npt_model(masked_tensors)

        print(f'Length of outputs: {len(outputs)}')
        print(f'Length of ground_truth_tensors: {len(ground_truth_tensors)}')

        ground_truth_tensors_target = ground_truth_tensors[-1]
        outputs_target = outputs[-1] 
        print(ground_truth_tensors_target)

        long_data = torch.argmax(torch_cast_to_dtype(obj=ground_truth_tensors_target,dtype_name=c.data_dtype),
                                 dim=1)
        #print(long_data.shape)
        #preds = torch.cat(outputs_target,dim=0)
        #print(preds.shape)
        softmax = torch.nn.Softmax(dim=1)
        preds = softmax(outputs_target)
        #print(preds)
        #true=torch.cat(long_data,dim=0)
        true=long_data
        roc_auc = lightning_auroc(preds[:,-1],true,task="binary")
        print(f'ROC AUC: {roc_auc}')
        total_auc+=roc_auc
        batch_counter+=1
if batch_counter>0:
    avg_roc_auc = total_auc / batch_counter
else:
    avg_roc_auc=0
    print("No elements in dataset.")
print(f'Average ROC AUC: {avg_roc_auc}')

wandb.finish()

