import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.naive_bayes import GaussianNB
from functools import partial

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestRegressor, RandomForestClassifier)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor

from baselines.utils.hyper_tuning_utils import modified_tabnet
from baselines.c45 import C45
from baselines.smo_optimizer import SVM
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import label_binarize

from transfer_learning_methods import watanabe_transform, cruz_transform, knn_filter, \
    he_method, herbold_method, ma_transfer_naive_bayes, get_split_train_dataset
MAX_ITER = 1000
SKLEARN_CLASSREG_MODELS = {
    'XGBoost': {
        'reg': XGBRegressor,
        'class': partial(XGBClassifier, use_label_encoder=False)},
    'GradientBoosting': {
        'reg': GradientBoostingRegressor,
        'class': GradientBoostingClassifier},
    'RandomForest': {
        'reg': RandomForestRegressor,
        'class': RandomForestClassifier},
    'CatBoost': {
        'reg': CatBoostRegressor,
        'class': CatBoostClassifier},
    'MLP': {
        'reg': MLPRegressor,
        'class': MLPClassifier},
    'LightGBM': {
        'reg': LGBMRegressor,
        'class': LGBMClassifier
    },
    'TabNet': {
        'reg': modified_tabnet(TabNetRegressor),
        'class': modified_tabnet(TabNetClassifier),
    },
    'KNN': {
        'reg': KNeighborsRegressor,
        'class': KNeighborsClassifier
    },
    'SVM': {
        'class': lambda verbose: CalibratedClassifierCV(
            svm.LinearSVC(verbose=True,C=0.1,max_iter=MAX_ITER))  #svm.SVC(kernel='linear',probability=True,verbose=True) # #SVM
    },
    'C45': {
        'class': lambda verbose: C45()
    },
    'NaiveBayes': {
        'class': lambda verbose: GaussianNB()
    }
}

TRANSFER_LEARNING_METHODS = {
    'watanabe'  : lambda x: watanabe_transform,
    'cruz'      : lambda x: cruz_transform,
    'turhan'    : lambda x: knn_filter,
    'he_method' : lambda x: he_method,
    'herbold'   : lambda x: herbold_method,
    'ma_nb'     : lambda x: ma_transfer_naive_bayes
    
}

DEFAULT_TARGET_COL = "label"
DEFAULT_INPUT_FEATURES = lambda x: x[4:-1]

HYPER_PARAMS_SEARCH = {
    'RandomForest':
        {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        },
    'NaiveBayes':
        {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        },
    'SVM':
        {
            'C': [1e-3, 1e-2, 0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
            'kernel': [ 'linear']
        },
    'C45':
        {
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'criterion': ['gini', 'entropy']
        },
    'MLP':
        {
            'hidden_layer_sizes': [(64,), (128,), (64, 64)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01]
        }
}

import wandb
from datetime import datetime

use_wandb = False

def save_results_to_csv(project_folder, model_name, task_type, test_results):
    """Save results to a CSV file in the project folder."""
    results_list = []
    for test_file, performance in test_results.items():
        result = {
            'model': model_name,
            'task_type': task_type,
            'test_file': test_file,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **performance
        }
        results_list.append(result)
    
    results_df = pd.DataFrame(results_list)
    
    # Create results folder if it doesn't exist
    results_folder = os.path.join(project_folder, 'results')
    os.makedirs(results_folder, exist_ok=True)
    
    # Save to CSV
    csv_path = os.path.join(results_folder, f'results_{model_name}_{task_type}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

def load_data(file_path):
    """Load data from a CSV file."""
    print(f"Load file {file_path}")
    return pd.read_csv(file_path,index_col=0)

def preprocess_data(data, target_column=DEFAULT_TARGET_COL,scaler=True):
    """Preprocess the data."""
    #print(data.iloc[0])
    #print(data.columns)
    X = data.drop(columns=[target_column])
    X = X.iloc[:,3:-1] #data.apply(DEFAULT_INPUT_FEATURES,axis=1)
    #print(X.iloc[0])
    y = data[target_column]
    if scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y
    return X,y

def train_model(model, X_train, y_train, params=None):
    """Train a model with optional parameters."""
    if params:
        model.set_params(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, task_type, y_pred=None, y_proba=None):
    """Evaluate the model and return comprehensive metrics."""
    if y_pred == None:
        y_pred = model.predict(X_test)
    
    if task_type == 'class':
        # Classification metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Add ROC AUC score for binary classification
        #print(len(np.unique(y_test)))
        if len(np.unique(y_test)) == 2:
            if y_proba == None:
                y_proba  = model.predict_proba(X_test)[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_test,y_proba)
        else:
            # For multi-class, calculate ROC AUC for each class
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            y_pred_proba = model.predict_proba(X_test)
            metrics['roc_auc'] = roc_auc_score(y_test_bin, y_pred_proba, average='macro', multi_class='ovr')
    
    elif task_type == 'reg':
        # Regression metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred)
        }
    
    return metrics

def handle_transfer_learning(method_name, model, train_data, test_data, target_column):
    """
    Transforms the data according to the given method.

    Returns: Modified_X_train, Y_train, Modified_X_test, Y_predictions
    """
    X_test, Y_test = preprocess_data(test_data)
    if method_name not in TRANSFER_LEARNING_METHODS.keys(): 
        X_train, y_train = preprocess_data(train_data, target_column)
        print(f'ERROR: {method_name} does not exist as a transfer learning method')
        return X_train, y_train, X_test, None, None
    
    if method_name == 'he_method' or method_name == 'herbold':

        #method = TRANSFER_LEARNING_METHODS[method_name](0)
        X_train_ps, Y_train_ps = get_split_train_dataset(train_data)
        if method_name == 'he_method':
            X_test, Y_test = preprocess_data(test_data, scaler=False)
            final_predictions, final_probas = he_method(X_train_ps,X_test, Y_train_ps,model)
            #print(final_predictions)
            #print(final_probas)
            return X_train_ps, Y_train_ps, X_test, final_predictions.tolist(),final_probas.tolist()
        
        else:

            selected_projects, X_train_filtered, y_train_filtered =\
                herbold_method(X_train_ps, Y_train_ps, X_test)

            return X_train_filtered, y_train_filtered, X_test, None, None
    
    elif method_name == 'ma_nb':

        X_train, y_train = preprocess_data(train_data, target_column)
        final_predictions, proba = ma_transfer_naive_bayes(X_train,X_test,y_train)
        return X_train, y_train, X_test, final_predictions.tolist(), proba.tolist()
    
    elif method_name == 'turhan':

        X_train, y_train = preprocess_data(train_data, target_column)
        method = TRANSFER_LEARNING_METHODS[method_name](0)
        print(X_train.shape)
        X_train_filtered,y_train = method(X_train, X_test,y_train)
        print(X_train_filtered.shape)
        return X_train_filtered,y_train, X_test, None, None
    
    else:

        X_train, y_train = preprocess_data(train_data, target_column)
        method = TRANSFER_LEARNING_METHODS[method_name](0)
        X_test_modified = method(X_train, X_test)
        return X_train, y_train, X_test_modified, None, None

def perform_grid_search(model, X_train, y_train, param_grid, task_type):
    """Perform grid search for hyperparameter tuning."""
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy' if task_type=='class' else 'neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

def evaluate_on_test_files(model, test_folder, task_type):
    """Evaluate the model on all CSV files in the test folder."""
    results = {}
    for file in os.listdir(test_folder):
        if file.endswith('.csv'):
            test_data = load_data(os.path.join(test_folder, file))
            X_test, y_test = preprocess_data(test_data)
            performance = evaluate_model(model, X_test, y_test, task_type)
            results[file] = performance
    return results

def process_project(project_folder, model_name, task_type,
                     target_col=DEFAULT_TARGET_COL, grid_search=False):
    """Process a single project folder."""
    train_file = [f for f in os.listdir(project_folder) if f.endswith('.csv')][0]
    train_data = load_data(os.path.join(project_folder, train_file))
    
    X_train, y_train = preprocess_data(train_data, target_col)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # experiment with smaller data
    #X_train = X_train[:10000,:]
    #y_train = y_train[:10000]
    model_class = SKLEARN_CLASSREG_MODELS[model_name][task_type]
    model = model_class(verbose=True)
    #print(f'ROC_AUC: {roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])}')

    if grid_search:
        # Define a simple param_grid for demonstration
        if model_name in HYPER_PARAMS_SEARCH:
            param_grid = HYPER_PARAMS_SEARCH[model_name]
        else:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10]
            }
        best_params = perform_grid_search(model, X_train, y_train, param_grid, task_type)
        model = train_model(model, X_train, y_train, best_params)
    else:
        model = train_model(model, X_train, y_train)
    
    #performance = evaluate_model(model, X_test, y_test,task_type)
    
    test_folder = os.path.join(project_folder, 'test')
    test_results  = evaluate_on_test_files(model, test_folder,task_type)
    
    # Log to wandb
    if use_wandb:
        for test_file, performance in test_results.items():
            wandb.log({
                'project': os.path.basename(project_folder),
                'model': model_name,
                'task_type': task_type,
                'test_file': test_file,
                **performance
            })
    
    # Save results to CSV
    save_results_to_csv(project_folder, model_name, task_type, test_results)
    
    return model, test_results

def process_project_with_transfer_learning(project_folder, model_name, task_type,transfer_learning_method,
                     target_col=DEFAULT_TARGET_COL, grid_search=False):
    """Process a single project folder."""
    train_file = [f for f in os.listdir(project_folder) if f.endswith('.csv')][0]
    train_data = load_data(os.path.join(project_folder, train_file))
    
    model_class = SKLEARN_CLASSREG_MODELS[model_name][task_type]
    model = model_class(verbose=True)
    test_folder = os.path.join(project_folder, 'test')
    test_results = {}
    for test_file in os.listdir(test_folder):
        if not test_file.endswith('.csv'): continue
        
        test_data = load_data(os.path.join(test_folder, test_file))
        
        X_train, y_train, X_test, y_pred, y_probs = handle_transfer_learning(
            transfer_learning_method,model, train_data,test_data,target_col)
        #print(type(y_pred))
        
        if y_pred != None:
            #print(y_probs)
            performance = evaluate_model(model, X_test, test_data[target_col], task_type,
                                          y_pred=y_pred, y_proba=y_probs)
            test_results[test_file] = performance
            continue
        #X_train, y_train, scaler = preprocess_data(train_data, target_col)
        if grid_search:
            # Define a simple param_grid for demonstration
            if model_name in HYPER_PARAMS_SEARCH:
                param_grid = HYPER_PARAMS_SEARCH[model_name]
            else:
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10]
                }
            best_params = perform_grid_search(model, X_train, y_train, param_grid, task_type)
            model = train_model(model, X_train, y_train, best_params)
        
        else:
            model = train_model(model, X_train, y_train)
        
        performance = evaluate_model(model, X_test, test_data[target_col], task_type)
        test_results[test_file] = performance
    #performance = evaluate_model(model, X_test, y_test,task_type)
    #test_results  = evaluate_on_test_files(model, test_folder,task_type)
    # Log to wandb
    if use_wandb:
        for test_file, performance in test_results.items():
            wandb.log({
                'project': os.path.basename(project_folder),
                'model': model_name,
                'task_type': task_type,
                'test_file': test_file,
                **performance
            })
    
    # Save results to CSV
    save_results_to_csv(project_folder, model_name, f'{task_type}_{transfer_learning_method}', test_results)
    
    return model, test_results

def main(base_folder, model_name, task_type, grid_search=False, specific_project=None, transfer_learning=None):
    """Main function to process all projects or a specific project in the base folder."""
    if use_wandb:
        wandb.init(project=f"cross_project_defect{specific_project if specific_project != None else ''}", 
               config={
                    "base_folder": base_folder,
                    "model": model_name,
                    "task_type": task_type,
                    "grid_search": grid_search,
                    "specific_project": specific_project
                })
    
    results = {}
    if specific_project:
        projects_to_process = [os.path.join(base_folder, specific_project)]
    else:
        projects_to_process = [os.path.join(base_folder, p) for p in os.listdir(base_folder)]
    for pf in projects_to_process:
        if os.path.isdir(pf):
            print(f"Processing project: {pf}")
            if transfer_learning != None:
                model,test_results = process_project_with_transfer_learning(
                    pf, model_name, task_type, transfer_learning,grid_search=grid_search)
            else:
                model, test_results = process_project(pf,
                    model_name, task_type,grid_search=grid_search)
            results[pf] = {
                'model': model,
                'test_results': test_results
            }
        else:
            print(f"Error: Project folder '{pf}' not found in '{base_folder}'")
    # if specific_project:
    #     project_folder = os.path.join(base_folder, specific_project)
    #     if os.path.isdir(project_folder):
    #         print(f"Processing project: {specific_project}")
    #         if transfer_learning != None:
    #             model,test_results = process_project_with_transfer_learning(
    #                 project_folder, model_name, task_type, transfer_learning,grid_search=grid_search)
    #         else:
    #             model, test_results = process_project(project_folder,
    #                 model_name, task_type,grid_search=grid_search)
    #         results[specific_project] = {
    #             'model': model,
    #             'test_results': test_results
    #         }
    #     else:
    #         print(f"Error: Project folder '{specific_project}' not found in '{base_folder}'")
    # else:
    #     for project in os.listdir(base_folder):
    #         project_folder = os.path.join(base_folder, project)
    #         if os.path.isdir(project_folder):
    #             print(f"Processing project: {project}")
    #             model, test_results = process_project(project_folder,
    #             model_name, task_type, grid_search=grid_search)
    #             results[specific_project] = {
    #                 'model': model,
    #                 'test_results': test_results
    #             }
    
    if use_wandb: wandb.finish()
    
    return results

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate machine learning models on multiple projects.")
    parser.add_argument("--base_folder", type=str, default="Train-Test Data", help="Base folder containing project directories")
    parser.add_argument("--model", type=str, default="RandomForest", choices=SKLEARN_CLASSREG_MODELS.keys(), help="Model to use for training")
    parser.add_argument("-tl","--transfer_learning", type=str, default=None, choices=TRANSFER_LEARNING_METHODS.keys(), help="Model to use for training")
    parser.add_argument("--task", type=str, default="class", choices=["class", "reg"], help="Task type: 'class' for classification, 'reg' for regression")
    parser.add_argument("--grid_search", action="store_true", help="Perform grid search for hyperparameter tuning")
    parser.add_argument("--project", type=str, help="Specific project to process (if not specified, all projects will be processed)")
    parser.add_argument("--wandb",action="store_true",help="Use wandb for logging. Needs an account and a key")

    args = parser.parse_args()

    if args.wandb:
        use_wandb=True
    results = main(args.base_folder, args.model, args.task, args.grid_search,args.project,args.transfer_learning)
    for project, data in results.items():
        print(f"\nResults for project: {project}")
        for test_file, performance in data['test_results'].items():
            print(f"Test file: {test_file}")
            for metric, value in performance.items():
                print(f"  {metric}: {value:.4f}")