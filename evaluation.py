#pip install torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define layers
        self.fc = nn.Linear(26, 1)  # Example layer

    def forward(self, x):
        return self.fc(x)

# Loading the model from checkpoint
def load_model(checkpoint_path):
    model = MyModel()
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
    X_test = test_data.iloc[:, :-1].values  # Features
    y_test = test_data.iloc[:, -1].values  # Target labels

    X_test = X_test.astype(float)  # Convert features to float bcs of error
    y_test = y_test.astype(int)
    print(y_test)

    # PyTorch DataLoader
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    auroc, accuracy = evaluate_model(model, test_loader)

    print("AUROC on test data:", auroc)
    print("Accuracy on test data:", accuracy)
else:
    print(f"Test data file not found: {test_data_path}")