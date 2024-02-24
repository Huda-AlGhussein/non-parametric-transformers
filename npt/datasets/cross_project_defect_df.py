# -*- coding: utf-8 -*-
"""cross-project-defect-df.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r-d3LrFHOlt21VnWSsuaHKkDTfpLGX3T
"""

!git clone https://github.com/Huda-AlGhussein/non-parametric-transformers

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/non-parametric-transformers

from google.colab import drive
drive.mount('/content/drive')

path_to_data= '/content/drive/MyDrive/Software Defect Detection - Phase 4 /defect_prediction_data.csv'

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from npt.datasets.base import BaseDataset

"""#Repository code"""

class CrossProjectDefectDataset(BaseDataset):
    def __init__(self, c):
        super().__init__(
            fixed_test_set_index=None)
        self.c = c

    def load(self):
        """
        Software Defect Diction dataset.

        Target in last column.
        110917 rows.
        30 attributes.
        1 target (labels) (2 unique numbers)
        Features:
        Project                         moa
        Version                         mfa
        Class                           cam
        wmc                             ic
        dit                             cbm
        noc                             amc
        cbo                             ndc
        rfc                             nml
        lcom                            isOld
        ca                              ndpv
        ce                              max(cc)
        npm                             avg(cc)
        lcom3                           bugs
        loc                             label
        dam

        Std of Target Col 0.3446975399556955.
        """

        # Load data from drive
        data_home = path_to_data
        df = pd.read_csv(path_to_data, index_col=0)
        df= df.drop('bugs', axis=1)
        x = df

        if isinstance(x, np.ndarray):
            pass
        elif isinstance(x, pd.DataFrame):
            x = x.to_numpy()

        self.data_table = x
        self.N = self.data_table.shape[0]
        self.D = self.data_table.shape[1]

        # Target col is the last feature
        self.num_target_cols = [self.D - 1]
        self.cat_target_cols = []

        self.num_features = list(range(2, self.D - 1))
        self.cat_features = [0,1,2]

        # TODO: add missing entries to sanity check
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)
        self.is_data_loaded = True