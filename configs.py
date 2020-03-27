"""
configs.py


Created by limsi on 23/03/2020
"""

import os
import pandas as pd
import numpy as np

from data_formatters import power, batchers

# Main folder path
ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))

# CHANGE THIS FOR SERIALISED FOLDERS
_storage_root = os.path.join(ROOT_FOLDER, "..")

# Folders
DATA_PATH = os.path.join(_storage_root, "Data")
MODEL_PATH = os.path.join(_storage_root, "Models", 'RNF')
OUTPUT_PATH = os.path.join(ROOT_FOLDER, "outputs")

# OTHER PARAMS
VALID_EXPT_NAMES = ["power"]


# ------------------------------------------------------------------------
# General functions
def check_valid_expt(expt_name):
    if expt_name not in VALID_EXPT_NAMES:
        raise ValueError("Unrecognised experiment: {}".format(expt_name))


def init_storage_folders(expt_name):
    """Create folders if they don't exist"""
    print("Initialising storage folders")

    check_valid_expt(expt_name)

    folders = [ os.path.join(folder, expt_name) for expt_name in VALID_EXPT_NAMES
                    for folder in [DATA_PATH, MODEL_PATH, OUTPUT_PATH]]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

# ------------------------------------------------------------------------
# Dataset specific functions
def load_dataset(expt_name):

    print("Loading data for: {}".format(expt_name))

    check_valid_expt(expt_name)

    filename = {'power': 'power.csv'}[expt_name]

    data_file = os.path.join(DATA_PATH, expt_name, filename)

    df = pd.read_csv(data_file, index_col=0)

    return df

def get_dataformatter(expt_name, model_name):

    if expt_name == "power":

        return power.PowerFormatters(model_name)

    else:
        raise ValueError("Unrecognised experiment name: {}".format(expt_name))

# ------------------------------------------------------------------------
# Model specific functions
def get_batcher(model_name):

    if 'rnf' in model_name:
        print("Getting special batcher for RNF")
        return batchers.EfficientRnfBatcher

    else:
        print("Return default autoregressive batcher")
        return batchers.EfficientAutoregressiveBatcher

def get_default_hyperparams(model_name):

    defaults =  {'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
                    'hidden_layer_size': [5, 10, 25, 50, 100, 150],
                    'minibatch_size': [256, 512, 1024],
                    'learning_rate': np.logspace(-4, 0, 5),
                    'max_norm': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                }

    if 'rnf' in model_name:
        print("Using RNF params")
        params = dict(defaults)
        params['skip_rate'] = [0.25, 0.5, 0.75]

    else:
        print("Returning default params")
        params = dict(defaults)
    #else:
    #    raise ValueError("Unrecognised experiment name: {}".format(model_name))

    return params