#!/bin/bash

# Step 1: Setup environment.
echo
echo Running RNF Training Script
echo

set -e


# Step 1: Download & format UCI power dataset
echo
python -m script_download_data

# Step 2: Train with default RNF parameters
echo
python -m script_train_fixed_params rnf power

# Uncomment below to kickstart hyperparamter optimisation
# python -m script_hyperparam_opt rnf power yes   # To train RNF
# python -m script_hyperparam_opt lstm power yes  # To train DeepAR

# Step 3: Evaluate multistep forecasts for RNF
# N.b. user-defined settings can be modified in script
python -m script_forecasting