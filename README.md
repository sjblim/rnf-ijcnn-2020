# Recurrent Neural Filters for Time Series Prediction

**Reference:** Bryan Lim, Stefan Zohren and Stephen Roberts. *Recurrent Neural Filters: Learning Independent Bayesian Filtering Steps for Time Series Prediction*. Proceedings of the International Joint Conference on Neural Networks (IJCNN) 2020

**Paper link:** https://arxiv.org/abs/1901.08096

## Abstract
> Despite the recent popularity of deep generative state space models, few comparisons have been made between network architectures and the inference steps of the Bayesian filtering framework -- with most models simultaneously approximating both state transition and update steps with a single recurrent neural network (RNN). In this paper, we introduce the Recurrent Neural Filter (RNF), a novel recurrent autoencoder architecture that learns distinct representations for each Bayesian filtering step, captured by a series of encoders and decoders. Testing this on three real-world time series datasets, we demonstrate that the decoupled representations learnt improve the accuracy of one-step-ahead forecasts while providing realistic uncertainty estimates, and also facilitate multistep prediction through the separation of encoder stages.


## Code Usage
This code repository contains the demo code for the Standard RNF.


### Quick Start
To download default UCI electricity dataset, and train using the default RNF hyperparameters, run:
```bash
bash run.sh
```
Please refer to run.sh for additional usage instructions.

### Script Organisation
The key scripts are divided into:
* **script_download_data.py**: Downloads the default UCI power dataset and formats the data
* **script_hyperparam_opt.py**: Runs the full hyperparameter optimisation for the RNF
* **script_train_fixed_params.py**: Trains the RNF using pre-defined hyperparameters.
* **script_forecasting**: Performs the one-step-ahead and multistep forecasting evaluations.