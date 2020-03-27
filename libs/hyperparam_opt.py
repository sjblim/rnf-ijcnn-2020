"""
hyperparam_opt.py


Created by limsi on 23/03/2020
"""


# Lint as: python3
"""Classes used for hyperparameter optimisation.
Two main classes exist:
1) HyperparamOptManager used for optimisation on a single machine/GPU.
2) DistributedHyperparamOptManager for multiple GPUs on different machines.
"""

import collections
import os
import shutil
import numpy as np
import pandas as pd

Deque = collections.deque


class HyperparamOptManager:
  """Manages hyperparameter optimisation using random search for a single GPU.
  Attributes:
    param_ranges: Discrete hyperparameter range for random search.
    results: Dataframe of validation results.
    fixed_params: Fixed model parameters per experiment.
    saved_params: Dataframe of parameters trained.
    best_score: Minimum validation loss observed thus far.
    optimal_name: Key to best configuration.
    hyperparam_folder: Where to save optimisation outputs.
  """

  def __init__(self,
               param_ranges,
               fixed_params,
               model_folder,
               override_w_fixed_params=True):
    """Instantiates model.
    Args:
      param_ranges: Discrete hyperparameter range for random search.
      fixed_params: Fixed model parameters per experiment.
      model_folder: Folder to store optimisation artifacts.
      override_w_fixed_params: Whether to override serialsed fixed model
        parameters with new supplied values.
    """

    self.param_ranges = param_ranges

    self._max_tries = 1000
    self.results = pd.DataFrame()
    self.fixed_params = fixed_params
    self.saved_params = pd.DataFrame()

    self.best_score = np.Inf
    self.optimal_name = ""

    # Setup
    # Create folder for saving if its not there
    self.hyperparam_folder = model_folder
    self.checkpoint_path = os.path.join(model_folder, "model")
    os.makedirs(self.hyperparam_folder, exist_ok=True)

    self._override_w_fixed_params = override_w_fixed_params

  def load_results(self):
    """Loads results from previous hyperparameter optimisation.
    Returns:
      A boolean indicating if previous results can be loaded.
    """
    print("Loading results from", self.hyperparam_folder)

    results_file = os.path.join(self.hyperparam_folder, "results.csv")
    params_file = os.path.join(self.hyperparam_folder, "params.csv")

    if os.path.exists(results_file) and os.path.exists(params_file):

      self.results = pd.read_csv(results_file, index_col=0)
      self.saved_params = pd.read_csv(params_file, index_col=0)

      if not self.results.empty:
        self.results.at["loss"] = self.results.loc["loss"].apply(float)
        self.best_score = self.results.loc["loss"].min()

        is_optimal = self.results.loc["loss"] == self.best_score
        self.optimal_name = self.results.T[is_optimal].index[0]

        return True

    return False

  def _get_params_from_name(self, name):
    """Returns previously saved parameters given a key."""
    params = self.saved_params

    selected_params = dict(params[name])

    if self._override_w_fixed_params:
      for k in self.fixed_params:
        selected_params[k] = self.fixed_params[k]

    return selected_params

  def get_best_params(self):
    """Returns the optimal hyperparameters thus far."""

    optimal_name = self.optimal_name

    return self._get_params_from_name(optimal_name)

  def clear(self):
    """Clears all previous results and saved parameters."""
    shutil.rmtree(self.hyperparam_folder)
    os.makedirs(self.hyperparam_folder)
    self.results = pd.DataFrame()
    self.saved_params = pd.DataFrame()

  def _check_params(self, params):
    """Checks that parameter map is properly defined."""

    valid_fields = list(self.param_ranges.keys()) + list(
        self.fixed_params.keys())
    invalid_fields = [k for k in params if k not in valid_fields]
    missing_fields = [k for k in valid_fields if k not in params]

    if invalid_fields:
      raise ValueError("Invalid Fields Found {} - Valid ones are {}".format(
          invalid_fields, valid_fields))
    if missing_fields:
      raise ValueError("Missing Fields Found {} - Valid ones are {}".format(
          missing_fields, valid_fields))

  def _get_name(self, params):
    """Returns a unique key for the supplied set of params."""

    self._check_params(params)

    fields = list(params.keys())
    fields.sort()

    return "_".join([str(params[k]) for k in fields])

  def get_next_parameters(self, ranges_to_skip=None):
    """Returns the next set of parameters to optimise.
    Args:
      ranges_to_skip: Explicitly defines a set of keys to skip.
    """
    if ranges_to_skip is None:
      ranges_to_skip = set(self.results.index)

    if not isinstance(self.param_ranges, dict):
      raise ValueError("Only works for random search!")

    param_range_keys = list(self.param_ranges.keys())
    param_range_keys.sort()

    def _get_next():
      """Returns next hyperparameter set per try."""

      parameters = {
          k: np.random.choice(self.param_ranges[k]) for k in param_range_keys
      }

      # Adds fixed params
      for k in self.fixed_params:
        parameters[k] = self.fixed_params[k]

      return parameters

    for _ in range(self._max_tries):

      parameters = _get_next()
      name = self._get_name(parameters)

      if name not in ranges_to_skip:
        return parameters

    raise ValueError("Exceeded max number of hyperparameter searches!!")

  def update_score(self, parameters, loss, model, info=""):
    """Updates the results from last optimisation run.
    Args:
      parameters: Hyperparameters used in optimisation.
      loss: Validation loss obtained.
      model: Model to serialised if required.
      info: Any ancillary information to tag on to results.
    Returns:
      Boolean flag indicating if the model is the best seen so far.
    """

    if np.isnan(loss):
      loss = np.Inf

    if not os.path.isdir(self.hyperparam_folder):
      os.makedirs(self.hyperparam_folder)

    name = self._get_name(parameters)

    is_optimal = self.results.empty or loss < self.best_score

    # save the first model
    if is_optimal:
      # Try saving first, before updating info
      if model is not None:
        print("Optimal model found, updating")

        model.save_weights(self.checkpoint_path, save_format='tf')
      self.best_score = loss
      self.optimal_name = name

    self.results[name] = pd.Series({"loss": loss, "info": info})
    self.saved_params[name] = pd.Series(parameters)

    self.results.to_csv(os.path.join(self.hyperparam_folder, "results.csv"))
    self.saved_params.to_csv(os.path.join(self.hyperparam_folder, "params.csv"))

    return is_optimal

