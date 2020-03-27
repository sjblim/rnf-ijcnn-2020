"""
script_download_data.py.py


Created by limsi on 23/03/2020
"""

import os
import shutil
import pyunpack
import wget
import pandas as pd
import numpy as np
import time

import configs

from libs.utils import recreate_folder

# ------------------------------------------------------------------------------
# General functions for data downloading & aggregation.
def download_from_url(url, output_path):
  """Downloads a file froma url."""

  print('Pulling data from {} to {}'.format(url, output_path))
  wget.download(url, output_path)
  print('done')


def unzip(zip_path, output_file, data_folder):
  """Unzips files and checks successful completion."""

  print('Unzipping file: {}'.format(zip_path))
  pyunpack.Archive(zip_path).extractall(data_folder)

  # Checks if unzip was successful
  if not os.path.exists(output_file):
    raise ValueError(
        'Error in unzipping process! {} not found.'.format(output_file))


def download_and_unzip(url, zip_path, csv_path, data_folder):
  """Downloads and unzips an online csv file.
  Args:
    url: Web address
    zip_path: Path to download zip file
    csv_path: Expected path to csv file
    data_folder: Folder in which data is stored.
  """

  download_from_url(url, zip_path)

  unzip(zip_path, csv_path, data_folder)

  print('Done.')


# ------------------------------------------------------------------------------
# Download scripts
def download_power(data_folder):

  # Reinitialise
  recreate_folder(data_folder)

  url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip'

  base_path = os.path.join(data_folder, 'household_power_consumption')
  zip_path = base_path + r'.zip'
  csv_path = base_path + r'.txt'
  output_path = os.path.join(data_folder, "power.csv")

  download_and_unzip(url, zip_path, csv_path, data_folder)

  df = pd.read_csv(csv_path, sep=";")

  df.index = pd.to_datetime(df['Date'] + " " + df['Time'])
  df = df.sort_index()

  # Add additional cols
  df['Target_active_power'] = df['Global_active_power'].shift(-1)
  df['t'] = [i for i in range(len(df))]
  df['id'] = 0 # only on id col

  df = df.replace("?", np.nan).fillna(method='ffill').dropna()

  df.to_csv(output_path)

# ------------------------------------------------------------------------------
if __name__ == "__main__":

  print("Running download script")
  expt_name = "power"

  configs.init_storage_folders(expt_name)

  data_path = os.path.join(configs.DATA_PATH, expt_name)
  download_power(data_path)

