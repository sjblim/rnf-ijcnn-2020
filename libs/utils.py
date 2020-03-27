"""
utils.py


Created by limsi on 24/03/2020
"""


import os
import shutil
import time

# Generic
def recreate_folder(path):
  """Deletes and recreates folder."""
  try:
    shutil.rmtree(path)
    time.sleep(2)
  except:
    pass
  os.makedirs(path)
