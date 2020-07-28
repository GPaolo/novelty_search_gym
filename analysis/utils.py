# Created by Giuseppe Paolo 
# Date: 28/07/2020

import numpy as np
import os
from scipy.spatial.distance import jensenshannon
from core.population import Archive
import pickle as pkl
from environments.environments import registered_envs
import re
from parameters import Params
import multiprocessing as mp

def calculate_coverage(occupied_grid):
  """
  This function calculated the coverage percentage from the grid
  :param occupied_grid
  :return:
  """
  coverage = np.sum(occupied_grid)/occupied_grid.size
  return coverage

def calculate_uniformity(normed_grid):
  """
  This function calculates the uniformity of the normed grid, that is the histogram
  :param normed_grid
  :return:
  """
  uniform_grid = np.ones_like(normed_grid)/normed_grid.size
  return 1-jensenshannon(normed_grid.flatten(), uniform_grid.flatten())

def get_grid(points, grid_parameters):
  """
  This function calculates the normed histogram and the grid of occupied cells.
  :param points:
  :param grid_parameters
  :return: histogram, occupied_grid
  """
  hist, x_edges, y_edges = np.histogram2d(points[:, 0], points[:, 1],
                                          bins=[grid_parameters['bins'], grid_parameters['bins']],
                                          range=[[grid_parameters['min_coord'][0], grid_parameters['max_coord'][0]],
                                                 [grid_parameters['min_coord'][1], grid_parameters['max_coord'][1]]],
                                          density=False)
  hist = hist.T[::-1, :]
  occupied_grid = hist.copy()
  occupied_grid[occupied_grid > 0] = 1
  return hist, occupied_grid

def extract_exp_cov(exp_folder): # TODO needs to be improved
  """
  This function loads the trajectories and extract coverage and unif by generation
  :param exp_folder: Folder of the experiment
  :param gt_bd: Function to extract the GT_BD from the traj
  :param grid_params: Parameters to calculate the grid
  """
  print("Working on exp: {}".format(exp_folder))
  # Load archive traj and parameters
  with open(os.path.join(exp_folder, 'archive_traj.pkl'), 'rb') as f:
    trajs = pkl.load(f)
  params = Params()
  params.load(os.path.join(exp_folder, '_params.json'))
  grid_params = registered_envs[params.env_name]['grid']

def get_runs_list(path):
  """
  This function returns the list of run folders in the path, by verifying that they are in the right format
  :param path:
  :return:
  """
  assert os.path.exists(path), "The path {} does not exists!".format(path)
  r = re.compile(".{4}_.{2}_.{2}_.{2}:.{2}_.{6}")
  runs = [run for run in os.listdir(path) if r.match(run)]
  return runs

def load_arch_data(folder, info=None, generation=None, params=None):
  """
  This function loads all the archives, by generation in an experiment folder.
  :param folder: The folder of the experiment
  :param info: List of information to load from the archive. If None the whole archive is loaded
  :param generation: generation for which to load the archive. If None, load all the archives
  """
  archives = {}
  r = re.compile('archive_gen_.*.pkl')
  files = [file for file in os.listdir(folder) if 'archive' in file and r.match(file) is not None]
  for arch_file in files:
    arch = Archive(params)
    gen = int(arch_file.split('_')[-1].split('.')[0])
    if gen == generation or generation is None:
      arch.load(os.path.join(folder, arch_file))
      if info is None:
        archives[gen] = arch # Load whole archive
      else:
        archives[gen] = {}
        for label in info: # Save only the needed info
          archives[gen][label] = arch[label] # Each info is saved as a list of values
  return archives