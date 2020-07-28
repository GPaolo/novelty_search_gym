# Created by Giuseppe Paolo 
# Date: 28/07/2020

import numpy as np

def dummy_obs(traj):
  """
  Get observations from the trajectory coming from the dummy environment
  :param traj:
  :return:
  """
  return np.array([traj[-1][0]]) # Returns the observation of the last element

def walker_2D_obs(traj):
  """
  Get observations from the trajectory coming from the Walker2D environment
  :param traj:
  :return:
  """
  return np.array([t[0] for t in traj]) # t[0] selects the observation part