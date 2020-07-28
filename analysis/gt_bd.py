# Created by Giuseppe Paolo 
# Date: 28/07/2020

import numpy as np

# In this file we specify the functions that from the traj of observations and from the infos extract the ground truth BD.
# On this BD we calculate the coverage and uniformity of the archive.
# But they are not used during the search process. Only at the final evaluation process in the evaluate archive script.

def dummy_gt_bd(traj, info, max_steps, ts=1):
  """
  This function extract the ground truth BD for the dummy environment
  :param traj:
  :param max_steps: Maximum number of steps the traj can have
  :param ts: percentage of the traj len at which the BD is extracted. In the range [0, 1]. Default: 1
  :return:
  """
  if ts == 1:
    index = max_steps - 1
  else:
    index = int(max_steps * ts)
  if index >= len(traj): index = -1  # If the trajectory is shorted, consider it as having continued withouth changes
  obs = traj[index]
  return obs