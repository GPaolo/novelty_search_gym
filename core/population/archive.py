# Created by Giuseppe Paolo 
# Date: 27/07/2020

import os
import pickle as pkl
import numpy as np
from collections import deque

class Archive(object):
  """
  This class implements the archive. It only stores the BD and traj of each agent in unordered sets.
  We do not need to know anything else nor we need an order.
  """
  # ---------------------------------
  def __init__(self, parameters):
    self.data = deque() # This contains tuples with (agent_id, gt_bd bd, traj). The agent_id is necessary in case there are same bd-traj
    # self.data = set() # If wanna use set
    self.params = parameters
    self.stored_info = self.params.archive_stored_info # Stuff is stored according to this order
  # ---------------------------------

  # ---------------------------------
  def __len__(self):
    """
    Returns the length of the archive
    """
    return self.size

  def __iter__(self):
    """
    Allows to directly iterate the pop.
    :return:
    """
    return self.data.__iter__()

  def __next__(self):
    """
    During iteration returns the next element of the iterator
    :return:
    """
    return self.data.__next__()

  def __getitem__(self, item):
    """
    Returns the asked item
    :param item: item to return. Can be an agent or a key
    :return: returns the corresponding agent or the column of the dict
    """
    if type(item) == str:
      try:
        index = self.stored_info.index(item) # Get index of item to get
        return [np.array(x[index]) for x in self.data] # Return list of those items
      except ValueError:
        raise ValueError('Wrong key given. Available: {} - Given: {}'.format(self.stored_info, item))
    else:
      return self.data[item]

  @property
  def size(self):
    """
    Size of the archive
    """
    return len(self.data)
  # ---------------------------------

  # ---------------------------------
  def update(self, data):
    """
    This function updates the archive.
    :param data: dictionary containing the data to update. They keys can be the ones from the archive stored_info
    The values are lists of the same length of the archive
    :return:
    """
    for d in list(data):
      assert d in self.stored_info, print("Can't update. {} not among archive stored data. Available: {}".format(d, self.stored_info))
      assert len(data[d]) == self.size, print("Updated {} size mismatch. Given: {} - Archive size: {}".format(d, len(data[d]), self.size))

    # This is slow, but is done not that often, so it is not so important for it to be fast
    for idx in range(self.size):
      for d in list(data):
        info = self.stored_info.index(d) # Get index of info we want to update
        self.data[idx][info] = data[d][idx]
  # ---------------------------------

  # ---------------------------------
  def store(self, agent):
    """
    Store data in the archive as a list of: (genome, gt_bd, bd, traj).
    No need to store the ID given that we store the genome.
    Saving as a tuple instead of a dict makes the append operation faster
    :param agent: agent to store
    :return:
    """
    # If want to use set
    # self.data.add((self._totuple(agent['genome']), self._totuple(agent['gt_bd']), self._totuple(agent['bd']),
    #                self._totuple(agent['traj'])))
    self.data.append([agent[info] for info in self.stored_info])
  # ---------------------------------

  # ---------------------------------
  def _totuple(self, a):
    """
    This function converts trajectories to tuples to be added to the set
    :param a:
    :return:
    """
    try:
      return tuple(self._totuple(i) for i in a)
    except TypeError:
      return a
  # ---------------------------------

  # ---------------------------------
  def save(self, filepath, filename):
    """
    This function saves the population as a pkl file
    :param filepath:
    :param name: Name of the file
    :return:
    """
    try:
      with open(os.path.join(filepath, 'archive_{}.pkl'.format(filename)), 'wb') as file:
        pkl.dump(self.data, file)
    except Exception as e:
      print('Cannot Save archive {}.'.format(filename))
      print('Exception {}'.format(e))
  # ---------------------------------

  # ---------------------------------
  def load(self, filepath):
    """
    This function loads the population
    :param filepath: File from where to load the population
    :return:
    """
    if not os.path.exists(filepath):
      print('File to load not found.')
      return

    if self.params is not None and self.params.verbose:
      print('Loading archive from {}'.format(filepath))
    with open(filepath, 'rb') as file:
      self.data = pkl.load(file)
  # ---------------------------------