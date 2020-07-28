# Created by Giuseppe Paolo 
# Date: 06/03/2020

import numpy as np
from scipy.spatial.distance import squareform, cdist
import matplotlib.pyplot as plt
plt.style.use('seaborn')



def novelty(distances, neighs):
  """
  Calculates the novelty for agent i from the distances given
  :param distances: i row of distance matrix
  :param neighs: number of neighbors used to calculate novelty
  :return: (i, mean_k_dist)
  """
  idx = np.argsort(distances) # Get list of idx from closest to farthest
  mean_k_dist = np.mean(distances[idx[1:neighs + 1]])  # the 1:+1 is necessary cause the position 0 is occupied by the index of the considered element
  return mean_k_dist

def calculate_novelties(bd_set, reference_set, distance_metric='euclidean', novelty_neighs=15, pool=None):
  """
  This function calculates the novelty for each element in the BD set wrt the Reference set
  :param bd_set:
  :param reference_set:
  :param distance_metric: Distance metric with which the novelty is calculated. Default: euclidean
  :param novelty_neighs: Number of neighbors used for novelty calculation. Default: 15
  :param pool: Pool for multiprocessing
  :return:
  """
  distance_matrix = calculate_distances(bd_set, reference_set, distance_metric=distance_metric)
  if pool is not None:
    novelties = [pool.apply(novelty, args=(distance, novelty_neighs,)) for distance in distance_matrix]
  else:
    novelties = [novelty(distance, novelty_neighs) for distance in distance_matrix]
  return novelties

def calculate_distances(bd_set, reference_set, distance_metric='euclidean'):
  """
  This function is used to calculate the distances between the sets
  :param bd_set:
  :param reference_set:
  :param distance_metric: Distance metric to use. Default: euclidean
  :return:
  """
  if distance_metric == 'euclidean':
    # TODO this operation might become slower when the archive grows. Might have to parallelize as well by doing it myself
    distance_matrix = cdist(bd_set, reference_set, metric='euclidean')
  elif distance_metric == 'mahalanobis':
    distance_matrix = cdist(bd_set, reference_set, metric='mahalanobis')
  elif distance_metric == 'manhattan':
    distance_matrix = cdist(bd_set, reference_set, metric='cityblock')
  elif distance_metric == 'mink_0.1':
    distance_matrix = cdist(bd_set, reference_set, metric='minkowski', p=0.1)
  elif distance_metric == 'mink_1':
    distance_matrix = cdist(bd_set, reference_set, metric='minkowski', p=1)
  elif distance_metric == 'mink_0.01':
    distance_matrix = cdist(bd_set, reference_set, metric='minkowski', p=.01)
  else:
    raise ValueError('Specified distance {} not available.'.format(distance_metric))
  return distance_matrix

def plot_pareto_fronts(points):
  import matplotlib.colors as colors
  cmap = plt.get_cmap('jet')
  plt.figure()
  color = cmap(np.linspace(0, 1, len(points)))

  for idx, front in enumerate(points):
    pp = np.array(points[front])
    plt.scatter(pp[:, 0], pp[:, 1], cmap=cmap, label=front)
    # plt.plot(pp[:, 0], pp[:, 1], '-o', c=color[idx], label=front)
  plt.legend()
  plt.show()


