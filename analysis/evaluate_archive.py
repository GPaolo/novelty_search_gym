# Created by Giuseppe Paolo 
# Date: 28/07/2020

import os
import parameters
import multiprocessing as mp
from core import Evaluator
from analysis import utils
from environments.environments import registered_envs
import pickle as pkl
from progress.bar import Bar
import argparse
import gc
from core.behavior_descriptors.behavior_descriptors import BehaviorDescriptor
from analysis.gt_bd import *

evaluator = None
main_pool = None

class EvalArchive(object):
  """
  This function is used to evaluate the archive of an experiment
  """
  def __init__(self, exp_path, multip=False, agents=None):
    """
    Constructor
    """
    self.params = parameters.Params()
    self.params.load(os.path.join(exp_path, '_params.json'))
    self.exp_path = exp_path
    self.agents = agents
    self.bd_extractor = BehaviorDescriptor(self.params)

    self.traj_to_obs = registered_envs[self.params.env_name]['traj_to_obs']
    self.mp = multip
    if self.mp:
      global main_pool
      main_pool = mp.Pool(initializer=self.init_process, processes=int(os.cpu_count()))
    else:
      self.evaluator = Evaluator(self.params)

    if not os.path.exists(os.path.join(self.exp_path, 'analyzed_data')):
      os.mkdir(os.path.join(self.exp_path, 'analyzed_data'))

  def init_process(self):
    """
    This function is used to initialize the pool so each process has its own instance of the evaluator
    :return:
    """
    global evaluator
    evaluator = Evaluator(self.params)

  def _get_eval_traj(self, genome):
    """
    This function feeds the archive genome to the evaluator to get the traj of observations
    :param genome:
    :return:
    """
    global evaluator
    _, data_traj = evaluator.evaluate({'genome':genome})
    obs_traj = self.traj_to_obs(data_traj)
    infos = []
    for idx, t in enumerate(data_traj):
      infos.append(t[3])

    return (obs_traj, infos)

  def load_eval_archive(self, generation=None):
    """
    This function loads and evaluates the archives in the exp folder
    :param generation: Generation to evaluate. If None, evaluates the archive for all the generations
    :return: Trajectory of observations
    """
    genomes = utils.load_arch_data(self.exp_path, info=['genome'], generation=generation, params=self.params)
    if self.agents is not None:
      for generation in genomes:
        idx = list(range(len(genomes[generation]['genome'])))
        np.random.shuffle(idx)
        genomes[generation]['genome'] = [genomes[generation]['genome'][i] for i in idx[:self.agents]]

    gen_obs_traj = {}
    gen_info_traj = {}
    if generation is None:
      bar = Bar('Generations:', max=len(genomes), suffix='[%(index)d/%(max)d] - Avg time per epoch: %(avg).3fs - Elapsed: %(elapsed_td)s')
    else:
      bar = None

    global main_pool
    print("Starting evaluation...")

    for gen in genomes:
      if bar is not None:
        bar.next()
      if self.mp:
        trajs = main_pool.map(self._get_eval_traj, genomes[gen]['genome'])
        obs_trajs = [t[0] for t in trajs]
        info_trajs = [t[1] for t in trajs]
      else:
        obs_trajs = []
        info_trajs = []
        for genome in genomes[gen]['genome']:
          _, data_traj = self.evaluator.evaluate({'genome': genome})  # , action_coupled=self.action_coupled)

          obs_trajs.append(self.traj_to_obs(data_traj))
          info_trajs.append([t[3] for t in data_traj])

      gen_obs_traj[gen] = obs_trajs
      gen_info_traj[gen] = info_trajs
    print("Done")
    return gen_obs_traj, gen_info_traj

  def get_metrics(self, observations, infos, generation=None):
    """
    This function calculates the metrics used for plotting.
    :param trajectories: of all the agents in the run
    :param dist: Either Frechet or TW. Default Frechet
    :param generation: Generation from which the trajectories are from
    :return:
    """
    ts_bins = np.linspace(0, 1, num=100, endpoint=True)
    max_len = registered_envs[self.params.env_name]['max_steps']
    grid_parameters = registered_envs[self.params.env_name]['grid']

    # GT_BD is the gorund truth bd that is used to calculate the CVG
    if self.params.env_name == 'Dummy':
      self.gt_bd_extractor = dummy_gt_bd
    elif self.params.env_name == 'Walker2D':
      self.gt_bd_extractor = dummy_gt_bd

    descriptors = np.array([self.gt_bd_extractor(obs, info, max_len) for obs, info in zip(observations[generation], infos[generation])])

    if generation is None:
      name_gt_bd = 'gt_bd_all_gens.pkl'
    else:
      name_gt_bd = 'gt_bd_gen_{}.pkl'.format(generation)
    with open(os.path.join(self.exp_path, 'analyzed_data', name_gt_bd), 'wb') as f:
      pkl.dump(descriptors, f)

    print('Calculating CVG and UNIF')
    hist, grid = utils.get_grid(descriptors, grid_parameters)
    cvg = utils.calculate_coverage(grid)
    unif = utils.calculate_uniformity(hist)
    if generation is None:
      name_cvg = "cvg_all_gens.pkl"
      name_unif = "unif_all_gens.pkl"
    else:
      name_cvg = "cvg_gen_{}.pkl".format(generation)
      name_unif = "unif_gen_{}.pkl".format(generation)
    with open(os.path.join(self.exp_path, 'analyzed_data', name_cvg), 'wb') as f:
      pkl.dump(np.array(cvg), f)
    with open(os.path.join(self.exp_path, 'analyzed_data', name_unif), 'wb') as f:
      pkl.dump(np.array(unif), f)

    print("Done.")

  def save_trajectories(self, data, data_type='traj', generation=None):
    """
    Saves the trajectories in a single file
    :param data: trajectory dict
    :param data_type: name of type of data being saved
    :param generation: generation for which the trajs have been evaluated
    :return:
    """
    if generation is None:
      name = "archive_{}_all_gens.pkl".format(data_type)
    else:
      name = "archive_{}_gen_{}.pkl".format(data_type, generation)
    with open(os.path.join(self.exp_path, 'analyzed_data', name), 'wb') as f:
      pkl.dump(data, f)

if __name__ == "__main__":
  parser = argparse.ArgumentParser('Run archive eval script')
  parser.add_argument('-p', '--path', help='Path of experiment')
  parser.add_argument('-mp', '--multiprocessing', help='Multiprocessing', action='store_true')
  parser.add_argument('-g', '--generation', help='Generation for which to evaluate the archive', type=int, default=None)
  parser.add_argument('-a', '--agents', help='Agents to evaluate', type=int, default=None)
  parser.add_argument('--multi', help='Flag to give in case multiple runs have to be evaluated', action='store_true')

  args = parser.parse_args(["-p", "/home/giuseppe/src/cmans/experiment_data/Walker2D/Walker2D_NS/", '-g' '500', '-mp', '--multi'])

  if not args.multi:
    paths = [args.path]
  else:
    raw_paths = [x for x in os.walk(args.path)]
    raw_paths = raw_paths[0]
    paths = [os.path.join(raw_paths[0], p) for p in raw_paths[1]]

  for path in paths:
    print('Working on: {}'.format(path))
    arch_eval = EvalArchive(path, multip=args.multiprocessing, agents=args.agents)

    obs_traj, infos_traj = arch_eval.load_eval_archive(args.generation)
    gc.collect()
    arch_eval.save_trajectories(obs_traj, data_type='obs', generation=args.generation)
    arch_eval.save_trajectories(infos_traj, data_type='info', generation=args.generation)

    arch_eval.get_metrics(obs_traj, infos_traj, generation=args.generation)
    del arch_eval
    gc.collect()