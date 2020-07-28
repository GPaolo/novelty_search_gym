# Created by Giuseppe Paolo 
# Date: 28/07/2020

import sys, os
from parameters import Params
import numpy as np
import traceback
from progress.bar import Bar
import argparse
import multiprocessing as mp
from parameters import params
from core.searcher import Searcher

import datetime
from environments.environments import registered_envs


if __name__ == "__main__":
  # To check why these options are here:
  # 1. https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
  # 2. https://pytorch.org/docs/stable/multiprocessing.html#sharing-strategies
  mp.set_start_method('spawn')
#  mp.set_sharing_strategy('file_system') # Fundamental otherwise crashes complaining that too many files are open

  parser = argparse.ArgumentParser('Run evolutionary script')
  parser.add_argument('-env', '--environment', help='Environment to use', choices=list(registered_envs.keys()))
  parser.add_argument('-exp', '--experiment', help='Experiment type. Defines the behavior descriptor', choices=['NS'])
  parser.add_argument('-sp', '--save_path', help='Path where to save the experiment')
  parser.add_argument('-mp', '--multiprocesses', help='How many parallel workers need to use', type=int)
  parser.add_argument('-p', '--pop_size', help='Size of the population', type=int)
  parser.add_argument('-g', '--generations', help='Number of generations', type=int)
  parser.add_argument('-v', '--verbose', help='Verbose', action='store_true')
  parser.add_argument('--restart_gen', help='Generation at which to restart. It will load from the savepath', type=int)

  args = parser.parse_args()

  if args.environment is not None: params.env_name = args.environment
  if args.experiment is not None: params.exp_type = args.experiment
  if args.save_path is not None: params.save_path = os.path.join(args.save_path, params.save_dir)
  if args.multiprocesses is not None: params.multiprocesses = args.multiprocesses
  if args.pop_size is not None: params.pop_size = args.pop_size
  if args.generations is not None: params.generations = args.generations
  if args.verbose is True: params.verbose = args.verbose

  print("SAVE PATH: {}".format(params.save_path))
  params.save()

  if params.seed is not None:
    np.random.seed(params.seed)

  bar = Bar('Generation:', max=params.generations, suffix='[%(index)d/%(max)d] - Avg time per gen: %(avg).3fs - Elapsed: %(elapsed_td)s')

  searcher = Searcher(params)

  if args.restart_gen is not None:
    print("Restarting:")
    print("\t Restarting from generation {}".format(args.restart_gen))
    print("\t Loading from: {}".format(args.save_path))
    searcher.load_generation(args.restart_gen, args.save_path)
    print("\t Loading done.")

  gen_times = []
  for k in range(params.generations):
    if params.verbose: print("Generation: {}".format(k))
    bar.next()
    try:
      gen_times.append(searcher.generational_step())
    except KeyboardInterrupt:
      print('User interruption. Saving.')
      searcher.population.save(params.save_path, 'gen_{}'.format(searcher.generation))
      searcher.offsprings.save(params.save_path, 'gen_{}'.format(searcher.generation))
      searcher.evolver.archive.save(params.save_path, 'gen_{}'.format(searcher.generation))
      searcher.close()
      bar.finish()
      break
    except Exception as e:
      print('Exception occurred.')
      print(traceback.print_exc())
      searcher.close()
      bar.finish()
      sys.exit()
  print('Done.')
  total_time = np.sum(gen_times)
  print("Total time: {}".format(str(datetime.timedelta(seconds=total_time))))



