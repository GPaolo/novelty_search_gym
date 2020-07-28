# Created by Giuseppe Paolo 
# Date: 27/07/2020

#Here I make the class that creates everything. I pass the parameters as init arguments, this one creates the param class, and the popu, arch, opt alg

import os
from core.population import Population
from core.evolvers import NoveltySearch
from core.behavior_descriptors.behavior_descriptors import BehaviorDescriptor
from core import Evaluator
import multiprocessing as mp
from timeit import default_timer as timer

evaluator = None
main_pool = None # Using pool as global prevents the creation of new environments at every generation


class Searcher(object):
  """
  This class creates the instance of the NS algorithm and everything related
  """
  def __init__(self, parameters):
    self.parameters = parameters
    self.bd_extractor = BehaviorDescriptor(self.parameters)

    self.generation = 0

    if self.parameters.multiprocesses:
      global main_pool
      main_pool = mp.Pool(initializer=self.init_process, processes=self.parameters.multiprocesses)
    else:
      self.evaluator = Evaluator(self.parameters)

    self.evolver = NoveltySearch(self.parameters)
    self.population = Population(self.parameters, init_size=self.parameters.pop_size)
    self.offsprings = None
    self.ns_archive = self.evolver.archive

  def init_process(self):
    """
    This function is used to initialize the pool so each process has its own instance of the evaluator
    :return:
    """
    global evaluator
    evaluator = Evaluator(self.parameters)

  def _feed_eval(self, agent):
    """
    This function feeds the agent to the evaluator and returns the updated agent
    :param agent:
    :return:
    """
    global evaluator
    eval_agent = evaluator(agent, self.bd_extractor.__call__) # Is passing the call better than init the evaluator with the bd inside???
    return eval_agent

  def evaluate_in_env(self, pop, pool=None):
    """
    This function evaluates the population in the environment by passing it to the parallel evaluators.
    :return:
    """
    if self.parameters.verbose: print('Evaluating {} in environment.'.format(pop.name))
    if self.parameters.multiprocesses:
      pop.pop = pool.map(self._feed_eval, pop.pop) # As long as the ID is fine, the order of the element in the list does not matter
    else:
      for i in range(pop.size):
        if self.parameters.verbose: print(".", end = '') # The end prevents the newline
        pop[i] = self.evaluator(pop[i], self.bd_extractor)
      if self.parameters.verbose: print()

  def generational_step(self):
    """
    This function performs all the calculations needed for one generation.
    Generates offsprings, evaluates them and the parents in the environment, calculates the performance metrics,
    updates archive and population and finally saves offsprings, population and archive.
    :return: time taken for running the generation
    """
    global main_pool
    start_time = timer()

    # NB if you pass a pool here the offsprings are not well generated. This is probably due to the fact that
    # the sampling for the mutation are done in parallel so multiple offsprings will have same mutations.
    self.offsprings = self.evolver.generate_offspring(self.population, pool=None) # Generate offsprings

    # Evaluate population and offsprings in the environment
    self.evaluate_in_env(self.population, pool=main_pool)
    self.evaluate_in_env(self.offsprings, pool=main_pool)

    # Do evolution stuff #TODO maybe can wrap these 3 functions into a single ea.step function
    self.evolver.evaluate_performances(self.population, self.offsprings, pool=main_pool)  # Calculate novelty/fitness/curiosity etc
    self.evolver.update_archive(self.offsprings)
    self.evolver.update_population(self.population, self.offsprings)

    self.generation += 1

    # Save pop, archive and off
    self.population.save(self.parameters.save_path, 'gen_{}'.format(self.generation))
    self.evolver.archive.save(self.parameters.save_path, 'gen_{}'.format(self.generation))
    self.offsprings.save(self.parameters.save_path, 'gen_{}'.format(self.generation))
    return timer() - start_time

  def load_generation(self, generation, path):
    """
    This function loads the population, the offsprings and the archive at a given generation, so it can restart the
    search from there.
    :param generation:
    :param path: experiment path
    :return:
    """
    self.generation = generation

    self.population.load(os.path.join(path, 'population_gen_{}.pkl'.format(self.generation)))
    self.offsprings.load(os.path.join(path, 'offsprings_gen_{}.pkl'.format(self.generation)))
    self.evolver.archive.load(os.path.join(path, 'archive_gen_{}.pkl'.format(self.generation)))

  def close(self):
    """
    This function closes the pool and deletes everything.
    :return:
    """
    if self.parameters.multiprocesses:
      global main_pool
      main_pool.close()
      main_pool.join()
