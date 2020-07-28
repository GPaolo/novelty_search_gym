# Created by Giuseppe Paolo 
# Date: 09/03/2020

import numpy as np
from core.population import Population
from core.population import Archive

class BaseEvolver(object):
  """
  This class implements a base evolver that generates a population of offsprings given a population of parents
  """
  def __init__(self, parameters):
    """
    Constructor
    """
    self.params = parameters
    self.sigma = self.params.mutation_parameters['sigma']
    self.mu = self.params.mutation_parameters['mu']
    self.mutation_operator = np.random.normal
    self.archive = Archive(self.params)
    self.update_criteria = None # ['fitness', 'novelty', 'surprise']
    self.agent_template = self.params.agent_template

  def mutate_genome(self, genome):
    """
    This function mutates the genome by using the mutation operator.
    NB: The genome is clipped in the range [-1, 1]
    :param genome:
    :return:
    """
    genome = np.clip(genome + self.mutation_operator(self.mu, self.sigma, np.shape(genome)),
                     self.params.genome_limit[0], self.params.genome_limit[1])
    return genome

  def _generate_off(self, parent_id_gen):
    """
    This function generates the offsprings from a given parent
    :param parent_id_gen: A tuple containing (parent_id, parent_genome)
    :return: list of offsprings
    """
    offsprings = []
    for k in range(self.params.offsprings_per_parent):
      off = self.agent_template.copy() # Get new agent
      off['genome'] = self.mutate_genome(parent_id_gen[1]) # Add parent mutate genome
      off['parent'] = parent_id_gen[0] # Add parent ID
      offsprings.append(off)
    return offsprings

  def generate_offspring(self, parents, pool=None):
    """
    This function generates the offspring from the population
    :return: Population of offsprings
    """
    offsprings = Population(self.params, init_size=0, name='offsprings')

    parent_genome = parents['genome']
    parent_ids = parents['id']

    if pool is not None:
      offs = pool.map(self._generate_off, zip(parent_ids, parent_genome))
    else:
      offs = []
      for id_gen in zip(parent_ids, parent_genome): # Generate offsprings from each parent
        offs.append(self._generate_off(id_gen))

    offsprings.pop = [off for p_off in offs for off in p_off]  # Unpack list of lists and add it to offsprings
    offs_ids = parents.agent_id + np.array(range(len(offsprings)))  # Calculate offs IDs
    offsprings['id'] = offs_ids  # Update offs IDs
    parents.agent_id = max(offs_ids) + 1 # This saves the maximum ID reached till now
    return offsprings

  def evaluate_performances(self, population, offsprings, pool=None):
    raise NotImplementedError("This needs to be implemented")

  def update_archive(self, offsprings):
    """
    Updates the archive according to the strategy and the criteria given.
    :param offsprings:
    :return:
    """
    # Get list of ordered indexes according to selection strategy
    if self.params.selection_operator == 'random':
      idx = list(range(offsprings.size))
      np.random.shuffle(idx)
    elif self.params.selection_operator == 'best':
      performances = offsprings[self.update_criteria]
      idx = np.argsort(performances)[::-1]  # Order idx according to performances. (From highest to lowest)
    else:
      raise ValueError(
        'Please specify a valid selection operator for the archive. Given {} - Valid: ["random", "best"]'.format(
          self.params.selection_operator))

    # TODO This part gets slower with time.
    # Add to archive the first lambda offsprings in the idx list
    for i in idx[:self.params._lambda]:
      self.archive.store(offsprings[i])

  def update_population(self, population, offsprings):
    """
    This function updates the population according to the given criteria
    :param population:
    :param offsprings:
    :return:
    """
    performances = population[self.update_criteria] + offsprings[self.update_criteria]
    idx = np.argsort(performances)[::-1]  # Order idx according to performances.
    parents_off = population.pop + offsprings.pop
    # Update population list by going through it and putting an agent from parents+off at its place
    for new_pop_idx, old_pop_idx in zip(range(population.size), idx[:population.size]):
      population.pop[new_pop_idx] = parents_off[old_pop_idx]