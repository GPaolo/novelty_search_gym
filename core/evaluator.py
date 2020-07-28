# Created by Giuseppe Paolo 
# Date: 27/07/2020

from environments.environments import registered_envs
import gym

class Evaluator(object):
  """
  This class evaluates the controller in the environment
  """
  def __init__(self, params):
    """
    Constructor
    :param params:
    """
    self.params = params
    if self.params.env_name not in registered_envs:
      raise NameError("Unknown environment. Given {} - Available {}".format(self.params.env_name, list(registered_envs.keys())))
    self.env = registered_envs[self.params.env_name]
    print("Instantiating environment: {}".format(self.env['gym_name']))
    self.gym_env = gym.make(self.env['gym_name'])

    if self.params.seed is not None:
      self.gym_env.seed(self.params.seed)
      self.gym_env.action_space.seed(self.params.seed)
      self.gym_env.observation_space.seed(self.params.seed)

    self.controller = self.env['controller']['controller'](input_size=self.env['controller']['input_size'],
                                                           output_size=self.env['controller']['output_size'],
                                                           name=self.env['controller']['name'])
    self.max_steps = self.env['max_steps']

  def evaluate(self, agent):
    """
    This function evaluates the agent in the env
    :param agent:
    :param data_collection: If flag is given we collect the data to train the AE
    :return:
    """
    self.controller.load_genome(agent['genome'])
    traj = []
    done = False
    cumulated_reward = 0

    if self.params.seed is not None:
      self.gym_env.seed(self.params.seed)
      self.gym_env.action_space.seed(self.params.seed)
      self.gym_env.observation_space.seed(self.params.seed)
    obs = self.gym_env.reset()
    traj.append((obs, 0, done, {}))
    t = 0

    while not done:
      agent_input = self.env['controller']['input_formatter'](t/self.max_steps, obs)
      action = self.env['controller']['output_formatter'](self.controller(agent_input))

      obs, reward, done, info = self.gym_env.step(action)
      cumulated_reward += reward

      if t >= self.max_steps:
        done = True
      traj.append((obs, reward, done, info))
      t += 1
    return cumulated_reward, traj

  def __call__(self, agent, bd_extractor=None):
    """
    This function evaluates the agent in the environment
    :param agent:
    :return:
    """
    cumulated_rew, traj = self.evaluate(agent)
    agent['reward'] = cumulated_rew
    bd, surprise = bd_extractor(traj, agent)
    agent['bd'] = bd
    agent['surprise'] = surprise
    return agent
