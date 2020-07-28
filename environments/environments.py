# Created by Giuseppe Paolo 
# Date: 27/07/2020

from core.controllers import *
from environments.io_formatters import *
from core.behavior_descriptors.trajectory_to_observations import *

try: import gym_dummy
except: print("Gym dummy not installed")

registered_envs = {}

registered_envs['NAME'] = {
  'gym_name': None,
  'controller': None,
  'input_formatter': None,
  'output_formatter': None,
  "traj_to_obs": None,
}


registered_envs['Dummy'] = {
  'gym_name': 'Dummy-v0',
  'controller': {
    'controller': DummyController,
    'input_formatter': dummy_input_formatter,
    'output_formatter': output_formatter,
    'input_size': 2,
    'output_size':2,
    'name': 'dummy',
  },
  'traj_to_obs': dummy_obs,
  'max_steps': 1,
  'grid':{
    'min_coord':[-1,-1],
    'max_coord':[1, 1],
    'bins':50
  }
}

registered_envs['Walker2D'] = {
  'gym_name': 'Walker2D-v0',
  'controller': {
    'controller': FFNeuralController,
    'input_formatter': walker_input_formatter,
    'output_formatter': output_formatter,
    'input_size': 2,
    'output_size': 2,
    'name': 'dummy',
  },
  'traj_to_obs': walker_2D_obs,
  'max_steps': 50,
  'grid':{
    'min_coord':[-1,-1],
    'max_coord':[1, 1],
    'bins':50
  }
}