# Created by Giuseppe Paolo 
# Date: 28/07/2020

# -----------------------------------------------
# INPUT FORMATTERS
# -----------------------------------------------
def dummy_input_formatter(t, obs):
  """
  This function formats the data to give as input to the controller
  :param t:
  :param obs:
  :return: None
  """
  return None

def walker_input_formatter(t, obs):
  """
  This function formats the data to give as input to the controller
  :param t:
  :param obs:
  :return: None
  """
  return obs

# -----------------------------------------------
# OUTPUT FORMATTERS
# -----------------------------------------------
def output_formatter(action):
  """
  This function formats the output of the controller to extract the action for the env
  :param action:
  :return:
  """
  return action