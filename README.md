# Novelty Search Gym

A (fairly modular and easily expandable) novelty search implementation for gym-based environments

This repo comes with a very simple example environment in the `environments/assets` folder.

---
To install run:
```
pipenv shell --three
python setup.py install
```
---
This code can be used in two ways:
* By taking advantage of the `run_experiment.py` script in the `script` folder
* By importing and using the `Searcher` class in the `core/searcher.py` file. This one can be easily used in the following way:

```python
from parameters import params
from core.searcher import Searcher

searcher = Searcher(params)
for k in range(params.generations):
    searcher.generational_step()
searcher.close()
```

---

## Performing an experiment
To run the algorithm you just need to launch:
```bash
ipython script/run_experiment.py
```

If you want to change the experiment parameters, go to: `parameters.py`

You can also specify some of the parameters on the command line. If you want to check which ones, launch:
```bash
ipython script/run_experiment.py -- -h
```

For each generation the script saves: `population`, `offsprings`, `archive` as pkl files in 
folders whose name is formatted as: 
`<env_name>_<exp_type>/<year>_<month>_<day>_<hour>:<minute>_<random_seed_used>`. 
The time corresponds to when the experiment has been launched.

In this folder you find the parameters, in a file called: `_params.json`,
the archive for each generation, in files called: `archive_gen_<generation>.pkl`,
the population for each generation, in files called: `population_gen_<generation>.pkl`,
and the offsprings for each generation, in files called: `offsprings_gen_<generation>.pkl`.

## Evaluating the archive
Once the experiment is finished, if you want to study the behavior descriptors of the
agents in the archive you have to evaluate the archive first by running it in the
environment and saving the trajectories of images and observations.

You can do that by launching:
```bash
ipython scripts/evaluate_archive.py -- -e EXPERIMENT_PATH
```

This script will evaluate the archive of the given generation (The default is the 500).

As for the `run_experiment.py` script, you can change the parameters or provide some on the command line.

Once the evaluation is done, the trajectories will be saved in the experiment folder, inside a folder called `analyzed_data` as:
* `archive_info_gen_<generation>.pkl`: these are the infos given by the gym environment every time a step is performed;
* `archive_obs_gen_<generation>.pkl`: these are the observations given by the gym environment every time a step is performed.
* `cvg_gen_<generation>.pkl`: the coverage of the ground truth behavior description space obtained by the archive.
* `unif_gen_<generation>.pkl`: the uniformity of the archive in the behavior descriptor space.
* `gt_bd_gen_<generation>.pkl`: the behavior descriptor's points for every agent in the archive.

### Plotting the results
Finally you can plot your results by using the jupyter notebook `archive_analysis` located in the `analysis` folder.

# Extending it
As said, this code is fairly modular and can be extended easily, both by adding new kind of experiments/metrics
or by adding new gym environments.

### Adding environments
If you want to add an environment you have to do:
1. Add the gym environment in the `environments/assets` folder
2. Register the environment in the `environments/environments.py` file as an entry in the `registered_envs` dictionary
3. Add an input formatter (and in case also an output formatter) in the `environments/io_formatters.py` files.
These formatters are used to interface the environment with the controllers. 
    * The input formatters prepares the observation to be fed to the controller. 
    * The output formatters takes the controller output and formats it as an action for the environment.
4. Add the ground truth behavior descriptor in the `analysis/gt_behavior_descriptors.py` and in the `get_metrics` function in `analysis/evaluate_archive.py`.
5. Add the observations extraction function from the trajectory in the `core/behavior_descriptors/trajectory_to_observations.py`.

### Adding experiments
If you want to add an experiment type you have to:
1. The behavior descriptor in the `core/behavior_descriptors/behavior_descriptor.py`. You have to both add the 
option in the `__init__` of the class and the actual descriptor function as a member function of the class.
2. If you want to define an evolution algorithm you can do it in `core/evolvers`
4. Add the name you chose for the experiment in the list of possible choices in the parser in `scripts/run_experiment.py`

---

If you find this code useful and you use it in a project, please cite it:
```
@misc{NSPaolo2020,
  author = {Paolo, Giuseppe},
  title = {Novelty Search Gym},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/GPaolo/novelty_search_gym}},
}
```
