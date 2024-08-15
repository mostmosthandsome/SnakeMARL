from sacred import Experiment, SETTINGS
from pymarl.src.utils.logging import get_logger
import os
from copy import deepcopy
from os.path import dirname, abspath
import numpy as np
import torch as th
from pymarl.src.run import run
from pymarl.src.main import config_copy,_get_config,recursive_dict_update
import sys
import yaml
from sacred.observers import FileStorageObserver
from env.WrappedSnake import SnakeEnv

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()
ex = Experiment("pymarl")
ex.logger = logger

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")

@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    env = SnakeEnv(config)
    run(_run, config, _log, env)

if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    config_path = os.path.join(dirname(__file__), "config")
    with open(os.path.join(config_path, "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    alg_config = _get_config(os.path.join(config_path,"algs", "{}.yaml".format(config_dict["alg_type"])))
    env_config = _get_config(os.path.join(config_path,"env/config.json"))
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    ex.run_commandline(params)