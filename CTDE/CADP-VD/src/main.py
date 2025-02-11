
import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy


from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_lifefeeds


import sys
import torch
from utils.logging import get_logger
import yaml


from run import run


SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()


ex = Experiment("pymarl")
ex.logger = logger
ex.capture_out_filter = apply_backspaces_and_lifefeeds


results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):

    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, _config, _log)



def _get_config(params, arg_name, subfolder):
    config_name = None
    
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break
    
    if config_name is not None:
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "config",
                subfolder,
                "{}.yaml",format(config_name)
            ), "r"
        ) as f:
            try:
                config_dict = yaml.load(f)
            
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        
        return config_dict


