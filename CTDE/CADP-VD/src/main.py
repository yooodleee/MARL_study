
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



