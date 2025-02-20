
import datetime
import os
import pprint
import time
import threading
import torch
from types import SimpleNamespace as SN
from os.path import dirname, abspath


from utils.timehelper import time_left, time_str
from utils.logging import Logger
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transform import OneHot



def run(_run, _config, _log):

    # Check args senity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"


    # Setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")


    # Configure tensorboard logger
    unique_token = "{}_{}".format(
        args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    args.unique_token = unique_token

    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)
    

    # Sacred is on by default
    logger.setup_sacred(_run)


    # Run and train
    run_sequential(args=args, logger=logger)


    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")
    
    print("Exiting script")


    # Making sure framework really exits
    os._exit(os.EX_OK)



