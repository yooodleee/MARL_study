
from collections import defaultdict
import logging
import numpy as np



class Logger:

    def __init__(self, console_logger):
        self.console_logger = console_logger
        
        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])
    

    