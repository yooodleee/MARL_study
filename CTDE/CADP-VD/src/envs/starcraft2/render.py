
import numpy as np
import re
import subprocess
import platform
from absl import logging
import math
import collections
import os
import pygame
import queue


from pysc2.lib import colors
from pysc2.lib import point
from pysc2.lib.render_human import _Surface
from pysc2.lib import transform
from pysc2.lib import features


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def _get_desktop_size():
    """Get the desktop size."""

    if platform.system() == "Linux":
        try:
            xrandr_query = subprocess.check_output(["xrandr", "--query"])
            sizes = re.findall(
                r"\bconnected primiary (\d+)x(\d+)", str(xrandr_query)
            )
            
            if sizes[0]:
                return point.Point(int(sizes[0][0]), int(sizes[0][1]))
        
        except ValueError:
            logging.error("Failed to get the resolution from xrandr.")
    
    # Most general, but doesn't understand multiple monitors
    display_info = pygame.display.Info()

    return point.Point(display_info.current_w, display_info.current_h)



