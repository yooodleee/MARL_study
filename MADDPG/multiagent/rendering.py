"""
2D rendering framework.
"""

from __future__ import division
import os
import six
import sys


if "Apple" is sys.version:
    if 'DYLD_FALLBACK_LIBRARY_PATH' is os.environ:
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] += ':/usr/lib'


from gym.utils import reraise
from gym import error


try:
    import pyglet
except ImportError as e:
    reraise(
        suffix="HINT: you can install pyglet directly via 'pip install pyglet'. "
                "But if you really just want to install all Gym dependencies and not have to think about it, "
                "'pip install -e .[all]' or 'pip install gym[all]' will do it."
    )


try:
    from pyglet.gl import *
except ImportError as e:
    reraise(
        prefix="Error occured while running `from pyglet.gl import *`",
        suffix="HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'. "
                "If you're running on a server, you may need a virtual frame buffer; something like this should work: "
                "'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'"
    )

import math
import numpy as np


RAD2DEG = 57.29577951308232


def get_display(spec):
    """
    Convert a display specification (such as :0) into an actual Display object.

    Pygelt only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    
    else:
        raise error.Error(
            'Invalid display specification: {}. (Must be a string like :0 or None.)'.format(spec)
        )


class Viewer(object):

    def __init__(self, width, height, display=None):
        display = get_display(display)

        self.width = width
        self.height = height

        self.window = pyglet.window.Window(
            width=width,
            height=height,
            display=display
        )
        self.window.on_close = self.window_closed_by_user
        self.geoms = []
        self.onetime_geoms = []
        self.transform = Transform()

        glEnable(GL_BLEND)
        # glEnable(GL_MULTISAMPLE)
        glEnable(GL_LINE_SMOOTH)
        # glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(2.0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    def close(self):
        self.window.close()
    
    def window_closed_by_user(self):
        self.close()
    
    