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
    
    def set_bounds(
            self,
            left,
            right,
            bottom,
            top):
        
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation = (-left * scalex, -bottom * scaley),
            scale = (scalex, scaley),
        )
    
    def add_geom(self, geom):
        self.geoms.append(geom)
    
    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)
    
    def render(
            self,
            return_rgb_array=False):
        
        glClearColor(1, 1, 1, 1)

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()

        for geom in self.geoms:
            geom.render()
        
        for geom in self.onetime_geoms:
            geom.render()
        
        self.transform.disable()
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.fromstring(
                image_data.data, dtype=np.uint8, sep=''
            )
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        
        self.window.flip()
        self.onetime_geoms = []
        return arr
    
    def draw_circle(
            self,
            radius=10,
            res=30,
            filled=True,
            **attrs):
        """
        Convenience.
        """
        geom = make_circle(radius=radius, res=res, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)

        return geom
    
    def draw_polygon(
            self,
            v,
            filled=True,
            **attrs):
        
        geom = make_polygon(v=v, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)

        return geom
    
    def draw_polyline(
            self,
            v,
            **attrs):
        
        geom = make_polyline(v=v)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)

        return geom
    
    def draw_line(
            self,
            start,
            end,
            **attrs):
        
        geom = Line(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)

        return geom
    
    def get_array(self):
        self.window.flip()
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        self.window.flip()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        arr = arr.reshape(self.height, self.width, 4)

        return arr[::-1, :, 0:3]


def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])


class Geom(object):

    def __init__(self):
        self._color = Color((0, 0, 0, 1.0))
        self.attrs = [self._color]
    
    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        
        self.render1()
        for attr in self.attrs:
            attr.disable()
    
    def render1(self):
        raise NotImplementedError
    
    def add_attr(self, attr):
        self.attrs.append(attr)
    
    def set_color(
            self,
            r,
            g,
            b,
            alpha=1):

        self._color.vec4 = (r, g, b, alpha)


