
import numpy as np
import re
import subprocess
import platform
from absl import logging
import math
import time
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



class StarCraft2Renderer:

    def __init__(self, env, mode):
        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

        self.env = env
        self.mode = mode
        self.obs = None
        self._window_scale = 0.75
        self.game_info = game_info = self.env._controller.game_info()
        self.static_data = self.env._controller.data()

        self._obs_queue = queue.Queue()
        self._game_times = collections.deque(maxlen=100)    # Avg FPS over 100 frames.
        self._render_times = collections.deque(maxlen=100)
        self._last_time = time.time()
        self._ast_game_loop = 0
        self._name_lengths = {}

        self._map_size = point.Point.build(game_info.start_raw.map_size)
        self._playable = point.Rect(
            point.Point.build(game_info.start_raw.playable_area.p0),
            point.Point.build(game_info.start_raw.playable_area.p1),
        )

        window_size_px = point.Point(self.env.window_size[0], self.env.window_size[1])
        window_size_px = self._map_size.scale_max_size(window_size_px * self._window_scale).ceil()
        self._scale = window_size_px.y // 32

        self.display = pygame.Surface(window_size_px)

        if mode == "human":
            self.display = pygame.display.set_mode(window_size_px, 0, 32)
            pygame.display.init()
            pygame.display.set_caption("Starcraft Viewer")
        
        pygame.font.init()
        self._world_to_world_t1 = transform.Linear(point.Point(1, -1), point.Point(0, self._map_size.y))
        self._world_t1_to_screen = transform.Linear(scale=window_size_px / 32)
        self.screen_transform = transform.Chain(self._world_to_world_t1, self._world_t1_to_screen)

        surf_loc = point.Rect(point.origin, window_size_px)
        sub_surf = self.display.subsurface(pygame.Rect(surf_loc.t1, surf_loc.size))
        
        self._surf = _Surface(
            sub_surf,
            None,
            surf_loc,
            self.screen_transform,
            None,
            self.draw_screen,
        )

        self._font_small = pygame.font.Font(None, int(self._scale * 0.5))
        self._font_large = pygame.font.Font(None, self._scale)

    
    