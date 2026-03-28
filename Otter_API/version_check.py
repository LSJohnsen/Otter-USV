import ast
import sys
import importlib
import importlib.metadata

IMPORT_BLOCK = r"""
import Otter_api
from DRL_control import Otter_simulator_DRL
from lib.plotTimeSeries import *
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import pandas as pd
from gymnasium.spaces import Box
import torch
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor 
from stable_baselines3.common.vec_env import VecNormalize
from collections import deque
from torch import nn
from lib.Performance_metrics import PerformanceMetrics
from logs.IO import log_to_csv, log_params as io_log_params
import csv
import time
from DRL_control.reward_callback_plot import append_reward_training_progress


import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OTTER_API_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if OTTER_API_DIR not in sys.path:
    sys.path.insert(0, OTTER_API_DIR)



import numpy as np
import math
from lib.gnc import Smtrx, Hmtrx, Rzyx, m2c, crossFlowDrag, sat, attitudeEuler
import pandas as pd
import pathlib
from numba import jit, cuda
from pathlib import Path
from lib.Performance_metrics import PerformanceMetrics
from logs.IO import log_params


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import time


import lib.Control as Control
import lib.Connector as Connector
import time
import pymap3d as pm
import math


import time
import socket
import select
import numpy as np
from numpy import pi
import pymap3d as pm
from copy import copy


import numpy as np
from numpy import round, pi
import math
from scipy.interpolate import CubicSpline
import pandas as pd
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import os
import math

import numpy as np
import time
import math
import datetime
import pandas as pd
import os
import requests
import threading
import pymap3d as pm

import numpy as np
import math
from lib.gnc import Smtrx, Hmtrx, Rzyx, m2c, crossFlowDrag, sat, attitudeEuler
import pandas as pd
from numba import jit, cuda
from pathlib import Path

import casadi as ca
from casadi_control.lib.casadi_utils import B2N, CRB6sx, CA3sx, crossFlowDrag3
from casadi_control.lib.usv_params import usv_params_6dof
"""

def get_top_level_imports(code: str):
    tree = ast.parse(code)
    modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split(".")[0])

    return sorted(modules)

def get_version(module_name: str):
    try:
        pkg_map = importlib.metadata.packages_distributions()
        if module_name in pkg_map and pkg_map[module_name]:
            dist_name = pkg_map[module_name][0]
            return dist_name, importlib.metadata.version(dist_name)
    except Exception:
        pass

    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", None)
        if version is not None:
            return module_name, str(version)
        return module_name, "unknown"
    except Exception:
        return module_name, "not found"

def main():
    modules = get_top_level_imports(IMPORT_BLOCK)

    print(f"{'Module':<22}{'Distribution':<28}{'Version'}")
    print("-" * 70)

    for mod in modules:
        dist_name, version = get_version(mod)
        print(f"{mod:<22}{dist_name:<28}{version}")

    print("\n" + f"{'Python':<50}{sys.version.split()[0]}")

if __name__ == "__main__":
    main()