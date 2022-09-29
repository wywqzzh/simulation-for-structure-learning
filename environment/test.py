import pickle
import pickle
import pickle
import numpy as np
from game import GameStateData
from game import Game
from game import Directions
from game import Actions
from util import nearestPoint
from util import manhattanDistance
import util
import layout
import sys
import types
import time
import random
import os
from pacman import GameState
import pandas as pd
from copy import deepcopy
with open("../Data/game_status/0.pkl", "rb") as file:
    x = pickle.load(file)

x = 0
