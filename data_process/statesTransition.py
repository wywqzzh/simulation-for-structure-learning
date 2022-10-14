import pickle
import pickle
import pickle
import numpy as np
import sys

sys.path.append("../environment")
from FeatureExtractor.ExtractGameFeatures import featureExtractor
from game import GameStateData
from game import Game
from game import Directions
from game import Actions
from util import nearestPoint
from util import manhattanDistance
import util
import layout

import types
import time
import random
import os
from environment.pacman import GameState
import pandas as pd
from copy import deepcopy


def change_pos(pos, numRow):
    temp_pos = (pos[0] + 1, numRow - pos[1])
    return temp_pos


def state_to_feature(state):
    """
    将游戏state转变为 game_status和feature
    :param state: 
    :return: 
    """
    game_status = {"PacmanPos": [], "ghost_data": [], "ghost_status": [],
                   "bean_data": [], "energizer_data": [], "Reward": []}

    dir_dict = {"Stop": None, "North": "up", "South": "down", "West": "left", "East": "right"}

    # get game_status
    numRow = len(state.data.layout.layoutText)
    game_status["PacmanPos"] = change_pos(state.data.agentStates[0].configuration.pos, numRow)
    for i in range(1, len(state.data.agentStates)):
        ghost_data = change_pos(state.data.agentStates[i].configuration.pos, numRow)
        if state.data._eaten[i] == True:
            ghost_status = 1
        elif state.data.agentStates[i].scaredTimer == 0:
            ghost_status = 0
        else:
            ghost_status = 2
        game_status["ghost_data"].append(ghost_data)
        game_status["ghost_status"].append(ghost_status)
    for i in range(2 - len(game_status["ghost_data"])):
        game_status["ghost_data"].append(None)
        game_status["ghost_status"].append(None)
    food = np.array(state.data.food.data)
    food_position = np.where(food)
    temp = []
    for k in range(len(food_position[0])):
        temp.append(change_pos((food_position[0][k], food_position[1][k]), numRow))
    game_status["bean_data"] = temp
    if len(state.data.capsules) != 0:
        game_status["energizer_data"] = deepcopy([change_pos(i, numRow) for i in state.data.capsules])

    Series_data = {
        "pacmanPos": game_status["PacmanPos"],
        "ghost1Pos": game_status["ghost_data"][0],
        "ghost2Pos": game_status["ghost_data"][1],
        "energizers": game_status["energizer_data"],
        "beans": game_status["bean_data"],
        "ifscared1": game_status["ghost_status"][0],
        "ifscared2": game_status["ghost_status"][1],
    }
    return Series_data


def direction_transition(x):
    if x == None:
        return None
    dir_dict = {"Stop": None, "North": "up", "South": "down", "West": "left", "East": "right"}
    return dir_dict[x]


def transition():
    gram = "bi"
    start_num = 0
    end_num = 10

    number = 1

    state = []
    action = []
    reward = []
    dead = []
    strategy = []
    strategy_utility = []
    files = []
    # 整合 所有trail的ganme_status
    for i in range(start_num, end_num):
        with open("../Data/game_status/" + gram + "_" + str(i) + ".pkl", "rb") as file:
            data = pickle.load(file)
            for j in range(len(data["states"])):
                state += data["states"][j]
                action += data["actions"][j]
                reward += data["rewards"][j]
                dead += data["deads"][j]
                strategy += data["strategy_sequences"][j]
                strategy_utility += data["strategy_utility_sequences"][j]
                files += ["biStrategy_" + str(number)] * len(data["deads"][j])
                number += 1
    data = {
        "state": state,
        "pacman_dir": action,
        "Reward": reward,
        "dead": dead,
        "file": files,
        "strategy": strategy,
        "strategy_utility": strategy_utility
    }
    data = pd.DataFrame(data)
    game_status = data["state"].apply(lambda x: state_to_feature(x))

    pacmanPos = []
    ghost1Pos = []
    ghost2Pos = []
    energizers = []
    beans = []
    ifscared1 = []
    ifscared2 = []
    for i in game_status:
        pacmanPos.append(i["pacmanPos"])
        ghost1Pos.append(i["ghost1Pos"])
        ghost2Pos.append(i["ghost2Pos"])
        energizers.append(i["energizers"])
        beans.append(i["beans"])
        ifscared1.append(i["ifscared1"])
        ifscared2.append(i["ifscared2"])
    data["pacmanPos"] = pacmanPos
    data["ghost1Pos"] = ghost1Pos
    data["ghost2Pos"] = ghost2Pos
    data["ifscared1"] = ifscared1
    data["ifscared2"] = ifscared2
    data["beans"] = beans
    data["energizers"] = energizers
    data["pacman_dir"] = data["pacman_dir"].apply(lambda x: direction_transition(x))
    data["last_dir"] = [None] + list(data["pacman_dir"])[:-1]
    data = data[
        ["file", "pacmanPos", "ghost1Pos", "ghost2Pos", "ifscared1", "ifscared2", "beans", "energizers", "Reward",
         "pacman_dir", "last_dir", "strategy", "strategy_utility"]]

    data.to_pickle("../Data/process/bi-gram.pkl")


if __name__ == '__main__':
    transition()
