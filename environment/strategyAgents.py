import random
import time

from game import Agent
from game import Directions
import numpy as np
from primitiveStrategy.Strategy import *
from primitiveStrategy.simpleGlobalStrategy import simpleGlobalStrategy
from Utils.FileUtils import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath
from strategyPolicyTable import strategyPolicyTable
from copy import deepcopy
from Utils.ComputationUtils import scaleOfNumber, makeChoice
import pandas as pd
from FeatureExtractor.ExtractGameFeatures import featureExtractor


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--depth', type=int, default=10, help='The maximum depth of tree.')
    parser.add_argument('--ignore_depth', type=int, default=0, help=' Ignore this depth of nodes.')
    parser.add_argument('--ghost_attractive_thr', type=int, default=34, help='Ghost attractive threshold.')
    parser.add_argument('--ghost_repulsive_thr', type=int, default=34, help='Ghost repulsive threshold.')
    parser.add_argument('--reward_coeff', type=float, default=1.0, help='Coefficient for the reward.')
    parser.add_argument('--risk_coeff', type=float, default=0.0, help='Coefficient for the risk.')
    parser.add_argument('--randomness_coeff', type=float, default=1.0, help='Coefficient for the randomness.')
    parser.add_argument('--laziness_coeff', type=float, default=1.0, help='Coefficient for the laziness.')
    config = parser.parse_args(args=[])
    return config


def get_paramater_of_strategy(strategy_name, h, w):
    args = get_args()
    L = max(h, w)
    if strategy_name == "local":
        args.depth = int(L / 3)
        args.ghost_attractive_thr = int(L / 3)
        args.ghost_repulsive_thr = int(L / 3)
    elif strategy_name == "global":
        args.depth = h+w
        args.ignore_depth = int(L / 2)
        args.ghost_attractive_thr = L
        args.ghost_repulsive_thr = L
    elif strategy_name == "evade":
        args.depth = 3
        args.ghost_attractive_thr = L
        args.ghost_repulsive_thr = L
        args.reward_coeff = 0.0
        args.risk_coeff = 1.0
    elif strategy_name == "energizer":
        args.depth = int(L / 3)
        args.ghost_attractive_thr = 0
        args.ghost_repulsive_thr = 0
    elif strategy_name == "approach":
        args.depth = int(L / 4)
        args.ghost_attractive_thr = L
        args.ghost_repulsive_thr = L
    return args


def change_pos(pos, numRow):
    temp_pos = (pos[0] + 1, numRow - pos[1])
    return temp_pos


class singleStartegyAgent(Agent):
    def __init__(self, map_name="smallGrid", index=0, **arg):

        self.lastMove = Directions.STOP
        self.map_name = map_name
        # self.index = index
        self.keys = []
        self.index = index
        self.get_strategies(map_name)
        self.strategy_choice = strategyPolicyTable()
        self.featureExtractor = featureExtractor(map_name)

    def get_strategies(self, map_name):
        import layout
        layout = layout.getLayout(map_name).layoutText
        h = len(layout)
        w = len(layout[0])
        locs_df = readLocDistance("../Data/mapMsg/dij_distance_map_" + map_name + ".csv")
        adjacent_data = readAdjacentMap("../Data/mapMsg/adjacent_map_" + map_name + ".csv")
        reward_amount = readRewardAmount()

        local_strategy = Strategy("local", adjacent_data, locs_df, reward_amount,
                                  get_paramater_of_strategy("local", h, w))
        global_strategy = Strategy("global", adjacent_data, locs_df, reward_amount,
                                   get_paramater_of_strategy("global", h, w))
        evade_strategy = Strategy("evade", adjacent_data, locs_df, reward_amount,
                                  get_paramater_of_strategy("evade", h, w))
        energizer_strategy = Strategy("energizer", adjacent_data, locs_df, reward_amount,
                                      get_paramater_of_strategy("energizer", h, w))
        approach_strategy = Strategy("approach", adjacent_data, locs_df, reward_amount,
                                     get_paramater_of_strategy("approach", h, w))
        self.startegies = {
            "local": local_strategy, "global": global_strategy, "evade": evade_strategy,
            "energizer": energizer_strategy, "approach": approach_strategy
        }

    def state_to_feature(self, state):
        """
        将游戏state转变为 game_status和feature
        :param state: 
        :return: 
        """
        game_status = {"PacmanPos": [], "ghost_data": [], "ghost_status": [],
                       "bean_data": [], "energizer_data": [], "Reward": [], "last_dir": self.lastMove}

        dir_dict = {"Stop": None, "North": "up", "South": "down", "West": "left", "East": "right"}
        game_status["last_dir"] = dir_dict[game_status["last_dir"]]

        # get game_status
        numRow = len(state.data.layout.layoutText)
        game_status["PacmanPos"] = change_pos(state.data.agentStates[0].configuration.pos, numRow)
        for i in range(1, len(state.data.agentStates)):
            ghost_data = change_pos(state.data.agentStates[i].configuration.pos, numRow)
            if state.data._eaten[i] == True:
                ghost_status = 1
            elif state.data.agentStates[1].scaredTimer == 0:
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

        # get feature
        Series_data = {
            "pacmanPos": [game_status["PacmanPos"]],
            "ghost1Pos": [game_status["ghost_data"][0]],
            "ghost2Pos": [game_status["ghost_data"][1]],
            "energizers": [game_status["energizer_data"]],
            "beans": [game_status["bean_data"]],
            "ifscared1": [game_status["ghost_status"][0]],
            "ifscared2": [game_status["ghost_status"][1]],
            "pacman_dir": [game_status["last_dir"]]
        }
        if len(Series_data["energizers"][0]) == 0:
            Series_data["energizers"][0] = np.nan
        data = pd.DataFrame(Series_data)
        feature = self.featureExtractor.extract_feature(data)
        return game_status, feature

    def getAction(self, state):

        legal = state.getLegalActions(self.index)
        if 'Stop' in legal:
            legal.remove('Stop')

        game_status, feature = self.state_to_feature(state)

        # choose strategy
        strategy_name = self.strategy_choice.get_strategy(feature)
        print(strategy_name)
        strategy = self.startegies[strategy_name]
        # strategy = self.startegies["approach"]
        strategy.set_state(game_status)
        _, Q = strategy.nextDir(return_Q=True)
        choice = strategy.mapStatus["dir_list"][makeChoice(Q)]

        dir_dict = {"left": Directions.WEST, "right": Directions.EAST, "up": Directions.NORTH, "down": Directions.SOUTH
                    }
        move = dir_dict[choice]
        print(Q)
        return move

# if __name__ == '__main__':
#     x = singleStartegyAgent()
