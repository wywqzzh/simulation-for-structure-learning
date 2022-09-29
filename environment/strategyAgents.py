import random
import time

from game import Agent
from game import Directions
import numpy as np
from primitiveStrategy.Strategy import *
from primitiveStrategy.simpleGlobalStrategy import simpleGlobalStrategy
from Utils.FileUtils import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath
from strategyPolicyTable import strategyPolicyTable, twoStrategyPolicyTable
from copy import deepcopy
from Utils.ComputationUtils import scaleOfNumber, makeChoice
import pandas as pd
from FeatureExtractor.ExtractGameFeatures import featureExtractor

import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--depth', type=int, default=10, help='The maximum depth of tree.')
    parser.add_argument('--ignore_depth', type=int, default=0, help=' Ignore this depth of nodes.')
    parser.add_argument('--ghost_attractive_thr', type=int, default=34, help='Ghost attractive threshold.')
    parser.add_argument('--ghost_repulsive_thr', type=int, default=34, help='Ghost repulsive threshold.')
    parser.add_argument('--reward_coeff', type=float, default=1.0, help='Coefficient for the reward.')
    parser.add_argument('--risk_coeff', type=float, default=0.0, help='Coefficient for the risk.')
    parser.add_argument('--randomness_coeff', type=float, default=0.0, help='Coefficient for the randomness.')
    parser.add_argument('--laziness_coeff', type=float, default=0.0, help='Coefficient for the laziness.')
    config = parser.parse_args(args=[])
    return config


def get_paramater_of_strategy(strategy_name, h, w):
    args = get_args()
    L = max(h, w)
    if strategy_name == "local":
        args.depth = 10
    elif strategy_name == "global":
        args.depth = L
        args.ignore_depth = 0
    elif strategy_name == "evade":
        args.depth = 3
        args.reward_coeff = 0.0
        args.risk_coeff = 1.0
    elif strategy_name == "energizer":
        args.depth = 10
    elif strategy_name == "approach":
        args.depth = 15
    elif strategy_name == "counterattack":
        args.depth = 10
        args.reward_coeff = 1.0
        args.risk_coeff = 1.0
    return args


def change_pos(pos, numRow):
    temp_pos = (pos[0] + 1, numRow - pos[1])
    return temp_pos


class StartegyAgents(Agent):
    def __init__(self, map_name="smallGrid", index=0, **arg):
        self.lastMove = Directions.STOP
        self.map_name = map_name
        self.index = index
        self.get_strategies(map_name)
        self.featureExtractor = featureExtractor(map_name)

    def get_strategies(self, map_name):
        import layout
        layout = layout.getLayout(map_name).layoutText
        h = len(layout)
        w = len(layout[0])
        self.layout_h = h
        locs_df = readLocDistance("../Data/mapMsg/dij_distance_map_" + map_name + ".csv")
        adjacent_data = readAdjacentMap("../Data/mapMsg/adjacent_map_" + map_name + ".csv")
        self.intersection_data = pd.read_pickle("../Data/mapMsg/intersection_map_" + map_name + ".pkl")["pos"]
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
        counterattack_strategy = Strategy("counterattack", adjacent_data, locs_df, reward_amount,
                                          get_paramater_of_strategy("approach", h, w))
        self.startegies = {
            "local": local_strategy, "global": global_strategy, "evade": evade_strategy,
            "energizer": energizer_strategy, "approach": approach_strategy, "counterattack": counterattack_strategy
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


class singleStartegyAgents(StartegyAgents):
    def __init__(self, map_name="smallGrid", index=0, **arg):
        super(singleStartegyAgents, self).__init__(map_name)
        self.strategy_choice = strategyPolicyTable()
        self.last_strategy_name = "local"

    def getAction(self, state):
        # TODO: approach会自杀

        legal = state.getLegalActions(self.index)
        if 'Stop' in legal:
            legal.remove('Stop')

        game_status, feature = self.state_to_feature(state)

        # choose strategy
        strategy_name = self.strategy_choice.get_strategy(feature)
        # strategy_name = "approach"
        cur_pos = change_pos(state.data.agentStates[0].configuration.pos, self.layout_h)
        if cur_pos in self.intersection_data or strategy_name == "evade" or self.last_strategy_name == "evade":
            # print("change strategy,", cur_pos)
            self.last_strategy_name = strategy_name
        else:
            strategy_name = self.last_strategy_name
        # print(strategy_name)
        strategy = self.startegies[strategy_name]
        # strategy = self.startegies["approach"]
        strategy.set_state(game_status)
        _, Q = strategy.nextDir(return_Q=True)
        choice = strategy.mapStatus["dir_list"][makeChoice(Q)]

        dir_dict = {"left": Directions.WEST, "right": Directions.EAST, "up": Directions.NORTH, "down": Directions.SOUTH
                    }
        move = dir_dict[choice]
        # if strategy_name == "approach":
        #     print(Q)
        return move, strategy_name


class twoStartegyAgents(StartegyAgents):
    def __init__(self, map_name="smallGrid", index=0, **arg):
        super(twoStartegyAgents, self).__init__(map_name)
        self.strategy_choice = twoStrategyPolicyTable()
        self.featureExtractor = featureExtractor(map_name)

    def getAction(self, state):
        # TODO: approach会自杀

        legal = state.getLegalActions(self.index)
        if 'Stop' in legal:
            legal.remove('Stop')

        game_status, feature = self.state_to_feature(state)
        # feature={'PG1': 1, 'GS1': 0, 'PG2': 2, 'GS2': 0, 'PE': 1, 'BW': 0, 'BB': 1, 'ZBW': 1, 'ZBB': 0}
        # choose strategy
        two_strategy_name = self.strategy_choice.get_two_strategy(feature)
        strategy_name = self.strategy_choice.get_single_strategy(feature)
        if self.strategy_choice.two_strategy_end == True:
            self.strategy_choice.two_strategy = two_strategy_name
            self.strategy_choice.two_strategy_end = False
            self.strategy_choice.strategy = None
            strategy_name = self.strategy_choice.get_single_strategy(feature)
            if two_strategy_name == "EA" and strategy_name == None:
                print("feature:", feature)
                strategy_name = self.strategy_choice.get_single_strategy(feature)
            if two_strategy_name == "eC" and strategy_name == None:
                strategy_name = self.strategy_choice.get_single_strategy(feature)
        print(self.strategy_choice.two_strategy, strategy_name)
        strategy = self.startegies[strategy_name]
        strategy.set_state(game_status)
        _, Q = strategy.nextDir(return_Q=True)
        choice = strategy.mapStatus["dir_list"][makeChoice(Q)]

        dir_dict = {"left": Directions.WEST, "right": Directions.EAST, "up": Directions.NORTH, "down": Directions.SOUTH
                    }
        move = dir_dict[choice]
        if strategy_name == "approach":
            print(Q)
        return move
