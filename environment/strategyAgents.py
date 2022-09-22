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
    L = max(h,w)
    if strategy_name == "local":
        args.depth = int(L / 3)
        args.ghost_attractive_thr = int(L / 3)
        args.ghost_repulsive_thr = int(L / 3)
    elif strategy_name == "global":
        args.depth = L
        args.ignore_depth = int(L / 3)
        args.ghost_attractive_thr = L
        args.ghost_repulsive_thr = L
    elif strategy_name == "evade":
        args.depth = int(L / 3)
        args.ghost_attractive_thr = L
        args.ghost_repulsive_thr = L
        args.reward_coeff = 0.0
        args.risk_coeff = 1.0
    elif strategy_name == "energizer":
        args.depth = int(L / 3)
        args.ghost_attractive_thr = 0
        args.ghost_repulsive_thr = 0
    elif strategy_name == "approach":
        args.depth = int(L / 3)
        args.ghost_attractive_thr = int(L / 3)
        args.ghost_repulsive_thr = int(L / 3)
    return args


def change_pos(pos, numRow):
    temp_pos = (pos[0] + 1, numRow - pos[1])
    return temp_pos


class singleStartegyAgent(Agent):
    def __init__(self, map_name="smallGrid", index=0, **arg):

        self.lastMove = Directions.STOP
        # self.index = index
        self.keys = []
        self.index = index
        self.get_strategies(map_name)
        self.strategy_choice = strategyPolicyTable()

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
        global_strategy = simpleGlobalStrategy(adjacent_data, locs_df, reward_amount,
                                               get_paramater_of_strategy("local", h, w))
        evade_strategy = Strategy("evade", adjacent_data, locs_df, reward_amount,
                                  get_paramater_of_strategy("evade", h, w))
        energizer_strategy = Strategy("energizer", adjacent_data, locs_df, reward_amount,
                                      get_paramater_of_strategy("energizer", h, w))
        approach_strategy = Strategy("approach", adjacent_data, locs_df, reward_amount,
                                     get_paramater_of_strategy("approach", h, w))
        self.startegies = [local_strategy, global_strategy, evade_strategy, energizer_strategy, approach_strategy]

    def state_to_feature(self, state):
        game_status = {"PacmanPos": [], "ghost_data": [], "ghost_status": [],
                       "bean_data": [], "energizer_data": [], "Reward": [], "last_dir": self.lastMove}

        dir_dict = {"Stop": None, "North": "up", "South": "down", "West": "left", "East": "right"}
        game_status["last_dir"] = dir_dict[game_status["last_dir"]]

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

        return game_status

    def getAction(self, state):

        legal = state.getLegalActions(self.index)
        if 'Stop' in legal:
            legal.remove('Stop')



        game_status = self.state_to_feature(state)
        # choose strategy
        strategy = self.startegies[2]
        strategy.set_state(game_status)
        _, Q = strategy.nextDir(return_Q=True)
        choice = strategy.mapStatus["dir_list"][makeChoice(Q)]
        if (choice == 'down' and game_status["PacmanPos"][1] == 10) or (
                choice == 'up' and game_status["PacmanPos"][1] == 9):
            x = 0

        # strategy.set_state(game_status)
        # _, Q = strategy.nextDir(return_Q=True)

        dir_dict = {"left": Directions.WEST, "right": Directions.EAST, "up": Directions.NORTH, "down": Directions.SOUTH
                    }
        move = dir_dict[choice]
        print(Q)
        # print(game_status["PacmanPos"], move)
        # if random.random() < 0.8:
        #     move = dir_dict[choice]
        # else:
        #     move = np.random.choice(legal, 1)[0]

        # move = np.random.choice(legal, 1)[0]
        # print(move)
        # if move == "West":
        #     move = Directions.WEST
        # elif move == 'Stop':
        #     move = Directions.STOP
        # elif move == 'East':
        #     move = Directions.EAST
        # elif move == "Up":
        #     move = Directions.NORTH
        # elif move == "Down":
        #     move = Directions.SOUTH
        return move

    def getMove(self, legal):
        move = Directions.STOP
        if (self.WEST_KEY in self.keys or 'Left' in self.keys) and Directions.WEST in legal:
            move = Directions.WEST
        if (self.EAST_KEY in self.keys or 'Right' in self.keys) and Directions.EAST in legal:
            move = Directions.EAST
        if (self.NORTH_KEY in self.keys or 'Up' in self.keys) and Directions.NORTH in legal:
            move = Directions.NORTH
        if (self.SOUTH_KEY in self.keys or 'Down' in self.keys) and Directions.SOUTH in legal:
            move = Directions.SOUTH
        return move

# if __name__ == '__main__':
#     x = singleStartegyAgent()
