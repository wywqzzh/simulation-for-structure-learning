from game import Agent
from game import Directions
import numpy as np
from primitiveStrategy.Strategy import *
from primitiveStrategy.simpleGlobalStrategy import simpleGlobalStrategy


def argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--depth', type=int, default=10, help='The maximum depth of tree.')
    parser.add_argument('--ignore_depth', type=int, default=0, help=' Ignore this depth of nodes.')
    parser.add_argument('--ghost_attractive_thr', type=int, default=34, help='Ghost attractive threshold.')
    parser.add_argument('--ghost_repulsive_thr', type=int, default=34, help='Ghost repulsive threshold.')
    parser.add_argument('--reward_coeff', type=float, default=1.0, help='Coefficient for the reward.')
    parser.add_argument('--risk_coeff', type=float, default=0.0, help='Coefficient for the risk.')
    parser.add_argument('--randomness_coeff', type=float, default=1.0, help='Coefficient for the randomness.')
    parser.add_argument('--laziness_coeff', type=float, default=1.0, help='Coefficient for the laziness.')
    config = parser.parse_args()
    return config


def get_paramater_of_strategy(strategy_name, h, w):
    args = argparser()
    L = max(h, w)
    if strategy_name == "local":
        args.depth = int(L / 3)
        args.ghost_attractive_thr = int(L / 3)
        args.ghost_repulsive_thr = int(L / 3)
    elif strategy_name == "global":
        args.depth = int(L / 2)
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


class singleStartegyAgent(Agent):
    def __init__(self, index=0, map_name="smallGrid", **args):

        locs_df = readLocDistance("../Data/constant/dij_distance_map_" + map_name + ".csv")
        adjacent_data = readAdjacentMap("../Data/constant/adjacent_map_" + map_name + ".csv")
        adjacent_path = readAdjacentPath("../Data/constant/dij_distance_map_" + map_name + ".csv")
        reward_amount = readRewardAmount()

        self.startegies = [Strategy("local", adjacent_data, locs_df, reward_amount, args),
                           simpleGlobalStrategy(adjacent_data, locs_df, reward_amount, args)]
        self.lastMove = Directions.STOP
        self.index = index
        self.keys = []
        self.startegies = []

    def getAction(self, state):
        legal = state.getLegalActions(self.index)
        move = np.random.choice(legal, 1)[0]
        print(move)
        if move == "West":
            move = Directions.WEST
        elif move == 'Stop':
            move = Directions.STOP
        elif move == 'East':
            move = Directions.EAST
        elif move == "Up":
            move = Directions.NORTH
        elif move == "Down":
            move = Directions.SOUTH
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


if __name__ == '__main__':
    x = singleStartegyAgent()
