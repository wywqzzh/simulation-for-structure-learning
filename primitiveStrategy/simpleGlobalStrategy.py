import numpy as np
import anytree
from collections import deque
from copy import deepcopy
import copy
import sys

sys.path.append("../Utils")
from Utils.ComputationUtils import scaleOfNumber, makeChoice
import argparse


class simpleGlobalStrategy:
    def __init__(self, adjacent_data, locs_df, reward_amount, args):
        """
        :param root:
        :param energizer_data:
        :param bean_data:
        :param ghost_data:
        :param ghost_status:
        :param adjacent_data:
        :param locs_df:
        :param reward_amount:
        :param last_dir:
        :param args:
        """
        # Parameter type check

        if not isinstance(args.depth, int):
            raise TypeError("The depth should be a integer, but got a {}.".format(type(args.depth)))
        if args.depth <= 0:
            raise ValueError("The depth should be a positive integer.")

        # trade-off args
        self.args = args

        # pre-computed map status data
        self.mapStatus = {
            "adjacent_data": adjacent_data,
            "locs_df": locs_df,
            "reward_amount": reward_amount,
            "dir_list": ['left', 'right', 'up', 'down']
        }
        # Utility (Q-value) for every direction

    def set_state(self, root, energizer_data, bean_data, ghost_data, ghost_status, last_dir):

        if not isinstance(root, tuple):
            raise TypeError("The root should be a 2-tuple, but got a {}.".format(type(root)))

        self.gameStatus = {"cur_pos": root, "energizer_data": energizer_data, "bean_data": bean_data,
                           "ghost_data": ghost_data,
                           "ghost_status": ghost_status, "existing_bean": bean_data,
                           "existing_energizer": energizer_data, "last_dir": last_dir}
        self.Q_value = [0, 0, 0, 0]

        self.available_dir = []
        self.adjacent_pos = self.mapStatus["adjacent_data"][self.gameStatus["cur_pos"]]
        for dir in ["left", "right", "up", "down"]:
            if None != self.adjacent_pos[dir] and not isinstance(self.adjacent_pos[dir], float):
                self.available_dir.append(dir)
        if 0 == len(self.available_dir) or 1 == len(self.available_dir):
            raise ValueError("The position {} has {} adjacent positions.".format(self.gameStatus["cur_pos"],
                                                                                 len(self.available_dir)))
        self.adjacent_pos = [self.adjacent_pos[each] for each in self.available_dir]

    def _dirArea(self, dir):
        # x: 1~28 | y: 1~33
        left_bound = 1
        right_bound = 28
        upper_bound = 1
        lower_bound = 33
        # Area corresponding to the direction
        if dir == "left":
            area = [
                (left_bound, upper_bound),
                (max(1, self.gameStatus["cur_pos"][0] - 1), lower_bound)
            ]
        elif dir == "right":
            area = [
                (min(right_bound, self.gameStatus["cur_pos"][0] + 1), upper_bound),
                (right_bound, lower_bound)
            ]
        elif dir == "up":
            area = [
                (left_bound, upper_bound),
                (right_bound, min(lower_bound, self.gameStatus["cur_pos"][1] + 1))
            ]
        elif dir == "down":
            area = [
                (left_bound, min(lower_bound, self.gameStatus["cur_pos"][1] + 1)),
                (right_bound, lower_bound)
            ]
        else:
            raise ValueError("Undefined direction {}!".format(dir))
        return area

    def _countBeans(self, upper_left, lower_right):
        area_loc = []
        # Construct a grid area
        for i in range(upper_left[0], lower_right[0] + 1):
            for j in range(upper_left[1], lower_right[1] + 1):
                area_loc.append((i, j))
        if isinstance(self.gameStatus["bean_data"], float) or self.gameStatus["bean_data"] is None:
            return 0
        else:
            beans_num = 0
            for each in self.gameStatus["bean_data"]:
                if each in area_loc:
                    beans_num += 1
            return beans_num

    def nextDir(self, return_Q=False):
        available_directions_index = [self.mapStatus["dir_list"].index(each) for each in self.available_dir]
        self.Q_value = [0.0, 0.0, 0.0, 0.0]
        for dir in self.available_dir:
            area = self._dirArea(dir)
            beans_num = self._countBeans(area[0], area[1])
            self.Q_value[self.mapStatus["dir_list"].index(dir)] = beans_num
        self.Q_value = np.array(self.Q_value, dtype=np.float)
        # self.Q_value[available_directions_index] += 1.0 # avoid 0 utility
        # Add randomness and laziness
        Q_scale = scaleOfNumber(np.max(np.abs(self.Q_value)))
        if len(available_directions_index) > 0:
            # randomness = np.random.normal(loc=0, scale=0.1, size=len(available_directions_index)) * Q_scale
            randomness = np.random.uniform(low=0, high=0.1, size=len(available_directions_index)) * Q_scale
            self.Q_value[available_directions_index] += (self.args.randomness_coeff * randomness)
        if self.gameStatus["last_dir"] is not None and self.gameStatus["last_dir"].index(
                self.gameStatus["last_dir"]) in available_directions_index:
            self.Q_value[self.gameStatus["last_dir"].index(self.gameStatus["last_dir"])] += (
                    self.args.laziness_coeff * Q_scale)
        if return_Q:
            return makeChoice(self.Q_value), self.Q_value
        else:
            return makeChoice(self.Q_value)


def argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--depth', type=int, default=10, help='The maximum depth of tree.')
    parser.add_argument('--ignore_depth', type=int, default=0, help=' Ignore this depth of nodes.')
    parser.add_argument('--ghost_attractive_thr', type=int, default=34, help='Ghost attractive threshold.')
    parser.add_argument('--ghost_repulsive_thr', type=int, default=10, help='Ghost repulsive threshold.')
    parser.add_argument('--reward_coeff', type=float, default=1.0, help='Coefficient for the reward.')
    parser.add_argument('--risk_coeff', type=float, default=0.0, help='Coefficient for the risk.')
    parser.add_argument('--randomness_coeff', type=float, default=0.0, help='Coefficient for the randomness.')
    parser.add_argument('--laziness_coeff', type=float, default=0.0, help='Coefficient for the laziness.')
    config = parser.parse_args()
    return config


if __name__ == '__main__':
    from Utils.FileUtils import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath
    from Utils.ComputationUtils import makeChoice

    locs_df = readLocDistance("../Data/constant/dij_distance_map.csv")
    adjacent_data = readAdjacentMap("../Data/constant/adjacent_map.csv")
    adjacent_path = readAdjacentPath("../Data/constant/dij_distance_map.csv")
    reward_amount = readRewardAmount()

    import pickle

    with open("../Data/10_trial_data_Omega.pkl", "rb") as file:
        result = pickle.load(file)

    args = argparser()
    args.depth = 15
    args.ignore_depth = 5
    args.ghost_attractive_thr = 34
    args.ghost_repulsive_thr = 34
    args.reward_coeff = 1.0
    args.risk_coeff = 0.0
    strategy = simpleGlobalStrategy(adjacent_data, locs_df, reward_amount, args)
    for index in range(len(result)):
        # if index != 645:
        #     continue
        print(index)
        cur_pos = result["pacmanPos"][index]
        ghost_data = [result["ghost1Pos"][index], result["ghost2Pos"][index]]
        ghost_status = [result["ifscared1"][index], result["ifscared2"][index]]
        energizer_data = result["energizers"][index]
        bean_data = result["beans"][index]
        last_dir = result["pacman_dir"][index]

        strategy.set_state(cur_pos, energizer_data, bean_data, ghost_data, ghost_status, last_dir)
        _, Q = strategy.nextDir(return_Q=True)
        choice = strategy.mapStatus["dir_list"][makeChoice(Q)]
        print("Global Choice : ", choice, Q)
