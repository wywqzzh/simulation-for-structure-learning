import argparse
import copy
from collections import deque

import anytree
import numpy as np

from Utils.ComputationUtils import scaleOfNumber
from copy import deepcopy


class Strategy:

    def __init__(self, strategy_type, adjacent_data, locs_df, reward_amount, \
                 args):
        if not isinstance(args.depth, int):
            raise TypeError("The depth should be a integer, but got a {}.".format(type(args.depth)))
        if args.depth <= 0:
            raise ValueError("The depth should be a positive integer.")

        self.strategy_type = strategy_type
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
        self.Q_value = [0, 0, 0, 0]
        # Pacman is eaten? If so, the path will be ended
        self.is_eaten = False

    def set_state(self, gameStatus):

        self.gameStatus = gameStatus
        self.gameStatus.update({"existing_bean": deepcopy(gameStatus["bean_data"]),
                                "existing_energizer": deepcopy(gameStatus["energizer_data"])})
        # self.gameStatus = {"energizer_data": energizer_data, "bean_data": bean_data, "ghost_data": ghost_data,
        #                    "ghost_status": ghost_status, "existing_bean": bean_data,
        #                    "existing_energizer": energizer_data, "last_dir": last_dir}
        self.Q_value = [0, 0, 0, 0]
        self.is_eaten = False
        self.root = anytree.Node(gameStatus["PacmanPos"],
                                 cur_len=1,
                                 cur_utility=0.0,
                                 cumulative_utility=0.0,
                                 cur_reward=0.0,
                                 cumulative_reward=0.0,
                                 cur_risk=00.,
                                 cumulative_risk=0.0,
                                 existing_beans=copy.deepcopy(self.gameStatus["bean_data"]),
                                 existing_energizers=copy.deepcopy(self.gameStatus["energizer_data"]),
                                 ghost_status=copy.deepcopy(self.gameStatus["ghost_status"]),
                                 exact_reward_list=[],
                                 exact_risk_list=[],
                                 )
        # TODO: add game status for the node
        # The current node
        self.current_node = self.root
        # A queue used for append nodes on the tree
        self.node_queue = deque()
        self.node_queue.append(self.root)

    def _computeReward(self, cur_position):
        """
        计算reward,在local中只考虑豆子和energizers数
        :param cur_position:
        :return:
        """
        existing_beans = copy.deepcopy(self.current_node.existing_beans)
        existing_energizers = copy.deepcopy(self.current_node.existing_energizers)
        ghost_status = copy.deepcopy(self.current_node.ghost_status)
        exact_reward = 0.0
        # Bean reward
        if self.strategy_type == "local" or self.strategy_type == "global":
            if isinstance(existing_beans, float):
                exact_reward += 0.0
            elif cur_position in existing_beans:
                exact_reward += self.mapStatus["reward_amount"][1]
                existing_beans.remove(cur_position)
        # energizer reward
        if self.strategy_type == "energizer" or self.strategy_type == "counterattack":
            if isinstance(existing_energizers, float) or cur_position not in existing_energizers:
                exact_reward += 0.0
            elif cur_position in existing_energizers:
                # Reward for eating the energizer
                exact_reward += self.mapStatus["reward_amount"][2]
                existing_energizers.remove(cur_position)
                # TODO:改变ghost的状态
                ghost_status = [2 if each != 1 else 1 for each in ghost_status]  # change ghost status
        # eat ghost reward
        if self.strategy_type == "approach":
            # TODO:改变ghost的状态
            if isinstance(self.gameStatus["ghost_data"], float) or cur_position not in self.gameStatus["ghost_data"] \
                    or np.all(np.array(ghost_status) == 1):
                exact_reward += 0.0
            for index, ghost in enumerate(self.gameStatus["ghost_data"]):
                ghost = (int(ghost[0]), int(ghost[1]))
                if ghost_status[index] != 1:
                    if cur_position == ghost:
                        exact_reward += self.mapStatus["reward_amount"][8]
                        # print(index, ghost_status[index], exact_reward)
                        if ghost_status[index] > 1:
                            ghost_status[index] = 1
                        else:
                            exact_reward -= (self.mapStatus["reward_amount"][8] + 10)
                            # print(exact_reward)
                            self.is_eaten = True
        return exact_reward, existing_beans, existing_energizers, ghost_status

    def _computeRisk(self, cur_position):
        """
        计算risk，在local中不考虑risk
        :param cur_position:
        :return:
        """
        ghost_status = copy.deepcopy(self.current_node.ghost_status)
        # Compute ghost risk when ghosts are normal
        ifscared1 = ghost_status[0] if not isinstance(ghost_status[0], float) else 0
        ifscared2 = ghost_status[1] if not isinstance(ghost_status[1], float) else 0
        exact_risk = 0.0
        # TODO: 改变ghost的状态
        if ifscared1 == 0 and cur_position == self.gameStatus["ghost_data"][0]:
            self.is_eaten = True
            if self.strategy_type == "evade":
                exact_risk = -self.mapStatus["reward_amount"][9]
        if ifscared2 == 0 and cur_position == self.gameStatus["ghost_data"][1]:
            self.is_eaten = True
            if self.strategy_type == "evade":
                exact_risk = -self.mapStatus["reward_amount"][9]
        return exact_risk

    def _attachNode(self, cur_depth=0, ignore=False):
        if 0 == cur_depth:  # TODO: cur_depth is useless for now
            raise ValueError("The depth should not be 0!")
        tmp_data = self.mapStatus["adjacent_data"][self.current_node.name]
        for each in ["left", "right", "up", "down"]:
            # do not walk on the wall or walk out of boundary
            # do not turn back
            if None == self.current_node.parent and isinstance(tmp_data[each], float):
                continue
            elif None != self.current_node.parent and (
                    isinstance(tmp_data[each], float) or tmp_data[each] == self.current_node.parent.name):
                continue
            else:
                # Compute utility
                cur_pos = tmp_data[each]
                if ignore:
                    exact_reward = 0.0
                    exact_risk = 0.0
                    existing_beans = copy.deepcopy(self.current_node.existing_beans)
                    existing_energizers = copy.deepcopy(self.current_node.existing_energizers)
                    ghost_status = copy.deepcopy(self.current_node.ghost_status)
                else:
                    # Compute reward
                    exact_reward, existing_beans, existing_energizers, ghost_status = self._computeReward(cur_pos)
                    # Compute risk
                    # if the position is visited before, do not add up the risk to cumulative
                    if cur_pos in [each.name for each in self.current_node.path]:
                        exact_risk = 0.0
                    else:
                        exact_risk = self._computeRisk(cur_pos)
                        # Construct the new node
            exact_reward_list = copy.deepcopy(self.current_node.exact_reward_list)
            exact_risk_list = copy.deepcopy(self.current_node.exact_risk_list)

            exact_reward_list.append(exact_reward)
            exact_risk_list.append(exact_risk)
            if self.strategy_type == "local":
                cumulative_utility = self.current_node.cumulative_utility + self.args.reward_coeff * exact_reward / (
                        self.current_node.cur_len + 1) + self.args.risk_coeff * exact_risk
            else:
                cumulative_utility = self.current_node.cumulative_utility + self.args.reward_coeff * exact_reward + self.args.risk_coeff * exact_risk
            new_node = anytree.Node(
                cur_pos,
                cur_len=self.current_node.cur_len + 1,
                parent=self.current_node,
                dir_from_parent=each,
                cur_utility={
                    "exact_reward": exact_reward,
                    "exact_risk": exact_risk,
                },
                cur_reward={
                    "exact_reward": exact_reward,
                },
                cur_risk={
                    "exact_risk": exact_risk,
                },
                cumulative_reward=self.current_node.cumulative_reward + exact_reward,
                cumulative_risk=self.current_node.cumulative_risk + exact_risk,
                cumulative_utility=cumulative_utility,
                existing_beans=existing_beans,
                existing_energizers=existing_energizers,
                ghost_status=ghost_status,
                exact_reward_list=exact_reward_list,
                exact_risk_list=exact_risk_list,
            )
            # If the Pacman is eaten, end this path
            if self.is_eaten:
                # if self.strategy_type=="approach":
                #     self.node_queue.append(new_node)
                self.is_eaten = False
            else:
                self.node_queue.append(new_node)

    def _construct(self):
        """
         Construct the utility tree.
        :return:
        """
        # construct the first layer firstly (depth = 1)
        self._attachNode(cur_depth=1,
                         ignore=True if self.args.ignore_depth > 0 else False)  # attach all children of the root (depth = 1)
        self.node_queue.append(None)  # the end of layer with depth = 1
        self.node_queue.popleft()
        self.current_node = self.node_queue.popleft()
        cur_depth = 2
        # construct the other parts
        while cur_depth <= self.args.depth:
            if cur_depth <= self.args.ignore_depth:
                ignore = True
            else:
                ignore = False
            while None != self.current_node:
                self._attachNode(cur_depth=cur_depth, ignore=ignore)
                self.current_node = self.node_queue.popleft()
            self.node_queue.append(None)
            if 0 == len(self.node_queue):
                break
            self.current_node = self.node_queue.popleft()
            cur_depth += 1

        # Add potential reward/risk for every path
        for each in self.root.leaves:
            if self.strategy_type == "global" or self.strategy_type == "approach":
                each.path_utility = each.cumulative_utility / each.cur_len
            elif self.strategy_type == "evade":
                # 向能走的最远的方向躲避，避免进入死胡同
                if each.cumulative_utility >= 0:
                    each.path_utility = each.cumulative_utility + each.cur_len
                else:
                    each.path_utility = each.cumulative_utility
            else:
                each.path_utility = each.cumulative_utility
        # Find the best path with the highest utility
        best_leaf = self.root.leaves[0]
        for leaf in self.root.leaves:
            if leaf.path_utility > best_leaf.path_utility:
                best_leaf = leaf
        highest_utility = best_leaf.path_utility
        best_path = best_leaf.ancestors
        best_path = [(each.name, each.dir_from_parent) for each in best_path[1:]]
        if best_path == []:  # only one step is taken
            best_path = [(best_leaf.name, best_leaf.dir_from_parent)]
        return self.root, highest_utility, best_path

    def _descendantUtility(self, node):
        leaves_utility = []
        for each in node.leaves:
            leaves_utility.append(each.path_utility)
        return sum(leaves_utility) / len(leaves_utility)

    def nextDir(self, return_Q=False):
        _, highest_utility, best_path = self._construct()
        available_directions = [each.dir_from_parent for each in self.root.children]
        available_dir_utility = np.array([self._descendantUtility(each) for each in self.root.children])
        for index, each in enumerate(available_directions):
            self.Q_value[self.mapStatus["dir_list"].index(each)] = available_dir_utility[index]
        unavailable_directions = [each for each in self.mapStatus["dir_list"] if each not in available_directions]
        for each in unavailable_directions:
            self.Q_value[self.mapStatus["dir_list"].index(each)] = -np.inf
        self.Q_value = np.array(self.Q_value)
        available_directions_index = [self.mapStatus["dir_list"].index(each) for each in available_directions]
        # self.Q_value[available_directions_index] += 1.0 # avoid 0 utility
        # Add randomness and laziness
        Q_scale = scaleOfNumber(np.max(np.abs(self.Q_value)))
        # randomness = np.random.normal(loc=0, scale=0.1, size=len(available_directions_index)) * Q_scale
        randomness = np.random.uniform(low=0, high=0.1, size=len(available_directions_index)) * Q_scale
        self.Q_value[available_directions_index] += (self.args.randomness_coeff * randomness)
        if self.gameStatus["last_dir"] is not None and self.mapStatus["dir_list"].index(
                self.gameStatus["last_dir"]) in available_directions_index:
            self.Q_value[self.mapStatus["dir_list"].index(self.gameStatus["last_dir"])] += (
                    self.args.laziness_coeff * Q_scale)
        if return_Q:
            return best_path[0][1], self.Q_value
        else:
            return best_path[0][1]


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
    strategy_type = "evade"

    args = argparser()
    if strategy_type == "local":
        args.depth = 10
        args.ghost_attractive_thr = 10
        args.ghost_repulsive_thr = 10
        args.reward_coeff = 1.0
        args.risk_coeff = 0.0
    elif strategy_type == "global":
        args.depth = 15
        args.ignore_depth = 5
        args.ghost_attractive_thr = 34
        args.ghost_repulsive_thr = 34
        args.reward_coeff = 1.0
        args.risk_coeff = 0.0
    elif strategy_type == "evade":
        args.depth = 5
        args.ignore_depth = 0
        args.ghost_attractive_thr = 34
        args.ghost_repulsive_thr = 34
        args.reward_coeff = 0.0
        args.risk_coeff = 1.0
    elif strategy_type == "energizer":
        args.depth = 5
        args.ignore_depth = 0
        args.ghost_attractive_thr = 0
        args.ghost_repulsive_thr = 0
        args.reward_coeff = 1.0
        args.risk_coeff = 0.0
    elif strategy_type == "approach":
        args.depth = 10
        args.ignore_depth = 0
        args.ghost_attractive_thr = 10
        args.ghost_repulsive_thr = 10
        args.reward_coeff = 1.0
        args.risk_coeff = 0.0

    strategy = Strategy(strategy_type, adjacent_data, locs_df, reward_amount, args)
    for index in range(len(result)):
        # print(index)
        cur_pos = result["pacmanPos"][index]
        ghost_data = [result["ghost1Pos"][index], result["ghost2Pos"][index]]
        ghost_status = [result["ifscared1"][index], result["ifscared2"][index]]
        energizer_data = result["energizers"][index]
        bean_data = result["beans"][index]
        last_dir = result["pacman_dir"][index]

        strategy.set_state(cur_pos, energizer_data, bean_data, ghost_data, ghost_status, last_dir)
        _, Q = strategy.nextDir(return_Q=True)
        choice = strategy.mapStatus["dir_list"][makeChoice(Q)]
        # print(strategy_type + " Choice : ", choice, Q)
