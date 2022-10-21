import sys

sys.path.append("../")
from primitiveStrategy.Strategy import *
from Utils.FileUtils import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath
import pandas as pd
import sys
from functools import partial
import multiprocessing

sys.path.append("../environment")
from environment import layout


class utilityEstimator:
    def __init__(self, map_name="originalClassic"):
        self.get_strategies(map_name=map_name)

    def get_args(self):
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

    def get_paramater_of_strategy(self, strategy_name, h, w):
        args = self.get_args()
        L = max(h, w)
        if strategy_name == "local":
            args.depth = 10
        elif strategy_name == "global":
            args.depth = L
            args.ignore_depth = 0
        elif strategy_name == "evade":
            args.depth = 3
            args.risk_coeff = 1
            args.reward_coeff = 0
        elif strategy_name == "energizer":
            args.depth = 10
        elif strategy_name == "approach":
            args.depth = 15
        elif strategy_name == "counterattack":
            args.depth = 10
        return args

    def get_strategies(self, map_name):
        # import layout
        from environment import layout
        layout = layout.getLayout(map_name).layoutText
        h = len(layout)
        w = len(layout[0])
        self.layout_h = h
        locs_df = readLocDistance("../Data/mapMsg/dij_distance_map_" + map_name + ".csv")
        adjacent_data = readAdjacentMap("../Data/mapMsg/adjacent_map_" + map_name + ".csv")
        self.intersection_data = pd.read_pickle("../Data/mapMsg/intersection_map_" + map_name + ".pkl")["pos"]
        reward_amount = readRewardAmount()

        local_strategy = Strategy("local", adjacent_data, locs_df, reward_amount,
                                  self.get_paramater_of_strategy("local", h, w))
        global_strategy = Strategy("global", adjacent_data, locs_df, reward_amount,
                                   self.get_paramater_of_strategy("global", h, w))
        evade_strategy = Strategy("evade", adjacent_data, locs_df, reward_amount,
                                  self.get_paramater_of_strategy("evade", h, w))
        energizer_strategy = Strategy("energizer", adjacent_data, locs_df, reward_amount,
                                      self.get_paramater_of_strategy("energizer", h, w))
        approach_strategy = Strategy("approach", adjacent_data, locs_df, reward_amount,
                                     self.get_paramater_of_strategy("approach", h, w))
        counterattack_strategy = Strategy("counterattack", adjacent_data, locs_df, reward_amount,
                                          self.get_paramater_of_strategy("approach", h, w))
        self.startegies = {
            "local": local_strategy, "global": global_strategy, "evade": evade_strategy,
            "energizer": energizer_strategy, "approach": approach_strategy
        }

    def get_Q(self, game_status):

        strategies_name = ["local", "global", "evade", "energizer", "approach"]
        strategy_Q = {"local": [], "global": [], "evade": [], "energizer": [], "approach": []}
        for strategy_name in strategies_name:
            if strategy_name == "evade":
                x = 0
            strategy = self.startegies[strategy_name]
            strategy.set_state(game_status)
            strategy.strategy_type = strategy_name
            _, Q = strategy.nextDir(return_Q=True)
            strategy_Q[strategy_name] = Q
        return strategy_Q


utility_estimator = utilityEstimator(map_name="originalClassic1")


def estimateUnitility_parallelize(game_status):
    Q = utility_estimator.get_Q(game_status)
    return Q


def estimateUnitility(filename):
    data = pd.read_pickle(filename)
    print(filename)
    local_Q = []
    global_Q = []
    energizer_Q = []
    approach_Q = []
    evade_Q = []
    game_status_ = []
    for i in range(len(data)):
        each = data.iloc[i]
        game_status = {
            "PacmanPos": each["pacmanPos"],
            "ghost_data": [each["ghost1Pos"], each["ghost2Pos"]],
            "ghost_status": [each["ifscared1"], each["ifscared2"]],
            "bean_data": each["beans"],
            "energizer_data": each["energizers"],
            "Reward": [],
            "last_dir": each["last_dir"]

        }
        game_status_.append(game_status)
    with multiprocessing.Pool(processes=12) as pool:
        Q = pool.map(partial(estimateUnitility_parallelize), game_status_)
    for q in Q:
        local_Q.append(q["local"])
        global_Q.append(q["global"])
        evade_Q.append(q["evade"])
        energizer_Q.append(q["energizer"])
        approach_Q.append(q["approach"])
    data["local_Q"] = local_Q
    data["global_Q"] = global_Q
    data["evade_Q"] = evade_Q
    data["energizer_Q"] = energizer_Q
    data["approach_Q"] = approach_Q
    data.to_pickle("../Data/process/bi-20_Q.pkl")
    pass


if __name__ == '__main__':
    filepath = "../Data/process/bi-20.pkl"
    estimateUnitility(filepath)

    # data=pd.read_pickle("../Data/process/10trial_Q.pkl")
    # temp_data=pd.read_pickle("../Data/process/10trial_gameStatus.pkl")
    # data["strategy"]=temp_data["strategy"]
    # data.to_pickle("../Data/process/10trial_Q.pkl")
