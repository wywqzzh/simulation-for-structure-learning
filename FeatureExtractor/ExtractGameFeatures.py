import pandas as pd
import numpy as np
import sys
from environment import layout
from copy import deepcopy

sys.path.append("../../Utils/")
from Utils.FileUtils import readAdjacentMap, readLocDistance, readAdjacentPath

inf_val = 50  # 空数据的默认值（e.g., 地图中不存在水果时，Pacman和水果的距离就设置为50）


# reborn_pos = (14, 27)  # Pacman重生位置


class featureExtractor:
    def __init__(self, map_name=None):
        self.map_name = map_name
        self.locs_df = readLocDistance("../Data/mapMsg/dij_distance_map_" + map_name + ".csv")
        self.adjacent_data = readAdjacentMap("../Data/mapMsg/adjacent_map_" + map_name + ".csv")
        self.layout = layout.getLayout(map_name)
        self.layout_h = len(self.layout.layoutText)
        self.layout_w = len(self.layout.layoutText[0])
        self.get_map_const()

    def _ghostModeDist(self, ifscared1, ifscared2, PG1, PG2, mode):
        '''
        在考虑鬼的状态下，Pacman和鬼的距离
        '''
        if mode == "normal":
            ifscared1 = ifscared1.apply(lambda x: x < 1)
            ifscared2 = ifscared2.apply(lambda x: x < 1)
        elif mode == "scared":
            ifscared1 = ifscared1.apply(lambda x: x > 1)
            ifscared2 = ifscared2.apply(lambda x: x > 1)
        else:
            raise ValueError("Undefined ghost mode {}!".format(mode))
        res = []
        for i in range(ifscared1.shape[0]):
            ind = np.where(np.array([ifscared1[i], ifscared2[i]]) == True)[0]
            res.append(np.min(np.array([PG1[i], PG2[i]])[ind]) if len(ind) > 0 else inf_val)
        return pd.Series(res)

    def _adjacentBeans(self, pacmanPos, beans, type):
        '''
        Pacman某个相邻位置和豆子的距离
        '''
        # print("_adjacentBeans pacmanPos:", pacmanPos)
        if isinstance(pacmanPos, float):
            return 0
        # Find adjacent positions
        if type == "left":
            adjacent = ((pacmanPos[0] - 1 - 1) % self.layout_w + 1, pacmanPos[1])
        elif type == "right":
            adjacent = ((pacmanPos[0] - 1 + 1) % self.layout_w + 1, pacmanPos[1])
        elif type == "up":
            adjacent = (pacmanPos[0], pacmanPos[1] - 1)
        elif type == "down":
            adjacent = (pacmanPos[0], pacmanPos[1] + 1)
        else:
            raise ValueError("Undefined direction {}!".format(type))
        # Adjacent beans num
        if adjacent not in self.locs_df:
            bean_num = 0
        else:
            bean_num = (
                0 if isinstance(beans, float) else len(np.where(
                    np.array([0 if adjacent == each else self.locs_df[adjacent][each] for each in beans]) <= 10)[0]
                                                       )
            )
        return bean_num

    def _adjacentDist(self, pacmanPos, ghostPos, type, adjacent_data, locs_df):
        '''
        Pacman某个相邻位置和鬼的距离
        '''
        # print("_adjacentDist pacmanPos:", pacmanPos)
        if isinstance(ghostPos, list):
            ghostPos = ghostPos[0]
        if isinstance(pacmanPos, float) or isinstance(adjacent_data[pacmanPos][type], float):
            return inf_val
        # Find adjacent positions
        if type == "left":
            adjacent = ((pacmanPos[0] - 1 - 1) % self.layout_w + 1, pacmanPos[1])
        elif type == "right":
            adjacent = ((pacmanPos[0] - 1 + 1) % self.layout_w + 1, pacmanPos[1])
        elif type == "up":
            adjacent = (pacmanPos[0], pacmanPos[1] - 1)
        elif type == "down":
            adjacent = (pacmanPos[0], pacmanPos[1] + 1)
        else:
            raise ValueError("Undefined direction {}!".format(type))
        ghostPos = (int(ghostPos[0]), int(ghostPos[1]))
        return 0 if adjacent == ghostPos else locs_df[adjacent][ghostPos]

    def extractBehaviorFeature(self, trial):
        '''
        从数据中提取行为相关的数据，包括Pacman-energizer距离、pacman-ghost距离、眼动信息等。
        '''
        trial = trial.reset_index(drop=True)
        PG1_left = trial[["pacmanPos", "ghost1Pos"]].apply(
            lambda x: self._adjacentDist(x.pacmanPos, x.ghost1Pos, "left", self.adjacent_data, self.locs_df),
            axis=1
        )
        PG1_right = trial[["pacmanPos", "ghost1Pos"]].apply(
            lambda x: self._adjacentDist(x.pacmanPos, x.ghost1Pos, "right", self.adjacent_data, self.locs_df),
            axis=1
        )
        PG1_up = trial[["pacmanPos", "ghost1Pos"]].apply(
            lambda x: self._adjacentDist(x.pacmanPos, x.ghost1Pos, "up", self.adjacent_data, self.locs_df),
            axis=1
        )
        PG1_down = trial[["pacmanPos", "ghost1Pos"]].apply(
            lambda x: self._adjacentDist(x.pacmanPos, x.ghost1Pos, "down", self.adjacent_data, self.locs_df),
            axis=1
        )
        # Pacman-Clyde distance
        PG2_left = trial[["pacmanPos", "ghost2Pos"]].apply(
            lambda x: self._adjacentDist(x.pacmanPos, x.ghost2Pos, "left", self.adjacent_data, self.locs_df),
            axis=1
        )
        PG2_right = trial[["pacmanPos", "ghost2Pos"]].apply(
            lambda x: self._adjacentDist(x.pacmanPos, x.ghost2Pos, "right", self.adjacent_data, self.locs_df),
            axis=1
        )
        PG2_up = trial[["pacmanPos", "ghost2Pos"]].apply(
            lambda x: self._adjacentDist(x.pacmanPos, x.ghost2Pos, "up", self.adjacent_data, self.locs_df),
            axis=1
        )
        PG2_down = trial[["pacmanPos", "ghost2Pos"]].apply(
            lambda x: self._adjacentDist(x.pacmanPos, x.ghost2Pos, "down", self.adjacent_data, self.locs_df),
            axis=1
        )
        # Pacman-energizer distance
        PE_left = trial[["pacmanPos", "energizers"]].apply(
            lambda x: inf_val if isinstance(x.energizers, float)
            else np.min(
                [self._adjacentDist(x.pacmanPos, each, "left", self.adjacent_data, self.locs_df) for each in
                 x.energizers]),
            axis=1
        )
        PE_right = trial[["pacmanPos", "energizers"]].apply(
            lambda x: inf_val if isinstance(x.energizers, float)
            else np.min(
                [self._adjacentDist(x.pacmanPos, each, "right", self.adjacent_data, self.locs_df) for each in
                 x.energizers]),
            axis=1
        )
        PE_up = trial[["pacmanPos", "energizers"]].apply(
            lambda x: inf_val if isinstance(x.energizers, float)
            else np.min([self._adjacentDist(x.pacmanPos, each, "up", self.adjacent_data, self.locs_df) for each in
                         x.energizers]),
            axis=1
        )
        PE_down = trial[["pacmanPos", "energizers"]].apply(
            lambda x: inf_val if isinstance(x.energizers, float)
            else np.min(
                [self._adjacentDist(x.pacmanPos, each, "down", self.adjacent_data, self.locs_df) for each in
                 x.energizers]),
            axis=1
        )
        # Pacman附近5步内豆子数
        beans_5step = trial[["pacmanPos", "beans"]].apply(
            lambda x: 0 if isinstance(x.beans, float)
            else len(
                np.where(
                    np.array([0 if x.pacmanPos == each
                              else self.locs_df[x.pacmanPos][each] for each in x.beans]) <= 5
                )[0]
            ),
            axis=1
        )
        beans_10step = trial[["pacmanPos", "beans"]].apply(
            lambda x: 0 if isinstance(x.beans, float)
            else len(
                np.where(
                    np.array([0 if x.pacmanPos == each
                              else self.locs_df[x.pacmanPos][each] for each in x.beans]) <= 10
                )[0]
            ),
            axis=1
        )
        # Pacman附近5~10步内豆子数
        beans_5to10step = trial[["pacmanPos", "beans"]].apply(
            lambda x: 0 if isinstance(x.beans, float)
            else len(
                np.where(
                    (np.array([0 if x.pacmanPos == each
                               else self.locs_df[x.pacmanPos][each] for each in x.beans]) > 5) &
                    (np.array([0 if x.pacmanPos == each
                               else self.locs_df[x.pacmanPos][each] for each in x.beans]) <= 10)
                )[0]
            ),
            axis=1
        )
        # Pacman附近10步外豆子数
        beans_over_10step = trial[["pacmanPos", "beans"]].apply(
            lambda x: 0 if isinstance(x.beans, float)
            else len(
                np.where(
                    np.array([0 if x.pacmanPos == each else self.locs_df[x.pacmanPos][each] for each in x.beans]) > 10
                )[0]
            ),
            axis=1
        )
        # 盘面豆子总数
        beans_num = trial[["beans"]].apply(
            lambda x: 0 if isinstance(x.beans, float)
            else len(x.beans),
            axis=1
        )

        # Pacman四个相邻位置10步内豆子数
        beans_left = trial[["pacmanPos", "beans"]].apply(
            lambda x: self._adjacentBeans(x.pacmanPos, x.beans, "left"),
            axis=1
        )
        beans_right = trial[["pacmanPos", "beans"]].apply(
            lambda x: self._adjacentBeans(x.pacmanPos, x.beans, "right"),
            axis=1
        )
        beans_up = trial[["pacmanPos", "beans"]].apply(
            lambda x: self._adjacentBeans(x.pacmanPos, x.beans, "up"),
            axis=1
        )
        beans_down = trial[["pacmanPos", "beans"]].apply(
            lambda x: self._adjacentBeans(x.pacmanPos, x.beans, "down"),
            axis=1
        )
        # 鬼的状态
        ifscared1 = trial.ifscared1
        ifscared2 = trial.ifscared2
        processed_trial_data = pd.DataFrame(
            data=
            {
                "ifscared1": ifscared1,
                "ifscared2": ifscared2,

                "PG_normal_left": self._ghostModeDist(trial.ifscared1, trial.ifscared2, PG1_left, PG2_left, "normal"),
                "PG_normal_right": self._ghostModeDist(trial.ifscared1, trial.ifscared2, PG1_right, PG2_right,
                                                       "normal"),
                "PG_normal_up": self._ghostModeDist(trial.ifscared1, trial.ifscared2, PG1_up, PG2_up, "normal"),
                "PG_normal_down": self._ghostModeDist(trial.ifscared1, trial.ifscared2, PG1_down, PG2_down, "normal"),

                "PG_scared_left": self._ghostModeDist(trial.ifscared1, trial.ifscared2, PG1_left, PG2_left, "scared"),
                "PG_scared_right": self._ghostModeDist(trial.ifscared1, trial.ifscared2, PG1_right, PG2_right,
                                                       "scared"),
                "PG_scared_up": self._ghostModeDist(trial.ifscared1, trial.ifscared2, PG1_up, PG2_up, "scared"),
                "PG_scared_down": self._ghostModeDist(trial.ifscared1, trial.ifscared2, PG1_down, PG2_down, "scared"),

                "PG1_left": PG1_left,
                "PG1_right": PG1_right,
                "PG1_up": PG1_up,
                "PG1_down": PG1_down,

                "PG2_left": PG2_left,
                "PG2_right": PG2_right,
                "PG2_up": PG2_up,
                "PG2_down": PG2_down,

                "PE_left": PE_left,
                "PE_right": PE_right,
                "PE_up": PE_up,
                "PE_down": PE_down,

                "beans_left": beans_left,
                "beans_right": beans_right,
                "beans_up": beans_up,
                "beans_down": beans_down,

                "beans_within_5": beans_5step,
                "beans_within_10": beans_10step,
                "beans_between_5and10": beans_5to10step,
                "beans_beyond_10": beans_over_10step,
                "beans_num": beans_num,
                # "beans_diff": beans_diff,

                "true_dir": trial.pacman_dir,
            }
        )
        self.behavior_features = processed_trial_data

    def predictor4Prediction(self):
        feature_data = deepcopy(self.behavior_features)
        '''
        用来预测的属性。
        '''
        dir_list = ["left", "right", "up", "down"]
        df = feature_data.copy()
        # 移动方向的beans数量
        df["beans_dir"] = [
            df.loc[idx, "beans_" + i] if type(i) == str else np.nan
            for idx, i in df.true_dir.iteritems()
        ]
        # reward features
        df["PE"] = df[["PE_{}".format(d) for d in dir_list]].apply(lambda x: x.min(), axis=1)
        # ghost features
        df["PG1"] = df[["PG1_{}".format(d) for d in dir_list]].apply(lambda x: x.min(), axis=1)
        df["PG2"] = df[["PG2_{}".format(d) for d in dir_list]].apply(lambda x: x.min(), axis=1)
        if np.array(df["ifscared1"] == 1)[0]:
            x = 0
        if np.array(df["ifscared2"] == 1)[0]:
            x = 0
        df.loc[df.ifscared1 == 1, "PG1"] = inf_val
        df.loc[df.ifscared2 == 1, "PG2"] = inf_val
        df["if_normal1"] = (df.ifscared1 <= 0).astype(int)
        df["if_dead1"] = (df.ifscared1 == 1).astype(int)
        df["if_scared1"] = (df.ifscared1 >= 2).astype(int)
        df["if_normal2"] = (df.ifscared2 <= 0).astype(int)
        df["if_dead2"] = (df.ifscared2 == 1).astype(int)
        df["if_scared2"] = (df.ifscared2 >= 2).astype(int)
        df["zero_beans_within_10"] = (df.beans_within_10 == 0).astype(int)
        df["zero_beans_beyond_10"] = (df.beans_beyond_10 == 0).astype(int)
        predictors = df[[
            "PG1", "PG2", "PE",
            "beans_dir", "beans_num", "beans_within_10", "beans_between_5and10", "beans_beyond_10",
            "if_scared1", "if_scared2", "if_normal1", "if_normal2", "if_dead1", "if_dead2", "zero_beans_within_10",
            "zero_beans_beyond_10"]]
        # normalization
        continuous_cols = [
            "PG1", "PG2", "PE",
            "beans_dir", "beans_num", "beans_within_10", "beans_between_5and10", "beans_beyond_10"
        ]
        max_beans = self.map_num_const["max_beans"]
        max_dist = self.map_num_const["max_dist"]
        continuous_col_max = np.array(
            [max_dist] * 3 + [int(max_beans / 4), max_beans, min(40, max_beans), min(75, max_beans), max_beans])
        category_cols = ["if_scared1", "if_scared2", "if_normal1", "if_normal2", "if_dead1", "if_dead2"]
        # predictors[continuous_cols] = predictors[continuous_cols] / predictors[continuous_cols].max()
        predictors[continuous_cols] = predictors[continuous_cols] / continuous_col_max
        # predictors.loc[predictors.beans_num > 0.1, "beans_diff"] = np.nan
        for i in category_cols:
            predictors[i] = predictors[i].astype(int).astype("category")

        return predictors

    def get_map_const(self):

        max_dist = self.layout_w + self.layout_h
        max_beans = len(np.where(np.array(self.layout.food.data) == True)[0])
        self.map_num_const = {
            "max_dist": max_dist,
            "max_beans": max_beans
        }

    def discretize(self, data):
        numerical_cols1 = ["PG1", "PG2", "PE", "beans_within_10", "beans_beyond_10"]
        bin = [0, 0.1, 0.4, 10]
        numerical_encode1 = pd.concat(
            [pd.cut(data[i], bin, right=False, labels=[0, 1, 2]) for i in numerical_cols1
             ],
            axis=1,
        )
        numerical_encode1.columns = numerical_cols1
        # numerical_cols2 = []
        # print(data["beans_within_10"])
        # bin = [0, 0.33, 0.67, 10]
        # numerical_encode2 = pd.concat(
        #     [pd.cut(data[i], bin, right=False, labels=[0, 1, 2]) for i in numerical_cols2
        #      ],
        #     axis=1,
        # )
        # numerical_encode2.columns = numerical_cols2
        numerical_encode = pd.DataFrame()
        numerical_encode[numerical_cols1] = numerical_encode1[numerical_cols1]
        # numerical_encode[numerical_cols2] = numerical_encode2[numerical_cols2]
        is_encode = pd.DataFrame()
        for ghost in [1, 2]:
            cols = ["if_" + i + str(ghost) for i in ["normal", "dead", "scared"]]
            is_encode["GS" + str(ghost)] = np.argmax(data[cols].values, 1)

        numerical_encode[["GS1", "GS2"]] = is_encode[["GS1", "GS2"]]
        numerical_encode[["zero_beans_within_10", "zero_beans_beyond_10"]] = data[
            ["zero_beans_within_10", "zero_beans_beyond_10"]]
        numerical_cols = ["PG1", "PG2", "PE", "BN5", "BN10"]
        numerical_encode.columns = numerical_cols + ["GS1", "GS2", "ZBN5", "ZBN10"]
        numerical_encode = numerical_encode[["PG1", "GS1", "PG2", "GS2", "PE", "BN5", "BN10", "ZBN5", "ZBN10"]]
        print(numerical_encode)
        feature = numerical_encode.iloc[0].to_dict()
        return feature

    def extract_feature(self, data):
        self.extractBehaviorFeature(data)
        predictors = self.predictor4Prediction()
        feature = self.discretize(predictors)
        print(feature)
        return feature
# if __name__ == '__main__':
#     data = pd.read_pickle("../Data/10_trial_data_Omega.pkl")
#     behavior_features = extractBehaviorFeature(data)
#     predictors = predictor4Prediction(behavior_features)
