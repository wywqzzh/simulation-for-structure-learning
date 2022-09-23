import pandas as pd
import numpy as np
import sys

sys.path.append("../../Utils/")
from Utils.FileUtils import readAdjacentMap, readLocDistance, readAdjacentPath

inf_val = 50  # 空数据的默认值（e.g., 地图中不存在水果时，Pacman和水果的距离就设置为50）
reborn_pos = (14, 27)  # Pacman重生位置


def _ghostModeDist(ifscared1, ifscared2, PG1, PG2, mode):
    '''
    在考虑鬼的状态下，Pacman和鬼的距离
    '''
    if mode == "normal":
        ifscared1 = ifscared1.apply(lambda x: x < 3)
        ifscared2 = ifscared2.apply(lambda x: x < 3)
    elif mode == "scared":
        ifscared1 = ifscared1.apply(lambda x: x > 3)
        ifscared2 = ifscared2.apply(lambda x: x > 3)
    else:
        raise ValueError("Undefined ghost mode {}!".format(mode))
    res = []
    for i in range(ifscared1.shape[0]):
        ind = np.where(np.array([ifscared1[i], ifscared2[i]]) == True)[0]
        res.append(np.min(np.array([PG1[i], PG2[i]])[ind]) if len(ind) > 0 else inf_val)
    return pd.Series(res)


def _adjacentBeans(pacmanPos, beans, type, locs_df):
    '''
    Pacman某个相邻位置和豆子的距离
    '''
    if isinstance(pacmanPos, float):
        return 0
    # Pacman in tunnel
    if pacmanPos == (29, 18):
        pacmanPos = (28, 18)
    if pacmanPos == (0, 18):
        pacmanPos = (1, 18)
    # Find adjacent positions
    if type == "left":
        adjacent = (pacmanPos[0] - 1, pacmanPos[1])
    elif type == "right":
        adjacent = (pacmanPos[0] + 1, pacmanPos[1])
    elif type == "up":
        adjacent = (pacmanPos[0], pacmanPos[1] - 1)
    elif type == "down":
        adjacent = (pacmanPos[0], pacmanPos[1] + 1)
    else:
        raise ValueError("Undefined direction {}!".format(type))
    # Adjacent beans num
    if adjacent not in locs_df:
        bean_num = 0
    else:
        bean_num = (
            0 if isinstance(beans, float) else len(np.where(
                np.array([0 if adjacent == each else locs_df[adjacent][each] for each in beans]) <= 10)[0]
                                                   )
        )
    return bean_num


def _processPacmanPos(pacmanPos):
    if pacmanPos == (30, 18):
        pacmanPos = (29, 18)
    if pacmanPos == (-1, 18):
        pacmanPos == (0, 18)
    return pacmanPos


def _adjacentDist(pacmanPos, ghostPos, type, adjacent_data, locs_df):
    '''
    Pacman某个相邻位置和鬼的距离
    '''
    if isinstance(ghostPos, list):
        ghostPos = ghostPos[0]
    # Pacman in tunnel
    if pacmanPos == (30, 18):
        pacmanPos = (29, 18)
    if pacmanPos == (-1, 18):
        pacmanPos = (0, 18)
    if isinstance(pacmanPos, float) or isinstance(adjacent_data[pacmanPos][type], float):
        return inf_val
    # Find adjacent positions
    if type == "left":
        adjacent = (pacmanPos[0] - 1, pacmanPos[1])
    elif type == "right":
        adjacent = (pacmanPos[0] + 1, pacmanPos[1])
    elif type == "up":
        adjacent = (pacmanPos[0], pacmanPos[1] - 1)
    elif type == "down":
        adjacent = (pacmanPos[0], pacmanPos[1] + 1)
    else:
        raise ValueError("Undefined direction {}!".format(type))
    # Adjacent positions in the tunnel
    # 因为map和dij distance的数据中没有tunnel内的数据，所以这部分的目的是把处于tunnel里的位置强制设定到地图边界上
    # 猴子数据和人的数据中tunnel位置不同，需要按照具体情况修改这部分代码
    if adjacent == (-2, 18):
        adjacent = (0, 18)
    if adjacent == (-1, 18):
        adjacent = (0, 18)
    if adjacent == (30, 18):
        adjacent = (29, 18)
    if adjacent == (31, 18):
        adjacent = (29, 18)
    return 0 if adjacent == ghostPos else locs_df[adjacent][ghostPos]


def extractBehaviorFeature(trial):
    '''
    从数据中提取行为相关的数据，包括Pacman-energizer距离、pacman-ghost距离、眼动信息等。
    '''
    locs_df = readLocDistance("../Data/constant/dij_distance_map.csv")
    adjacent_data = readAdjacentMap("../Data/constant/adjacent_map.csv")
    trial = trial.reset_index(drop=True)
    trial.pacmanPos = trial.pacmanPos.apply(_processPacmanPos)
    # ---------------------------------------------
    # Features for the estimation
    # Pacman-Blinky distance
    PG1_left = trial[["pacmanPos", "ghost1Pos"]].apply(
        lambda x: _adjacentDist(x.pacmanPos, x.ghost1Pos, "left", adjacent_data, locs_df),
        axis=1
    )
    PG1_right = trial[["pacmanPos", "ghost1Pos"]].apply(
        lambda x: _adjacentDist(x.pacmanPos, x.ghost1Pos, "right", adjacent_data, locs_df),
        axis=1
    )
    PG1_up = trial[["pacmanPos", "ghost1Pos"]].apply(
        lambda x: _adjacentDist(x.pacmanPos, x.ghost1Pos, "up", adjacent_data, locs_df),
        axis=1
    )
    PG1_down = trial[["pacmanPos", "ghost1Pos"]].apply(
        lambda x: _adjacentDist(x.pacmanPos, x.ghost1Pos, "down", adjacent_data, locs_df),
        axis=1
    )
    # Pacman-Clyde distance
    PG2_left = trial[["pacmanPos", "ghost2Pos"]].apply(
        lambda x: _adjacentDist(x.pacmanPos, x.ghost2Pos, "left", adjacent_data, locs_df),
        axis=1
    )
    PG2_right = trial[["pacmanPos", "ghost2Pos"]].apply(
        lambda x: _adjacentDist(x.pacmanPos, x.ghost2Pos, "right", adjacent_data, locs_df),
        axis=1
    )
    PG2_up = trial[["pacmanPos", "ghost2Pos"]].apply(
        lambda x: _adjacentDist(x.pacmanPos, x.ghost2Pos, "up", adjacent_data, locs_df),
        axis=1
    )
    PG2_down = trial[["pacmanPos", "ghost2Pos"]].apply(
        lambda x: _adjacentDist(x.pacmanPos, x.ghost2Pos, "down", adjacent_data, locs_df),
        axis=1
    )
    # Pacman-energizer distance
    PE_left = trial[["pacmanPos", "energizers"]].apply(
        lambda x: inf_val if isinstance(x.energizers, float)
        else np.min(
            [_adjacentDist(x.pacmanPos, each, "left", adjacent_data, locs_df) for each in x.energizers]),
        axis=1
    )
    PE_right = trial[["pacmanPos", "energizers"]].apply(
        lambda x: inf_val if isinstance(x.energizers, float)
        else np.min(
            [_adjacentDist(x.pacmanPos, each, "right", adjacent_data, locs_df) for each in x.energizers]),
        axis=1
    )
    PE_up = trial[["pacmanPos", "energizers"]].apply(
        lambda x: inf_val if isinstance(x.energizers, float)
        else np.min([_adjacentDist(x.pacmanPos, each, "up", adjacent_data, locs_df) for each in x.energizers]),
        axis=1
    )
    PE_down = trial[["pacmanPos", "energizers"]].apply(
        lambda x: inf_val if isinstance(x.energizers, float)
        else np.min(
            [_adjacentDist(x.pacmanPos, each, "down", adjacent_data, locs_df) for each in x.energizers]),
        axis=1
    )
    # Pacman-fruit distance
    PF_left = trial[["pacmanPos", "fruitPos"]].apply(
        lambda x: inf_val if isinstance(x.fruitPos, float)
        else _adjacentDist(x.pacmanPos, x.fruitPos, "left", adjacent_data, locs_df),
        axis=1
    )
    PF_right = trial[["pacmanPos", "fruitPos"]].apply(
        lambda x: inf_val if isinstance(x.fruitPos, float)
        else _adjacentDist(x.pacmanPos, x.fruitPos, "right", adjacent_data, locs_df),
        axis=1
    )
    PF_up = trial[["pacmanPos", "fruitPos"]].apply(
        lambda x: inf_val if isinstance(x.fruitPos, float)
        else _adjacentDist(x.pacmanPos, x.fruitPos, "up", adjacent_data, locs_df),
        axis=1
    )
    PF_down = trial[["pacmanPos", "fruitPos"]].apply(
        lambda x: inf_val if isinstance(x.fruitPos, float)
        else _adjacentDist(x.pacmanPos, x.fruitPos, "down", adjacent_data, locs_df),
        axis=1
    )
    # Pacman附近5步内豆子数
    beans_5step = trial[["pacmanPos", "beans"]].apply(
        lambda x: 0 if isinstance(x.beans, float)
        else len(
            np.where(
                np.array([0 if x.pacmanPos == each
                          else locs_df[x.pacmanPos][each] for each in x.beans]) <= 5
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
                           else locs_df[x.pacmanPos][each] for each in x.beans]) > 5) &
                (np.array([0 if x.pacmanPos == each
                           else locs_df[x.pacmanPos][each] for each in x.beans]) <= 10)
            )[0]
        ),
        axis=1
    )
    # Pacman附近10步外豆子数
    beans_over_10step = trial[["pacmanPos", "beans"]].apply(
        lambda x: 0 if isinstance(x.beans, float)
        else len(
            np.where(
                np.array([0 if x.pacmanPos == each else locs_df[x.pacmanPos][each] for each in x.beans]) > 10
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
    # 重生位置附近10步豆子数减去Pacman当前位置10步内豆子数
    beans_diff = trial[["pacmanPos", "beans"]].apply(
        lambda x: 0 if isinstance(x.beans, float)
        else np.sum(
            np.where(
                np.array([0 if reborn_pos == each else locs_df[reborn_pos][each] for each in x.beans]) <= 10
            )[0]
        ) - np.sum(
            np.where(
                np.array([0 if x.pacmanPos == each else locs_df[x.pacmanPos][each] for each in x.beans]) <= 10
            )[0]
        ),
        axis=1
    )
    # Pacman四个相邻位置10步内豆子数
    beans_left = trial[["pacmanPos", "beans"]].apply(
        lambda x: _adjacentBeans(x.pacmanPos, x.beans, "left", locs_df),
        axis=1
    )
    beans_right = trial[["pacmanPos", "beans"]].apply(
        lambda x: _adjacentBeans(x.pacmanPos, x.beans, "right", locs_df),
        axis=1
    )
    beans_up = trial[["pacmanPos", "beans"]].apply(
        lambda x: _adjacentBeans(x.pacmanPos, x.beans, "up", locs_df),
        axis=1
    )
    beans_down = trial[["pacmanPos", "beans"]].apply(
        lambda x: _adjacentBeans(x.pacmanPos, x.beans, "down", locs_df),
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

            "PG_normal_left": _ghostModeDist(trial.ifscared1, trial.ifscared2, PG1_left, PG2_left, "normal"),
            "PG_normal_right": _ghostModeDist(trial.ifscared1, trial.ifscared2, PG1_right, PG2_right, "normal"),
            "PG_normal_up": _ghostModeDist(trial.ifscared1, trial.ifscared2, PG1_up, PG2_up, "normal"),
            "PG_normal_down": _ghostModeDist(trial.ifscared1, trial.ifscared2, PG1_down, PG2_down, "normal"),

            "PG_scared_left": _ghostModeDist(trial.ifscared1, trial.ifscared2, PG1_left, PG2_left, "scared"),
            "PG_scared_right": _ghostModeDist(trial.ifscared1, trial.ifscared2, PG1_right, PG2_right, "scared"),
            "PG_scared_up": _ghostModeDist(trial.ifscared1, trial.ifscared2, PG1_up, PG2_up, "scared"),
            "PG_scared_down": _ghostModeDist(trial.ifscared1, trial.ifscared2, PG1_down, PG2_down, "scared"),

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

            "PF_left": PF_left,
            "PF_right": PF_right,
            "PF_up": PF_up,
            "PF_down": PF_down,

            "beans_left": beans_left,
            "beans_right": beans_right,
            "beans_up": beans_up,
            "beans_down": beans_down,

            "beans_within_5": beans_5step,
            "beans_between_5and10": beans_5to10step,
            "beans_beyond_10": beans_over_10step,
            "beans_num": beans_num,
            "beans_diff": beans_diff,

            "true_dir": trial.pacman_dir,
        }
    )
    return processed_trial_data

def predictor4Prediction(feature_data):
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
    df["PF"] = df[["PF_{}".format(d) for d in dir_list]].apply(lambda x: x.min(), axis=1)
    # ghost features
    df["PG1"] = df[["PG1_{}".format(d) for d in dir_list]].apply(lambda x: x.min(), axis=1)
    df["PG2"] = df[["PG2_{}".format(d) for d in dir_list]].apply(lambda x: x.min(), axis=1)
    df.loc[df.ifscared1 == 3, "PG1"] = np.nan
    df.loc[df.ifscared2 == 3, "PG2"] = np.nan
    df["if_normal1"] = (df.ifscared1 <= 2).astype(int)
    df["if_dead1"] = (df.ifscared1 == 3).astype(int)
    df["if_scared1"] = (df.ifscared1 >= 4).astype(int)
    df["if_normal2"] = (df.ifscared2 <= 2).astype(int)
    df["if_dead2"] = (df.ifscared2 == 3).astype(int)
    df["if_scared2"] = (df.ifscared2 >= 4).astype(int)
    predictors = df[[
        "PG1", "PG2", "PE", "PF",
        "beans_dir", "beans_num", "beans_within_5", "beans_between_5and10", "beans_beyond_10", "beans_diff",
        "if_scared1", "if_scared2", "if_normal1", "if_normal2", "if_dead1", "if_dead2"]]
    # normalization
    continuous_cols = [
        "PG1", "PG2", "PE", "PF",
        "beans_dir", "beans_num", "beans_within_5", "beans_between_5and10", "beans_beyond_10", "beans_diff"
    ]
    category_cols = ["if_scared1", "if_scared2", "if_normal1", "if_normal2", "if_dead1", "if_dead2"]
    predictors[continuous_cols] = predictors[continuous_cols] / predictors[continuous_cols].max()
    predictors.loc[predictors.beans_num > 0.1, "beans_diff"] = np.nan
    for i in category_cols:
        predictors[i] = predictors[i].astype(int).astype("category")
    return predictors



if __name__ == '__main__':
    data = pd.read_pickle("../Data/10_trial_data_Omega.pkl")
    behavior_features = extractBehaviorFeature(data)
    predictors = predictor4Prediction(behavior_features)
    x = 0
