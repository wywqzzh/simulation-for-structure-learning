import numpy as np
import pandas as pd

from Utils.FileUtils import readAdjacentMap, readLocDistance, readAdjacentPath
import pickle


def skeleton():
    filepath = "../environment/layouts/originalClassic.lay"
    locs_df = readLocDistance("../Data/mapMsg/dij_distance_map_originalClassic.csv")
    adjacent_data = readAdjacentMap("../Data/mapMsg/adjacent_map_originalClassic.csv")
    adjacent_path = readAdjacentPath("../Data/mapMsg/dij_distance_map_originalClassic.csv")
    map_skeleton = []
    with open(filepath) as f:
        line = f.readline()
        while line:
            map_skeleton.append(list(line[:-1]))
            line = f.readline()

    map_skeleton = [['%' if i == '%' else ' ' for i in t] for t in map_skeleton]
    map_skeleton = np.array(map_skeleton)
    return map_skeleton, locs_df, adjacent_data, adjacent_path


def write_map(map):
    map = ["".join(m) + "\n" for m in map]
    filepath = "../environment/layouts/originalClassic_test.lay"
    with open(filepath, "w+") as f:
        for m in map:
            f.write(m)


def direction_two_position(pos1, pos2, L):
    if (pos1[0] + 1) % L == pos2[0]:
        return "right"
    elif (pos1[0] - 1) % L == pos2[0]:
        return "left"
    elif pos1[1] + 1 == pos2[1]:
        return "down"
    elif pos1[1] - 1 == pos2[1]:
        return "up"
    else:
        return None


def delete_inavailable(available, availabled):
    for p in availabled:
        if p in available:
            available.remove(p)

    return available


def eEA():
    map_skeleton, locs_df, adjacent_data, adjacent_path = skeleton()
    availabled_position = []

    # 随机挑选一个Pac-man的位置
    available_position = np.where(map_skeleton == ' ')
    available_position = list(zip(list(available_position[1] + 1), list(available_position[0] + 1)))
    need_resample = True
    while need_resample:
        choice_index = \
            np.random.choice(a=list(range(len(available_position))),
                             p=[1 / len(available_position)] * len(available_position),
                             size=1)[0]
        Pacman_position = available_position[choice_index]
        need_resample = False
        if not adjacent_data.__contains__(Pacman_position):
            need_resample = True
        else:
            values = np.array(list(adjacent_data[Pacman_position].values()))
            values = [isinstance(x, float) for x in values]
            # 如果该位置只有一个方向可以走则重新挑选Pac-man的位置
            if np.sum(values) < 2:
                need_resample = True
    availabled_position.append(Pacman_position)
    map_skeleton[Pacman_position[1] - 1, Pacman_position[0] - 1] = "P"

    # 挑选鬼1的位置
    distance = locs_df[Pacman_position]
    position = np.array(list(distance.keys()))
    distance = np.array(list(distance.values()))
    # 鬼1的位置在Pac-man3格以内
    available_position = position[np.where(distance <= 3)[0]]
    available_position = list(zip(available_position[:, 0], available_position[:, 1]))
    available_position = delete_inavailable(available_position, availabled_position)
    choice_index = \
        np.random.choice(a=list(range(len(available_position))),
                         p=[1 / len(available_position)] * len(available_position),
                         size=1)[0]
    ghost_position = available_position[choice_index]
    availabled_position.append(ghost_position)
    map_skeleton[ghost_position[1] - 1, ghost_position[0] - 1] = "G"

    # 挑选鬼2的位置
    distance = locs_df[Pacman_position]
    position = np.array(list(distance.keys()))
    distance = np.array(list(distance.values()))
    # 鬼2的位置在Pac-man15格以外
    available_position = position[np.where(distance <= 15)[0]]
    available_position = list(zip(available_position[:, 0], available_position[:, 1]))
    available_position = delete_inavailable(available_position, availabled_position)
    choice_index = \
        np.random.choice(a=list(range(len(available_position))),
                         p=[1 / len(available_position)] * len(available_position),
                         size=1)[0]
    ghost2_position = available_position[choice_index]
    availabled_position.append(ghost2_position)
    map_skeleton[ghost2_position[1] - 1, ghost2_position[0] - 1] = "G"

    # 判断鬼在Pacman的哪个方向
    path_index = np.where((adjacent_path.pos1 == Pacman_position) & (adjacent_path.pos2 == ghost_position))[0][0]
    path = adjacent_path["path"][path_index][0]
    direction = direction_two_position(path[0], path[1], map_skeleton.shape[1])

    # 挑选energizer位置
    available_position = position[np.where((distance > 3) & (distance < 6))[0]]
    available_position = list(zip(available_position[:, 0], available_position[:, 1]))
    available_position = delete_inavailable(available_position, availabled_position)
    # 删除与鬼的方向相同的点
    new_available_position = []
    for p in available_position:
        path_index = np.where((adjacent_path.pos1 == Pacman_position) & (adjacent_path.pos2 == p))[0][0]
        path = adjacent_path["path"][path_index][0]
        d = direction_two_position(path[0], path[1], map_skeleton.shape[1])
        if d != direction:
            new_available_position.append(p)
    available_position = new_available_position
    choice_index = \
        np.random.choice(a=list(range(len(available_position))),
                         p=[1 / len(available_position)] * len(available_position),
                         size=1)[0]
    energizer_position = available_position[choice_index]
    availabled_position.append(energizer_position)
    map_skeleton[energizer_position[1] - 1, energizer_position[0] - 1] = "o"

    # 生成豆子簇

    # 生成豆子中心
    available_position = position[np.where(distance > 10)[0]]
    available_position = list(zip(available_position[:, 0], available_position[:, 1]))
    available_position = delete_inavailable(available_position, availabled_position)
    choice_index = \
        np.random.choice(a=list(range(len(available_position))),
                         p=[1 / len(available_position)] * len(available_position),
                         size=1)[0]
    beans_position_center = available_position[choice_index]
    availabled_position.append(beans_position_center)
    map_skeleton[beans_position_center[1] - 1, beans_position_center[0] - 1] = "."

    # 在豆子中心生成多个豆子
    distance = locs_df[beans_position_center]
    position = np.array(list(distance.keys()))
    distance = np.array(list(distance.values()))
    available_position = position[np.where(distance <= 4)[0]]
    available_position = list(zip(available_position[:, 0], available_position[:, 1]))
    available_position = delete_inavailable(available_position, availabled_position)
    choice_index = \
        np.random.choice(a=list(range(len(available_position))),
                         p=[1 / len(available_position)] * len(available_position),
                         size=3)
    beans_position = list(np.array(available_position)[choice_index])
    for p in beans_position:
        map_skeleton[p[1] - 1, p[0] - 1] = "."

    # 生产较远的单独豆子
    available_position = position[np.where(distance > 18)[0]]
    available_position = list(zip(available_position[:, 0], available_position[:, 1]))
    available_position = delete_inavailable(available_position, availabled_position)
    choice_index = \
        np.random.choice(a=list(range(len(available_position))),
                         p=[1 / len(available_position)] * len(available_position),
                         size=1)[0]
    beans_position_ = available_position[choice_index]
    availabled_position.append(beans_position_)
    map_skeleton[beans_position_[1] - 1, beans_position_[0] - 1] = "."

    # write_map(map_skeleton)

if __name__ == '__main__':
    while True:
        eEA()
