import numpy as np
from Utils.FileUtils import readAdjacentMap, readLocDistance


def skeleton():
    filepath = "../environment/layouts/originalClassic.lay"
    locs_df = readLocDistance("../Data/mapMsg/dij_distance_map_originalClassic.csv")
    adjacent_data = readAdjacentMap("../Data/mapMsg/adjacent_map_originalClassic.csv")
    map_skeleton = []
    with open(filepath) as f:
        line = f.readline()
        while line:
            map_skeleton.append(list(line[:-1]))
            line = f.readline()

    map_skeleton = [['%' if i == '%' else ' ' for i in t] for t in map_skeleton]
    map_skeleton = np.array(map_skeleton)
    return map_skeleton, locs_df, adjacent_data


def eEA():
    map_skeleton, locs_df, adjacent_data = skeleton()

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
            if np.sum(values) < 2:
                need_resample = True
    map_skeleton[Pacman_position[1] - 1, Pacman_position[0] - 1] = "P"

    # 挑选鬼的位置
    distance = locs_df[Pacman_position]
    position=list(distance.keys())
    distance=list(distance.values())
    x = 0


if __name__ == '__main__':
    eEA()
