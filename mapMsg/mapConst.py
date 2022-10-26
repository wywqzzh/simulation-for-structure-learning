import pickle

import numpy as np
import pandas as pd
import networkx as nx
import math


# TODO:预测鬼的行为轨迹
def runMapConst(filename):
    f = open(filename)
    map = [line.strip('\n') for line in f]
    mapCol = len(map[0])
    mapRow = len(map)
    map_info = np.zeros((mapRow * mapCol, 12), dtype=np.int)

    # index all the path and wall
    ct = 0
    for i in range(mapCol):
        for j in range(mapRow):
            if i == 1 and j == 13:
                x = 0
            map_info[ct, 0] = i + 1
            map_info[ct, 1] = j + 1
            # print(j, i)
            if map[j][i] == '%':
                map_info[ct, 2] = 1
            else:
                map_info[ct, 2] = 0
            ct += 1

    for ct in np.where(map_info[:, 2] == 0)[0]:
        x = map_info[ct, 0] - 1
        y = map_info[ct, 1] - 1
        counter = 0
        if map[y - 1][x] != '%':
            map_info[ct, 4] = x + 1
            map_info[ct, 5] = y
            counter += 1
        if map[y + 1][x] != '%':
            map_info[ct, 6] = x + 1
            map_info[ct, 7] = y + 2
            counter += 1
        if map[y][(x - 1) % mapCol] != '%':
            map_info[ct, 8] = (x - 1) % mapCol + 1
            map_info[ct, 9] = y + 1
            counter += 1
        if map[y][(x + 1) % mapCol] != '%':
            map_info[ct, 10] = (x + 1) % mapCol + 1
            map_info[ct, 11] = y + 1
            counter += 1
        map_info[ct, 3] = counter
    column_names = ['Pos1', 'Pos2', 'iswall', 'NextNum', 'UpX', 'UpY', 'DownX', 'DownY', 'LeftX', 'LeftY', 'RightX',
                    'RightY']
    data = pd.DataFrame(map_info, columns=column_names)
    data.to_csv("../Data/mapMsg/map_info_" + filename.split("/")[-1][:-4] + ".csv")


def create_adjacent_map(map_name):
    def tuple_list(l):
        return [tuple(a) for a in l]

    MAP_INFO = pd.read_csv("../Data/mapMsg/map_info_" + map_name + ".csv")
    MAP_INFO = MAP_INFO.loc[MAP_INFO.iswall == 0].reset_index()
    data = dict({"pos": tuple_list(MAP_INFO[["Pos1", "Pos2"]].values),
                 "left": tuple_list(MAP_INFO[["LeftX", "LeftY"]].values),
                 "right": tuple_list(MAP_INFO[["RightX", "RightY"]].values),
                 "up": tuple_list(MAP_INFO[["UpX", "UpY"]].values),
                 "down": tuple_list(MAP_INFO[["DownX", "DownY"]].values)})
    for key, value in data.items():
        for index, i in enumerate(value):
            if i == (0, 0):
                data[key][index] = np.nan
    T = pd.DataFrame(data)
    T.to_csv("../Data/mapMsg/adjacent_map_" + map_name + ".csv")
    T.to_pickle("../Data/mapMsg/adjacent_map_" + map_name + ".pkl")


def create_dij_distance_map(map_name):
    T = pd.read_pickle("../Data/mapMsg/adjacent_map_" + map_name + ".pkl")
    G = nx.Graph()
    G.add_nodes_from(T.pos)
    for i in range(0, T.shape[0]):
        k = T.pos[i]
        G.add_edges_from(([(k, t) for t in T.iloc[i, 1:5].values if t is not np.nan]))

    def get_relative_dir(pos1, pos2):
        res = tuple(map(lambda i, j: j - i, Source, Target))
        if res[0] == 0 and res[0] > 0:
            return ['down']
        if res[0] == 0 and res[0] < 0:
            return ['up']
        if res[0] > 0 and res[1] == 0:
            return ['right']
        if res[0] < 0 and res[1] == 0:
            return ['left']
        if res[0] > 0 and res[1] > 0:
            return ['right', 'down']
        if res[0] > 0 and res[1] < 0:
            return ['right', 'up']
        if res[0] < 0 and res[1] > 0:
            return ['left', 'down']
        if res[0] < 0 and res[1] < 0:
            return ['left', 'up']

    # In[5]:

    Tr = {"pos1": [], "pos2": [], "dis": [], "path": [], "relative_dir": []}
    print(len(T.pos) ** 2)
    for Source in T.pos:
        print(Source)
        for Target in T.pos:

            if Source == Target:
                continue
            Tr['pos1'].append(Source)
            Tr['pos2'].append(Target)
            Tr['dis'].append(nx.shortest_path_length(G, Source, Target))
            Tr['path'].append([x for x in nx.all_shortest_paths(G, Source, Target)])
            Tr['relative_dir'].append(get_relative_dir(Source, Target))
    pos1 = Tr.get("pos1")
    pos2 = Tr.get("pos2")
    dis = Tr.get("dis")
    path = Tr.get("path")
    relative_dir = Tr.get("relative_dir")
    df = pd.DataFrame({"pos1": pos1, "pos2": pos2, "dis": dis, "path": path, "relative_dir": relative_dir})
    df.to_csv("../Data/mapMsg/dij_distance_map_" + map_name + ".csv")
    df.to_pickle("../Data/mapMsg/dij_distance_map_" + map_name + ".pkl")


def get_intersection(map_name):
    adjacent = pd.read_pickle("../Data/mapMsg/adjacent_map_" + map_name + ".pkl")
    position = []
    for i in range(len(adjacent)):
        pos = adjacent["pos"][i]
        if pos == (2, 14):
            x = 0
        left = adjacent["left"][i]
        right = adjacent["right"][i]
        up = adjacent["up"][i]
        down = adjacent["down"][i]
        t = np.ones(4)
        if isinstance(left, float):
            t[0] = 0
        if isinstance(right, float):
            t[1] = 0
        if isinstance(up, float):
            t[2] = 0
        if isinstance(down, float):
            t[3] = 0
        if np.sum(t) >= 3 or np.sum(t) == 1:
            position.append(pos)
        if np.sum(t) == 2 and (np.sum(t[:2]) > 0 and np.sum(t[2:]) > 0):
            position.append(pos)
    position = {"pos": position}
    with open("../Data/mapMsg/intersection_map_" + map_name + ".pkl","wb") as file:
        pickle.dump(position,file)
    # df = pd.DataFrame(position)
    # df.to_csv("../Data/mapMsg/intersection_map_" + map_name + ".csv")
    # df.to_pickle("../Data/mapMsg/intersection_map_" + map_name + ".pkl")


if __name__ == '__main__':
    filename = "../environment/layouts/originalClassic2.lay"
    runMapConst(filename)
    create_adjacent_map(filename.split("/")[-1][:-4])
    create_dij_distance_map(filename.split("/")[-1][:-4])
    get_intersection(filename.split("/")[-1][:-4])
