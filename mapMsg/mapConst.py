import numpy as np
import pandas as pd


def runMapConst(filename):
    f = open(filename)
    map = [line.strip() for line in f]
    mapCol = len(map[0])
    mapRow = len(map)
    map_info = np.zeros((mapRow * mapCol, 12), dtype=np.int)

    # index all the path and wall
    ct = 0
    for i in range(mapCol):
        for j in range(mapRow):
            map_info[ct, 0] = i + 1
            map_info[ct, 1] = j + 1
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
        if map[y][x - 1] != '%':
            map_info[ct, 8] = x
            map_info[ct, 9] = y + 1
            counter += 1
        if map[y][x + 1] != '%':
            map_info[ct, 10] = x + 2
            map_info[ct, 11] = y + 1
            counter += 1
        map_info[ct, 3] = counter
    column_names = ['Pos1', 'Pos2', 'iswall', 'NextNum', 'UpX', 'UpY', 'DownX', 'DownY', 'LeftX', 'LeftY', 'RightX',
                    'RightY']
    data=pd.DataFrame(map_info, columns=column_names)
    data.to_csv("../Data/mapMsg/map_info.csv")


if __name__ == '__main__':
    filename = "../environment/layouts/smallGrid.lay"
    runMapConst(filename)
