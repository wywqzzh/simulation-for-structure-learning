import pickle
import pandas as pd
import numpy as np
from Utils.FileUtils import readAdjacentMap, readLocDistance


class weightFitter:
    """
    fitting the strategy weight
    """

    def __init__(self, filename, map_name="originalClassic1"):
        self.filename = filename
        self.all_dir_list = ["left", "right", "up", "down"]
        self.adjacent_data = readAdjacentMap("../Data/mapMsg/adjacent_map_" + map_name + ".csv")
        self.adjacent_data = readAdjacentMap("../Data/constant/adjacent_map.csv")
        self._readData()

    def _normalizeWithInf(self, x):
        res_x = x.copy()
        tmp_x_idx = np.where(~np.isinf(x))[0]
        if set(x[tmp_x_idx]) == {0}:
            res_x[tmp_x_idx] = 0
        else:
            res_x[tmp_x_idx] = res_x[tmp_x_idx] / np.max(np.abs(res_x[tmp_x_idx]))
        return res_x

    def _positivePessi(self, pess_Q, offset, pos):
        '''
        Make evade agent Q values non-negative.
        '''
        non_zero = []
        if pos == (29, 18) or pos == (30, 18):
            pos = (28, 18)
        if pos == (0, 18) or pos == (-1, 18):
            pos = (1, 18)
        for dir in self.all_dir_list:
            if None != self.adjacent_data[pos][dir] and not isinstance(self.adjacent_data[pos][dir], float):
                non_zero.append(self.all_dir_list.index(dir))
        pess_Q[non_zero] = pess_Q[non_zero] - offset
        return self._normalizeWithInf(pess_Q)

    def _readData(self):
        df = pd.read_pickle(self.filename)
        filename_list = df.file.unique()
        selected_data = pd.concat([df[df.file == i] for i in filename_list]).reset_index(drop=True)
        df = selected_data

        if "pacman_dir" in df.columns.values and "next_pacman_dir_fill" not in df.columns.values:
            df["next_pacman_dir_fill"] = df.pacman_dir.apply(lambda x: x if x is not None else np.nan)
        trial_name_list = np.unique(df.file.values)
        trial_data = []
        for each in trial_name_list:
            pac_dir = df[df.file == each].next_pacman_dir_fill
            if np.sum(pac_dir.apply(lambda x: isinstance(x, float))) == len(pac_dir):
                # all the directions are none
                print("({}) Pacman No Move ! Shape = {}".format(each, pac_dir.shape))
                continue
            else:
                trial_data.append(df[df.file == each])
        df = pd.concat(trial_data).reset_index(drop=True)
        for c in df.filter(regex="_Q").columns:
            tmp_val = df[c].explode().values
            offset_num = np.min(tmp_val[tmp_val != -np.inf])
            offset_num = min(0, offset_num)
            # offset_num = df[c].explode().min()
            df[c + "_norm"] = df[[c, "pacmanPos"]].apply(
                lambda x: self._positivePessi(x[c], offset_num, x.pacmanPos)
                if set(x[c]) != {0}
                else [0, 0, 0, 0],
                axis=1
            )
        self.df = df

    def dynamicStrategyFitting(self):
        print("=== Dynamic Strategy Fitting ====")
        self.suffix = "_Q_norm"
        trial_name_list = np.unique(self.df.file.values)
        print("The num of trials : ", len(trial_name_list))
        print("-" * 50)
        


if __name__ == '__main__':
    filename = "../Data/10_trial_data_Omega.pkl"
    weight_fitter = weightFitter(filename)
