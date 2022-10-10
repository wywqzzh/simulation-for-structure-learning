import numpy as np
import pandas as pd
import pickle
import scipy
import copy
import ruptures as rpt
import os
from sklearn.model_selection import KFold
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

import sys

sys.path.append("../Utils")
from Utils.FileUtils import readAdjacentMap

agents = [
    "global",
    "local",
    "evade_blinky",
    "evade_clyde",
    "approach",
    "energizer",
]


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

    def change_dir_index(self, x):
        '''
        Find the position where the Pacman changes its direction.
        '''
        temp = pd.Series((x != x.shift()).where(lambda x: x == True).dropna().index)
        return temp[(temp - temp.shift()) > 1].values

    def add_cutoff_pts(self, cutoff_pts, df_monkey):
        '''
        Initialize cut-off points at where the ghosts and energizers are eaten.
        '''
        eat_ghost = (
            (
                    ((df_monkey.ifscared1 == 3) & (df_monkey.ifscared1.diff() < 0))
                    | ((df_monkey.ifscared2 == 3) & (df_monkey.ifscared2.diff() < 0))
            )
                .where(lambda x: x == True)
                .dropna()
                .index.tolist()
        )
        eat_energizers = (
            (
                    df_monkey.energizers.apply(
                        lambda x: len(x) if not isinstance(x, float) else 0
                    ).diff()
                    < 0
            )
                .where(lambda x: x == True)
                .dropna()
                .index.tolist()
        )
        cutoff_pts = sorted(list(cutoff_pts) + eat_ghost + eat_energizers)
        return cutoff_pts

    def _combine(self, cutoff_pts, dir):
        '''
        Combine cut off points when necessary.
        '''
        if len(cutoff_pts) > 1:
            temp_pts = [cutoff_pts[0]]
            for i in range(1, len(cutoff_pts)):
                if cutoff_pts[i][1] - cutoff_pts[i][0] > 3:
                    if np.all(
                            dir.iloc[cutoff_pts[i][0]:cutoff_pts[i][1]].apply(lambda x: isinstance(x, float)) == True):
                        temp_pts[-1] = (temp_pts[-1][0], cutoff_pts[i][1])
                    else:
                        temp_pts.append(cutoff_pts[i])
                else:
                    temp_pts[-1] = (temp_pts[-1][0], cutoff_pts[i][1])
            cutoff_pts = temp_pts
        return cutoff_pts

    def _oneHot(self, val):
        """
        Convert the direction into a one-hot vector.
        :param val: (str) The direction ("left", "right", "up", "down").
        :return: (list) One-hotted vector.
        """
        dir_list = ["left", "right", "up", "down"]
        # Type check
        if val not in dir_list:
            raise ValueError("Undefined direction {}!".format(val))
        if not isinstance(val, str):
            raise TypeError("Undefined direction type {}!".format(type(val)))
        # One-hot
        onehot_vec = [0, 0, 0, 0]
        onehot_vec[dir_list.index(val)] = 1
        return onehot_vec

    def _makeChoice(self, prob):
        '''
        Chose a direction based on estimated Q values.
        :param prob: (list) Q values of four directions (lef, right, up, down).
        :return: (int) The chosen direction.
        '''
        copy_estimated = copy.deepcopy(prob)
        if np.any(prob) < 0:
            available_dir_index = np.where(prob != 0)
            copy_estimated[available_dir_index] = (
                    copy_estimated[available_dir_index]
                    - np.min(copy_estimated[available_dir_index])
                    + 1
            )
        return np.random.choice([idx for idx, i in enumerate(prob) if i == max(prob)])

    def negativeLikelihood(
            self, param, all_data, true_prob, agents_list, return_trajectory=False, suffix="_Q"
    ):
        """
        Compute the negative log-likelihood.
        :param param: (list) Model parameters, which are agent weights.
        :param all_data: (pandas.DataFrame) A table of data.
        :param agent_list: (list) Names of all the agents.
        :param return_trajectory: (bool) Set to True when making predictions.
        :return: (float) Negative log-likelihood
        """
        if 0 == len(agents_list) or None == agents_list:
            raise ValueError("Undefined agents list!")
        else:
            agent_weight = [param[i] for i in range(len(param))]
        # Compute negative log likelihood
        nll = 0
        num_samples = all_data.shape[0]
        agents_list = [("{}" + suffix).format(each) for each in agents_list]
        pre_estimation = all_data[agents_list].values
        # raise KeyboardInterrupt
        agent_Q_value = np.zeros((num_samples, 4, len(agents_list)))
        for each_sample in range(num_samples):
            for each_agent in range(len(agents_list)):
                agent_Q_value[each_sample, :, each_agent] = pre_estimation[each_sample][
                    each_agent
                ]
        dir_Q_value = agent_Q_value @ agent_weight
        dir_Q_value[np.isnan(dir_Q_value)] = -np.inf
        true_dir = true_prob.apply(lambda x: self._makeChoice(x)).values
        exp_prob = np.exp(dir_Q_value)
        for each_sample in range(num_samples):
            if np.isnan(dir_Q_value[each_sample][0]):
                continue
            log_likelihood = dir_Q_value[each_sample, true_dir[each_sample]] - np.log(np.sum(exp_prob[each_sample]))
            nll = nll - log_likelihood
        if not return_trajectory:
            return nll
        else:
            return (nll, dir_Q_value)

    def caculate_correct_rate(self, result_x, all_data, true_prob, agents, suffix="_Q"):
        '''
        Compute the estimation correct rate of a fitted model.
        '''
        _, estimated_prob = self.negativeLikelihood(
            result_x, all_data, true_prob, agents, return_trajectory=True, suffix=suffix
        )
        true_dir = np.array([np.argmax(each) for each in true_prob])
        estimated_dir = np.array([self._makeChoice(each) for each in estimated_prob])
        correct_rate = np.sum(estimated_dir == true_dir) / len(estimated_dir)
        return correct_rate

    def caculate_correct_rate(self, result_x, all_data, true_prob, agents, suffix="_Q"):
        '''.ct rate of a fitted model.
        '''
        _, estimated_prob = self.negativeLikelihood(
            result_x, all_data, true_prob, agents, return_trajectory=True, suffix=suffix
        )
        true_dir = np.array([np.argmax(each) for each in true_prob])
        estimated_dir = np.array([self._makeChoice(each) for each in estimated_prob])
        correct_rate = np.sum(estimated_dir == true_dir) / len(estimated_dir)
        return correct_rate

    def _calculate_is_correct(self, result_x, all_data, true_prob, agents, suffix="_Q"):
        '''
        Determine whether the estimation of each time step is correct.
        '''
        _, estimated_prob = self.negativeLikelihood(
            result_x, all_data, true_prob, agents, return_trajectory=True, suffix=suffix
        )
        true_dir = np.array([np.argmax(each) for each in true_prob])
        estimated_dir = np.array([self._makeChoice(each) for each in estimated_prob])
        is_correct = (estimated_dir == true_dir)
        return is_correct

    def caculate_strategy_correct(self, ):
        pass

    def fit_func(self, df_monkey, cutoff_pts, suffix="_Q", is_match=False,
                 agents=["global", "local", "evade_blinky", "evade_clyde", "approach", "energizer"]):
        '''
        Fit model parameters (i.e., agent weights).
        '''
        result_list = []
        is_correct = []
        bounds = [[0, 1000], [0, 1000], [0, 1000], [0, 1000], [0, 1000], [0, 1000]]
        params = [0.0] * 6
        cons = []  # construct the bounds in the form of constraints

        for par in range(len(bounds)):
            l = {"type": "ineq", "fun": lambda x: x[par] - bounds[par][0]}
            u = {"type": "ineq", "fun": lambda x: bounds[par][1] - x[par]}
            cons.append(l)
            cons.append(u)

        prev = 0
        total_loss = 0
        is_correct = np.zeros((df_monkey.shape[0],))
        is_correct[is_correct == 0] = np.nan

        for prev, end in cutoff_pts:
            all_data = df_monkey[prev:end]
            temp_data = copy.deepcopy(all_data)
            temp_data["nan_dir"] = temp_data.next_pacman_dir_fill.apply(lambda x: isinstance(x, float))
            all_data = all_data[temp_data.nan_dir == False]
            if all_data.shape[0] == 0:
                print("All the directions are nan from {} to {}!".format(prev, end))
                continue
            ind = np.where(temp_data.nan_dir == False)[0] + prev

            true_prob = all_data.next_pacman_dir_fill.fillna(method="ffill").apply(self._oneHot)
            func = lambda params: self.negativeLikelihood(
                params, all_data, true_prob, agents, return_trajectory=False, suffix=suffix
            )
            res = scipy.optimize.minimize(
                func,
                x0=params,
                method="SLSQP",
                bounds=bounds,  # exclude bounds and cons because the Q-value has different scales for different agents
                tol=1e-5,
                constraints=cons,
            )
            if set(res.x) == {0}:
                print("Failed optimization at ({},{})".format(prev, end))
                params = [0.1] * 6
                for i, a in enumerate(agents):
                    if set(np.concatenate(all_data["{}{}".format(a, suffix)].values)) == {0}:
                        params[i] = 0.0
                res = scipy.optimize.minimize(
                    func,
                    x0=params,
                    method="SLSQP",
                    bounds=bounds,
                    # exclude bounds and cons because the Q-value has different scales for different agents
                    tol=1e-5,
                    constraints=cons,
                )
            total_loss += self.negativeLikelihood(
                res.x / res.x.sum(),
                all_data,
                true_prob,
                agents,
                return_trajectory=False,
                suffix=suffix,
            )
            cr = self.caculate_correct_rate(res.x, all_data, true_prob, agents, suffix=suffix)
            result_list.append(res.x.tolist() + [cr] + [prev] + [end])
            phase_is_correct = self._calculate_is_correct(res.x, all_data, true_prob, agents, suffix=suffix)
            is_correct[ind] = phase_is_correct
        if is_match:
            return result_list, total_loss, is_correct
        else:
            return result_list, total_loss

    def normalize_weights(self, result_list, df_monkey):
        '''
        Normalize fitted agent weights.
        '''
        agents = [
            "global",
            "local",
            "pessimistic_blinky",
            "pessimistic_clyde",
            "suicide",
            "planned_hunting",
        ]
        df_result = (
            pd.DataFrame(
                result_list,
                columns=[i + "_w" for i in agents] + ["accuracy", "start", "end"],
            )
                .set_index("start")
                .reindex(range(df_monkey.shape[0]))
                .fillna(method="ffill")
        )
        df_plot = df_result.filter(regex="_w").divide(
            df_result.filter(regex="_w").sum(1), 0
        )
        return df_plot, df_result

    def dynamicStrategyFitting(self):
        print("=== Dynamic Strategy Fitting ====")
        self.suffix = "_Q_norm"
        trial_name_list = np.unique(self.df.file.values)
        print("The num of trials : ", len(trial_name_list))
        print("-" * 50)

        results = []
        data = []
        for t, trial_name in enumerate(trial_name_list):
            df = self.df[self.df.file == trial_name].reset_index()
            print("| ({}) {} | Data shape {}".format(t, trial_name, df.shape))
            cutoff_pts = self.add_cutoff_pts(self.change_dir_index(df.next_pacman_dir_fill),
                                             df)
            if cutoff_pts[-1] != len(df) - 1:
                cutoff_pts.append(len(df))
            else:
                cutoff_pts[-1] = len(df)
            cutoff_pts = list(zip([0] + list(cutoff_pts[:-1]), cutoff_pts))
            cutoff_pts = self._combine(cutoff_pts, df.next_pacman_dir_fill)
            result_list, _, is_correct = self.fit_func(df, cutoff_pts, suffix=self.suffix, agents=agents, is_match=True)
            try:
                df_plot, df_result = self.normalize_weights(result_list, df)
            except:
                print("Error occurs in weight normalization.")
                continue

            trial_weight = []
            trial_contribution = []
            for res in result_list:
                weight = res[:len(agents)]
                prev = res[-2]
                end = res[-1]
                for _ in range(prev, end):
                    trial_weight.append(weight)
                    trial_contribution.append(weight / np.linalg.norm(weight))
            if len(trial_weight) != df.shape[0]:
                df["weight"] = [np.nan for _ in range(df.shape[0])]
                df["contribution"] = [np.nan for _ in range(df.shape[0])]
                df["is_correct"] = [np.nan for _ in range(df.shape[0])]
            elif len(trial_weight) > 0:
                df["weight"] = trial_weight
                df["contribution"] = trial_contribution
                df["is_correct"] = is_correct
            else:
                pass
            data.append(df)
        data = pd.concat(data).reset_index(drop=True)
        x = 0


if __name__ == '__main__':
    filename = "../Data/10_trial_data_Omega.pkl"
    weight_fitter = weightFitter(filename)
    weight_fitter.dynamicStrategyFitting()
