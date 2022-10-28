import copy
import pickle

import numpy as np
import pandas as pd


def predict_strategy(weight):
    """
    预测strategy ,如果最大weight的策略与次大weight的策略直接的weight差值小于0.2则表示该场景策略模糊，return "None"
    :param weight: 
    :return: 
    """
    strategy_dict = {
        0: "global",
        1: "local",
        2: "evade",
        3: "approach",
        4: "energizer",
    }
    index = np.argmax(np.array(weight))
    Max = np.max(weight)
    Max2 = -1
    for i in range(len(weight)):
        if i != index and weight[i] > Max2:
            Max2 = weight[i]
    if Max - Max2 < 0.2:
        return "None"
    else:
        return strategy_dict[index]


def strategy_equal_prediction(x, y):
    """
    判断预测strategy与真实strategy是否一致
    :param x: 
    :param y: 
    :return: 
    """
    if x == y:
        return True
    else:
        return False


def get_action_accuracy(df):
    """
    计算action的预测准确度
    :param df: 
    :return: 
    """
    action_dir = {
        "left": 0,
        "right": 1,
        "up": 2,
        "down": 3
    }

    suffix = "_Q_norm"
    agents_list = ["global", "local", "evade", "approach", "energizer"]
    data_columns = [a + suffix for a in agents_list]
    pre_estimation = df[data_columns].values

    num_samples = len(df)
    # 提取Q value
    agent_Q_value = np.zeros((num_samples, 4, len(agents_list)))
    for each_sample in range(num_samples):
        for each_agent in range(len(agents_list)):
            agent_Q_value[each_sample, :, each_agent] = pre_estimation[each_sample][
                each_agent
            ]
    agent_weight = np.array([np.array(a) for a in df["weight"]])
    num = 0
    for i in range(num_samples):
        weight = agent_weight[i]
        Q = agent_Q_value[i]
        dir_Q_value = Q @ weight
        dir_Q_value[np.isnan(dir_Q_value)] = -np.inf
        exp_prob = np.exp(dir_Q_value)
        exp_prob = exp_prob / np.sum(exp_prob)

        true_action = action_dir[df["pacman_dir"][i]]

        # 如果最大值唯一，则选取最大值，如果最大值不唯一则从最大值中随机选择一个
        MAX_index = np.argmax(exp_prob)
        MAX = exp_prob[MAX_index]
        indexs = [MAX_index]
        MAXS = [MAX]
        for i, e in enumerate(exp_prob):
            if e == MAX and i != MAX_index:
                indexs.append(i)
        prediction = np.random.choice(indexs, p=[1 / len(indexs)] * len(indexs), size=1)[0]
        if prediction == true_action:
            num += 1
    print("action accuracy:", num / num_samples)


def weight_normalize(weight):
    """
    normalize weight
    :param weight: 
    :return: 
    """
    weight = np.array([np.array(w) for w in weight])
    weight_normaliztion = np.sum(weight, axis=1)
    for i in range(len(weight)):
        weight[i] /= weight_normaliztion[i]
    weight_norm = []
    for i in range(len(weight)):
        weight_norm.append(list(weight[i]))
    return weight_norm


def MAX_equal_MAX2(x):
    MAX_index = np.argmax(x)
    MAX = x[MAX_index]
    indexs = [MAX_index]
    for i, e in enumerate(x):
        if e == MAX and i != MAX_index:
            indexs.append(i)
    if len(indexs) > 1:
        return True
    else:
        return False


def drop_bad_context(data):
    """
    删除采用某策略，但该策略的Q value却不能确定运动方向的片段
    :return: 
    """
    # 确定changepoint
    true_strategy = data["strategy"]
    cutoff_pts = list(np.where((true_strategy == true_strategy.shift()) == False)[0])[1:]
    cutoff_pts.append(len(true_strategy))
    cutoff_pts = list(zip([0] + list(cutoff_pts[:-1]), cutoff_pts))
    data["vague"] = data["strategy_utility"].apply(lambda x: MAX_equal_MAX2(x))
    for s, e in cutoff_pts:
        temp = data["vague"][s:e]
        temp = np.where(temp == True)[0]
        rate = len(temp) / (e - s)
        if rate >= 0.9:
            print("rate:", rate)
            data["predict_strategy"][s:e] = ["None"] * (e - s)

    return data


def strategy_accuracy():
    with open("../Data/process/tri-gram_weight_norm.pkl", "rb") as file:
        data = pickle.load(file)
    
    print(len(data))
    import warnings
    warnings.filterwarnings("ignore")

    data["weight_norm"] = weight_normalize(np.array(data["weight"]))
    get_action_accuracy(data)

    # strategy = data["strategy"]
    # temp = []
    # for i, s in enumerate(strategy):
    #     ss = s + "_Q"
    #     temp.append(data[ss][i])
    # 
    # temp = np.array([temp, data["strategy_utility"]])
    data["predict_strategy"] = data["weight_norm"].apply(lambda x: predict_strategy(x))
    data = drop_bad_context(data)
    index = np.where((data["predict_strategy"] != "None") == True)[0]
    data = data.iloc[index]
    new_data = pd.DataFrame()
    columns = list(data.columns)[1:]
    for c in columns:
        new_data[c] = copy.deepcopy(data[c])
    new_data.reset_index(inplace=True, drop=True)
    data = new_data

    data["index"] = list(range(len(data)))
    index = np.where((data["strategy"] == "global") == True)[0]
    temp_data = data[["strategy", "predict_strategy", "index"]].iloc[index]
    index = []
    for i in range(len(temp_data)):
        d = temp_data.iloc[i]
        if "global" != d["predict_strategy"]:
            index.append(d["index"])
    # index = temp_data["index"].iloc[index]
    temp_data = data.iloc[index]
    temp = data[["strategy", "predict_strategy"]]
    data["equal"] = temp.apply(
        lambda x: strategy_equal_prediction(x.strategy, x.predict_strategy), axis=1)
    accuracy = len(np.where(data["equal"] == True)[0]) / len(data)
    print("strategy accuracy:", accuracy)
    print("data num:", len(data))
    accuracy = np.zeros((5, 5))
    strategy_num = {
        "global": 0,
        "local": 1,
        "evade": 2,
        "approach": 3,
        "energizer": 4,

    }
    for i in range(len(data)):
        true_strategy = data["strategy"][i]
        predicted_strategy = data["predict_strategy"][i]
        accuracy[strategy_num[true_strategy]][strategy_num[predicted_strategy]] += 1

        # print(strategy_num[true_strategy])
        # print(strategy_num[predict_strategy])

    for i in range(len(accuracy)):
        accuracy[i] = accuracy[i] / np.sum(accuracy[i])
    print(accuracy)


if __name__ == '__main__':
    strategy_accuracy()
