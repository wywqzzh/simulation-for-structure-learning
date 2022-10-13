import copy
import pickle

import numpy as np
import pandas as pd


def predict_strategy(weight):
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
    if x in y:
        return True
    else:
        return False


def get_action_accuracy(df):
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
        prediction = np.random.choice([0, 1, 2, 3], p=exp_prob, size=1)[0]
        if prediction == true_action:
            num += 1
    print("action accuracy:", num / num_samples)


def weight_normalize(weight):
    weight = np.array([np.array(w) for w in weight])
    weight_normaliztion = np.sum(weight, axis=1)
    for i in range(len(weight)):
        weight[i] /= weight_normaliztion[i]
    weight_norm = []
    for i in range(len(weight)):
        weight_norm.append(list(weight[i]))
    return weight_norm


def strategy_accuracy():
    with open("../Data/process/two_1_weight_norm.pkl", "rb") as file:
        data = pickle.load(file)

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
    index = np.where((data["predict_strategy"] != "None") == True)[0]
    data = data.iloc[index]
    new_data = pd.DataFrame()
    columns = list(data.columns)[1:]
    for c in columns:
        new_data[c] = copy.deepcopy(data[c])
    new_data.reset_index(inplace=True, drop=True)
    data = new_data

    data["index"] = list(range(len(data)))
    index = np.where((data["strategy"] == "energizer") == True)[0]
    temp_data = data[["strategy", "predict_strategy", "index"]].iloc[index]
    index = []
    for i in range(len(temp_data)):
        d = temp_data.iloc[i]
        if "energizer" != d["predict_strategy"]:
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
