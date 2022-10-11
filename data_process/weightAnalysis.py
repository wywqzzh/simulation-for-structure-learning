import pickle

import numpy as np


def predict_strategy(weight):
    index = np.argmax(np.array(weight))
    num = 0
    temp = []
    for i, w in enumerate(weight):
        if w == weight[index]:
            if i == 0:
                temp.append("global")
            elif i == 1:
                temp.append("local")
            elif i == 2:
                temp.append("approach")
            elif i == 3:
                temp.append("energizer")
    return temp[0]


def strategy_equal_prediction(x, y):
    if x in y:
        return True
    else:
        return False


with open("../Data/process/10_weight__.pkl", "rb") as file:
    data = pickle.load(file)

strategy = data["strategy"]

temp = []
for i, s in enumerate(strategy):
    ss = s + "_Q"
    temp.append(data[ss][i])

temp = np.array([temp, data["strategy_utility"]])
data["predict_strategy"] = data["weight"].apply(lambda x: predict_strategy(x))
temp = data[["strategy", "predict_strategy"]]
data["equal"] = temp.apply(
    lambda x: strategy_equal_prediction(x.strategy, x.predict_strategy), axis=1)
x = len(np.where(data["equal"] == True)[0]) / len(data)
print(x)
accuracy = np.zeros((4, 4))
strategy_num = {
    "global": 0,
    "local": 1,
    "approach": 2,
    "energizer": 3,

}
for i in range(len(data)):
    true_strategy = data["strategy"][i]
    predict_strategy = data["predict_strategy"][i]
    # print(strategy_num[true_strategy])
    # print(strategy_num[predict_strategy])
    accuracy[strategy_num[true_strategy]][strategy_num[predict_strategy]] += 1

for i in range(len(accuracy)):
    accuracy[i]=accuracy[i]/np.sum(accuracy[i])
print(accuracy)
