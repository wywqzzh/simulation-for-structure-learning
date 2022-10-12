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
    return temp


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
# index = np.where((data["strategy"] == "global") == True)[0]
# temp_data = data[["strategy", "predict_strategy", "index"]].iloc[index]
# index = np.where((temp_data["predict_strategy"] == "local") == True)[0]
# index=temp_data["index"].iloc[index]
# temp_data = data.iloc[index]
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
    predicted_strategy = data["predict_strategy"][i]
    flag = True
    for p in predicted_strategy:
        x = strategy_num[p]
        if x == strategy_num[true_strategy]:
            accuracy[strategy_num[true_strategy]][x] += 1
            flag = False
            break
    if flag == True:
        accuracy[strategy_num[true_strategy]][strategy_num[predicted_strategy[0]]] += 1

    # print(strategy_num[true_strategy])
    # print(strategy_num[predict_strategy])

for i in range(len(accuracy)):
    accuracy[i] = accuracy[i] / np.sum(accuracy[i])
print(accuracy)
