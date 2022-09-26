import numpy as np


class strategyPolicyTable:
    def __init__(self):
        self.state_num = {
            "PG1": 3, "GS1": 3,
            "PG2": 3, "GS2": 3,
            "PE": 3, "BN5": 3, "BN10": 3, "ZBN5": 2, "ZBN10": 2
        }
        self.local_table = {}
        for i in range(self.state_num["PG1"]):
            for j in range(self.state_num["GS1"]):
                for k in range(self.state_num["ZBN5"]):
                    if (j != 0 and k == 0) or (j == 0 and i >= 1 and k == 0):
                        self.local_table.update({str(i) + str(j) + str(k): 1})
                    else:
                        self.local_table.update({str(i) + str(j) + str(k): 0.5})

        self.global_table = {}
        for i in range(self.state_num["ZBN5"]):
            for j in range(self.state_num["ZBN10"]):
                if i == 1 and j == 0:
                    self.global_table.update({str(i) + str(j): 1})
                else:
                    self.global_table.update({str(i) + str(j): 0.5})

        self.evade_table = {}
        for i in range(self.state_num["PG1"]):
            for j in range(self.state_num["GS1"]):
                for k in range(self.state_num["PG2"]):
                    for l in range(self.state_num["GS2"]):
                        if (j == 0 and i == 0) or (k == 0 and l == 0):
                            self.evade_table.update({str(i) + str(j) + str(k) + str(l): 1})
                        else:
                            self.evade_table.update({str(i) + str(j) + str(k) + str(l): 0.5})

        self.energizer_table = {}
        for i in range(self.state_num["PG1"]):
            for j in range(self.state_num["GS1"]):
                for k in range(self.state_num["PE"]):
                    if (j != 0 and k <= 0) or (j == 0 and i >= 1 and k <= 0):
                        self.energizer_table.update({str(i) + str(j) + str(k): 1})
                    else:
                        self.energizer_table.update({str(i) + str(j) + str(k): 0.5})

        self.approach_table = {}
        for i in range(self.state_num["PG1"]):
            for j in range(self.state_num["GS1"]):
                for k in range(self.state_num["PG2"]):
                    for l in range(self.state_num["GS2"]):
                        if (j == 2 and i <= 0) or (l == 2 and k <= 0):
                            self.approach_table.update({str(i) + str(j) + str(k) + str(l): 1})
                        else:
                            self.approach_table.update({str(i) + str(j) + str(k) + str(l): 0.5})

    def get_strategy(self, state):
        local_state = str(state["PG1"]) + str(state["GS1"]) + str(state["ZBN5"])
        global_state = str(state["ZBN5"]) + str(state["ZBN10"])
        evade_state = str(state["PG1"]) + str(state["GS1"]) + str(state["PG2"]) + str(state["GS2"])
        energizer_state = str(state["PG1"]) + str(state["GS1"]) + str(state["PE"])
        approach_state = str(state["PG1"]) + str(state["GS1"]) + str(state["PG2"]) + str(state["GS2"])

        prob_local = self.local_table[local_state]
        prob_global = self.global_table[global_state]
        prob_evade = self.evade_table[evade_state]
        prob_energizer = self.energizer_table[energizer_state]
        prob_approach = self.approach_table[approach_state]

        stragetegy_name = ["local", "global", "evade", "energizer", "approach"]
        prob = np.array([prob_local, prob_global, prob_evade, prob_energizer, prob_approach])
        index = np.where(prob == np.max(prob))[0]

        stragetegy = np.array(stragetegy_name)[index]
        if "evade" in stragetegy:
            return "evade"
        elif "approach" in stragetegy:
            return "approach"
        elif "local" in stragetegy:
            return "local"
        elif "energizer" in stragetegy:
            return "energizer"
        elif "global" in stragetegy:
            return "global"
        # stragetegy = np.random.choice(index, 1)[0]

        # return stragetegy_name[stragetegy]


if __name__ == '__main__':
    s = strategyPolicyTable()
    state = {
        "PG1": 1, "GS1": 1,
        "PG2": 1, "GS2": 1,
        "BN5": 1, "BN10": 1, "PE": 1
    }
    print(s.get_strategy(state))
