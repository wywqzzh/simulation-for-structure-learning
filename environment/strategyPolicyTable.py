import numpy as np


class strategyPolicyTable:
    def __init__(self):
        self.state_num = {
            "PG1": 3, "GS1": 3,
            "PG2": 3, "GS2": 3,
            "PE": 3, "BN5": 3, "BN10": 3
        }
        self.local_table = {}
        for i in range(self.state_num["PG1"]):
            for j in range(self.state_num["GS1"]):
                for k in range(self.state_num["BN5"]):
                    if (j != 0 and k <= 2) or (j == 0 and i >= 1 and k <= 2):
                        self.local_table.update({str(i) + str(j) + str(k): 1})
                    else:
                        self.local_table.update({str(i) + str(j) + str(k): 0.5})

        self.global_table = {}
        for i in range(self.state_num["BN5"]):
            for j in range(self.state_num["BN10"]):
                if i == 0 and j >= 1:
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
                    if (j != 0 and k <= 2) or (j == 0 and i >= 1 and k <= 2):
                        self.energizer_table.update({str(i) + str(j) + str(k): 1})
                    else:
                        self.energizer_table.update({str(i) + str(j) + str(k): 0.5})

        self.approach_table = {}
        for i in range(self.state_num["PG1"]):
            for j in range(self.state_num["GS1"]):
                for k in range(self.state_num["PG2"]):
                    for l in range(self.state_num["GS2"]):
                        if (j == 2 and i <= 1) or (l == 2 and k <= 1):
                            self.approach_table.update({str(i) + str(j) + str(k) + str(l): 1})
                        else:
                            self.approach_table.update({str(i) + str(j) + str(k) + str(l): 0.5})

    def get_strategy(self, state):
        local_state = str(state["PG1"]) + str(state["GS1"]) + str(state["BN5"])
        global_state = str(state["BN5"]) + str(state["BN10"])
        evade_state = str(state["PG1"]) + str(state["GS1"]) + str(state["PG2"]) + str(state["GS2"])
        energizer_state = str(state["PG1"]) + str(state["GS1"]) + str(state["PE"])
        approach_state = str(state["PG1"]) + str(state["GS1"]) + str(state["PG2"]) + str(state["GS2"])

        prob_local = self.local_table[local_state]
        prob_global = self.global_table[global_state]
        prob_evade = self.evade_table[evade_state]
        prob_energizer = self.energizer_table[energizer_state]
        prob_approach = self.approach_table[approach_state]

        prob = np.array([prob_local, prob_global, prob_evade, prob_energizer, prob_approach])
        index = np.where(prob == np.max(prob))[0]
        stragetegy = np.random.choice(index, 1)[0]
        stragetegy_name = ["local", "global", "evade", "energizer", "approach"]
        return stragetegy_name[stragetegy]



if __name__ == '__main__':
    s = strategyPolicyTable()
    state = {
        "PG1": 1, "GS1": 1,
        "PG2": 1, "GS2": 1,
        "BN5": 1, "BN10": 1, "PE": 1
    }
    print(s.get_strategy(state))
