import numpy as np


class strategyPolicyTable:
    def __init__(self):
        self.state_num = {
            "PG1": 3, "GS1": 3,
            "PG2": 3, "GS2": 3,
            "PE": 3, "BN5": 3, "BN10": 3, "ZBN5": 2, "ZBN10": 2
        }
        self.local_table = {}
        # for i in range(self.state_num["PG1"]):
        #     for j in range(self.state_num["GS1"]):
        #         for k in range(self.state_num["ZBN5"]):
        #             if (j != 0 and k == 0) or (j == 0 and i >= 1 and k == 0):
        #                 self.local_table.update({str(i) + str(j) + str(k): 1})
        #             else:
        #                 self.local_table.update({str(i) + str(j) + str(k): 0.5})
        for i in range(self.state_num["ZBN5"]):
            if i == 0:
                self.local_table.update({str(i): 1})
            else:
                self.local_table.update({str(i): 0.5})

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
        for i in range(self.state_num["PE"]):
            if i == 0:
                self.energizer_table.update({str(i): 1})
            else:
                self.energizer_table.update({str(i): 0.5})

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
        local_state = str(state["PG1"]) + str(state["GS1"]) + str(state["ZBN5"])
        local_state = str(state["ZBN5"])
        global_state = str(state["ZBN5"]) + str(state["ZBN10"])
        evade_state = str(state["PG1"]) + str(state["GS1"]) + str(state["PG2"]) + str(state["GS2"])
        energizer_state = str(state["PG1"]) + str(state["GS1"]) + str(state["PE"])
        energizer_state = str(state["PE"])
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


class twoStrategyPolicyTable:
    def __init__(self):
        self.state_num = {
            "PG1": 3, "GS1": 3,
            "PG2": 3, "GS2": 3,
            "PE": 3, "BN5": 3, "BN10": 3, "ZBN5": 2, "ZBN10": 2
        }
        self.two_strategy = "LG"
        self.strategy = "local"
        self.LG_table = {}
        for i in range(self.state_num["ZBN5"]):
            if i == 0:
                self.LG_table.update({str(i): 1})
            else:
                self.LG_table.update({str(i): 0.5})

        self.GL_table = {}
        for i in range(self.state_num["ZBN5"]):
            for j in range(self.state_num["ZBN10"]):
                if i == 1 and j == 0:
                    self.GL_table.update({str(i) + str(j): 1})
                else:
                    self.GL_table.update({str(i) + str(j): 0.5})

        self.EvE_table = {}
        for i in range(self.state_num["PG1"]):
            for j in range(self.state_num["GS1"]):
                for k in range(self.state_num["PG2"]):
                    for l in range(self.state_num["GS2"]):
                        for m in range(self.state_num["PE"]):
                            if (j == 0 and i == 0 and m == 0) or (k == 0 and l == 0 and m == 0):
                                self.EvE_table.update({str(i) + str(j) + str(k) + str(l): 1})
                            else:
                                self.EvE_table.update({str(i) + str(j) + str(k) + str(l): 0.5})

        self.EA_table = {}
        for i in range(self.state_num["PG1"]):
            for j in range(self.state_num["GS1"]):
                for k in range(self.state_num["PG2"]):
                    for l in range(self.state_num["GS2"]):
                        for m in range(self.state_num["PE"]):
                            if (j != 0 and m <= 0 and i <= 1) or (l != 0 and m <= 0 and k <= 1):
                                self.EA_table.update({str(i) + str(j) + str(k): 1})
                            else:
                                self.EA_table.update({str(i) + str(j) + str(k): 0.5})

    def get_two_strategy(self, state):
        LG_state = str(state["PG1"]) + str(state["GS1"]) + str(state["ZBN5"])
        GL_state = str(state["ZBN5"]) + str(state["ZBN10"])
        evade_state = str(state["PG1"]) + str(state["GS1"]) + str(state["PG2"]) + str(state["GS2"])
        EA_state = str(state["PG1"]) + str(state["GS1"]) + str(state["PG2"]) + str(state["GS2"]) + str(state["PE"])

        prob_LG = self.local_table[LG_state]
        prob_GL = self.global_table[GL_state]
        prob_evade = self.evade_table[evade_state]
        prob_EA = self.energizer_table[EA_state]

        stragetegy_name = ["LG", "GL", "evade", "EA"]
        prob = np.array([prob_LG, prob_GL, prob_evade, prob_EA])
        index = np.where(prob == np.max(prob))[0]

        stragetegy = np.array(stragetegy_name)[index]
        if "evade" in stragetegy:
            self.strategy = "evade"
            return "evade"
        elif "EA" in stragetegy:
            self.strategy = "energizer"
            return "EA"
        elif "LG" in stragetegy:
            self.strategy = "local"
            return "LG"
        elif "GL" in stragetegy:
            self.strategy = "global"
            return "GL"

    def get_single_strategy(self, state):
        if self.two_strategy == "LG":
            pass


if __name__ == '__main__':
    s = strategyPolicyTable()
    state = {
        "PG1": 1, "GS1": 1,
        "PG2": 1, "GS2": 1,
        "BN5": 1, "BN10": 1, "PE": 1
    }
    print(s.get_strategy(state))
