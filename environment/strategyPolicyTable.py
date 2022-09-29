import numpy as np


class strategyPolicyTable:
    def __init__(self):
        self.state_num = {
            "PG1": 3, "GS1": 3,
            "PG2": 3, "GS2": 3,
            "PE": 3, "BW": 3, "BB": 3, "ZBW": 2, "ZBB": 2
        }
        self.local_table = {}
        for i in range(self.state_num["ZBW"]):
            if i == 0:
                self.local_table.update({str(i): 1})
            else:
                self.local_table.update({str(i): 0.5})

        self.global_table = {}
        for i in range(self.state_num["ZBW"]):
            for j in range(self.state_num["ZBB"]):
                if i == 1 and j == 0:
                    self.global_table.update({str(i) + str(j): 1})
                else:
                    self.global_table.update({str(i) + str(j): 0.5})

        # self.evade_table = {}
        # for i in range(self.state_num["PG1"]):
        #     for j in range(self.state_num["GS1"]):
        #         for k in range(self.state_num["PG2"]):
        #             for l in range(self.state_num["GS2"]):
        #                 if (j == 0 and i == 0) or (k == 0 and l == 0):
        #                     self.evade_table.update({str(i) + str(j) + str(k) + str(l): 1})
        #                 else:
        #                     self.evade_table.update({str(i) + str(j) + str(k) + str(l): 0.5})

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
        local_state = str(state["ZBW"])
        global_state = str(state["ZBW"]) + str(state["ZBB"])
        # evade_state = str(state["PG1"]) + str(state["GS1"]) + str(state["PG2"]) + str(state["GS2"])
        energizer_state = str(state["PE"])
        approach_state = str(state["PG1"]) + str(state["GS1"]) + str(state["PG2"]) + str(state["GS2"])

        prob_local = self.local_table[local_state]
        prob_global = self.global_table[global_state]
        # prob_evade = self.evade_table[evade_state]
        prob_energizer = self.energizer_table[energizer_state]
        prob_approach = self.approach_table[approach_state]

        stragetegy_name = ["local", "global", "energizer", "approach"]
        prob = np.array([prob_local, prob_global, prob_energizer, prob_approach])
        index = np.where(prob == np.max(prob))[0]

        stragetegy = np.array(stragetegy_name)[index]
        if "approach" in stragetegy:
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
            "PE": 3, "BW": 3, "BB": 3, "ZBW": 2, "ZBB": 2
        }
        self.two_strategy = "LG"
        self.strategy = "local"
        self.two_strategy_end = True
        self.LG_table = {}
        for i in range(self.state_num["ZBW"]):
            if i == 0:
                self.LG_table.update({str(i): 1})
            else:
                self.LG_table.update({str(i): 0.5})

        self.GL_table = {}
        for i in range(self.state_num["ZBW"]):
            for j in range(self.state_num["ZBB"]):
                if i == 1 and j == 0:
                    self.GL_table.update({str(i) + str(j): 1})
                else:
                    self.GL_table.update({str(i) + str(j): 0.5})

        # self.eC_table = {}
        # for i in range(self.state_num["PG1"]):
        #     for j in range(self.state_num["GS1"]):
        #         for k in range(self.state_num["PG2"]):
        #             for l in range(self.state_num["GS2"]):
        #                 if (j == 0 and i == 0) or (k == 0 and l == 0):
        #                     self.eC_table.update({str(i) + str(j) + str(k) + str(l): 1})
        #                 else:
        #                     self.eC_table.update({str(i) + str(j) + str(k) + str(l): 0.5})

        self.EA_table = {}
        for i in range(self.state_num["PG1"]):
            for k in range(self.state_num["PG2"]):
                for m in range(self.state_num["PE"]):
                    if (m <= 1 and i <= 1) or (m <= 1 and k <= 1):
                        self.EA_table.update({str(i) + str(k) + str(m): 1})
                    else:
                        self.EA_table.update({str(i) + str(k) + str(m): 0.5})

    def get_two_strategy(self, state):
        LG_state = str(state["ZBW"])
        GL_state = str(state["ZBW"]) + str(state["ZBB"])
        # ec_state = str(state["PG1"]) + str(state["GS1"]) + str(state["PG2"]) + str(state["GS2"])
        EA_state = str(state["PG1"])  + str(state["PG2"])  + str(state["PE"])

        prob_LG = self.LG_table[LG_state]
        prob_GL = self.GL_table[GL_state]
        # prob_eC = self.eC_table[ec_state]
        prob_EA = self.EA_table[EA_state]

        stragetegy_name = ["LG", "GL", "EA"]
        prob = np.array([prob_LG, prob_GL, prob_EA])
        index = np.where(prob == np.max(prob))[0]

        stragetegy = np.array(stragetegy_name)[index]
        # if "eE" in stragetegy:
        #     return "eE"
        if prob[index][0] == 0.5:
            return "GL"
        else:
            if "EA" in stragetegy:
                return "EA"
            elif "LG" in stragetegy:
                return "LG"
            elif "GL" in stragetegy:
                return "GL"

    def get_single_strategy(self, state):
        if self.two_strategy == "LG":
            if state["ZBW"] == 1 and state["ZBB"] == 0:
                self.strategy = "global"
                return "global"
            elif state["ZBW"] == 0:
                if self.strategy == "global":
                    self.two_strategy_end = True
                    return None
                self.strategy = "local"
                return "local"
            else:
                self.two_strategy_end = True

        if self.two_strategy == "GL":
            if state["ZBW"] == 0:
                self.strategy = "local"
                return "local"
            elif state["ZBB"] == 0:
                if self.strategy == "global":
                    self.two_strategy_end = True
                    return None
                self.strategy = "global"
                return "global"
            else:
                self.two_strategy_end = True

        if self.two_strategy == "EA":
            if state["GS1"] == 2 or state["GS2"] == 2:
                self.strategy = "approach"
                return "approach"
            elif state["PE"] <= 1:
                if self.strategy == "approach":
                    self.two_strategy_end = True
                    return None
                self.strategy = "energizer"
                return "energizer"
            else:
                self.two_strategy_end = True

        # if self.two_strategy == "eC":
        #     if (state["PG1"] == 0 and state["GS1"] == 0 and state["PE"] <= 1) or (
        #             state["PG2"] == 0 and state["GS2"] == 0 and state["PE"] <= 1):
        #         return "counterattack"
        #     elif (state["PG1"] == 0 and state["GS1"] == 0) or (state["PG2"] == 0 and state["GS2"] == 0):
        #         if self.strategy == "counterattack":
        #             self.two_strategy_end = True
        #             return None
        #         return "evade"
        #     else:
        #         self.two_strategy_end = True


if __name__ == '__main__':
    s = strategyPolicyTable()
    state = {
        "PG1": 1, "GS1": 1,
        "PG2": 1, "GS2": 1,
        "BN5": 1, "BN10": 1, "PE": 1
    }
    print(s.get_strategy(state))
