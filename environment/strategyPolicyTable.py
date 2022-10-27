import numpy as np


class strategyPolicyTable:
    def __init__(self):
        """
              PG1:鬼1距离Pac-man的距离，分为三段，0表示近，1表示中，2表示远
              PG2:鬼1距离Pac-man的距离，分为三段，0表示近，1表示中，2表示远
              PE:最近energizer距离Pac-man的距离，分为三段，0表示近，1表示中，2表示远
              GS1:鬼1的状态，0表示正常，1表示scared,2表示scared
              GS1:鬼2的状态，0表示正常，1表示scared,2表示scared
              BW:Pac-man n步以内的豆子数，分三段，0表示少，1表示中，2表示多
              BB:Pac-man n步以外的豆子数，分三段，0表示少，1表示中，2表示多
              ZBW：Pac-man n步以内是否有豆子，0表示没有，1表示有
              ZBB：Pac-man n步以外是否有豆子，0表示没有，1表示有
        """
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
        local_state = str(state["ZBW"])
        global_state = str(state["ZBW"]) + str(state["ZBB"])
        evade_state = str(state["PG1"]) + str(state["GS1"]) + str(state["PG2"]) + str(state["GS2"])
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
        elif "energizer" in stragetegy:
            return "energizer"
        elif "local" in stragetegy:
            return "local"
        elif "global" in stragetegy:
            return "global"


class twoStrategyPolicyTable:
    def __init__(self):
        """
        PG1:鬼1距离Pac-man的距离，分为三段，0表示近，1表示中，2表示远
        PG2:鬼1距离Pac-man的距离，分为三段，0表示近，1表示中，2表示远
        PE:最近energizer距离Pac-man的距离，分为三段，0表示近，1表示中，2表示远
        GS1:鬼1的状态，0表示正常，1表示scared,2表示scared
        GS1:鬼2的状态，0表示正常，1表示scared,2表示scared
        BW:Pac-man n步以内的豆子数，分三段，0表示少，1表示中，2表示多
        BB:Pac-man n步以外的豆子数，分三段，0表示少，1表示中，2表示多
        ZBW：Pac-man n步以内豆子是否为0，0表示为0，1表示不为0
        ZBB：Pac-man n步以外豆子是否为0，0表示为0，1表示不为0
        """
        self.state_num = {
            "PG1": 3, "GS1": 3,
            "PG2": 3, "GS2": 3,
            "PE": 3, "BW": 3, "BB": 3, "ZBW": 2, "ZBB": 2
        }
        self.available = ["EA", "GL", "LG"]
        self.two_strategy = "LG"
        self.strategy = "local"
        self.two_strategy_end = True
        # 是否在evade
        self.evading = False
        # num用来debug,没有实际意义
        self.num = 0
        # local global
        self.LG_table = {}
        for i in range(self.state_num["ZBW"]):
            if i == 0:
                self.LG_table.update({str(i): 1})
            else:
                self.LG_table.update({str(i): 0.5})

        # global local
        self.GL_table = {}
        for i in range(self.state_num["ZBW"]):
            for j in range(self.state_num["ZBB"]):
                if i == 1 and j == 0:
                    self.GL_table.update({str(i) + str(j): 1})
                else:
                    self.GL_table.update({str(i) + str(j): 0.5})

        # evade
        self.evade_table = {}
        for i in range(self.state_num["PG1"]):
            for j in range(self.state_num["GS1"]):
                for k in range(self.state_num["PG2"]):
                    for l in range(self.state_num["GS2"]):
                        if (j == 0 and i == 0) or (k == 0 and l == 0):
                            self.evade_table.update({str(i) + str(j) + str(k) + str(l): 1})
                        else:
                            self.evade_table.update({str(i) + str(j) + str(k) + str(l): 0.5})
        # energizer approach
        self.EA_table = {}
        for i in range(self.state_num["PG1"]):
            for k in range(self.state_num["PG2"]):
                for m in range(self.state_num["PE"]):
                    if (m <= 1 and i <= 1) or (m <= 1 and k <= 1):
                        self.EA_table.update({str(i) + str(k) + str(m): 1})
                    else:
                        self.EA_table.update({str(i) + str(k) + str(m): 0.5})
        # energizer local
        self.EL_table = {}
        for i in range(self.state_num["PG1"]):
            for k in range(self.state_num["PG2"]):
                for m in range(self.state_num["PE"]):
                    for j in range(self.state_num["ZBW"]):
                        if m <= 1 and j == 0 and i > 1 and k > 1:
                            self.EL_table.update({str(i) + str(k) + str(m) + str(j): 1})
                        else:
                            self.EL_table.update({str(i) + str(k) + str(m) + str(j): 0})
        # energizer global
        self.EG_table = {}
        for i in range(self.state_num["PG1"]):
            for k in range(self.state_num["PG2"]):
                for m in range(self.state_num["PE"]):
                    for j in range(self.state_num["ZBW"]):
                        if m <= 1 and j != 0 and i > 1 and k > 1:
                            self.EG_table.update({str(i) + str(k) + str(m) + str(j): 1})
                        else:
                            self.EG_table.update({str(i) + str(k) + str(m) + str(j): 0})

    def is_evade(self, state):
        evade_state = str(state["PG1"]) + str(state["GS1"]) + str(state["PG2"]) + str(state["GS2"])
        prob_evade = self.evade_table[evade_state]
        if prob_evade > 0.5:
            self.evading = True
            return True
        else:
            return False

    def get_two_strategy(self, state):

        state = {
            "LG": str(state["ZBW"]),
            "GL": str(state["ZBW"]) + str(state["ZBB"]),
            "EA": str(state["PG1"]) + str(state["PG2"]) + str(state["PE"]),
            "EL": str(state["PG1"]) + str(state["PG2"]) + str(state["PE"]) + str(state["ZBW"]),
            "EG": str(state["PG1"]) + str(state["PG2"]) + str(state["PE"]) + str(state["ZBW"]),
        }
        prob_strategy = {
            "LG": self.LG_table[state["LG"]],
            "GL": self.GL_table[state["GL"]],
            "EA": self.EA_table[state["EA"]],
            "EL": self.EL_table[state["EL"]],
            "EG": self.EG_table[state["EG"]],
        }

        prob = []
        for s in self.available:
            prob.append(prob_strategy[s])
        prob = np.array(prob)

        index = np.where(prob == np.max(prob))[0]
        stragetegy = np.array(self.available)[index]

        if prob[index][0] == 0.5:
            return "GL"
        else:
            return stragetegy[0]

    def get_single_strategy(self, state):
        if self.two_strategy == "LG":
            if state["ZBW"] == 0:
                if self.strategy == "global":
                    self.two_strategy_end = True
                    self.num = 0
                    return None
                self.num = 1
                self.strategy = "local"
                return "local"
            elif state["ZBW"] == 1 and state["ZBB"] == 0:
                self.strategy = "global"
                self.num = 2
                return "global"
            else:
                self.num = 0
                self.two_strategy_end = True

        if self.two_strategy == "GL":
            if state["ZBB"] == 0:
                if self.strategy == "local":
                    self.two_strategy_end = True
                    self.num = 0
                    return None
                self.num = 1
                self.strategy = "global"
                return "global"
            elif state["ZBW"] == 0:
                self.strategy = "local"
                self.num = 2
                return "local"

            else:
                self.num = 0
                self.two_strategy_end = True

        if self.two_strategy == "EA":
            if state["PE"] <= 1:
                if self.strategy == "approach":
                    self.two_strategy_end = True
                    self.num = 0
                    return None
                self.num = 1
                self.strategy = "energizer"
                return "energizer"
            elif state["GS1"] == 2 or state["GS2"] == 2:
                self.strategy = "approach"
                if self.num == 0:
                    x = 0
                self.num = 2
                return "approach"
            else:
                self.num = 0
                self.two_strategy_end = True

        if self.two_strategy == "EL":
            if state["PE"] <= 1:
                if self.strategy == "local":
                    self.two_strategy_end = True
                    self.num = 0
                    return None
                self.num = 1
                self.strategy = "energizer"
                return "energizer"
            elif state["ZBW"] == 0:
                self.strategy = "local"
                self.num = 2
                return "local"
            else:
                self.num = 0
                self.two_strategy_end = True

        if self.two_strategy == "EG":
            if state["PE"] <= 1:
                if self.strategy == "global":
                    self.two_strategy_end = True
                    self.num = 0
                    return None
                self.num = 1
                self.strategy = "energizer"
                return "energizer"
            elif state["ZBB"] == 0:
                self.strategy = "global"
                self.num = 2
                return "global"
            else:
                self.num = 0
                self.two_strategy_end = True


class triStrategyPolicyTable:
    def __init__(self):
        """
              PG1:鬼1距离Pac-man的距离，分为三段，0表示近，1表示中，2表示远
              PG2:鬼1距离Pac-man的距离，分为三段，0表示近，1表示中，2表示远
              PE:最近energizer距离Pac-man的距离，分为三段，0表示近，1表示中，2表示远
              GS1:鬼1的状态，0表示正常，1表示scared,2表示scared
              GS1:鬼2的状态，0表示正常，1表示scared,2表示scared
              BW:Pac-man n步以内的豆子数，分三段，0表示少，1表示中，2表示多
              BB:Pac-man n步以外的豆子数，分三段，0表示少，1表示中，2表示多
              ZBW：Pac-man n步以内豆子是否为0，0表示为0，1表示不为0
              ZBB：Pac-man n步以外豆子是否为0，0表示为0，1表示不为0
              BD:Pac-man 出生点与当前点的local豆子差
              """
        self.state_num = {
            "PG1": 3, "GS1": 3,
            "PG2": 3, "GS2": 3,
            "PE": 3, "BW": 3, "BB": 3, "ZBW": 2, "ZBB": 2,
            "BD": 2,
        }
        self.available = ["eEA", "LGL", "GLG"]
        self.tri_strategy = "LGL"
        self.strategy = "local"
        self.tri_strategy_end = True
        # 是否在evade
        self.evading = False
        # num用来debug,没有实际意义
        self.num = 0
        # evade energizer approach
        self.eEA_table = {}
        for i in range(self.state_num["PG1"]):
            for j in range(self.state_num["PG2"]):
                for k in range(self.state_num["GS1"]):
                    for l in range(self.state_num["GS2"]):
                        for m in range(self.state_num["PE"]):
                            if ((i <= 0 and k == 0) or (j <= 0 and l == 0)) and m <= 1:
                                self.eEA_table.update({str(i) + str(j) + str(k) + str(l) + str(m): 1})
                            else:
                                self.eEA_table.update({str(i) + str(j) + str(k) + str(l) + str(m): 0})

        # # VAL
        # self.VAL_table = {}
        # for i in range(self.state_num["PG1"]):
        #     for j in range(self.state_num["PG2"]):
        #         for k in range(self.state_num["GS1"]):
        #             for l in range(self.state_num["GS2"]):
        #                 for m in range(self.state_num["PE"]):
        #                     if ((i <= 0 and k == 0) or (j <= 0 and l == 0)) and m > 1:
        #                         self.VAL_table.update({str(i) + str(j) + str(k) + str(l) + str(m): 1})
        #                     else:
        #                         self.VAL_table.update({str(i) + str(j) + str(k) + str(l) + str(m): 0})

        # LGL
        self.LGL_table = {}
        for i in range(self.state_num["ZBW"]):
            if i == 0:
                self.LGL_table.update({str(i): 1})
            else:
                self.LGL_table.update({str(i): 0.5})

        # GLG
        self.GLG_table = {}
        for i in range(self.state_num["ZBW"]):
            for j in range(self.state_num["ZBB"]):
                if i == 1 and j == 0:
                    self.GLG_table.update({str(i) + str(j): 1})
                else:
                    self.GLG_table.update({str(i) + str(j): 0.5})

    def get_tri_strategy(self, state):
        tri_state = {
            "eEA": str(state["PG1"]) + str(state["PG2"]) + str(state["GS1"]) + str(state["GS2"]) + str(state["PE"]),
            # "VAL": str(state["PG1"]) + str(state["PG2"]) + str(state["GS1"]) + str(state["GS2"]) + str(state["PE"]),
            "LGL": str(state["ZBW"]),
            "GLG": str(state["ZBW"]) + str(state["ZBB"]),
        }
        prob_strategy = {
            "eEA": self.eEA_table[tri_state["eEA"]],
            # "VAL": self.VAL_table[state["VAL"]],
            "LGL": self.LGL_table[tri_state["LGL"]],
            "GLG": self.GLG_table[tri_state["GLG"]],
        }
        prob = []
        for s in self.available:
            prob.append(prob_strategy[s])
        prob = np.array(prob)

        index = np.where(prob == np.max(prob))[0]
        stragetegy = np.array(self.available)[index]
        if prob[index][0] == 0.5:
            return "GLG"
        else:
            return stragetegy[0]

    def get_single_strategy(self, state):
        if self.tri_strategy == "eEA":
            if state["PE"] == 1:
                self.num = 1
                self.strategy = "evade"
            elif state["PE"] == 0:
                self.num = 2
                self.strategy = "energizer"
            elif state["GS1"] == 2 and state["GS2"] == 2:
                self.strategy = "approach"
                self.num = 3
            else:
                self.num = 0
                self.tri_strategy_end = True
        if self.tri_strategy == "LGL":
            if state["ZBW"] == 0:
                if self.num == 0:
                    self.num = 1
                if self.num == 2:
                    self.num = 3
                self.strategy = "local"
            elif state["ZBW"] == 1 and state["ZBB"] == 0:
                if self.num == 3:
                    self.tri_strategy_end = True
                    self.num = 0
                    return None
                self.num = 2
                self.strategy = "global"
            else:
                self.num = 0
                self.tri_strategy_end = True

        if self.tri_strategy == "GLG":
            if state["ZBW"] == 1 and state["ZBB"] == 0:
                if self.num == 0:
                    self.num = 1
                if self.num == 2:
                    self.num = 3
                self.strategy = "global"
            elif state["ZBW"] == 0:
                if self.num == 3:
                    self.tri_strategy_end = True
                    self.num = 0
                    return None
                self.num = 2
                self.strategy = "local"

            else:
                self.num = 0
                self.tri_strategy_end = True

        return self.strategy
