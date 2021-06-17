class MspspInstance:
    def __init__(self, filepath):
        with open(filepath) as f:
            f.readline()
            content = f.read()
            for c in ['\n', ' ', '\t', '[', ']']:
                content = content.replace(c, "")
            content = content.split(";")
            for index, entry in enumerate(content):
                content[index] = entry.split("=")
            for t in content:
                x = None
                if t[0] in ["nActs", "nSkills", "nResources", "nPrecs", "mint", "maxt", "nUnrels"]:
                    x = int(t[1])
                if t[0] in ["dur", "pred", "succ", "unpred", "unsucc"]:
                    x = t[1].split(',')
                    for index, entry in enumerate(x):
                        x[index] = int(entry) - (0 if t[0] == "dur" else 1)
                if t[0] == "sreq":
                    x = t[1][1:-2].split(",|")
                    for index, entry in enumerate(x):
                        x[index] = entry.split(',')
                        for jndex, fntry in enumerate(x[index]):
                            x[index][jndex] = int(fntry)
                if t[0] == "mastery":
                    x = t[1][1:-2].split(",|")
                    for index, entry in enumerate(x):
                        x[index] = entry.split(',')
                        for jndex, fntry in enumerate(x[index]):
                            if fntry == "true":
                                x[index][jndex] = True
                            if fntry == "false":
                                x[index][jndex] = False
                if t[0] in ["USEFUL_RES", "POTENTIAL_ACT"]:
                    x = t[1][1:-1].split("},{")
                    for index, entry in enumerate(x):
                        if entry == "":
                            x[index] = []
                        else:
                            x[index] = entry.split(',')
                            for jndex, fntry in enumerate(x[index]):
                                x[index][jndex] = int(fntry) - 1
                if x:
                    exec(f"self.{t[0]} = x")


class MspspSolution:
    def __init__(self, instance, start, resources, contributedSkill):
        self.instance = instance
        self.start = start
        self.end = list(map(lambda x, y: x + y, self.start, self.instance.dur))
        self.resources = resources
        self.contributedSkill = contributedSkill

    def show(self):
        for i in range(len(self.start)):
            print(
                f"Activity {i}:\t start = {self.start[i]}\t"
                f"resources/contributed skills = {[f'{res + 1}/{self.contributedSkill[i, res] + 1}' for res in self.resources[i]]}" 
            )

    def isValid(self):
        for act1 in range(self.instance.nActs):
            for act2 in range(0, act1):
                if len(set(self.resources[act1]) & set(self.resources[act2])) != 0 and min(self.end[act1] - self.start[act2], self.end[act2] - self.start[act1]) > 0:
                    print("overlap")
                    return False
                for prec in range(self.instance.nPrecs):
                    if (
                        self.end[act1] > self.start[act2] and
                        self.instance.pred[prec] == act1 and
                        self.instance.succ[prec] == act2
                    ) or (
                        self.end[act2] > self.start[act1] and
                        self.instance.pred[prec] == act2 and
                        self.instance.succ[prec] == act1
                    ):
                        print("precedence")
                        return False
            for skill, count in enumerate(self.instance.sreq[act1]):
                if count - len(list(filter(
                        lambda r: self.contributedSkill[act1, r] == skill,
                        self.resources[act1]
                ))) > 0:
                    print("skill")
                    return False
        return True
