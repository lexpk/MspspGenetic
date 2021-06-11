from ortools.sat.python import cp_model
from minizinc import Model, Solver, Instance
import os
from datetime import timedelta


class MspspSolver:
    """
    A solver class for MSPSP instances as described in the lecture.
    """
    def __init__(self, filepath, solvername='ortools', search_ann=''):
        """
        Constructor for MspspSolver.
        Possible solvernames are 'ortools' and all solvers supported by python minizinc.
        Search annotations only work for the minizinc solvers.
        """
        try:
            self.data = read_dzn(filepath)
        except KeyError:
            print(f"{filepath} does not have the correct format.")
            return
        self.solvername = solvername
        if self.solvername == 'ortools':
            self.solver = cp_model.CpSolver()
            self.model = cp_model.CpModel()
            self.start = {}
            self.res_assignment = {}

            for act in range(self.data['nActs']):
                self.start[act] = self.model.NewIntVar(0, self.data['maxt'], f"{act}_start")

            for rel in range(self.data['nPrecs']):
                self.model.Add(self.start[self.data["pred"][rel]] + self.data['dur'][self.data['pred'][rel]] <= self.start[self.data['succ'][rel]])

            for act in range(self.data['nActs']):
                for res in range(self.data['nResources']):
                    self.res_assignment[act, res] = self.model.NewBoolVar("{act}_{res}")
                    if res not in self.data['USEFUL_RES'][act]:
                        self.model.Add(self.res_assignment[act, res] == 0)

            for res in range(self.data['nResources']):
                intervals = []
                for act in self.data['POTENTIAL_ACT'][res]:
                    intervals.append(self.model.NewOptionalIntervalVar(self.start[act], self.data["dur"][act], self.model.NewIntVar(0, self.data['maxt'], f"t_{res}_{act}"), self.res_assignment[act, res], f"int_{res}_{act}"))
                self.model.AddNoOverlap(intervals)

            for act in range(self.data['nActs']):
                for skill in range(self.data['nSkills']):
                    self.model.Add(self.data['sreq'][act][skill] <= sum(self.res_assignment[act, res] for res in self.data['USEFUL_RES'][act] if self.data['mastery'][res][skill]))

            self.model.Minimize(self.start[self.data['nActs'] - 1])
        else:
            self.solver = Solver.lookup(solvername)
            self.model = Model()
            self.model.add_file(filepath)
            self.model.add_file(os.path.dirname(__file__) + "\\parameter_model.mzn")
            self.model.add_string(
                f"""
                array[ACT] of var 0..maxt: start;
                array[ACT, RESOURCE] of var bool: resource_assignment;

                constraint forall (i in PREC) (start[pred[i]] + dur[pred[i]] <= start[succ[i]]);
                constraint forall (i in ACT, r in RESOURCE where not (r in USEFUL_RES[i])) (not resource_assignment[i, r]);
                constraint forall (r in RESOURCE, i, j in POTENTIAL_ACT[r] where i < j) (resource_assignment[i, r] /\ resource_assignment[j, r] -> start[i] + dur[i] <= start[j] \/ start[j] + dur[j] <= start[i]);
                constraint forall (s in SKILL, i in ACT) (sreq[i, s] <= sum (r in RESOURCE where mastery[r, s]) (bool2int(resource_assignment[i, r])));
                
                solve {search_ann}minimize start[nActs];
                """
            )


    def solve(self, timeout = None):
        """
        Solves the given MSPSP instance.
        """
        if self.solvername == 'ortools':
            if timeout is not None:
                self.solver.parameters.max_time_in_seconds = timeout
            self.status = self.solver.Solve(self.model)
        else:
            self.instance = Instance(self.solver, self.model)
            if timeout is not None:
                self.result = self.instance.solve(timeout=timedelta(seconds = timeout))
            else:
                self.result = self.instance.solve()
    def report(self):
        """
        Call after solve. Presents the solution.
        """
        if self.solvername == 'ortools':
            if self.solver.StatusName(self.status) == "OPTIMAL":
                for act in range(self.data['nActs']):
                    print(
                        f"Activity {act}:\t start = {self.solver.Value(self.start[act])}\t end = {self.solver.Value(self.start[act]) + self.data['dur'][act]}\t"
                        f"resources = {[res + 1 for res in self.data['USEFUL_RES'][act] if self.solver.Value(self.res_assignment[act, res])]}" 
                    )
            else:
                print("NO OPTIMAL SOLUTION FOUND")
        else:
            for (i, start) in enumerate(self.result["start"]):
                    print(
                        f"Activity {i}:\t start = {start}\t end = {start + self.data['dur'][i]}\t"
                        f"resources = {[x + 1 for x in range(len(self.result['resource_assignment'][i])) if self.result['resource_assignment'][i][x]]}"
                    )


def read_dzn(filepath):
    """
    Parses a .dzn file conforming with the described format for MSPSP and returns a dictonary with it's contents.
    """ 
    data = {}
    with open(filepath) as f:
        f.readline()
        content = f.read()
        for c in ['\n', ' ', '\t', '[', ']']:
            content = content.replace(c, "")
        content = content.split(";")
        for index, entry in enumerate(content):
            content[index] = entry.split("=")
        for t in content:
            if t[0] in ["nActs", "nSkills", "nResources", "nPrecs", "mint", "maxt", "nUnrels"]:
                data[t[0]] = int(t[1])
            if t[0] in ["dur", "pred", "succ", "unpred", "unsucc"]:
                l = t[1].split(',')
                for index, entry in enumerate(l):
                    if t[0] == "dur":
                        l[index] = int(entry)
                    else:
                        l[index] = int(entry) - 1
                data[t[0]] = l
            if t[0] == "sreq":
                l = t[1][1:-2].split(",|")
                for index, entry in enumerate(l):
                    l[index] = entry.split(',')
                    for jndex, fntry in enumerate(l[index]):
                        l[index][jndex] = int(fntry)
                data[t[0]] = l
            if t[0] == "mastery":
                l = t[1][1:-2].split(",|")
                for index, entry in enumerate(l):
                    l[index] = entry.split(',')
                    for jndex, fntry in enumerate(l[index]):
                        if fntry == "true":
                            l[index][jndex] = True
                        if fntry == "false":
                            l[index][jndex] = False
                data[t[0]] = l
            if t[0] in ["USEFUL_RES", "POTENTIAL_ACT"]:
                l = t[1][1:-1].split("},{")
                for index, entry in enumerate(l):
                    if entry == "":
                        l[index] = []
                    else:
                        l[index] = entry.split(',')
                        for jndex, fntry in enumerate(l[index]):
                            l[index][jndex] = int(fntry) - 1
                data[t[0]] = l
        return data

example_file = "E:\\Alex\\Documents\\uni\\Wien\\Problem Solving and Search\\MSPSP\\example.dzn"
example_strategy = ":: int_search(start, input_order, indomain_min)\n\t"