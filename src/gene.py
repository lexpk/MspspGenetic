from functools import reduce
from operator import mul
from mspsp import MspspInstance, MspspSolution
from abc import ABC, abstractmethod
from typing import Sequence, Dict, Tuple
from random import choice, choices, randint, sample
import numpy as np
from numpy.random import binomial, permutation
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

class Gene(ABC):
    @abstractmethod
    def updateScore(self):
        pass
    
    @abstractmethod
    def recombine(parents):
        pass

    @abstractmethod
    def mutate(self):
        pass

    @abstractmethod
    def fromMspspSolution(solution : MspspSolution):
        pass

    @abstractmethod
    def toMspspSolution(self):
        pass

    @abstractmethod
    def random(instance):
        pass

    @abstractmethod
    def transform(instance):
        pass

    def show(self):
        self.toMspspSolution().show() 

class NaiveGene(Gene):
    def __init__(
            self, 
            instance : MspspInstance, 
            start : Sequence[int], 
            resources : Sequence[int],
            contributedSkill : Dict[Tuple[int, int], int]
        ):
        self.instance = instance
        self.start = start
        self.resources = resources
        self.contributedSkill = contributedSkill
        self.overlapFactor = instance.maxt
        self.precedenceFactor = instance.maxt**2
        self.skillFactor = instance.maxt
        self.updateScore()

    def updateScore(self):
        end = list(map(lambda x, y: x + y, self.start, self.instance.dur))
        overlappingPenalty = 0
        precedencePenalty = 0
        skillPenalty = 0
        for act1 in range(self.instance.nActs):
            for act2 in range(0, act1):
                if len(set(self.resources[act1]) & set(self.resources[act2])) != 0:
                    overlappingPenalty += self.overlapFactor * max(
                            min(
                                end[act1] - self.start[act2],
                                end[act2] - self.start[act1]
                            ),
                            0
                        )
                for prec in range(self.instance.nPrecs):
                    if (
                        end[act1] > self.start[act2] and
                        self.instance.pred[prec] == act1 and
                        self.instance.succ[prec] == act2
                    ) or (
                        end[act2] > self.start[act1] and 
                        self.instance.pred[prec] == act2 and 
                        self.instance.succ[prec] == act1
                    ):
                        precedencePenalty += self.precedenceFactor
            for skill, count in enumerate(self.instance.sreq[act1]):
                skillPenalty += self.skillFactor * max(
                    0,
                    count - len(list(filter(
                        lambda r: self.contributedSkill[act1, r] == skill,
                        self.resources[act1]
                    )))
                )
        self.score = - max(end) - overlappingPenalty - precedencePenalty - skillPenalty

    def recombine(parents):
        return reduce(lambda x, y: x * y, parents)

    def __mul__(self, parent):
        start = []
        resources = []
        contributedSkill = {}
        for act in range(self.instance.nActs):      
            start.append(
                min(self.start[act], parent.start[act]) +
                binomial(n=abs(self.start[act] - parent.start[act]), p=0.5)
            )
            inheritedResources = []
            for res in set(self.resources[act]) & set(parent.resources[act]):
                inheritedResources.append(res)
            remainder = list(
                (set(self.resources[act]) | set(parent.resources[act])) -
                (set(self.resources[act]) & set(parent.resources[act]))
            )
            resources.append(inheritedResources + sample(
                remainder,
                sum(self.instance.sreq[act]) - len(inheritedResources)
            ))
            for res in resources[act]:
                if (act, res) in self.contributedSkill:
                    selfchoice = self.contributedSkill[act, res]
                else:
                    selfchoice = parent.contributedSkill[act, res]
                if (act, res) in parent.contributedSkill:
                    parentchoice = parent.contributedSkill[act, res]
                else:
                    parentchoice = self.contributedSkill[act, res]
                contributedSkill[act, res] = choice([selfchoice, parentchoice])
        return NaiveGene(self.instance, start, resources, contributedSkill)

    def mutate(self, activityMutationPropablity = 0.1, resourceMutationPropablity = 0.1):
        for act in range(self.instance.nActs): 
            if choices([0, 1], k = 1, weights=[1 - activityMutationPropablity, 10*activityMutationPropablity])[0]:
                self.start[act] = randint(0, self.instance.maxt)
            if choices([0, 1], k = 1, weights=[1 - activityMutationPropablity, activityMutationPropablity])[0]:
                self.start[act] =  choice([
                    max(self.start[act] - 1, 0), 
                    min(self.start[act] + 1, self.instance.maxt - self.instance.dur[act])
                ])
            if self.resources[act]:
                if choices([0, 1], k = 1, weights=[1 - resourceMutationPropablity/10, resourceMutationPropablity/10])[0]:
                    self.resources[act] = sample(
                        self.instance.USEFUL_RES[act],
                        sum(self.instance.sreq[act])
                    )
                    for res in self.resources[act]:
                        self.contributedSkill[act, res] = choice(list(filter(
                            lambda i: self.instance.mastery[res][i],
                            range(self.instance.nSkills)
                        )))
                newRes = None
                if choices([0, 1], k = 1, weights=[1 - resourceMutationPropablity, resourceMutationPropablity])[0]:
                    self.resources[act].remove(choice(self.resources[act]))
                    newRes = choice(list(filter(
                        lambda res: res not in self.resources[act],
                        self.instance.USEFUL_RES[act]
                    )))
                    self.resources[act].append(newRes)
                for res in self.resources[act]:
                    if res == newRes or randint(1, NaiveGene.mutationFactor) <= 20:
                        self.contributedSkill[act, res] = choice(list(filter(
                            lambda index: self.instance.mastery[res][index],
                            range(self.instance.nSkills)
                        )))
            self.updateScore()
        return self


    def fromMspspSolution(solution: MspspSolution):
        result = NaiveGene()
        result.instance = solution.instance
        result.start = solution.start
        result.resources = solution.resources
        result.contributedSkill = solution.contributedSkill
        result.overlapFactor = solution.instance.maxt
        result.precedenceFactor = solution.instance.maxt**2
        result.skillFactor = solution.instsance.maxt
        result.updateScore()
        solution.instance, solution.start, solution.resources, solution.contributedSkill

    def toMspspSolution(self):
        return MspspSolution(self.instance, self.start, self.resources, self.contributedSkill)

    def random(instance : MspspInstance):
        start = []
        resources = []
        contributedSkill = {}
        for act in range(instance.nActs):
            start.append(randint(0, instance.maxt - instance.dur[act]))
            resources.append(sample(instance.USEFUL_RES[act], sum(instance.sreq[act])))
            for res in resources[act]:
                contributedSkill[act, res] = choice(list(filter(
                    lambda index: instance.mastery[res][index],
                    range(instance.nSkills)
                )))
        return NaiveGene(instance, start, resources, contributedSkill)

    def transform(instance : MspspInstance):
        return instance

class GraphGene(Gene):
    def __init__(
            self,
            instance,
            precedenceGraph,
            activityOrder,
            resourceGraph,
            matching     
        ):
        self.instance = instance
        self.precedenceGraph = precedenceGraph
        self.activityOrder = activityOrder
        self.resourceGraph = resourceGraph
        self.matching = matching
        self.age = 0
        self.score = None

    def updateScore(self):
        start = [0 for i in range(self.instance.nActs)]
        resources = [[] for i in range(self.instance.nActs)]
        resourceSchedule = [0 for i in range(self.instance.nResources)]
        for act in self.activityOrder:
            resources[act] = self.matching[act][1]
            skill = 0
            skillCount = 0
            for res in self.matching[act][1]:
                while skillCount == self.instance.sreq[act][skill]:
                    skill += 1
                    skillCount = 0
                else:
                    skillCount += 1
            start[act] = max(
                [resourceSchedule[i] for i in resources[act]] +
                [start[p] + self.instance.dur[p] for i, p in enumerate(self.instance.pred) if self.instance.succ[i] == act] + 
                [0]
            )
            for res in resources[act]:
                resourceSchedule[res] = start[act] + self.instance.dur[act]
        self.score = -start[self.instance.nActs - 1]
    
    def recombine(parents):
        activityOrder = sorted(
            range(parents[0].instance.nActs),
            key = lambda act: sum([parent.activityOrder.index(act) for parent in parents])
        )
        matching = []
        for act in range(parents[0].instance.nActs):
            targets = [parent.matching[act][1].copy() for parent in parents]
            for i in permutation(parents[0].matching[act][0]):
                target1 = choice(targets)
                for target2 in targets:
                    if target1[i] != target2[i]:
                        j = i
                        while target1[j] in target2:
                            nextj = np.where(target2 == target1[j])[0][0]
                            target2[j] = target1[j]
                            j = nextj
                        target2[j] = target1[j]
            matching.append((np.array(parents[0].matching[act][0], dtype='int32'), np.array(targets[0], dtype='int32')))
        return GraphGene(parents[0].instance, parents[0].precedenceGraph, activityOrder, parents[0].resourceGraph, matching)

    def mutate(self, activityMutationPropability = 0.1, resourceMutationPropability = 0.1):
        if choices([0, 1], k = 1, weights=[1 - activityMutationPropability, activityMutationPropability])[0]:
            start, end = self.randomMaxUnrelatedSection()
            length = randint(0, end - start)
            start = randint(start, end - length)
            end = start + length
            self.activityOrder[start:end + 1] = permutation(self.activityOrder[start:end + 1])
            if not self.toMspspSolution().isValid():
                pass
        if choices([0, 1], k = 1, weights=[1 - resourceMutationPropability, resourceMutationPropability])[0]:
            start, end = self.randomMaxUnrelatedSection()
            length = randint(0, end - start)
            start = randint(start, end - length)
            end = start + length
            order = permutation(range(start, end+1))
            weights = permutation(list(range(1, self.instance.nResources + 1)))
            for act in order:
                self.resourceGraph[act].data = np.array([weights[self.resourceGraph[act].col[res]] for res in range(self.resourceGraph[act].nnz)])
                self.matching[act] = min_weight_full_bipartite_matching(self.resourceGraph[act])
                for res in self.resourceGraph[act].col:
                    weights[res] += randint(1, 2*self.instance.nResources)
        self.updateScore()
        return self

    def randomMaxUnrelatedSection(self):
        start = choice(range(self.instance.nActs))
        end = start
        mostRecent = start
        excluded = []
        openStart = (start != 0)
        openEnd = (end != len(self.activityOrder) - 1)
        while openStart or openEnd:
            for i, act in enumerate(self.instance.succ):
                    if act == self.activityOrder[mostRecent]:
                        excluded.append(self.instance.pred[i])
            for i, act in enumerate(self.instance.pred):
                    if act == self.activityOrder[mostRecent]:
                        excluded.append(self.instance.succ[i])
            if openStart and (not openEnd or choice([0,1])):
                if self.activityOrder[start - 1] not in excluded:
                    start -= 1
                    mostRecent = start
                else:
                    openStart = False
            else:
                if self.activityOrder[end + 1] not in excluded:
                    end += 1
                    mostRecent = end
                else:
                    openEnd = False
        return start, end

    def fromMspspSolution(solution : MspspSolution):
        pass

    def toMspspSolution(self):
        start = [0 for i in range(self.instance.nActs)]
        resources = [[] for i in range(self.instance.nActs)]
        contributedSkill = {}
        resourceSchedule = [0 for i in range(self.instance.nResources)]
        for act in self.activityOrder:
            resources[act] = self.matching[act][1]
            skill = 0
            skillCount = 0
            for i, res in enumerate(self.matching[act][1]):
                while skillCount == self.instance.sreq[act][skill]:
                    skill += 1
                    skillCount = 0
                else:
                    skillCount += 1
                    contributedSkill[act, res] = skill
            start[act] = max(
                [resourceSchedule[i] for i in resources[act]] +
                [start[p] + self.instance.dur[p] for i, p in enumerate(self.instance.pred) if self.instance.succ[i] == act] + 
                [0]
            )
            for res in resources[act]:
                resourceSchedule[res] = start[act] + self.instance.dur[act]
        return MspspSolution(self.instance, start, resources, contributedSkill)


    def random(transformedInstance):
        instance, precedenceGraph, resourceGraph = transformedInstance
        activityOrder = []
        
        blocked = list(precedenceGraph.col.copy())
        remaining = list(range(0, instance.nActs))
        while remaining:
            next = choice(list(filter(
                lambda act: (act not in activityOrder) and (act not in blocked),
                remaining
            )))
            activityOrder.append(next)
            remaining.remove(next)
            for i in range(precedenceGraph.nnz):
                if precedenceGraph.row[i] == next:
                    blocked.remove(precedenceGraph.col[i])
        matching = []
        for act in range(instance.nActs):
            resourceGraph[act].data = permutation(list(range(1, resourceGraph[act].nnz + 1)))
            matching.append(min_weight_full_bipartite_matching(resourceGraph[act]))
        result = GraphGene(instance, precedenceGraph, activityOrder, resourceGraph, matching)
        result.updateScore()
        return result

    def transform(instance):
        precedenceGraph = coo_matrix(
            (
                np.ones(instance.nPrecs),
                (
                    np.array(instance.pred),
                    np.array(instance.succ)
                )
            ),
            shape = (instance.nPrecs, instance.nPrecs)
        )
        resourceGraph = []
        
        for act in range(instance.nActs):
            i = 0
            pred = []
            succ = []
            for skill, req in enumerate(instance.sreq[act]):
                for j in range(req):
                    for res in instance.USEFUL_RES[act]:
                        if instance.mastery[res][skill]:
                            pred.append(i)
                            succ.append(res)
                    i+=1
            resourceGraph.append(coo_matrix(
                (
                    np.ones(len(pred)),
                    (
                        np.array(pred),
                        np.array(succ)
                    )
                ),
                shape = (i, instance.nResources)
            ))
        return instance, precedenceGraph, resourceGraph
