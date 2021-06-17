from population import Population
from mspsp import MspspInstance, exampleFile
from gene import GraphGene
from time import time

class GeneticMspspSolver:
    def __init__(
        self,
        file,
        Gene=GraphGene,
        size=100,
        maxStagnation=20,
        activityMutationPropability=0.1,
        resourceMutationPropability=0.1,
        ageBiasFactor=1,
        parentalBiasFactor=1,
        parentCount=2
    ):
        self.population = Population(
            MspspInstance(file),
            Ge = Gene,
            size = size,
            activityMutationPropability = activityMutationPropability,
            resourceMutationPropability = resourceMutationPropability,
            ageBiasFactor=ageBiasFactor,
            parentalBiasFactor = parentalBiasFactor,
            parentCount = parentCount
        )
        self.maxStagnation = maxStagnation
        self.scores = [self.population.max]
        self.averages = [self.population.score]
        self.score = self.population.max
        self.solution = self.population.bestRecordedIndividual


    def solve(self, timeout = None, debug = False):
        if timeout:
            start = time()
        if debug:
            print(f"Generation 0: best = {self.population.max}, median = {self.population.median}, average = {self.population.score}")
            i = 1
        while True:
            self.population.age()
            self.scores.append(self.population.max)
            self.averages.append(self.population.score)
            if timeout:
                if start - time() > timeout:
                    break
            if self.population.stagnationPeriod > self.maxStagnation:
                break
            if debug:    
                print(f"Generation {i}: best = {self.population.max}, median = {self.population.median}, average = {self.population.score}")
                i += 1
        self.score = self.population.max
        self.solution = self.population.bestRecordedIndividual
        return self.solution

def genericTest():
    e = MspspInstance(exampleFile)
    p = Population(e, 100, GraphGene)
    s = GeneticMspspSolver(p, 100, 10)
    s.solve().show()