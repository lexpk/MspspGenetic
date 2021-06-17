from gene import Gene, GraphGene
import numpy as np

class Population:
    def __init__(
        self,
        instance,
        Ge : Gene=GraphGene,
        size : int=100,
        activityMutationPropability=0.1,
        resourceMutationPropability=0.1,
        ageBiasFactor=1,
        parentalBiasFactor=1,
        parentCount=2
    ):
        transformedInstance = Ge.transform(instance)
        self.Gene = Ge
        self.size = size
        self.population = [Ge.random(transformedInstance) for i in range(size)]
        self.activityMutationPropability = activityMutationPropability
        self.resourceMutationPropability = resourceMutationPropability
        self.ageBiasFactor = ageBiasFactor
        self.parentalBiasFactor = parentalBiasFactor
        self.parentCount = parentCount
        self.median = np.median([
            x.score for x in self.population
        ])
        self.max = max([
            x.score for x in self.population
        ])
        self.score = np.average([
            x.score for x in self.population if x.score >= self.median
        ])
        self.bestRecordedMax = self.max
        self.bestRecordedIndividual = next(filter(
            lambda x: x.score == self.max,
            self.population
        ))
        self.bestRecordedScore = self.score
        self.stagnationPeriod = 0


    def age(self):
        for individual in self.population:
            individual.age += 1
        temp_sum = sum([max((self.population[i].score - self.median)*self.parentalBiasFactor, 1) for i in range(self.size)])
        probabilities = [max((self.population[i].score - self.median)*self.parentalBiasFactor, 1)/temp_sum for i in range(self.size)]
        parentsCombinations = [[np.random.choice(self.population, p=probabilities) for i in range(self.parentCount)] for i in range(self.size)]
        children = [self.Gene.recombine(parents).mutate(activityMutationPropability=self.activityMutationPropability, resourceMutationPropability=self.resourceMutationPropability) for parents in parentsCombinations]
        self.population = sorted(self.population + children, key=lambda x: x.score - x.age*self.ageBiasFactor)[-self.size:]
        self.median = np.median([x.score for x in self.population])
        self.max = max([x.score for x in self.population])
        self.score = np.average([
            x.score for x in self.population if x.score >= self.median
        ])
        if self.max > self.bestRecordedMax:
            self.bestRecordedMax = self.max
            self.bestRecordedIndividual = next(filter(
                lambda x: x.score == self.max,
                self.population
            ))
            self.stagnationPeriod = -1
        if self.score > self.bestRecordedScore + 0.1:
            self.bestRecordedScore = self.score
            self.stagnationPeriod = -1
        self.stagnationPeriod += 1
