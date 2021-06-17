from numpy.core.shape_base import stack
from geneticMspspSolver import GeneticMspspSolver
from gene import GraphGene
import matplotlib.pyplot as plt
import numpy as np
import os
import time

parameters = {
    'populationSize': 128,
    'activityMutationPropability': 0.04681012215832443,
    'resourceMutationPropability': 0.43980729762565596,
    'ageBiasFactor': 6.75109938384272,
    'parentalBiasFactor': 9.156206937872355,
    'parentCount': 3
}

if __name__ == "__main__":
    benchmark_instances_dir = os.path.join(os.path.dirname(__file__), "..\\benchmark_instances")
    plot_dir = os.path.join(os.path.dirname(__file__), "..\\plots")

    bestScores = []
    averages = []
    standardDeviations = []
    bestTimes = []
    averageTimes = []

    if not os.path.exists(os.path.join(os.path.dirname(__file__), "..\\images")):
        os.mkdir(os.path.join(os.path.dirname(__file__), "..\\images"))

    for file in os.listdir(benchmark_instances_dir)[:20]:
        f = os.path.join(benchmark_instances_dir, file)
        print(f"Solving {file}...")
        start = time.time()
        scores = []
        times = []
        for i in range(5):
            start = time.time()
            solver = GeneticMspspSolver(f,
                GraphGene,
                size = parameters['populationSize'],
                activityMutationPropability = parameters['activityMutationPropability'],
                resourceMutationPropability = parameters['resourceMutationPropability'],
                ageBiasFactor=parameters['ageBiasFactor'],
                parentalBiasFactor=parameters['parentalBiasFactor'],
                parentCount=parameters['parentCount']
            )
            solver.solve()
            t = time.time() - start
            scores.append(solver.score)
            times.append(t)
            plt.plot(list(range(len(solver.scores))), solver.averages, label="run {number}".format(number = i))
            print(f"Attempt {i}: {solver.score} in {t}")
        plt.xlabel('Generation')
        plt.ylabel('Score')
        plt.savefig(os.path.join(os.path.dirname(__file__), f"..\\plots\\{file}.png"))
        plt.clf()
        print(f"""Best = {max(scores)} in {max(times, key=lambda e: scores[times.index(e)])}, Average = {sum(scores)/5} in {sum(times)/5}, Std = {np.std(scores)}""")
        bestScores.append(max(scores))
        bestTimes.append(max(times, key=lambda e: scores[times.index(e)]))
        averages.append(sum(scores)/5)
        averageTimes.append(sum(times)/5)
        standardDeviations.append(np.std(scores))
        bestTimes
    print(f"Best Scores: {bestScores}")
    print(f"Best Times: {bestTimes}")
    print(f"Averages = {averages}")
    print(f"Average Times = {averageTimes}")




