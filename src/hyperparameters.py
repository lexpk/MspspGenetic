from geneticMspspSolver import GeneticMspspSolver
from gene import GraphGene
from random import sample
import os
import time
import optuna

sizemax = 512
numberOfInstances = 4
n_trials = 128

benchmark_instances_dir = os.path.join(os.path.dirname(__file__), "..\\benchmark_instances")

class Objective(object):
        def __init__(self, numberOfInstances = numberOfInstances, timeBias = 0.1):
            self.benchFiles = sample(os.listdir(benchmark_instances_dir), k=numberOfInstances)
            self.timeBias = timeBias
            nl = '\n'
            print(
                f"""
                Selected instances:
                {nl.join(self.benchFiles)}            
                """
            )

        def __call__(self, trial):
            populationSize = trial.suggest_int("populationSize", 16, sizemax, 16)
            activityMutationPropability = trial.suggest_float("activityMutationPropability", 0, 1)
            resourceMutationPropability = trial.suggest_float("resourceMutationPropability", 0, 1)
            ageBiasFactor = trial.suggest_float("ageBiasFactor", 0, 10)
            parentalBiasFactor = trial.suggest_float("parentalBiasFactor", 0, 10)
            parentCount=trial.suggest_int("parentCount", 1, 10)
            score = 0
            for file in self.benchFiles:
                f = os.path.join(benchmark_instances_dir, file)
                start = time.time()
                solver = GeneticMspspSolver(f,
                    GraphGene,
                    size = populationSize,
                    activityMutationPropability = activityMutationPropability,
                    resourceMutationPropability = resourceMutationPropability,
                    ageBiasFactor=ageBiasFactor,
                    parentalBiasFactor=parentalBiasFactor,
                    parentCount=parentCount
                )
                solver.solve()
                score += solver.score
                score -= (time.time() - start)*self.timeBias
            return score

if __name__ == "__main__":
    benchmark_instances_dir = os.path.join(os.path.dirname(__file__), "..\\benchmark_instances")

    # Execute an optimization by using an `Objective` instance.
    study = optuna.create_study(direction="maximize")
    study.optimize(Objective(), n_trials=n_trials)
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "..\\images")):
        os.mkdir(os.path.join(os.path.dirname(__file__), "..\\images"))
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(os.path.join(os.path.dirname(__file__), "..\\images\\param_importances.png"))
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image(os.path.join(os.path.dirname(__file__), "..\\images\\parallel_coordinate.png"))
    fig = optuna.visualization.plot_slice(study)
    fig.write_image(os.path.join(os.path.dirname(__file__), "..\\images\\sclices.png"))
