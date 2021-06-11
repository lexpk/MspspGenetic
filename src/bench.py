from mspsp import MspspSolver
import os
import time

if __name__ == "__main__":
    benchmark_instances_dir = os.path.join(os.path.dirname(__file__), "..\\benchmark_instances")

    total = {}
    average = {}
    nbenches = len(os.listdir(benchmark_instances_dir))
    incomplete = {}
    waittime = 5

    problem = {}

    configurations = [
        ("ortools", ""),
        ("gecode",  ""),
        ("chuffed", ""),
        ("chuffed", ":: int_search(start, input_order, indomain_min)\n\t"),
        ("chuffed", ":: int_search(start, input_order, indomain_median)\n\t"),
        ("chuffed", ":: int_search(start, first_fail, indomain_min)\n\t"),
        ("chuffed", ":: int_search(start, first_fail, indomain_median)\n\t"),
        ("chuffed", ":: int_search(start, input_order, indomain_random)\n\t"),
    ]
    for (solver, annotation) in configurations:
        total[(solver, annotation)] = 0
        incomplete[(solver, annotation)] = 0
        for file in os.listdir(benchmark_instances_dir):
            f = os.path.join(benchmark_instances_dir, file)
            print(f"Solving {file} with {solver}, {annotation}...")
            problem[file] = MspspSolver(f, solvername=solver, search_ann=annotation)
            start = time.time()
            try:
                problem[file].solve(timeout=waittime)
                duration = time.time() - start
            except:
                duration = waittime
            if duration < waittime:
                total[(solver, annotation)] += duration
                print(f"Completed in {duration}s")
            else:
                print(f"Not completed within {waittime}s")
                incomplete[(solver, annotation)] += 1

        average[(solver, annotation)] = total[(solver, annotation)] / nbenches
        print(
            f"""
            Solver:\t {solver}
            Annotation: {annotation}
            Completed: {nbenches - incomplete[(solver, annotation)]} / {nbenches}
            Average: {average[(solver, annotation)]}"""
        )       



