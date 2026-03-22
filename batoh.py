import array
import os
import random
import numpy as np
from deap import algorithms, base, creator, tools

base_path = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(base_path, "input_100.txt")

def loadData():
    with open(filename, 'r') as f:
        n, W = map(int, f.readline().split())
        items = []
        for _ in range(n):
            line = f.readline().split()
            if line:
                items.append(list(map(int, line)))
    return n, W, items

N_ITEMS, MAX_WEIGHT, items = loadData()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool",  lambda: 1 if random.random() < 0.005 else 0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=N_ITEMS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalKnapsack(individual):
    weight = 0
    value = 0
    for i in range(len(individual)):
        if individual[i]: 
            value += items[i][0]
            weight += items[i][1]
    
    if weight > MAX_WEIGHT:
        return 0,
    return value,

toolbox.register("evaluate", evalKnapsack)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

def main():
    random.seed(42)

    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(3)     
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # Spuštění evolučního algoritmu
    pop, log = algorithms.eaSimple(pop, toolbox, 
                                   cxpb=0.8,   # pravděpodobnost křížení
                                   mutpb=0.3,  # pravděpodobnost mutace
                                   ngen=1000,    # počet generací
                                   stats=stats, 
                                   halloffame=hof, 
                                   verbose=True)

    # Výsledek
    best_ind = hof[0]
    total_weight = sum(items[i][1] for i in range(len(items)) if best_ind[i])
    total_value = sum(items[i][0] for i in range(len(items)) if best_ind[i])

    print("\n--- Nejlepší nalezené řešení ---")
    print(f"Jedinec (binárně): {list(best_ind)}")
    print(f"Celková váha: {total_weight} / {MAX_WEIGHT}")
    print(f"Celková cena: {total_value}")

if __name__ == "__main__":
    main()