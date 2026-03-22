import array
import os
import random
import numpy as np
from deap import algorithms, base, creator, tools
import matplotlib.pyplot as plt 

base_path = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(base_path, "input_1000.txt")

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

values_np = np.array([it[0] for it in items])
weights_np = np.array([it[1] for it in items])

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool",  lambda: 1 if random.random() < 0.005 else 0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=N_ITEMS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalKnapsack(individual):
    ind_np = np.array(individual)
    weight = np.dot(ind_np, weights_np)
    value = np.dot(ind_np, values_np)

    if weight > MAX_WEIGHT:
        penalty = (weight - MAX_WEIGHT) * 25 
        return value - penalty,

    return value,

toolbox.register("evaluate", evalKnapsack)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/N_ITEMS) 

def main():
    random.seed(42)

    pop = toolbox.population(n=400)
    hof = tools.HallOfFame(1)     
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 
                                   cxpb=0.8,   
                                   mutpb=0.6,  
                                   ngen=3000,    
                                   stats=stats, 
                                   halloffame=hof, 
                                   verbose=True)

    best_ind = hof[0]
    total_weight = sum(items[i][1] for i in range(len(items)) if best_ind[i])
    total_value = sum(items[i][0] for i in range(len(items)) if best_ind[i])

    print("\n--- Nejlepší nalezené řešení ---")
    print(f"Jedinec (binárně): {list(best_ind)}")
    print(f"Celková váha: {total_weight} / {MAX_WEIGHT}")
    print(f"Celková cena: {total_value}")

    gen = log.select("gen")      # Čísla generací
    fit_max = log.select("max")  # Nejlepší fitness v každé generaci
    fit_avg = log.select("avg")  # Průměrná fitness v každé generaci

    plt.figure(figsize=(10, 6))
    plt.plot(gen, fit_max, label="Maximální cena", color="green", linewidth=2)
    plt.plot(gen, fit_avg, label="Průměrná cena", color="blue", linestyle="--")
    
    plt.title("Průběh evoluce (Batoh 1000 předmětů)")
    plt.xlabel("Generace")
    plt.ylabel("Fitness (Cena)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()