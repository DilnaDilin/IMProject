import numpy as np
from InitialiationFile import binary_conversion
from fitness import fobj
from scipy.spatial.distance import euclidean


# Function to calculate the fitness of a solution (influence spread)
# def fobj(solution, candidates, G):
#     selected_nodes = [candidates[i] for i in range(len(solution)) if solution[i] == 1]
#     return simulate_influence_spread(G, selected_nodes)
#
#
# # Placeholder function for influence spread simulation
# def simulate_influence_spread(G, seed_nodes):
#     return len(seed_nodes)  # Replace this with actual influence spread calculation


# Function to initialize population
def initialization(noP, dim):
    return np.random.rand(noP, dim)


# Function to convert continuous values to binary based on the top-k nodes
# def binary_conversion(individual, k, candidates, budget, costs):
#     sorted_indices = np.argsort(individual)[::-1]
#     binary_solution = np.zeros_like(individual)
#     total_cost = 0
#     for i in sorted_indices:
#         if total_cost + costs[candidates[i]] <= budget and sum(binary_solution) < k:
#             binary_solution[i] = 1
#             total_cost += costs[candidates[i]]
#     return binary_solution


# Differential Evolution Algorithm
def DE(noP, maxite, candidates, budget, costs, G, F, CR, k):
    dim = len(candidates)
    PopPos = initialization(noP, dim)
    PopFit = np.zeros(noP)

    # Step 2: Convert to binary and adjust capuchins
    adjusted_population = []
    for capuchin in PopPos:
        adjusted_capuchin = binary_conversion(capuchin, k, candidates, budget, costs)
        adjusted_population.append(adjusted_capuchin)

    capuchinsFit = np.array(adjusted_population)

    # Calculate initial fitness



    for i in range(noP):
        PopFit[i] = fobj(capuchinsFit[i], candidates, G)

    BestFit = np.max(PopFit)
    BestPos = PopPos[np.argmax(PopFit)].copy()
    BestPosBin = adjusted_population[np.argmax(PopFit)].copy()

    cg_curve = np.zeros(maxite)

    for t in range(maxite):
        print(f"Iteration: {t}")
        for i in range(noP):
            # Mutation: Select 3 random individuals
            indices = list(range(noP))
            indices.remove(i)
            a, b, c = PopPos[np.random.choice(indices, 3, replace=False)]

            # Generate mutant vector
            mutant = np.clip(a + F * (b - c), 0, 1)

            # Crossover
            trial = np.where(np.random.rand(dim) < CR, mutant, PopPos[i])

            # Convert to binary and adjust based on budget
            trial_bin = binary_conversion(trial, k, candidates, budget, costs)

            # Calculate fitness after adjustment
            trial_fit = fobj(trial_bin, candidates, G)

            # Selection
            if trial_fit > PopFit[i]:  # Maximize influence
                PopPos[i] = trial
                PopFit[i] = trial_fit
                adjusted_population[i] = trial_bin

            if trial_fit > BestFit:
                BestFit = trial_fit
                BestPos = trial.copy()
                BestPosBin = trial_bin.copy()

        cg_curve[t] = BestFit

    # BestPos_bin = binary_conversion(BestPos, k, candidates, budget, costs)
    return BestFit, BestPos, BestPosBin, cg_curve









