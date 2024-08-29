import time
from datetime import datetime

import networkx as nx
import numpy as np
from scipy.integrate import quad
from math import sqrt, pi, exp
from Modified_LIE import LIE


# Initialize the crow positions (continuous space)
def init_population(N, num_nodes):
    return np.random.uniform(low=0.0, high=1.0, size=(N, num_nodes))


# Transfer function T(x) defined as the integral of e^(-t^2) from 0 to sqrt(pi)/2 * x
def transfer_function(x):
    integral, _ = quad(lambda t: exp(-t * t), 0, sqrt(pi) / 2 * x)
    return abs((sqrt(2) / pi) * integral)


# Convert continuous population to binary population using transfer function T(x) from Equation (8) and (9)
# Adjust the binary individual to ensure exactly k number of 1's
def adjust_individual(individual, k):
    # Count current number of 1's
    current_ones = int(np.sum(individual))
    k = int(k)  # Ensure k is an integer
    if current_ones > k:  # Too many 1's, need to turn some 1's to 0's
        indices = np.where(individual == 1)[0]
        np.random.shuffle(indices)
        individual[indices[:current_ones - k]] = 0
    elif current_ones < k:  # Too few 1's, need to turn some 0's to 1's
        indices = np.where(individual == 0)[0]
        np.random.shuffle(indices)
        individual[indices[:k - current_ones]] = 1
    return individual


# Convert continuous population to binary population using transfer function T(x) from Equation (8) and (9)
def continuous_to_binary_transfer(population, k):
    binary_population = np.zeros_like(population)
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            transfer_value = transfer_function(population[i, j])
            binary_population[i, j] = 1 if transfer_value > np.random.rand() else 0
        binary_population[i] = adjust_individual(binary_population[i], k)  # Ensure k 1's
    return binary_population


def continuous_to_binary_transfer_individual(individual, k):
    # Initialize the binary individual
    binary_individual = np.zeros_like(individual)

    # Apply the transfer function to each element in the individual
    for j in range(individual.shape[0]):
        transfer_value = transfer_function(individual[j])
        binary_individual[j] = 1 if transfer_value > np.random.rand() else 0

    # Adjust the binary individual to ensure it has exactly k 1's
    binary_individual = adjust_individual(binary_individual, k)

    return binary_individual

def is_dominated(solution_a, solution_b):
    """
    Check if solution_a is dominated by solution_b.

    Args:
    solution_a: A tuple (cost_a, LIE_a) representing the cost and LIE value of solution_a
    solution_b: A tuple (cost_b, LIE_b) representing the cost and LIE value of solution_b

    Returns:
    True if solution_a is dominated by solution_b, False otherwise.
    """
    cost_a, LIE_a = solution_a
    cost_b, LIE_b = solution_b

    # Solution A is dominated if:
    # 1. Solution B has a lower cost and a higher or equal LIE value, or
    # 2. Solution B has an equal cost and a strictly higher LIE value.
    if (cost_b <= cost_a and LIE_b >= LIE_a) and (cost_b < cost_a or LIE_b > LIE_a):
        return True
    return False


#
def find_non_dominated_solution(solution, node_costs, G, evaluated_solutions):
    """
    Check if a single solution is non-dominated based on cost and LIE values.

    Args:
    solution: A list of nodes representing the seed set
    node_costs: A dictionary where keys are node IDs and values are costs associated with those nodes
    G: A networkx graph representing the network
    evaluated_solutions: A list of tuples (solution, cost, LIE_value) of previously evaluated solutions

    Returns:
    True if the solution is non-dominated, False otherwise.
    The function also returns the cost and LIE value of the current solution.
    """
    # # Calculate cost and LIE for the current solution
    # for node in solution:
    #     cost_value = node_costs.get(node, None)  # Get cost or None if node not in dictionary
    #     print(f"Node: {node}, Cost: {cost_value}")
    cost = sum(node_costs[node] for node in solution)
    LIE_value = LIE(G, solution)
    # print("c",cost,LIE_value)

    # Check if the current solution is dominated by any previously evaluated solution
    for sol_b, cost_b, LIE_b in evaluated_solutions:
        # print("sol_b", sol_b,cost_b,LIE_b)
        if is_dominated((cost, LIE_value), (cost_b, LIE_b)):
            # print("no")
            return False, cost, LIE_value  # Solution is dominated
    # print("yes")
    return True, cost, LIE_value  # Solution is non-dominated


def find_local_best(non_dominated_solutions, budget):
    """
    Find the local best solution from the non-dominated solutions with the highest influence
    (LIE value) and the least budget violation.

    Args:
    non_dominated_solutions: A list of non-dominated solutions with their cost and LIE values.
    budget: The maximum allowed cost for a solution (budget constraint).

    Returns:
    The local best solution (seed set) that has the highest influence (LIE)
    within the budget constraint, or with the least budget violation.
    """
    local_best = None
    best_influence = -float('inf')
    least_violation = float('inf')

    # Iterate over the non-dominated solutions
    for solution, cost, LIE_value in non_dominated_solutions:
        violation = max(0, cost - budget)  # Calculate budget violation (0 if within budget)

        if (cost <= budget and LIE_value > best_influence) or (violation < least_violation):
            best_influence = LIE_value
            local_best = solution
            least_violation = violation

    return local_best


def DAP(iter, max_iter):
    return 1 - (iter / max_iter)


def DfL(iter, fl_max, fl_min, max_iter):
    return fl_max - (fl_max - fl_min) * (iter / max_iter)


def seed_set_to_binary(seed_set, n):
    # Initialize binary array with all zeros
    binary_individual = np.zeros(n, dtype=int)

    # Set positions corresponding to the seed set nodes to 1
    for node in seed_set:
        binary_individual[node] = 1

    return binary_individual


def black_hole_random_walk(current_position, black_hole_position, step_size=0.1):
    """
    Perform a random walk towards the black hole position.

    Args:
    current_position: The current position of the crow (agent).
    black_hole_position: The position of the best solution (black hole).
    step_size: The step size controlling the influence of the black hole.

    Returns:
    new_position: The updated position after the random walk.
    """
    # Random perturbation (small random noise)
    epsilon = np.random.uniform(-0.01, 0.01, size=current_position.shape)

    # Move towards the black hole with a small step and add random noise
    new_position = current_position + step_size * (black_hole_position - current_position) + epsilon

    return new_position


def check_event_horizon(new_position, black_hole_position, threshold=0.01):
    """
    Check if the new position is within the event horizon of the black hole.

    Args:
    new_position: The new position after the random walk.
    black_hole_position: The position of the black hole (best solution).
    threshold: The distance threshold defining the event horizon.

    Returns:
    True if the new position is within the event horizon, False otherwise.
    """
    distance = np.linalg.norm(new_position - black_hole_position)
    return distance < threshold
# Main MODPSO-IM-CM Algorithm Execution
if __name__ == "__main__":
    # Constants and parameters
    # Constants as given in the problem statement
    m = 1.1
    r = 5
    s = 1.1
    # set number of seed nodes to a threshold
    # Set the number of seed nodes to a threshold
    k = 40
    # Budget threshold
    budget = 350
    num_runs = 10
    beta = 0.5  # Example user-defined parameter

    # Dataset file
    # input_file = 'socfb-Reed98.mtx'
    # input_file = 'email-univ.edges'
    # input_file = 'soc-hamsterster.edges'
    input_file = 'soc-wiki-Vote.mtx'
    output_file = 'Mydata_with_costs.txt'

    # Read the graph
    G = nx.read_edgelist(input_file, nodetype=int, create_using=nx.Graph())
    assert isinstance(G, nx.Graph), "G must be a NetworkX graph"

    # Calculate the cost for each node
    cost = {}
    for node in G.nodes():
        di = G.degree(node)
        cost[node] = m ** (di / r) + (di / r) * s

    with open(output_file, 'w') as f:
        for node in G.nodes():
            f.write(f"{node} {G.degree(node)} {cost[node]:.4f}\n")

    n = G.number_of_nodes()

    # print("finish", ranked_nodes)
    searchAgents = 50  # Number of Capuchins (agents)
    dim = n  # Number of candidates (l)
    upper_bound = 1.0  # Upper bound of initialization
    lower_bound = 0.0  # Lower bound of initialization
    maxIter = 100
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"Crow_results_{current_time}.txt"
    # Open the output file to write results
    with open(output_file, 'w') as f:
        # Write the metadata before running the algorithm
        f.write(f"Algorithm name: CrowSA\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Budget: {budget}\n")
        f.write(f"Number of Nodes: {n}\n")
        f.write(f"k: {k}\n")
        f.write(f"Iterations: {maxIter}\n")
        f.write(f"Search Agents: {searchAgents}\n")
        f.write("\n")
        for run in range(num_runs):
            # Measure start time
            start_time = time.time()
            cg_curve = np.zeros(maxIter)

            # Initialize population and memory
            population = init_population(searchAgents, dim)
            memory = np.copy(population)

            # Convert continuous population to binary population using the transfer function T(x)
            binary_population = continuous_to_binary_transfer(population, k)
            memorybin = np.copy(binary_population)
            non_dominated_solutions = []

            # Loop through each position (solution) in the population
            for individual in binary_population:
                position = [j + 1 for j, val in enumerate(individual) if val == 1]
                # Check if the current position is non-dominated
                is_non_dominated, costval, LIE_value = find_non_dominated_solution(position, cost, G,
                                                                                   non_dominated_solutions)

                # If the solution is non-dominated, add it to the list of evaluated solutions
                if is_non_dominated:
                    non_dominated_solutions.append((position, costval, LIE_value))
                Lbesti = find_local_best(non_dominated_solutions, budget)
                best_spread = LIE(G, Lbesti)
            print("fit", best_spread)
            # Initialize dynamic parameters
            fl_max = 2.0  # Example maximum flight length, adjust as needed
            fl_min = 0.5  # Example minimum flight length, adjust as needed

            for t in range(maxIter):
                for i in range(searchAgents):
                    if np.random.rand() > DAP(t, maxIter):  # Exploration
                        # random_index = np.random.randint(0, len(non_dominated_solutions))
                        # randcrow = non_dominated_solutions[random_index][0]
                        # # print("rand",randcrow)
                        # # Initialize a binary vector with all zeros
                        # binary_position = np.zeros(n, dtype=int)
                        #
                        # # Set the positions corresponding to the seed set nodes to 1
                        # for ne in randcrow:
                        #     binary_position[ne - 1] = 1  # Subtract 1 to match zero-based indexing
                        random_crow = np.random.randint(0, searchAgents)
                        flight_length = DfL(t, fl_max, fl_min, maxIter)
                        new_position = population[i] + flight_length * (memorybin[random_crow] - population[i])
                        # new_position = population[i] + flight_length * (binary_position - population[i])



                    else:
                        # Intensification (local search)
                        # Perform random walk towards the black hole
                        local_bestnode = find_local_best(non_dominated_solutions, budget)
                        binary_positionnode = np.zeros(n, dtype=int)

                        # Set the positions corresponding to the seed set nodes to 1
                        for ne in local_bestnode:
                            binary_positionnode[ne - 1] = 1  # Subtract 1 to match zero-based indexing
                        # black_hole_position = memory[np.argmax(cg_curve)]  # The best solution so far
                        black_hole_position = binary_positionnode  # The best solution so far
                        new_position = black_hole_random_walk(population[i], black_hole_position)

                        # Check if the new position is within the event horizon
                        if check_event_horizon(new_position, black_hole_position):
                            # Reinitialize the crow's position randomly if it enters the event horizon
                            new_position = init_population(1, dim)[0]

                    # Ensure positions are within bounds
                    new_position = np.clip(new_position, lower_bound, upper_bound)
                    # Convert to binary and adjust to have exactly k 1's
                    # new_binary_position = continuous_to_binary_transfer(new_position.reshape(1, -1), k)[0]
                    new_binary_position = continuous_to_binary_transfer_individual(new_position, k)

                    # check old memory
                    # vv = [jk + 1 for jk, val in enumerate(memorybin[i) if val == 1]
                    vv = [jk + 1 for jk, va in enumerate(memorybin[i]) if va == 1]
                    mcost = sum(cost[nod] for nod in vv)
                    mLIE_value = LIE(G, vv)

                    # Evaluate the new position
                    position = [j + 1 for j, val in enumerate(new_binary_position) if val == 1]
                    is_non_dominated, costval, LIE_value = find_non_dominated_solution(position, cost, G,
                                                                                       non_dominated_solutions)
                    if mLIE_value < LIE_value:
                        memorybin[i] = new_binary_position

                    if is_non_dominated:
                        # print("non",non_dominated_solutions)
                        non_dominated_solutions.append((position, costval, LIE_value))

                local_best = find_local_best(non_dominated_solutions, budget)
                # print(Lbesti)
                best_fitness = LIE(G, local_best)

                print(f"Iteration {t + 1}, Best Fitness: {best_fitness}")
                local_best = find_local_best(non_dominated_solutions, budget)
                cg_curve[t] = best_fitness

            # Final evaluation
            Lbesti = local_best
            best_fitness = LIE(G, Lbesti)
            total_cost = sum(cost[node] for node in Lbesti)

            print("Convergence Curve:", cg_curve)
            print(f"Best Seed Set: {Lbesti}")
            print(f"Total Cost: {total_cost:.4f}")
            print(f"Best Fitness (LIE value): {best_fitness:.4f}")
            print(f"Number of Seeds (k): {k}")
            print(f"Budget: {budget}")
            # Measure end time
            end_time = time.time()
            execution_time = end_time - start_time
            from IC import IC

            # Set the propagation probability and the number of Monte Carlo simulations
            propagation_probability = 0.01
            monte_carlo_simulations = 1000

            # Calculate the spread using the IC model
            spread = IC(G, Lbesti, propagation_probability, mc=monte_carlo_simulations)

            # Output the results
            print(f"Final Seed Set: {Lbesti}")
            print(f"Spread of the Seed Set using IC: {spread}")

            # Output the results
            print(f"Final Seed Set: {Lbesti}")
            print(f"Spread of the Seed Set using IC: {spread}")
            final_seed_set = Lbesti
            convergence_curve = cg_curve
            # Write the results of this run to the file
            f.write(f"Run {run + 1}:\n")
            f.write(f"Best Seed Set: {final_seed_set}\n")
            f.write(f"Best Seed Set Cost: {total_cost:.4f}\n")
            f.write(f"Best Fitness Value (LIE): {best_fitness:.4f}\n")
            f.write(f"Spread: {spread}\n")
            f.write(f"Convergence Curve: {convergence_curve.tolist()}\n")
            f.write(f"Execution Time: {execution_time:.4f} seconds\n")
            f.write("\n")



