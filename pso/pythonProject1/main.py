import random
import numpy as np
import networkx as nx
from Modified_LIE import LIE


# Step 1: Initialize the population ensuring budget constraints are met
def initialize_population(graph, num_particles, k, node_costs, budget):
    # Sort nodes by degree in descending order
    degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)

    # Select top-k nodes with the highest degree
    top_k_nodes = [node for node, _ in degrees[:k]]

    # Initialize particles
    particles = []

    for _ in range(num_particles):
        # Start with the top-k nodes as the initial position of the particle
        position = top_k_nodes[:]

        # Introduce diversity in the population
        for i in range(len(position)):
            if random.random() > 0.5:
                # Replace the current node with a random node from the node set N
                node_set = set(G.nodes) - set(position)  # Exclude nodes already in the position vector
                new_node = random.choice(list(node_set))
                position[i] = new_node

        # Append this particle's position to the particle list
        particles.append(position)
    return particles


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



# def find_non_dominated_solutions_with_cost_LIE(solutions, node_costs, G, nondominated):
#     """
#     Find all non-dominated solutions based on cost and LIE values.
#
#     Args:
#     solutions: A list of solutions, where each solution is a list of nodes (seed set)
#     node_costs: A dictionary where keys are node IDs and values are costs associated with those nodes
#     graph: A networkx graph representing the network
#     pu: The probability of activation (default 0.1, similar to IC model)
#
#     Returns:
#     A list of non-dominated solutions with their cost and LIE values.
#     """
#     evaluated_solutions = []
#
#     for solution in solutions:
#         cost = sum(node_costs[node] for node in solution)
#         LIE_value = LIE(G, solution)
#         evaluated_solutions.append((solution, cost, LIE_value))
#
#     # Now find the non-dominated solutions based on (cost, LIE_value)
#     non_dominated = []
#
#     for i, (sol_a, cost_a, LIE_a) in enumerate(evaluated_solutions):
#         dominated = False
#         for j, (sol_b, cost_b, LIE_b) in enumerate(evaluated_solutions):
#             if i != j and is_dominated((cost_a, LIE_a), (cost_b, LIE_b)):
#                 dominated = True
#                 break
#         if not dominated:
#             non_dominated.append((sol_a, cost_a, LIE_a))
#
#     return non_dominated
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
    # Calculate cost and LIE for the current solution
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


# Step 2: Initialize velocities as random binary vectors
def initialize_velocities(population_size, k):
    velocities = []
    for _ in range(population_size):
        velocity = [random.randint(0, 1) for _ in range(k)]
        velocities.append(velocity)
    return velocities


# Step 3: Local best selection based on influence and budget
# def find_local_best(population, nondominated, graph, node_costs, budget):
#     local_best = None
#     best_influence = -float('inf')
#
#     for seed_set in archive:
#         if sum(node_costs[node] for node in seed_set) <= budget:
#             influence = LIE(graph, seed_set)
#             if influence > best_influence:
#                 best_influence = influence
#                 local_best = seed_set
#
#     return local_best
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

# def find_local_best(non_dominated_solutions, budget):
#     """
#     Find the local best solution from the non-dominated solutions that meets the budget constraint.
#
#     Args:
#     population: Current population of particles (list of solutions)
#     non_dominated_solutions: A list of non-dominated solutions with their cost and LIE values
#     graph: A networkx graph representing the network
#     node_costs: A dictionary where keys are node IDs and values are costs associated with those nodes
#     budget: The maximum allowed cost for a solution (budget constraint)
#
#     Returns:
#     The local best solution (seed set) that has the highest influence (LIE) within the budget constraint.
#     """
#     local_best = None
#     best_influence = -float('inf')
#
#     # Iterate over the non-dominated solutions to find the best within the budget
#     for solution, cost, LIE_value in non_dominated_solutions:
#         if cost <= budget and LIE_value > best_influence:
#             best_influence = LIE_value
#             local_best = solution
#     # Check if local_best is still None
#     if local_best is None:
#         local_best = non_dominated_solutions[0][0]
#     return local_best


# Step 4: Turbulence operator for diversity preservation
# def apply_turbulence(position, graph, node_costs, budget):
#     all_nodes = set(graph.nodes())
#     for i in range(len(position)):
#         if random.random() < pm:
#             available_nodes = list(all_nodes - set(position))
#             if available_nodes:
#                 candidate = random.choice(available_nodes)
#                 if sum(node_costs[node] for node in position if node != position[i]) + node_costs[candidate] <= budget:
#                     position[i] = candidate
#     return position
def turbulence_operator(particle, node_set, pm):
    """
    Apply the turbulence operator to the population of particles.

    Args:
    particles: List of particles, where each particle is a list of nodes (seed set).
    node_set: Set of all available nodes in the graph (N).
    pm: Mutation probability (probability of applying the turbulence operator to each element).

    Returns:
    Updated particles after applying the turbulence operator.
    """
    # Iterate over each particle in the population

    # Iterate over each element (node) in the particle
    for j in range(len(particle)):
        # Generate a random number between 0 and 1
        if random.random() < pm:
            # Replace the current element with another random node from node_set (excluding the current node)
            available_nodes = list(node_set - set(particle))  # Exclude nodes already in the particle
            if available_nodes:  # Ensure there are available nodes for replacement
                new_node = random.choice(available_nodes)
                particle[j] = new_node

        # Update the particle in the population after applying turbulence

    return particle


# Step 5: Local search strategy to refine solutions
def local_search(Xa, graph, node_costs, budget):
    L = []
    for i in range(len(Xa)):
        Flag = False
        Xb = Xa.copy()
        neighbors = list(graph.neighbors(Xb[i]))

        while not Flag:
            candidate = random.choice(neighbors)
            if candidate not in Xa:
                if sum(node_costs[node] for node in Xb if node != Xb[i]) + node_costs[candidate] <= budget:
                    Xb[i] = candidate

            if LIE(graph, Xa) < LIE(graph, Xb):
                Xa = Xb.copy()
            else:
                Flag = True

        L.append(Xb)

    return L


# Step 6: Update the velocity based on PSO rules
# def update_velocity(position, personal_best, local_best, velocity, w=0.5, c1=1.0, c2=1.0):
#     new_velocity = []
#     print("pppp",position)
#     for i in range(len(position)):
#         r1, r2 = random.random(), random.random()
#         inertia = w * velocity[i]
#         cognitive = c1 * r1 * (personal_best[i] - position[i])
#         social = c2 * r2 * (local_best[i] - position[i])
#         new_velocity.append(int(np.clip(inertia + cognitive + social, 0, 1)))
#     print("newve",new_velocity)
#     return new_velocity
# Step 6: Update the velocity based on PSO rules
def update_velocity(position, personal_best, local_best, velocity, w=0.5, c1=1.0, c2=1.0):
    new_velocity = []
    for i in range(len(position)):
        r1, r2 = random.random(), random.random()
        inertia = w * velocity[i]
        cognitive = c1 * r1 * (1 if personal_best[i] != position[i] else 0)
        social = c2 * r2 * (1 if local_best[i] != position[i] else 0)
        new_velocity.append(int(np.clip(inertia + cognitive + social, 0, 1)))
    return new_velocity

# Step 7: Update the position based on the velocity
# Step 7: Update the position based on the velocity
def update_position(position, velocity, graph):
    new_position = position.copy()
    all_nodes = set(graph.nodes())

    for i in range(len(position)):
        if velocity[i] == 1:  # If velocity suggests a change, modify the node
            available_nodes = list(all_nodes - set(new_position))
            if available_nodes:
                new_node = random.choice(available_nodes)
                new_position[i] = new_node

    return new_position

# Step 8: Write output to file
def write_output(file, best_solution, total_cost, best_fitness, k, budget):
    with open(file, 'w') as f:
        f.write("Best Seed Set: " + str(best_solution) + "\n")
        f.write(f"Total Cost: {total_cost:.4f}\n")
        f.write(f"Best Fitness (LIE value): {best_fitness:.4f}\n")
        f.write(f"Number of Seeds (k): {k}\n")
        f.write(f"Budget: {budget}\n")


# Main MODPSO-IM-CM Algorithm Execution
if __name__ == "__main__":
    # Constants and parameters
    m = 1.1
    r = 5
    s = 1.1
    k = 5
    budget = 90
    searchAgents = 50
    maxIter = 30
    pm = 0.2  # Mutation probability
    cg_curve = np.zeros(maxIter)

    # input_file = 'email-univ.edges'
    input_file = 'soc-hamsterster.edges'
    output_file = 'output.txt'

    G = nx.read_edgelist(input_file, nodetype=int, create_using=nx.Graph())
    assert isinstance(G, nx.Graph), "G must be a NetworkX graph"

    # Calculate node costs
    cost = {node: m ** (G.degree(node) / r) + (G.degree(node) / r) * s for node in G.nodes()}

    # Initialize population and velocities
    positions = initialize_population(G, searchAgents, k, cost, budget)
    # print("pp",positions)
    velocities = initialize_velocities(searchAgents, k)
    # Initialize empty list to store evaluated solutions
    non_dominated_solutions = []

    # Loop through each position (solution) in the population
    for position in positions:
        # Check if the current position is non-dominated
        is_non_dominated, costval, LIE_value = find_non_dominated_solution(position, cost, G, non_dominated_solutions)

        # If the solution is non-dominated, add it to the list of evaluated solutions
        if is_non_dominated:
            non_dominated_solutions.append((position, costval, LIE_value))
    # print("non",non_dominated_solutions)
    Pbesti = positions.copy()
    Lbesti = find_local_best(non_dominated_solutions, budget)
    # print("lb",Lbesti)
    # Example node set N
    # Main loop of the MODPSO-IM-CM algorithm
    for t in range(maxIter):
        print("Iteration", t)
        for i in range(searchAgents):
            # print("pobef", positions[i])
            velocities[i] = update_velocity(positions[i], Pbesti[i], Lbesti, velocities[i])
            positions[i] = update_position(positions[i], velocities[i], G)

            # # Apply turbulence
            # if t < maxIter * pm:
            #     positions[i] = apply_turbulence(positions[i], G, cost, budget, pm)
            # Apply the turbulence operator to the particles
            positions[i] = turbulence_operator(positions[i], set(G.nodes), pm=0.2)
            # print("po",positions[i])
            is_non_dominated, costval, LIE_value = find_non_dominated_solution(positions[i], cost, G,
                                                                               non_dominated_solutions)
            # If the solution is non-dominated, add it to the list of evaluated solutions

            if is_non_dominated:
                # print("non", positions[i],costval,LIE_value)
                non_dominated_solutions.append((positions[i], costval, LIE_value))


        # Update global best (local best in the population)
        Lbesti = find_local_best(non_dominated_solutions, budget)
        # print(Lbesti)
        best_fitness = LIE(G, Lbesti)
        print("fit",best_fitness)

        # # Local search
        # Xa = max(positions, key=lambda x: LIE(G, x))
        # local_solutions = local_search(Xa, G, cost, budget)
        # archive.extend(local_solutions)
        cg_curve[t] = best_fitness
    # Final evaluation
    best_fitness = LIE(G, Lbesti)
    total_cost = sum(cost[node] for node in Lbesti)
    write_output(output_file, Lbesti, total_cost, best_fitness, k, budget)
    print("Convergence Curve:", cg_curve)
    print(f"Best Seed Set: {Lbesti}")
    print(f"Total Cost: {total_cost:.4f}")
    print(f"Best Fitness (LIE value): {best_fitness:.4f}")
    print(f"Number of Seeds (k): {k}")
    print(f"Budget: {budget}")

    from IC import IC

    # Set the propagation probability and the number of Monte Carlo simulations
    propagation_probability = 0.01
    monte_carlo_simulations = 1000

    # Calculate the spread using the IC model
    spread = IC(G, Lbesti, propagation_probability, mc=monte_carlo_simulations)

    # Output the results
    print(f"Final Seed Set: {Lbesti}")
    print(f"Spread of the Seed Set using IC: {spread}")
