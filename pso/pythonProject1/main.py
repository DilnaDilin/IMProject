import random
import numpy as np
import networkx as nx
from Modified_LIE import LIE


def initialize_population_with_budget(graph, population_size, k, candidate_costs, budget):
    population = []
    all_nodes = list(graph.nodes())

    while len(population) < population_size:
        seed_set = random.sample(all_nodes, k)
        total_cost = sum(candidate_costs[node] for node in seed_set)
        if total_cost <= budget:
            population.append(seed_set)

    return population


def initialize_velocities(population_size, k):
    velocities = []
    for _ in range(population_size):
        velocity = [random.randint(0, 1) for _ in range(k)]
        velocities.append(velocity)
    return velocities


def find_local_best(population, graph):
    local_best = population[0]
    best_fitness = LIE(graph, local_best)
    for seed_set in population:
        fitness = LIE(graph, seed_set)
        if fitness > best_fitness:
            best_fitness = fitness
            local_best = seed_set
    return local_best


def update_velocity(position, personal_best, local_best, velocity, w=0.5, c1=1.0, c2=1.0):
    new_velocity = []
    for i in range(len(position)):
        r1, r2 = random.random(), random.random()
        inertia = w * velocity[i]
        cognitive = c1 * r1 * (personal_best[i] - position[i])
        social = c2 * r2 * (local_best[i] - position[i])
        new_velocity.append(int(np.clip(inertia + cognitive + social, 0, 1)))
    return new_velocity


def update_position(position, velocity, graph):
    new_position = position.copy()
    all_nodes = set(graph.nodes())

    for i in range(len(position)):
        if velocity[i] != 0:  # If velocity is non-zero, change the node
            available_nodes = list(all_nodes - set(new_position))
            if available_nodes:
                new_node = random.choice(available_nodes)
                new_position[i] = new_node

    return new_position


def write_output(file, best_solution, total_cost, best_fitness, k, budget):
    with open(file, 'w') as f:
        f.write("Best Seed Set: " + str(best_solution) + "\n")
        f.write(f"Total Cost: {total_cost:.4f}\n")
        f.write(f"Best Fitness (LIE value): {best_fitness:.4f}\n")
        f.write(f"Number of Seeds (k): {k}\n")
        f.write(f"Budget: {budget}\n")


if __name__ == "__main__":
    m = 1.1
    r = 5
    s = 1.1
    k = 5
    budget = 90
    beta = 0.5

    input_file = 'email-univ.edges'
    output_file = 'Mydata_with_costs.txt'

    G = nx.read_edgelist(input_file, nodetype=int, create_using=nx.Graph())
    assert isinstance(G, nx.Graph), "G must be a NetworkX graph"

    cost = {}
    for node in G.nodes():
        di = G.degree(node)
        cost[node] = m ** (di / r) + (di / r) * s

    with open(output_file, 'w') as f:
        for node in G.nodes():
            f.write(f"{node} {G.degree(node)} {cost[node]:.4f}\n")

    n = G.number_of_nodes()

    graph = G
    node_costs = cost

    searchAgents = 50
    population_size = searchAgents
    dim = k
    upper_bound = 1.0
    lower_bound = 0.0
    maxIter = 30
    max_iterations = maxIter

    positions = initialize_population_with_budget(graph, population_size, k, cost, budget)
    velocities = initialize_velocities(population_size, k)
    Pbesti = positions.copy()
    Lbesti = find_local_best(positions, graph)

    for iteration in range(max_iterations):
        print("iteration", iteration)
        for i in range(population_size):
            velocities[i] = update_velocity(positions[i], Pbesti[i], Lbesti, velocities[i])
            positions[i] = update_position(positions[i], velocities[i], graph)

            total_cost = sum(cost[node] for node in positions[i])

            if total_cost > budget:
                continue

            fitness = LIE(graph, positions[i])

            if fitness > LIE(graph, Pbesti[i]):
                Pbesti[i] = positions[i]

        Lbesti = find_local_best(positions, graph)

    best_fitness = LIE(graph, Lbesti)
    total_cost = sum(cost[node] for node in Lbesti)

    write_output("output.txt", Lbesti, total_cost, best_fitness, k, budget)

    print(f"Best Seed Set: {Lbesti}")
    print(f"Total Cost: {total_cost:.4f}")
    print(f"Best Fitness (LIE value): {best_fitness:.4f}")
    print(f"Number of Seeds (k): {k}")
    print(f"Budget: {budget}")

# import random
# import numpy as np
# import networkx as nx
# from Modified_LIE import LIE
#
#
# def initialize_population_with_budget(graph, population_size, k, candidate_costs, budget):
#     """
#     Initialize the population of seed sets while ensuring each seed set satisfies the budget constraint.
#     """
#     population = []
#     all_nodes = list(graph.nodes())
#
#     while len(population) < population_size:
#         # Randomly select k nodes as the initial seed set
#         seed_set = random.sample(all_nodes, k)
#         # Calculate the total cost of the seed set
#         total_cost = sum(candidate_costs[node] for node in seed_set)
#
#         # Check if the seed set satisfies the budget constraint
#         if total_cost <= budget:
#             population.append(seed_set)
#
#     return population
#
#
# def initialize_velocities(population_size, k):
#     # Initialize velocities as random binary vectors
#     velocities = []
#     for _ in range(population_size):
#         velocity = [random.randint(0, 1) for _ in range(k)]
#         velocities.append(velocity)
#     return velocities
#
#
# def find_local_best(population, costs):
#     # Find the local best by selecting the seed set with the best fitness in the neighborhood
#     local_best = population[0]
#     best_fitness = LIE(graph, local_best)
#     for seed_set in population:
#         fitness = LIE(graph, seed_set)
#         if fitness > best_fitness:
#             best_fitness = fitness
#             local_best = seed_set
#     return local_best
#
#
# # Step 3: Iterative update process
# def update_velocity(position, personal_best, local_best, velocity, w=0.5, c1=1.0, c2=1.0):
#     # Update velocity based on personal best and local best
#     new_velocity = []
#     for i in range(len(position)):
#         r1, r2 = random.random(), random.random()
#         inertia = w * velocity[i]
#         cognitive = c1 * r1 * (personal_best[i] - position[i])
#         social = c2 * r2 * (local_best[i] - position[i])
#         new_velocity.append(int(np.clip(inertia + cognitive + social, 0, 1)))
#     return new_velocity
#
#
#
# def update_position(position, velocity, graph):
#     """
#     Update position based on velocity.
#     If velocity is non-zero, replace the current node with a new node from the graph that is not in the current seed set.
#     """
#     new_position = position.copy()
#     all_nodes = set(graph.nodes())
#
#     for i in range(len(position)):
#         if velocity[i] != 0:  # If velocity is non-zero, change the node
#             # Find a new node that's not already in the seed set
#             available_nodes = list(all_nodes - set(new_position))
#             if available_nodes:  # Ensure there are available nodes
#                 new_node = random.choice(available_nodes)
#                 new_position[i] = new_node  # Replace the node at the current position with the new node
#
#     return new_position
#
#
# # def independent_cascade(graph, seed_set, activation_prob=0.1):
# #     # Simulate the Independent Cascade (IC) model
# #     activated_nodes = set(seed_set)
# #     new_activations = set(seed_set)
# #
# #     while new_activations:
# #         current_activations = set()
# #         for node in new_activations:
# #             neighbors = set(graph.neighbors(node))
# #             for neighbor in neighbors:
# #                 if neighbor not in activated_nodes and random.random() < activation_prob:
# #                     current_activations.add(neighbor)
# #         activated_nodes.update(current_activations)
# #         new_activations = current_activations
# #
# #     return len(activated_nodes) - len(seed_set)  # Return spread (exclude initial seeds)
#
#
# # Step 4: Write output
# def write_output(file, best_solution):
#     with open(file, 'w') as f:
#         f.write("Best Seed Set: " + str(best_solution) + "\n")
#
#
# # Main MODPSO Algorithm Execution
# if __name__ == "__main__":
#     # Parameters
#     # Constants as given in the problem statement
#     m = 1.1
#     r = 5
#     s = 1.1
#     # set number of seed nodes to a threshold
#     k = 5
#     # Budget threshold
#     budget = 90
#     beta = 0.5  # Example user-defined parameter
#
#     # Dataset file
#     # input_file = 'socfb-Reed98.mtx'
#     input_file = 'email-univ.edges'
#     output_file = 'Mydata_with_costs.txt'
#
#     # Read the graph
#     G = nx.read_edgelist(input_file, nodetype=int, create_using=nx.Graph())
#     assert isinstance(G, nx.Graph), "G must be a NetworkX graph"
#
#     # Calculate the cost for each node
#     cost = {}
#     for node in G.nodes():
#         di = G.degree(node)
#         cost[node] = m ** (di / r) + (di / r) * s
#
#     with open(output_file, 'w') as f:
#         for node in G.nodes():
#             f.write(f"{node} {G.degree(node)} {cost[node]:.4f}\n")
#
#     n = G.number_of_nodes()
#
#     # Step 1: Load the graph and compute node costs
#     graph = G
#     node_costs = cost
#     # print("finish", ranked_nodes)
#     searchAgents = 50  # Number of Capuchins (agents)
#     population_size = searchAgents
#     dim = k  # Number of candidates (l)
#     upper_bound = 1.0  # Upper bound of initialization
#     lower_bound = 0.0  # Lower bound of initialization
#     maxIter = 30
#     max_iterations = maxIter
#
#     # Step 2: Initialize the population and velocities
#     # positions = initialize_population(graph, population_size, k)
#     positions = initialize_population_with_budget(graph, population_size, k, cost, budget)
#     velocities = initialize_velocities(population_size, k)
#     Pbesti = positions.copy()  # Personal bests
#     Lbesti = find_local_best(positions, node_costs)  # Local bests
#     # Example usage
#
#     # Step 3: Iterative Update Process
#     for iteration in range(max_iterations):
#         for i in range(population_size):
#             # Update velocity and position
#             velocities[i] = update_velocity(positions[i], Pbesti[i], Lbesti, velocities[i])
#             positions[i] = update_position(positions[i], velocities[i])
#
#             # Calculate the total cost of the updated seed set
#             total_cost = sum(cost[node] for node in positions[i])
#
#             # Check if the new seed set satisfies the budget constraint
#             if total_cost > budget:
#                 # If the cost exceeds the budget, skip this solution
#                 continue
#
#             # If the position is valid (cost <= budget), evaluate fitness
#             fitness = LIE(graph, positions[i])
#
#             # Update personal best if the new fitness is better
#             if fitness > LIE( graph, Pbesti[i]):
#                 Pbesti[i] = positions[i]
#
#         # Update global best (local best in the population)
#         Lbesti = find_local_best(positions, node_costs)
#
#     # Step 4: Write the best solution to output
#     write_output("output.txt", Lbesti)
#     print("best fitness",Lbesti)
