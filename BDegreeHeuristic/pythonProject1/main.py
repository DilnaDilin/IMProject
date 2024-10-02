import time
from datetime import datetime

import networkx as nx


def max_degree_heuristic(graph, budget, node_costs):
    # Step 1: Sort nodes by degree in descending order
    sorted_nodes = sorted(graph.nodes(), key=lambda node: graph.degree(node), reverse=True)

    # Step 2: Select nodes until the budget is exhausted
    seed_set = []
    total_cost = 0

    for node in sorted_nodes:
        node_cost = node_costs[node]
        if total_cost + node_cost <= budget and len(seed_set) < k:
            seed_set.append(node)
            total_cost += node_cost
        else:
            break  # Stop when the budget is exhausted

    return seed_set, total_cost


import numpy as np
from Modified_LIE import  LIE
from IC import  IC
# Example usage
if __name__ == "__main__":
    # Constants and parameters
    m = 1.1
    r = 5
    s = 1.1
    # Set the number of seed nodes to a threshold
    k = 20
    # Budget threshold
    budget = 800
    num_runs = 5
    searchAgents = 50
    maxIter = 100
    pm = 0.2  # Mutation probability
    # input_file = 'email-univ.edges'
    input_file = 'soc-hamsterster.edges'
    # input_file = 'soc-wiki-Vote.mtx'
    # output_file = 'output.txt'
    # Generate a timestamp for the file name
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"Degree_results_{current_time}.txt"

    G = nx.read_edgelist(input_file, nodetype=int, create_using=nx.Graph())
    assert isinstance(G, nx.Graph), "G must be a NetworkX graph"
    n = G.number_of_nodes()
    # Calculate node costs
    cost = {node: m ** (G.degree(node) / r) + (G.degree(node) / r) * s for node in G.nodes()}
    # Open the output file to write results
    with open(output_file, 'w') as f:
        # Write the metadata before running the algorithm
        f.write(f"Algorithm name: DPSO\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Budget: {budget}\n")
        f.write(f"Number of Nodes: {n}\n")
        f.write(f"k: {k}\n")
        f.write(f"Iterations: {maxIter}\n")
        f.write(f"Search Agents: {searchAgents}\n")
        f.write("\n")
        for run in range(num_runs):
            cg_curve = np.zeros(maxIter)
            # Measure start time
            start_time = time.time()
            # Run the MAX_DEG heuristic
            seed_set, total_cost = max_degree_heuristic(G, budget, cost)

            best_fitness = LIE(G, seed_set)
            total_cost = sum(cost[node] for node in seed_set)
            # write_output(output_file, Lbesti, total_cost, best_fitness, k, budget)

            print(f"Best Seed Set: {seed_set}")
            print(f"Total Cost: {total_cost:.4f}")
            print(f"Best Fitness (LIE value): {best_fitness:.4f}")
            print(f"Number of Seeds (k): {k}")
            print(f"Budget: {budget}")
            # Measure end time
            end_time = time.time()
            execution_time = end_time - start_time
            from IC import IC

            # Set the propagation probability and the number of Monte Carlo simulations
            propagation_probability = 0.005
            monte_carlo_simulations = 1000

            # Calculate the spread using the IC model
            spread = IC(G, seed_set, propagation_probability, mc=monte_carlo_simulations)

            # Output the results
            print(f"Final Seed Set: {seed_set}")
            print(f"Spread of the Seed Set using IC: {spread}")
            final_seed_set = seed_set
            # Write the results of this run to the file
            f.write(f"Run {run + 1}:\n")
            f.write(f"Best Seed Set: {final_seed_set}\n")
            f.write(f"Best Seed Set Cost: {total_cost:.4f}\n")
            f.write(f"Best Fitness Value (LIE): {best_fitness:.4f}\n")
            f.write(f"Spread: {spread}\n")
            f.write(f"Execution Time: {execution_time:.4f} seconds\n")
            f.write("\n")




