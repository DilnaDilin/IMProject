import time
from datetime import datetime
from random import random

import networkx as nx
import numpy as np

from capSA import CapSA
# Constants as given in the problem statement
m = 1.1
r = 5
s = 1.1
# Set the number of seed nodes to a threshold
k = 20
# Budget threshold
budget = 800
num_runs = 5

beta = 0.5  # Example user-defined parameter

# Dataset file
# input_file = 'socfb-Reed98.mtx'
# input_file = 'email-univ.edges'
input_file = 'soc-hamsterster.edges'
# input_file = 'soc-wiki-Vote.mtx'
output_file = 'Mydata_with_costs.txt'

# Read the graph
G = nx.read_edgelist(input_file, nodetype=int, create_using=nx.Graph())
assert isinstance(G, nx.Graph), "G must be a NetworkX graph"

# Calculate the cost for each node
cost = {}
for node in G.nodes():
    di = G.degree(node)
    cost[node] = m ** (di/r) + (di/r) * s

with open(output_file, 'w') as f:
    for node in G.nodes():
        f.write(f"{node} {G.degree(node)} {cost[node]:.4f}\n")

n = G.number_of_nodes()

# Compute centrality measures
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G)

# Build the decision matrix
# centrality_measures = [degree_centrality, betweenness_centrality, closeness_centrality, eigenvector_centrality]
centrality_measures = [degree_centrality, betweenness_centrality]
decision_matrix = np.zeros((n, len(centrality_measures)))

nodes = list(G.nodes())
for i, node in enumerate(nodes):
    for j, centrality in enumerate(centrality_measures):
        decision_matrix[i, j] = centrality[node]

# Normalize the decision matrix using the sum method
normalized_matrix = decision_matrix / decision_matrix.sum(axis=0)

# Compute criteria weights using the provided equation
q = len(nodes)  # Sample size, could be a parameter
criteria_weights = np.zeros(len(centrality_measures))
for j in range(len(centrality_measures)):
    psi_j = np.sort(normalized_matrix[:, j])[-q:]
    criteria_weights[j] = psi_j.sum()

criteria_weights /= criteria_weights.sum()

# Calculate overall scores for each node using SAW
overall_scores = normalized_matrix.dot(criteria_weights)

# Rank the nodes based on overall scores
ranked_nodes = sorted(zip(nodes, overall_scores), key=lambda x: x[1], reverse=True)
ranked_nodes_list = [node for node, score in ranked_nodes]

# Calculate the number of candidates using the given equation
num_candidates = np.ceil(k + (n - k) * (beta * k / n) ** (1 - beta)).astype(int)

# Select the candidates
candidates = ranked_nodes_list[:num_candidates]
print("lenth:" , len(candidates))

# print("finish", ranked_nodes)
searchAgents = 50  # Number of Capuchins (agents)
dim = num_candidates          # Number of candidates (l)
upper_bound = 1.0  # Upper bound of initialization
lower_bound = 0.0  # Lower bound of initialization
maxIter = 100


# Open the output file to write results
# Generate a timestamp for the file name
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_file = f"results_{current_time}.txt"
with open(output_file, 'w') as f:
    # Write the metadata before running the algorithm
    f.write(f"Algorithm name: CapSA\n")
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
        # Call the CapSA function with the defined parameters
        best_fitness, best_seed_set, best_seed_set_bin, convergence_curve = CapSA(searchAgents, maxIter, candidates,
                                                                                  budget, cost, G, upper_bound,
                                                                                  lower_bound, dim, k)

        # Measure end time
        end_time = time.time()
        execution_time = end_time - start_time
        # Output the results
        print("K and cost threshold:", k, budget)
        print("Best Fitness Value:", best_fitness)
        # print("Best Seed Set (Binary):", best_seed_set)
        final_seed_set = [candidates[i] for i, val in enumerate(best_seed_set_bin) if val == 1]
        print("Best Seed Set :", final_seed_set)
        total_cost = sum(cost[node] for node in final_seed_set)
        print("Best Seed Set cost :", total_cost)
        print("Convergence Curve:", convergence_curve)

        #
        print("candid", candidates)

        from IC import IC

        # Set the propagation probability and the number of Monte Carlo simulations
        propagation_probability = 0.01
        monte_carlo_simulations = 1000

        # Calculate the spread using the IC model
        spread = IC(G, final_seed_set, propagation_probability, mc=monte_carlo_simulations)

        # Output the results
        print(f"Final Seed Set: {final_seed_set}")
        print(f"Spread of the Seed Set using IC: {spread}")

        # Write the results of this run to the file
        f.write(f"Run {run + 1}:\n")
        f.write(f"Best Seed Set: {final_seed_set}\n")
        f.write(f"Best Seed Set Cost: {total_cost:.4f}\n")
        f.write(f"Best Fitness Value (LIE): {best_fitness:.4f}\n")
        f.write(f"Spread: {spread}\n")
        f.write(f"Convergence Curve: {convergence_curve.tolist()}\n")
        f.write(f"Execution Time: {execution_time:.4f} seconds\n")
        f.write("\n")

