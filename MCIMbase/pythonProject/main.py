import networkx as nx
import numpy as np
from DEvolution import DE

# Constants as given in the problem statement
m = 1.1
r = 5
s = 1.1
# Set the number of seed nodes to a threshold
k = 5
# Budget threshold
budget = 90
beta = 0.5  # Example user-defined parameter

# Dataset file
input_file = 'socfb-Reed98.mtx'
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
centrality_measures = [degree_centrality, betweenness_centrality, closeness_centrality, eigenvector_centrality]
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

num_candidates = np.ceil(k + (n - k) * (beta * k / n) ** (1 - beta)).astype(int)

# Select the candidates
candidates = ranked_nodes_list[:num_candidates]

F = 0.8  # Differential weight
CR = 0.9  # Crossover probability
searchAgents = 50  # Number of Capuchins (agents)
dim = num_candidates          # Number of candidates (l)
upper_bound = 1.0  # Upper bound of initialization
lower_bound = 0.0  # Lower bound of initialization
maxIter = 30
# Run the Differential Evolution algorithm
BestFit, BestPos, BestPos_bin, cg_curve = DE(searchAgents, maxIter, candidates, budget, cost, G, F, CR, k)

print("Best Fitness:", BestFit)
print("Best Position (Continuous):", BestPos)
print("Best Position (Binary):", BestPos_bin)

# Output the results
print("K and cost threshold:", k, budget)
print("Best Fitness Value:", BestFit)
# print("Best Seed Set (Binary):", best_seed_set)
final_seed_set = [candidates[i] for i, val in enumerate(BestPos_bin) if val == 1]
print("Best Seed Set :", final_seed_set)
total_cost = sum(cost[node] for node in final_seed_set)
print("Best Seed Set cost :", total_cost)
print("Convergence Curve:", cg_curve)



# # Output the ranked nodes and candidates
# print("Ranked Nodes (Top 10):")
# for node, score in ranked_nodes[:10]:
#     print(f"Node: {node}, SAW Score: {score:.4f}")

print("\nSelected Candidate Nodes:")
print(candidates)

from IC import IC
# Set the propagation probability and the number of Monte Carlo simulations
propagation_probability = 0.01
monte_carlo_simulations = 1000

# Calculate the spread using the IC model
spread = IC(G, final_seed_set, propagation_probability, mc=monte_carlo_simulations)

# Output the results
print(f"Final Seed Set: {final_seed_set}")
print(f"Spread of the Seed Set using IC: {spread}")