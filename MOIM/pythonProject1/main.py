import networkx as nx
import numpy as np

from capSA import CapSA
# Constants as given in the problem statement
m = 1.1
r = 5
s = 1.1
# set number of seed nodes to a threshold
k = 5
# Budget threshold
budget = 90
beta = 0.5  # Example user-defined parameter

# Dateset file
#input_file = 'Mydata.mtx'
# input_file = 'socfb-Reed98.mtx'
input_file = 'socfb-Dartmouth6.mtx'
output_file = 'Mydata_with_costs.txt'


G = nx.read_edgelist(input_file, nodetype=int, create_using=nx.Graph())
assert isinstance(G, nx.Graph), "G must be a NetworkX graph"
# Calculate the cost for each node
cost = {}
for node in G.nodes():
    di = G.degree(node)
    cost[node] = m ** (di/r) + (di/r)*s

with open(output_file, 'w') as f:
  for node in G.nodes():
      f.write(f"{node} {G.degree(node)} {cost[node]:.4f}\n")

n = G.number_of_nodes()
# Compute centrality measures
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

# Rank nodes based on centrality
centrality_scores = {node: (degree_centrality[node] + betweenness_centrality[node]) / 2 for node in G.nodes()}
ranked_nodes = sorted(centrality_scores, key=centrality_scores.get, reverse=True)



# Calculate number of candidates using the given equation
num_candidates = np.ceil(k + (n - k) * (beta * k / n) ** (1 - beta)).astype(int)
# Select candidates
candidates = ranked_nodes[:num_candidates]
print("lenth:" , len(candidates))

# print("finish", ranked_nodes)
searchAgents = 50  # Number of Capuchins (agents)
dim = num_candidates          # Number of candidates (l)
upper_bound = 1.0  # Upper bound of initialization
lower_bound = 0.0  # Lower bound of initialization
maxIter = 30



# Call the CapSA function with the defined parameters
best_fitness, best_seed_set, best_seed_set_bin, convergence_curve = CapSA(searchAgents, maxIter, candidates, budget, cost, G,upper_bound, lower_bound, dim, k)

# Output the results
print("K and cost threshold:", k, budget)
print("Best Fitness Value:", best_fitness)
# print("Best Seed Set (Binary):", best_seed_set)
final_seed_set = [candidates[i] for i, val in enumerate(best_seed_set_bin) if val == 1]
print("Best Seed Set :", final_seed_set)
total_cost = sum(cost[node] for node in final_seed_set)
print("Best Seed Set cost :", total_cost)
print("Convergence Curve:", convergence_curve)



# # Step 1: Initialize the Capuchins
# capuchin_population = initialization(searchAgents, dim, upper_bound, lower_bound,k)
# capuchins = adjust_population(capuchin_population,candidates, budget,cost)


print("candid", candidates)
# print("candid", capuchin_population)
# print("candid", capuchins)
# Assuming you have the following variables from your CapSA run:
# - candidates: list of candidate nodes
# - best_seed_set_bin: binary array indicating the selected seed set
# - G: your NetworkX graph



from igraph import Graph
from IC import IC
from WC import WC

# Convert the NetworkX graph to an iGraph object for use with the IC model
ig_graph = Graph.TupleList(G.edges(), directed=False)

# Set the propagation probability and the number of Monte Carlo simulations
propagation_probability = 0.01
monte_carlo_simulations = 1000

# Calculate the spread using the IC model
spread = IC(G, final_seed_set, propagation_probability, mc=monte_carlo_simulations)

# Output the results
print(f"Final Seed Set: {final_seed_set}")
print(f"Spread of the Seed Set using IC: {spread}")

# Calculate the spread using the WC model
spread_wc = WC(ig_graph, final_seed_set, mc=1000)

# Output the results
print(f"Spread of the Seed Set using WC: {spread_wc}")
