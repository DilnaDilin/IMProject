import numpy as np
import networkx as nx


def IC(G, S, p, mc=1000):
    """
    Simulates the Independent Cascade (IC) model to estimate the spread of influence.

    Parameters:
    - G (networkx.Graph): The input graph.
    - S (list): List of seed nodes from which the influence starts.
    - p (float or dict): Propagation probability. Can be a uniform value or a dictionary of edge-specific probabilities.
    - mc (int): Number of Monte Carlo simulations to run (default is 1000).

    Returns:
    - float: The average number of nodes influenced (spread) by the seed nodes.
    """

    def propagate(G, active, p):
        """Propagate influence from the current set of active nodes."""
        new_active = set()
        for node in active:
            neighbors = G.neighbors(node)
            for neighbor in neighbors:
                # If a specific probability is assigned to an edge, use it; otherwise, use the global p.
                prob = p[(node, neighbor)] if isinstance(p, dict) else p
                if neighbor not in active and neighbor not in new_active:
                    if np.random.rand() <= prob:
                        new_active.add(neighbor)
        return new_active

    total_spread = 0

    for _ in range(mc):
        # Initial set of active nodes
        active_nodes = set(S)
        spread = len(active_nodes)

        while True:
            new_active = propagate(G, active_nodes, p)
            if not new_active:
                break
            active_nodes.update(new_active)
            spread += len(new_active)

        total_spread += spread

    # Return the average spread over all simulations
    return total_spread / mc


# import numpy as np
# import networkx as nx
#
#
#
#
# def calculate_custom_probability(G, node1, node2):
#     """Calculate the custom probability based on the degrees and common neighbors of node1 and node2."""
#     N = G.number_of_nodes()
#     pij = 0.01 + (G.degree(node1) + G.degree(node2)) / N + len(list(nx.common_neighbors(G, node1, node2))) / N
#     return pij
#
#
# def IC(G, S, mc=1000):
#     """
#     Simulates the Independent Cascade (IC) model with custom probabilities to estimate the spread of influence.
#
#     Parameters:
#     - G (networkx.Graph): The input graph.
#     - S (list): List of seed nodes from which the influence starts.
#     - mc (int): Number of Monte Carlo simulations to run (default is 1000).
#
#     Returns:
#     - float: The average number of nodes influenced (spread) by the seed nodes.
#     """
#
#     # Initialize a list to store the spread results for each simulation
#     spread = []
#
#     # Perform Monte Carlo simulations
#     for i in range(mc):
#
#         # Initialize the set of newly activated nodes and the set of all activated nodes
#         new_active = set(S)
#         A = set(S)
#
#         # Continue the spread process until no more nodes can be activated
#         while new_active:
#             new_ones = set()
#             for node in new_active:
#                 # Influence each neighbor with custom probability
#                 for neighbor in G.neighbors(node):
#                     if neighbor not in A:  # Only attempt to influence inactive nodes
#                         #pij = calculate_custom_probability(G, node, neighbor)
#                         pij = 0.5
#                         if np.random.uniform(0, 1) < pij:
#                             new_ones.add(neighbor)
#
#             # Update the set of newly activated nodes
#             new_active = new_ones
#
#             # Add the newly activated nodes to the total set of activated nodes
#             A.update(new_active)
#
#         # Record the total number of influenced nodes for this simulation
#         spread.append(len(A))
#
#     # Return the average spread over all simulations
#     return np.mean(spread)
#
#
# # Example usage with a sample graph and seed set:
# # G = nx.erdos_renyi_graph(100, 0.1)
# # seed_set = [0, 1, 2]
# # influence_spread = IC_custom(G, seed_set, mc=1000)
# # print(f"Estimated Influence Spread for Seed Set {seed_set}: {influence_spread}")
#
#
# # import numpy as np
# # from igraph import Graph
# #
# #
# # def IC(g, S, p=0.5, mc=1000):
# #     """
# #     Simulates the Independent Cascade (IC) model to estimate the spread of influence.
# #
# #     Parameters:
# #     - g (igraph.Graph): The input graph.
# #     - S (list): List of seed nodes from which the influence starts.
# #     - p (float): Propagation probability (default is 0.5).
# #     - mc (int): Number of Monte Carlo simulations to run (default is 1000).
# #
# #     Returns:
# #     - float: The average number of nodes influenced (spread) by the seed nodes.
# #     """
# #
# #     # Initialize a list to store the spread results for each simulation
# #     spread = []
# #
# #     # Perform Monte Carlo simulations
# #     for i in range(mc):
# #
# #         # Initialize the set of newly activated nodes and the set of all activated nodes
# #         new_active = S[:]
# #         A = S[:]
# #
# #         # Continue the spread process until no more nodes can be activated
# #         while new_active:
# #             new_ones = []
# #             for node in new_active:
# #                 # Determine neighbors that become infected
# #                 np.random.seed(i)
# #                 success = np.random.uniform(0, 1, len(g.neighbors(node, mode="all"))) < p
# #                 new_ones += list(np.extract(success, g.neighbors(node, mode="all")))
# #
# #             # Update the list of newly activated nodes
# #             new_active = list(set(new_ones) - set(A))
# #
# #             # Add the newly activated nodes to the total set of activated nodes
# #             A += new_active
# #
# #         # Record the total number of influenced nodes for this simulation
# #         spread.append(len(A))
# #
# #     # Return the average spread over all simulations
# #     return np.mean(spread)
