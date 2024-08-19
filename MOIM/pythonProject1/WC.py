import numpy as np
from igraph import Graph

def WC(g, S, mc=1000):
    """
    Simulates the Weighted Cascade (WC) model on an undirected graph to estimate the spread of influence.

    Parameters:
    - g (igraph.Graph): The input undirected graph.
    - S (list): List of seed nodes from which the influence starts.
    - mc (int): Number of Monte Carlo simulations to run (default is 1000).

    Returns:
    - float: The average number of nodes influenced (spread) by the seed nodes.
    """

    # Initialize a list to store the spread results for each simulation
    spread = []

    # Perform Monte Carlo simulations
    for i in range(mc):

        # Initialize the set of newly activated nodes and the set of all activated nodes
        new_active = S[:]
        A = S[:]

        # Continue the spread process until no more nodes can be activated
        while new_active:
            new_ones = []
            for node in new_active:

                # Get the neighbors of the active node (for undirected graph)
                neighbors = g.neighbors(node, mode="all")

                # Determine neighbors that become infected using WC model probabilities
                for neighbor in neighbors:
                    np.random.seed(i)
                    p_uv = 1.0 / g.degree(node)  # Degree in an undirected graph is the same for both directions
                    if np.random.uniform(0, 1) < p_uv:
                        if neighbor not in A:  # Activate this neighbor if not already activated
                            new_ones.append(neighbor)

            # Update the list of newly activated nodes
            new_active = list(set(new_ones) - set(A))

            # Add the newly activated nodes to the total set of activated nodes
            A += new_active

        # Record the total number of influenced nodes for this simulation
        spread.append(len(A))

    # Return the average spread over all simulations
    return np.mean(spread)

# import numpy as np
# from igraph import Graph
#
#
# def WC(g, S, mc=1000):
#     """
#     Simulates the Weighted Cascade (WC) model to estimate the spread of influence.
#
#     Parameters:
#     - g (igraph.Graph): The input graph.
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
#         new_active = S[:]
#         A = S[:]
#
#         # Continue the spread process until no more nodes can be activated
#         while new_active:
#             new_ones = []
#             for node in new_active:
#
#                 # Get the neighbors of the active node
#                 neighbors = g.neighbors(node, mode="out")
#
#                 # Determine neighbors that become infected using WC model probabilities
#                 for neighbor in neighbors:
#                     np.random.seed(i)
#                     p_uv = 1.0 / g.degree(node, mode="out")
#                     if np.random.uniform(0, 1) < p_uv:
#                         if neighbor not in A:  # Activate this neighbor if not already activated
#                             new_ones.append(neighbor)
#
#             # Update the list of newly activated nodes
#             new_active = list(set(new_ones) - set(A))
#
#             # Add the newly activated nodes to the total set of activated nodes
#             A += new_active
#
#         # Record the total number of influenced nodes for this simulation
#         spread.append(len(A))
#
#     # Return the average spread over all simulations
#     return np.mean(spread)
