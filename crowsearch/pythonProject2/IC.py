import numpy as np
import networkx as nx

def IC(G, S, p, mc=1000):
    """
    Simulates the Independent Cascade (IC) model to estimate the spread of influence,
    excluding the initial seed nodes.

    Parameters:
    - G (networkx.Graph): The input graph.
    - S (list): List of seed nodes from which the influence starts.
    - p (float or dict): Propagation probability. Can be a uniform value or a dictionary of edge-specific probabilities.
    - mc (int): Number of Monte Carlo simulations to run (default is 1000).

    Returns:
    - float: The average number of nodes influenced (spread) by the seed nodes, excluding the seeds.
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
        new_infected_nodes = set()

        while True:
            new_active = propagate(G, active_nodes, p)
            if not new_active:
                break
            new_infected_nodes.update(new_active)
            active_nodes.update(new_active)

        # Count only the new infected nodes, excluding the seed nodes
        total_spread += len(new_infected_nodes)


    # Return the average spread over all simulations
    return total_spread / mc

