from Modified_LIE import LIE
import networkx as nx
# from main import G

def fobj(capuchin,candidates,G):
    # assert isinstance(G, nx.Graph), "G must be a NetworkX graph"
    seed_set = [candidates[i] for i, val in enumerate(capuchin) if val == 1]
    # Calculate the LIE for the adjusted seed set
    lie_value = LIE(G,seed_set)
    # Calculate fitness without penalty, since it's adjusted
    fitness_value = lie_value

    return fitness_value
