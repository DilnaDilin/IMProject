import networkx as nx


# from main import G
# import matplotlib.pyplot as plt


# compute one hop nodes
def one_hop_area(G, seed):
    # assert isinstance(G, nx.Graph), "G must be a NetworkX graph"
    one_hop = set()
    for node in seed:
        one_hop.update(set(G.neighbors(node)))
    return list(one_hop - set(seed))


def two_hop_area(G, seed):
    two_hop = set()
    one_hop = one_hop_area(G, seed)
    for node in one_hop:
        two_hop.update(set(G.neighbors(node)))
    return list(two_hop - set(one_hop) - set(seed))


def calc_pcm_prob(G, nodes):
    """Calculate the Propagation Cascade Probability of the nodes."""
    total_nodes = G.number_of_nodes()
    return [G.degree(node) / total_nodes for node in nodes]
# def calc_pcm_prob(G, nodes):
#     p=0.01
#     """Calculate the Propagation Cascade Probability of the nodes using a fixed probability p."""
#     return [p for node in nodes]


def calc_edges(G, group1, group2):
    """Calculate the number of edges for each Group-2 node within Group-1 and Group-2."""
    edge_counts = []
    for node in group2:
        edges = list(G.edges(node))
        count = sum(1 for edge in edges if edge[1] in group1 or edge[1] in group2)
        edge_counts.append(count)
    return edge_counts


def sum_pd(list1, list2):
    """Compute the sum-product of two lists."""
    return sum(a * b for a, b in zip(list1, list2))


def calc_edge_prob(G, seed, one_hop):
    """Calculate the sum of edge probabilities for nodes with their One-Hop area nodes."""
    N = G.number_of_nodes()
    prob_sum = 0
    for node1 in one_hop:
        prob_prod = 1
        for node2 in seed:
            pij = 0.01 + (G.degree(node1) + G.degree(node2)) / N + len(list(nx.common_neighbors(G, node1, node2))) / N
            prob_prod *= (1 - pij)
        prob_sum += (1 - prob_prod)
    return prob_sum


def LIE(G, seed):
    """Calculate the Local Influence Spread Measure for a given capuchin (seed set)."""
    # assert isinstance(G, nx.Graph), "G must be a NetworkX graph"
    Ns1_S = one_hop_area(G, seed)
    # print(Ns1_S)
    Ns2_S = two_hop_area(G, seed)
    # print(Ns2_S)
    pu = calc_pcm_prob(G, Ns2_S)

    du = calc_edges(G, Ns1_S, Ns2_S)
    influence_spread = ((1 + (1 / len(Ns1_S)) * sum_pd(pu, du)) * calc_edge_prob(G, seed, Ns1_S))
    return influence_spread

# plt.figure(figsize=(10, 10))
# nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
# plt.title("Graph Visualization")
# plt.show()
# seed_set = [1, 2, 10, 3]  # Example seed set
# lie_value = LIE(G, seed_set)
# print(f"LIE Value for Seed Set {seed_set}: {lie_value}")
