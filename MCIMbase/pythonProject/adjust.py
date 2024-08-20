import numpy as np


def adjust_capuchin(capuchin, candidates, budget, candidate_costs):
    # Extract the seed set from binary representation
    seed_set = [candidates[i] for i, val in enumerate(capuchin) if val == 1]

    # Calculate the total cost of the current seed set
    total_cost = sum(candidate_costs[node] for node in seed_set)

    # If the total cost exceeds the budget, adjust the seed set
    if total_cost > budget:
        adjusted_seed_set, total_cost = adjust_seed_set(seed_set, budget, candidate_costs)
        # Convert the adjusted seed set back to binary representation
        new_binary_capuchin = np.zeros_like(capuchin)
        for node in adjusted_seed_set:
            new_binary_capuchin[candidates.index(node)] = 1
        return new_binary_capuchin
    else:
        return capuchin


def adjust_seed_set(seed_set, budget, costs):
    # Sort the seed set based on individual node cost in ascending order (least expensive first)
    seed_set = sorted(seed_set, key=lambda node: costs[node])

    # Calculate the total cost of the seed set
    total_cost = sum([costs[node] for node in seed_set])

    # Initialize an empty list to store nodes that fit within the budget
    final_seed_set = []
    tot_removed = []

    # While the total cost exceeds the budget
    while total_cost > budget and seed_set:
        # Remove the node with the least cost
        removed_node = seed_set.pop(0)
        tot_removed.append(removed_node)
        total_cost -= costs[removed_node]
    tot_removed = sorted(tot_removed, key=lambda node: costs[node])
    final_seed_set = seed_set
    # Add the removed nodes back if the budget allows, starting with the least expensive
    for node in tot_removed:
        if (total_cost + costs[node]) <= budget:
            final_seed_set.append(node)
            total_cost += costs[node]

    return final_seed_set, total_cost

    # # Add nodes back, starting with more expensive ones, until the budget is fully utilized
    # for node in sorted(seed_set + final_seed_set, key=lambda node: costs[node], reverse=True):
    #     if (total_cost + costs[node]) <= budget:
    #         final_seed_set.append(node)
    #         total_cost += costs[node]
    #     elif final_seed_set and (total_cost + costs[node] - costs[final_seed_set[-1]]) <= budget:
    #         # Replace a lower cost node with a higher cost one if it fits within the budget
    #         total_cost -= costs[final_seed_set.pop()]
    #         final_seed_set.append(node)
    #         total_cost += costs[node]
    # print("finalseed",final_seed_set)
    # return final_seed_set, total_cost

# def adjust_population(Capuchins, candidates, budget, candidate_costs):
#     adjusted_Capuchins = []
#     for capuchin in Capuchins:
#         # Extract the seed set from binary representation
#         seed_set = [candidates[i] for i, val in enumerate(capuchin) if val == 1]
#
#         # # Calculate the total cost of the current seed set
#         total_cost = sum(candidate_costs[node] for node in seed_set)
#         # print("tot",total_cost)
#         # print("before", seed_set)
#         if total_cost >budget:
#             # Adjust the seed set based on the budget
#             adjusted_seed_set, total_cost = adjust_seed_set(seed_set, budget, candidate_costs)
#             # Convert the adjusted seed set back to binary representation
#             new_binary_capuchin = np.zeros_like(capuchin)
#             for node in adjusted_seed_set:
#                 new_binary_capuchin[candidates.index(node)] = 1
#             # print("new seed",adjusted_seed_set)
#             # print("new",new_binary_capuchin)
#
#             # Add adjusted Capuchin to the list
#             adjusted_Capuchins.append(new_binary_capuchin)
#             # print("after",adjusted_seed_set)
#
#         else:
#             adjusted_Capuchins.append(capuchin)
#         Capuchins = adjusted_Capuchins
#         # print("tot2", total_cost)
#     return np.array(Capuchins)
# def adjust_seed_set(seed_set, budget, costs):
#     # Sort the seed set based on individual node cost in ascending order (least expensive first)
#     seed_set = sorted(seed_set, key=lambda node: costs[node])
#
#     # Calculate the total cost of the seed set
#     total_cost = sum([costs[node] for node in seed_set])
#     #print(total_cost)
#
#     # Initialize an empty list to store nodes that fit within the budget
#     final_seed_set = []
#     tot_removed=[]
#     # While the total cost exceeds the budget
#     while total_cost > budget and seed_set:
#         # Remove the node with the least cost
#         removed_node = seed_set.pop(0)
#         tot_removed.append(removed_node)
#         total_cost -= costs[removed_node]
#     # print("seedinfun",seed_set)
#     # Sort the removed nodes by cost in ascending order (least expensive first)
#     tot_removed = sorted(tot_removed, key=lambda node: costs[node])
#     final_seed_set=seed_set
#     # Add the removed nodes back if the budget allows, starting with the least expensive
#     for node in tot_removed:
#         if (total_cost + costs[node]) <= budget:
#             final_seed_set.append(node)
#             total_cost += costs[node]
#
#     # # Add nodes back, starting with more expensive ones, until the budget is fully utilized
#     # for node in sorted(seed_set + final_seed_set, key=lambda node: costs[node], reverse=True):
#     #     if (total_cost + costs[node]) <= budget:
#     #         final_seed_set.append(node)
#     #         total_cost += costs[node]
#     #     elif final_seed_set and (total_cost + costs[node] - costs[final_seed_set[-1]]) <= budget:
#     #         # Replace a lower cost node with a higher cost one if it fits within the budget
#     #         total_cost -= costs[final_seed_set.pop()]
#     #         final_seed_set.append(node)
#     #         total_cost += costs[node]
#     # print("finalseed",final_seed_set)
#     return final_seed_set, total_cost
#
#
