from random import random

import numpy as np
from adjust import adjust_capuchin
# np.random.seed(42)  # Set seed for numpy


def initialization(searchAgents, dim, upper_bound, lower_bound,k):
    """Initialize the first population of Capuchins."""
    pos = np.random.rand(searchAgents, dim) * (upper_bound - lower_bound) + lower_bound
    return pos

# def binary_conversion(population, k,candidates, budget, costs):
#     """Convert the continuous values to binary based on the top k highest values."""
#     binary_population = np.zeros_like(population)
#     for i in range(population.shape[0]):
#         # Get indices of the top k values
#         top_k_indices = np.argsort(population[i])[-k:]
#         # Set the top k indices to 1
#         binary_population[i, top_k_indices] = 1
#     capuchinsFit = adjust_capuchin(binary_population, candidates, budget, costs)
#     return capuchinsFit


def binary_conversion(capuchin, k, candidates, budget, costs):
    """Convert the continuous values of a single Capuchin to binary based on the top k highest values."""
    binary_capuchin = np.zeros_like(capuchin)

    # Get indices of the top k values
    top_k_indices = np.argsort(capuchin)[-k:]

    # Set the top k indices to 1
    binary_capuchin[top_k_indices] = 1

    # Adjust the Capuchin to fit within the budget
    adjusted_capuchin = adjust_capuchin(binary_capuchin, candidates, budget, costs)

    return adjusted_capuchin
