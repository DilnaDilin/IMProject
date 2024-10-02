import numpy as np
from collections import deque
from fitness import fobj
from InitialiationFile import initialization, binary_conversion


def generate_neighbors(solution, num_neighbors=5):
    neighbors = []
    for _ in range(num_neighbors):
        new_solution = solution.copy()
        # Swap two random positions in the binary vector
        idx1, idx2 = np.random.choice(len(solution), 2, replace=False)
        new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]
        neighbors.append(new_solution)
    return neighbors


def tabu_search(current_solution, current_fitness, tabu_list, max_tabu_size, candidates, G, k, budget, costs,
                num_neighbors=5):
    # Generate neighbors of the current solution
    neighbors = generate_neighbors(current_solution, num_neighbors)
    best_neighbor = None
    best_neighbor_fitness = current_fitness

    for neighbor in neighbors:
        adjusted_neighbor = binary_conversion(neighbor, k, candidates, budget, costs)
        fitness = fobj(adjusted_neighbor, candidates, G)

        # Check if the neighbor is tabu or not
        if tuple(neighbor) not in tabu_list and fitness > best_neighbor_fitness:
            best_neighbor = neighbor
            best_neighbor_fitness = fitness

    # If a better neighbor is found, update the current solution and add the old solution to the tabu list
    if best_neighbor is not None:
        tabu_list.append(tuple(current_solution))
        if len(tabu_list) > max_tabu_size:
            tabu_list.popleft()  # Remove the oldest entry if the tabu list exceeds its size limit
        return best_neighbor, best_neighbor_fitness

    return current_solution, current_fitness


def CapSA(noP, maxite, candidates, budget, costs, G, UB, LB, dim, k):
    # Initialize Capuchins' positions (binary vectors)
    CapPos = initialization(noP, dim, UB, LB, k)
    adjusted_population = []
    for capuchin in CapPos:
        adjusted_capuchin = binary_conversion(capuchin, k, candidates, budget, costs)
        adjusted_population.append(adjusted_capuchin)

    capuchinsFit = np.array(adjusted_population)
    v = 0.1 * CapPos  # Initial velocity
    v0 = np.zeros((noP, dim))
    CapFit = np.zeros(noP)

    # Calculate initial fitness
    for i in range(noP):
        CapFit[i] = fobj(capuchinsFit[i], candidates, G)

    Fit = CapFit.copy()
    fitCapSA = np.max(CapFit)
    index = np.argmax(CapFit)
    CapBestPos = CapPos.copy()  # Best position initialization
    CapBestPosBin = capuchinsFit.copy()
    Pos = CapPos.copy()
    gFoodPos = CapPos[index, :]
    gFoodPosBin = capuchinsFit[index, :]
    cg_curve = np.zeros(maxite)

    # CapSA Parameters
    bf = 0.70  # Balance factor
    cr = 11.0  # Modulus of elasticity
    g = 9.81
    a1 = 1.250
    a2 = 1.5
    beta = [2, 11, 2]
    wmax = 0.8
    wmin = 0.1

    # Tabu Search Parameters
    tabu_list = deque(maxlen=10)  # Maximum size of the tabu list
    max_tabu_size = 10
    num_neighbors = 5  # Number of neighbors to generate for Tabu Search
    tabu_search_frequency = 5  # Apply Tabu Search every 5 iterations

    for t in range(maxite):
        print("Iteration:", t)
        tau = beta[0] * np.exp(-beta[1] * t / maxite) ** beta[2]
        w = wmax - (wmax - wmin) * (t / maxite)
        fol = np.random.randint(0, noP, size=noP)

        # Velocity update
        for i in range(noP):
            for j in range(dim):
                v[i, j] = w * v[i, j] + a1 * (CapBestPos[i, j] - CapPos[i, j]) * np.random.rand() + a2 * (
                        gFoodPos[j] - CapPos[i, j]) * np.random.rand()

        # CapSA Position update
        for i in range(noP):
            if i < noP / 2:
                if np.random.rand() >= 0.1:
                    r = np.random.rand()
                    if r <= 0.15:
                        CapPos[i, :] = gFoodPos + bf * ((v[i, :] ** 2) * np.sin(
                            2 * np.random.rand() * 1.5)) / g  # Jumping (Projection)
                    elif 0.15 < r <= 0.30:
                        CapPos[i, :] = gFoodPos + cr * bf * (
                                (v[i, :] ** 2) * np.sin(2 * np.random.rand() * 1.5)) / g  # Jumping (Land)
                    elif 0.30 < r <= 0.9:
                        CapPos[i, :] = CapPos[i, :] + v[i, :]  # Movement on the ground
                    elif 0.9 < r <= 0.95:
                        CapPos[i, :] = gFoodPos + bf * np.sin(np.random.rand() * 1.5)  # Swing (Local search)
                    else:
                        CapPos[i, :] = gFoodPos + bf * (v[i, :] - v0[i, :])  # Climbing (Local search)
                else:
                    CapPos[i, :] = tau * (LB + np.random.rand(dim) * (UB - LB))
            elif noP / 2 <= i <= noP:
                eps = ((np.random.rand() + 2 * np.random.rand()) - (3 * np.random.rand())) / (
                        1 + np.random.rand())

                Pos[i, :] = gFoodPos + 2 * (CapBestPos[fol[i], :] - CapPos[i, :]) * eps + \
                            2 * (CapPos[i, :] - CapBestPos[i, :]) * eps

                CapPos[i, :] = (Pos[i, :] + CapPos[i - 1, :]) / 2

        v0 = v.copy()

        # Boundary check and fitness calculation
        for i in range(noP):
            u = UB - CapPos[i] < 0
            l = LB - CapPos[i] > 0
            CapPos[i, :] = LB * l + UB * u + CapPos[i, :] * ~np.logical_xor(u, l)
            # Convert to binary and adjust based on budget
            adjusted_capuchin = binary_conversion(CapPos[i], k, candidates, budget, costs)

            # Calculate fitness after adjustment
            CapFit[i] = fobj(adjusted_capuchin, candidates, G)

            if CapFit[i] > Fit[i]:  # Maximize influence
                CapBestPos[i, :] = CapPos[i, :].copy()
                CapBestPosBin[i, :] = adjusted_capuchin.copy()
                Fit[i] = CapFit[i]

        # Apply Tabu Search periodically
        if t % tabu_search_frequency == 0:
            for i in range(noP):
                # Apply Tabu Search to improve current solution
                CapPos[i, :], CapFit[i] = tabu_search(CapPos[i, :], CapFit[i], tabu_list, max_tabu_size, candidates, G,
                                                      k, budget, costs, num_neighbors)

                if CapFit[i] > Fit[i]:  # Maximize influence
                    CapBestPos[i, :] = CapPos[i, :].copy()
                    CapBestPosBin[i, :] = binary_conversion(CapPos[i, :], k, candidates, budget, costs).copy()
                    Fit[i] = CapFit[i]

        # Update global best position
        fmax = np.max(Fit)

        if fmax > fitCapSA:  # Maximize influence
            gFoodPos = CapBestPos[np.argmax(Fit)].copy()
            gFoodPosBin = CapBestPosBin[np.argmax(Fit)].copy()
            fitCapSA = fmax

        # Obtain the convergence curve
        cg_curve[t] = fitCapSA

    return fitCapSA, gFoodPos, gFoodPosBin, cg_curve
