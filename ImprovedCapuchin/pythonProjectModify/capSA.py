import numpy as np
from fitness import fobj
from InitialiationFile import initialization, binary_conversion
# np.random.seed(42)  # Set seed for numpy

def crossover(parent1, parent2):
    # Choose two crossover points
    crossover_point1 = np.random.randint(1, len(parent1) - 1)
    crossover_point2 = np.random.randint(crossover_point1, len(parent1))

    # Create offspring by swapping between the crossover points
    offspring1 = np.concatenate(
        (parent1[:crossover_point1], parent2[crossover_point1:crossover_point2], parent1[crossover_point2:]))
    offspring2 = np.concatenate(
        (parent2[:crossover_point1], parent1[crossover_point1:crossover_point2], parent2[crossover_point2:]))

    return offspring1, offspring2


def mutation(offspring, mutation_rate):
    val = 0
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            offspring[i] = 1 - offspring[i]  # Flip bit
            val = 1
    return offspring, val
def adaptive_rates(current_iter, max_iters, initial_rate):
    # Adapt the rate linearly based on iteration progress
    return initial_rate * (1 - (current_iter / max_iters))
def CapSA(noP, maxite, candidates, budget, costs, G, UB, LB, dim, k):
    # Initialize Capuchins' positions (binary vectors)
    CapPos = initialization(noP, dim, UB, LB, k)
    # Step 2: Convert to binary and adjust capuchins
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
    CapBestPos = CapPos.copy()
    CapBestPosBin = capuchinsFit.copy()
    Pos = CapPos.copy()
    gFoodPos = CapPos[index, :]
    gFoodPosBin = capuchinsFit[index, :]
    cg_curve = np.zeros(maxite)
    CapuchinPos = CapPos
    # CapSA Parameters
    bf = 0.70
    cr = 11.0
    g = 9.81
    a1 = 1.250
    a2 = 1.5
    beta = [2, 11, 2]
    wmax = 0.8
    wmin = 0.1
    # Initialize adaptive rates
    initial_crossover_rate = 0.9
    initial_mutation_rate = 0.01
    for t in range(maxite):
        print("Iteration:", t)
        tau = beta[0] * np.exp(-beta[1] * t / maxite) ** beta[2]
        w = wmax - (wmax - wmin) * (t / maxite)
        fol = np.random.randint(0, noP, size=noP)
        CapPos = CapuchinPos
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
            adjusted_capuchin = binary_conversion(CapPos[i], k, candidates, budget, costs)
            CapFit[i] = fobj(adjusted_capuchin, candidates, G)

            if CapFit[i] > Fit[i]:
                CapBestPos[i, :] = CapPos[i, :].copy()
                CapBestPosBin[i, :] = adjusted_capuchin.copy()
                CapuchinPos[i,:] = CapPos[i,:]
                Fit[i] = CapFit[i]

        fmax = np.max(Fit)
        if fmax > fitCapSA:
            gFoodPos = CapBestPos[np.argmax(Fit)].copy()
            gFoodPosBin = CapBestPosBin[np.argmax(Fit)].copy()
            fitCapSA = fmax

        cg_curve[t] = fitCapSA

        # Apply GA to a fixed percentage of the worst-performing population
        percentage = 0.3  # You can adjust this percentage
        num_worst = int(noP * percentage)
        num_best = num_worst
        # Get indices of the worst-performing individuals
        worst_indices = np.argsort(Fit)[:num_worst]
        # Get indices of the best-performing individuals, sorted in descending order
        best_indices = np.argsort(Fit)[-num_best:][::-1]

        # # Apply crossover and mutation to the worst-performing individuals
        crossover_rate = 0.6
        mutation_rate = 0.1

        for i in range(0, num_worst, 2):
            if i + 1 < num_worst and np.random.rand() < crossover_rate:
                offspring1, offspring2 = crossover(CapuchinPos[best_indices[i]], CapuchinPos[best_indices[i + 1]])
                CapuchinPos[worst_indices[i]] = offspring1
                CapuchinPos[worst_indices[i + 1]] = offspring2

        for i in worst_indices:
            CapuchinPos[i], val = mutation(CapuchinPos[i], mutation_rate)



        print("fit", cg_curve)

    return fitCapSA, gFoodPos, gFoodPosBin, cg_curve

