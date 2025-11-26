import numpy as np

def preference_function(diff):
    """Função de preferência usual."""
    return 1.0 if diff > 0 else 0.0

def promethee2(performance, weights):
    n_alts, n_crit = performance.shape
    P = np.zeros((n_alts, n_alts))
    for i in range(n_alts):
        for k in range(n_alts):
            if i == k:
                continue

            pref_sum = 0
            for j in range(n_crit):
                diff = performance[i, j] - performance[k, j]
                pj = preference_function(diff)
                pref_sum += weights[j] * pj

            P[i, k] = pref_sum

    # Calculo de fluxos 
    phi_plus = P.sum(axis=1)
    phi_minus = P.sum(axis=0)
    phi = phi_plus - phi_minus
    return phi 

if __name__ == "__main__":
    performance = np.array([
        [7, 9, 6],
        [8, 7, 7],
        [9, 6, 8],
    ])

    weights = np.array([0.5, 0.3, 0.2])

    phi = promethee2(performance, weights)
    print("Fluxos de preferência:", phi)
