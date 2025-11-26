import numpy as np

def classical_approach(F):
    A = np.zeros_like(F)
    K, q = F.shape

    for p in range(q):
        col = F[:, p]
        cmin, cmax = col.min(), col.max()

        if cmax == cmin:
            A[:, p] = 1.0
            continue

        A[:, p] = (col - cmin) / (cmax - cmin)

    # Bellman–Zadeh 
    D = A.min(axis=1)

    return D

if __name__ == "__main__":
    F = [
        [10, 5,  0,  8],   # X1
        [ 7, 9,  4,  6],   # X2
        [ 3, 12, 2, 11],   # X3
    ]

    result = classical_approach(np.array(F))
    print(result)