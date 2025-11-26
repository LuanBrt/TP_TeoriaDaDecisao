import pandas as pd


import numpy as np

values = np.random.normal(loc=0, scale=1, size=250*250)

A = values.reshape(250, 250)
A_min = A.min()
A_max = A.max()

A_scaled = (A - A_min) / (A_max - A_min)
df = pd.DataFrame(A_scaled)
df.to_csv("data/risk.csv", index=False, header=False)


N = 250

vehicle_emissions = {
    1: (20, 40),     
    2: (60, 90), 
    3: (110, 160), 
    4: (40, 70), 
    5: (90, 140), 
    6: (300, 500), 
}

vehicle_probabilities = {
    1: 0.15,
    2: 0.20,
    3: 0.25,
    4: 0.15,
    5: 0.15,
    6: 0.10
}

vehicle_ids = list(vehicle_probabilities.keys())
probabilities = list(vehicle_probabilities.values())

vehicle_matrix = np.zeros((N, N), dtype=int)

for i in range(N):
    for j in range(i+1, N):
        v = np.random.choice(vehicle_ids, p=probabilities)
        vehicle_matrix[i, j] = v
        vehicle_matrix[j, i] = v  # symmetric

co2_matrix = np.zeros((N, N), dtype=float)

for i in range(N):
    for j in range(i+1, N):
        vehicle_type = vehicle_matrix[i, j]
        low, high = vehicle_emissions[vehicle_type]
        emission = np.random.uniform(low, high)
        co2_matrix[i, j] = emission
        co2_matrix[j, i] = emission  # symmetric

df = pd.DataFrame(co2_matrix)
df.to_csv("data/co2.csv", index=False, header=False)