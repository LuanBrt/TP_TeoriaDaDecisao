import matplotlib.pyplot as plt 
import numpy as np
import random
from tqdm import tqdm
from parte1 import (custo, shake,
                  swap_vizinhanca, insert_vizinhanca,
                  dois_opt_vizinhanca, get_initial_solution)

import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed

np.random.seed(42)
random.seed(42)

def first_improvement_scalar(rota, vizinhancas, custo_eval):
    melhor_custo = custo_eval(rota)
    for viz in vizinhancas:
        for vizinho in viz(rota):
            c = custo_eval(vizinho)
            if c < melhor_custo:
                return vizinho
    return rota

def biobjetivo_custos(rota, matriz_tempo, matriz_dist):
    return custo(np.array(rota), matriz_tempo), custo(np.array(rota), matriz_dist)


def BVNS_scalarizado_EPS(matriz_tempo, matriz_dist, eps=None):
    penalty = 1e6  # grande penalidade
    def custo_scalar(rota):
        t, d = biobjetivo_custos(rota, matriz_tempo, matriz_dist)
        viol = max(0.0, d - eps)  # violação da restrição
        return t + penalty * viol

    sol = get_initial_solution(matriz_dist)
    melhor = sol[:]
    melhor_custo = custo_scalar(melhor)
    vizinhancas = [swap_vizinhanca, insert_vizinhanca, dois_opt_vizinhanca]

    for _ in tqdm(range(200)):
        k = 1
        while k <= 3:
            s_prime = shake(melhor, k)
            s_prime2 = first_improvement_scalar(s_prime, vizinhancas, custo_scalar)
            c = custo_scalar(s_prime2)
            if c < melhor_custo:
                melhor, melhor_custo = s_prime2, c
                k = 1
            else:
                k += 1

    return melhor, biobjetivo_custos(melhor, matriz_tempo, matriz_dist)


def BVNS_scalarizado_PW(matriz_tempo, matriz_dist, w=(0.5, 0.5), eps=None):
    def custo_scalar(rota):
        t, d = biobjetivo_custos(rota, matriz_tempo, matriz_dist)
        
        return w[0]*t + w[1]*d

    sol = get_initial_solution(matriz_dist)
    melhor = sol[:]
    melhor_custo = custo_scalar(melhor)
    vizinhancas = [swap_vizinhanca, insert_vizinhanca, dois_opt_vizinhanca]

    for _ in tqdm(range(100)):
        k = 1
        while k <= 3:
            s_prime = shake(melhor, k)
            s_prime2 = first_improvement_scalar(s_prime, vizinhancas, custo_scalar)
            c = custo_scalar(s_prime2)
            if c < melhor_custo:
                melhor, melhor_custo = s_prime2, c
                k = 1
            else:
                k += 1
    return melhor, biobjetivo_custos(melhor, matriz_tempo, matriz_dist)

def pareto_front(pontos):
    pontos = np.array(pontos)
    pareto = []
    for i, p in enumerate(pontos):
        if not np.any( np.all(pontos <= p, axis=1) & np.any(pontos < p, axis=1) ): # se algum ponto for <= p em todos os objetivos e < em pelo menos um
            pareto.append(p)

    pareto = np.array(sorted(pareto, key=lambda x: x[0]))
    return pareto


def run_pw(args):
    tempo, dist, pesos = args

    rota1, (f1_1, f2_1) = BVNS_scalarizado_PW(tempo, dist, w=(1.0, 0.0))
    z1 = (f1_1, f2_1)

    # z2: minimiza f2  (peso (0,1))
    rota2, (f1_2, f2_2) = BVNS_scalarizado_PW(tempo, dist, w=(0.0, 1.0))
    z2 = (f1_2, f2_2)

    tol = 1e-3
    max_depth = 3
    def almost_equal(a, b, eps=tol):
        return abs(a - b) <= eps

    def same_point(p, q, eps=tol):
        return almost_equal(p[0], q[0], eps) and almost_equal(p[1], q[1], eps)

    visited = set()  # para evitar duplicatas
    sol = []

    def add_point(r, p):
        key = (round(p[0], 10), round(p[1], 10))
        if key not in visited:
            visited.add(key)
            sol.append((r, p))

    def split(zl, zr, depth):
        if depth > max_depth:
            return
        # Passo 2: λ1 e λ2
        lam1 = zl[1] - zr[1]
        lam2 = zr[0] - zl[0]

        # linha vertical/horizontal
        if almost_equal(lam1, 0.0) and almost_equal(lam2, 0.0):
            return

        rota, (f1, f2) = BVNS_scalarizado_PW(tempo, dist, w=(lam1, lam2))
        z = (f1, f2)

        # Critérios de parada
        if same_point(z, zl) or same_point(z, zr):
            return

        add_point(rota, z)

        split(zl, z, depth + 1)
        split(z, zr, depth + 1)

    add_point(rota1, z1)
    add_point(rota2, z2)
    split(z1, z2, depth=0)

    sol.sort(key=lambda p: (p[0], p[1]))
    return sol

def run_eps(args):
    tempo, dist, fronteiras_pw = args

    # utopia f*
    f2_star = min(d for _, d in fronteiras_pw)  # best distance

    # nadir (worst) f_o
    f2_o = max(d for _, d in fronteiras_pw)

    eps_vals = np.linspace(f2_star, f2_o, 20)

    sol_set = []
    for eps in eps_vals:
        rota, (t, d) = BVNS_scalarizado_EPS(tempo, dist, eps=eps)
        if d <= eps:
            sol_set.append((rota, (t, d)))

    return sol_set

# ----------- Experimentos -----------

def main():
    tempo = np.loadtxt("tempo.csv", delimiter=",")
    dist = np.loadtxt("distancia.csv", delimiter=",")
    runs = 5
    pesos = np.linspace(0, 1, 20)
    

    print("Executando método Soma Ponderada (Pw)...")
    fronteiras_pw = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_pw, (tempo, dist, pesos)) for _ in range(runs)]
        all_rows_pw = []
        for run_id, future in enumerate(as_completed(futures), start=1):
            result = future.result()  
            for traj, cost in result:
                all_rows_pw.append({
                    "run": run_id,
                    "trajectory": traj,
                    "cost": cost
                })
            fronteiras_pw.append([cost for _, cost in result])

    
    df_pw = pd.DataFrame(all_rows_pw)
    df_pw.to_csv("rotas_pw.csv", index=False)

    print("Executando método Epsilon Restrito (Pε)...")
    fronteiras_eps = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_eps, (tempo, dist, fronteiras_pw[0])) for _ in range(runs)] # Passamos a fronteira Pw da primeira run, para nao recomputar as solucoes nadir e utopia
        all_rows_eps = []
        for run_id, future in enumerate(as_completed(futures), start=1):
            result = future.result()  
            for traj, cost in result:
                all_rows_eps.append({
                    "run": run_id,
                    "trajectory": traj,
                    "cost": cost
                })
            fronteiras_eps.append([cost for _, cost in result])
    
    df_eps = pd.DataFrame(all_rows_eps)
    df_eps.to_csv("rotas_eps.csv", index=False)

    # --- (c) Plot das fronteiras ---
    plt.figure(figsize=(8,6))
    for i, fronteira in enumerate(fronteiras_pw):
        if len(fronteira) == 0:
            continue
        pareto = pareto_front(fronteira)

        print(pareto)
        if len(pareto) > 0:
            t, d = pareto[:,0], pareto[:,1]
            plt.plot(t, d, '-o', label=f'Run {i+1}',
                     alpha=0.8, markersize=4, linewidth=1.5)
    plt.xlabel("Tempo")
    plt.ylabel("Distância")
    plt.title("Fronteiras - Soma Ponderada (Pw)")
    plt.legend()
    plt.savefig("fronteira_pw2.png", dpi=150)

    plt.figure(figsize=(8,6))
    for i, fronteira in enumerate(fronteiras_eps):
        if len(fronteira) == 0:
            continue
        pareto = pareto_front(fronteira)

        if len(pareto) > 0:
            t, d = pareto[:,0], pareto[:,1]
            plt.plot(t, d, '-o', label=f'Run {i+1}',
                     alpha=0.8, markersize=4, linewidth=1.5)
    plt.xlabel("Tempo")
    plt.ylabel("Distância")
    plt.title("Fronteiras - Epsilon Restrito (Pε)")
    plt.legend()
    plt.savefig("fronteira_eps2.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()

