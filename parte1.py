import random
import numpy as np
import pandas as pd
from numba import njit

# numba pra otimizacao
@njit
def custo(rota, matriz):
    valor = 0
    for i in range(len(rota)-1):
        valor += matriz[rota[i], rota[i+1]]
    # custo de voltar pro inicio
    valor += matriz[rota[-1], rota[0]]
    return valor

def shake(rota, k):
    rota_nova = rota[:]
    for _ in range(k):
        i, j = random.sample(range(1, len(rota)), 2)
        rota_nova[i], rota_nova[j] = rota_nova[j], rota_nova[i]
    return rota_nova


def first_improvement(rota, matriz, vizinhancas):
    melhor_custo = custo(np.array(rota), matriz)
    for viz in vizinhancas:
        for vizinho in viz(rota):
            c = custo(np.array(vizinho), matriz)
            if c < melhor_custo:
                return vizinho
    return rota

def swap_vizinhanca(rota):
    for i in range(1, len(rota)-1):
        for j in range(i+1, len(rota)):
            r = rota[:]
            r[i], r[j] = r[j], r[i]
            yield r

def insert_vizinhanca(rota):
    for i in range(1, len(rota)):
        for j in range(1, len(rota)):
            if i != j:
                r = rota[:]
                cidade = r.pop(i)
                r.insert(j, cidade)
                yield r

def dois_opt_vizinhanca(rota):
    for i in range(1, len(rota)-2):
        for j in range(i+1, len(rota)-1):
            r = rota[:]
            r[i:j] = reversed(r[i:j])
            yield r

def get_initial_solution(matriz):
    n = len(matriz)
    nao_visitados = set(range(n))
    start = random.randrange(n)
    rota = [start]
    nao_visitados.remove(0)
    atual = 0
    while nao_visitados:
        prox = min(nao_visitados, key=lambda j: matriz[atual, j])
        rota.append(prox)
        nao_visitados.remove(prox)
        atual = prox
    return rota



def BVNS(matriz, kmax=3, max_iter=1000):
    n = len(matriz)
    sol = get_initial_solution(matriz)  # solução inicial
    melhor = sol[:]
    melhor_custo = custo(np.array(melhor), matriz)

    vizinhancas = [swap_vizinhanca, insert_vizinhanca, dois_opt_vizinhanca]

    historico = [melhor_custo]

    for _ in range(max_iter):
        k = 1
        while k <= kmax:
            s_prime = shake(melhor, k)
            s_prime2 = first_improvement(s_prime, matriz, vizinhancas)
            c = custo(np.array(s_prime2), matriz)
            if c < melhor_custo:
                melhor = s_prime2
                melhor_custo = c
                historico.append(melhor_custo)
                k = 1
            else:
                k += 1
    return melhor, melhor_custo, historico

def main():
    task = 'tempo'
    matriz_c = np.loadtxt(f"{task}.csv", delimiter=",")

    resultados = []
    historicos = []
    rotas = []

    for run in range(5):
        rota, valor, hist = BVNS(matriz_c)
        resultados.append(valor)
        historicos.append(hist)
        rotas.append(" ".join(map(str, rota)))
        print(f"Execução {run+1}: custo = {valor}, rota = {rota}")

    # Estatísticas
    stats = {
        "min": np.min(resultados),
        "max": np.max(resultados),
        "std": np.std(resultados),
        "mean": np.mean(resultados)
    }

    df_rotas = pd.DataFrame({"execucao": list(range(1,6)), "rota": rotas})
    df_rotas.to_csv(f'rotas_{task}.csv')

    # Salvar CSV com resultados finais
    df_res = pd.DataFrame({"execucao": list(range(1,6)), "custo": resultados})
    for k,v in stats.items():
        df_res.loc[len(df_res)] = [k, v]
    df_res.to_csv(f"resultados_finais_{task}.csv", index=False)

    # CSV com historico de convergencia
    max_len = max(len(h) for h in historicos)
    for h in historicos:
        while len(h) < max_len:
            h.append(h[-1])
    df_hist = pd.DataFrame({f"run_{i+1}": historicos[i] for i in range(5)})
    df_hist.to_csv(f"convergencia_{task}.csv", index=False)


if __name__ == '__main__':
    main()
