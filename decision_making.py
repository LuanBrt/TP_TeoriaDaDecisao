import numpy as np 
import pandas as pd
from multi_objective import pareto_front
from utils.promethee import promethee2
from utils.classical_approach import classical_approach

def calculate_risk(rota, risk_matrix):
    risk = 0
    for i in range(len(rota)-1):
        risk -= np.log(risk_matrix[rota[i], rota[i+1]])
    # custo de voltar pro inicio
    risk -= np.log(risk_matrix[rota[-1], rota[0]])
    return risk

def calculate_co2(rota, co2_matrix):
    emission = 0
    for i in range(len(rota)-1):
        emission += co2_matrix[rota[i], rota[i+1]]
    # custo de voltar pro inicio
    emission += co2_matrix[rota[-1], rota[0]]
    return emission

if __name__ == "__main__":
    df_run_eps = pd.read_csv("runs/rotas_eps.csv")
    df_run_pw = pd.read_csv("runs/rotas_pw.csv")

    risk_matrix = np.loadtxt("data/risk.csv", delimiter=",")
    co2_matrix = np.loadtxt("data/co2.csv", delimiter=",")

    df_run_eps['cost'] = df_run_eps['cost'].apply(eval)
    df_run_pw['cost'] = df_run_pw['cost'].apply(eval)

    df = pd.concat([df_run_eps.assign(source="eps"),
                    df_run_pw.assign(source="pw")],
                   ignore_index=True)
    
    df['dist'] = df['cost'].apply(lambda x: x[0])
    df['time'] = df['cost'].apply(lambda x: x[1])
    df['risk'] = df['trajectory'].apply(
        lambda traj: calculate_risk(eval(traj), risk_matrix))
    df['co2'] = df['trajectory'].apply(
        lambda traj: calculate_co2(eval(traj), co2_matrix))
    
    frontier = df['cost'].tolist()
    pf = pareto_front(frontier)
    pf = pf.tolist()
    pf = [tuple(x) for x in pf]

    pareto_df = df[df['cost'].isin(pf)]
    pareto_df = pareto_df.reset_index(drop=True)
    pareto_df = pareto_df.drop_duplicates(subset=['dist', 'time', 'risk', 'co2'])
    lt_df = pareto_df[['dist', 'time', 'risk', 'co2']].to_latex(index=True)
    print(lt_df)

    performance = -pareto_df[['dist', 'time', 'risk', 'co2']].to_numpy()
    weights = np.array([0.1, 0.4, 0.3, 0.2])

    scores_prom = promethee2(performance, weights)
    scores_ca = classical_approach(performance)

    pareto_df['promethee_score'] = scores_prom
    pareto_df['CA_score'] = scores_ca
    print(scores_ca)

    pareto_df['rankingPromethee'] = pareto_df['promethee_score'].rank(
        ascending=False, method='dense'
    ).astype(int)

    pareto_df['rankingCA'] = pareto_df['CA_score'].rank(
        ascending=False, method='dense'
    ).astype(int)

    print(pareto_df.drop(columns=['run', 'trajectory', 'cost', 'source', 'dist', 'time', 'risk', 'co2']).to_latex())

    
