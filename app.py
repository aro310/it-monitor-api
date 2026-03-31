from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from scipy import stats

app = Flask(__name__)

def determiner_etat(cpu, ram, response_time):
    if cpu > 85 or ram > 85 or response_time > 500:
        return 2  # Critique
    elif cpu > 70 or ram > 70 or response_time > 300:
        return 1  # Dégradé
    else:
        return 0  # Normal

@app.route('/analyser', methods=['POST'])
def analyser():
    data = request.json
    df = pd.DataFrame(data)

    resultats = []

    for serveur in df['serveur'].unique():
        df_srv = df[df['serveur'] == serveur].copy()

        # 1. Chaîne de Markov
        matrice = np.array([
            [0.85, 0.12, 0.03],
            [0.40, 0.45, 0.15],
            [0.20, 0.30, 0.50]
        ])
        etat = determiner_etat(
            df_srv['cpu'].iloc[-1],
            df_srv['ram'].iloc[-1],
            df_srv['response_time'].iloc[-1]
        )
        # Probabilité panne après 2 étapes
        matrice_2 = np.linalg.matrix_power(matrice, 2)
        prob_panne = round(matrice_2[etat][2] * 100, 1)

        # 2. Monte Carlo - Probabilité surcharge CPU
        simulations = []
        for _ in range(1000):
            bruit = np.random.normal(0, df_srv['cpu'].std(), 24)
            simulations.append(df_srv['cpu'].iloc[-1] + np.cumsum(bruit))
        simulations = np.array(simulations)
        prob_surcharge = round(
            float(np.mean(simulations.max(axis=1) > 90) * 100), 1
        )

        # 3. MTTR
        mttr = round(float(df_srv['resolution_time'].mean()), 2)

        # 4. Z-Score anomalie
        if len(df_srv) > 2:
            z = np.abs(stats.zscore(df_srv['response_time']))
            anomalie = bool(z.iloc[-1] > 2.0)
        else:
            anomalie = False

        # 5. Alerte
        alerte = prob_panne > 30 or prob_surcharge > 50 or anomalie

        resultats.append({
            "serveur": serveur,
            "prob_panne_48h": prob_panne,
            "prob_surcharge_cpu": prob_surcharge,
            "mttr": mttr,
            "anomalie_detectee": anomalie,
            "alerte": alerte
        })

    return jsonify(resultats)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
