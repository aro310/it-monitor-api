from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from scipy import stats

app = Flask(__name__)

def determiner_etat(cpu, ram, response_time):
    if cpu > 85 or ram > 85 or response_time > 500:
        return 2
    elif cpu > 70 or ram > 70 or response_time > 300:
        return 1
    else:
        return 0

@app.route('/analyser', methods=['POST'])
def analyser():
    try:
        data = request.get_json(force=True)

        if isinstance(data, str):
            import json
            data = json.loads(data)

        df = pd.DataFrame(data)

        # Supprimer row_number si elle existe
        if 'row_number' in df.columns:
            df = df.drop(columns=['row_number'])

        # Convertir les colonnes numériques
        cols_numeriques = ['cpu', 'ram', 'disk', 'response_time',
                          'incidents', 'resolution_time']
        for col in cols_numeriques:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.').str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)

        resultats = []

        for serveur in df['serveur'].unique():
            df_srv = df[df['serveur'] == serveur].copy().reset_index(drop=True)

            # 1. Chaîne de Markov
            matrice = np.array([
                [0.85, 0.12, 0.03],
                [0.40, 0.45, 0.15],
                [0.20, 0.30, 0.50]
            ])
            etat = determiner_etat(
                float(df_srv['cpu'].iloc[-1]),
                float(df_srv['ram'].iloc[-1]),
                float(df_srv['response_time'].iloc[-1])
            )
            matrice_2 = np.linalg.matrix_power(matrice, 2)
            prob_panne = round(float(matrice_2[etat][2] * 100), 1)

            # 2. Monte Carlo
            cpu_std = float(df_srv['cpu'].std()) + 0.1
            cpu_last = float(df_srv['cpu'].iloc[-1])
            simulations = []
            for _ in range(1000):
                bruit = np.random.normal(0, cpu_std, 24)
                simulations.append(cpu_last + np.cumsum(bruit))
            simulations = np.array(simulations)
            prob_surcharge = round(
                float(np.mean(simulations.max(axis=1) > 90) * 100), 1
            )

            # 3. MTTR
            mttr = round(float(df_srv['resolution_time'].mean()), 2)

            # 4. Z-Score — corrigé
            anomalie = False
            if len(df_srv) > 2:
                valeurs = df_srv['response_time'].values.astype(float)
                moyenne = np.mean(valeurs)
                ecart_type = np.std(valeurs)
                if ecart_type > 0:
                    z_scores = np.abs((valeurs - moyenne) / ecart_type)
                    anomalie = bool(z_scores[-1] > 2.0)

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

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)