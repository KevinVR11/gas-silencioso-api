from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import requests
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import os

app = Flask(__name__)

# Cargar modelo y preprocesadores
# modelo = load_model("modelo_gas_entrenado.h5")
modelo = load_model("modelo_gas_entrenado.h5", compile=False)
modelo.compile(loss=MeanSquaredError())
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Diccionario de altitud por sector (puedes ajustar estos valores)
ALTITUDES = {
    "Calderón": 2850,
    "Carapungo": 2840,
    "La Ofelia": 2860,
    "Cotocollao": 2870,
    "El Condado": 2880
}

# Token de WeatherAPI
WEATHER_API_KEY = "cd1b592c33c84a1c97a150918251806"

@app.route("/predecir", methods=["POST"])
def predecir():
    try:
        data = request.get_json()
        campos_requeridos = ["sector", "hora_dia", "stock_actual", "dia_semana", "es_laboral", "demanda_comercial"]

        if not all(k in data for k in campos_requeridos):
            return jsonify({"error": "Faltan campos en el JSON"}), 400

        # Datos base del usuario
        sector = data["sector"]
        if sector not in ALTITUDES:
            return jsonify({"error": f"Sector no reconocido: {sector}"}), 400

        hora_dia = data["hora_dia"]
        stock_actual = data["stock_actual"]
        dia_semana = data["dia_semana"]
        es_laboral = data["es_laboral"]
        demanda_comercial = data["demanda_comercial"]
        altitud = ALTITUDES[sector]

        # Consulta a WeatherAPI
        url = f"https://api.weatherapi.com/v1/current.json?q=Quito&lang=es&key={WEATHER_API_KEY}"
        r = requests.get(url)
        clima = r.json()
        temperatura = clima["current"]["temp_c"]
        humedad = clima["current"]["humidity"] / 100
        presion = clima["current"]["pressure_mb"] / 1013.25

        # Preparar input
        df_input = pd.DataFrame([{
            "sector": sector,
            "hora_dia": hora_dia,
            "dia_semana": dia_semana,
            "es_laboral": es_laboral,
            "demanda_comercial": demanda_comercial,
            "temperatura": temperatura,
            "humedad": humedad,
            "presion": presion,
            "altitud": altitud,
            "stock_actual": stock_actual
        }])

        # Codificar sector
        X_cat = encoder.transform(df_input[["sector"]])
        df_input = df_input.drop(columns=["sector"])
        X_num = df_input.values
        X = np.hstack([X_num, X_cat])

        # Escalar
        X_scaled = scaler.transform(X)

        # Predicción
        y_pred = modelo.predict(X_scaled)
        cilindros_predichos = float(y_pred[0][0])
        tiempo_estimado = float(y_pred[0][1])
        urgencia_prob = float(y_pred[0][1])

        # Lógica de urgencia
        es_urgente = int((cilindros_predichos > 70) and (tiempo_estimado < 4.8) and (int(demanda_comercial) == 1))
        estado_urgencia = "Urgente" if es_urgente == 1 else "Normal"

        return jsonify({
            "prediccion_cilindros": int(cilindros_predichos),
            "prediccion_urgencia": estado_urgencia,
            "probabilidad_urgencia": round(urgencia_prob, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render usa PORT, por defecto 10000
    app.run(host="0.0.0.0", port=port)
