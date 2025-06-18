from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np
import requests
import os

app = Flask(__name__)

# Cargar modelo
modelo = load_model("modelo_gas_completo.h5", compile=False)
modelo.compile(loss={'demanda': MeanSquaredError(), 'urgencia': BinaryCrossentropy()})

# Definir categorías posibles para que el encoder tenga la misma estructura
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
categorias = [
    ['0', '1', '2', '3', '4', '5', '6'],  # dia_semana
    ['0', '1'],                           # es_laboral
    ['0', '1']                            # demanda_comercial
]
encoder.fit(np.array(np.meshgrid(*categorias)).T.reshape(-1, 3))

# Dummy scaler fit (importante que tenga 14 columnas)
scaler = MinMaxScaler()
scaler.fit(np.random.rand(10, 14))  # reemplaza esto con scaler real si lo tienes

# 🔑 Tu clave de WeatherAPI
WEATHER_API_KEY = "cd1b592c33c84a1c97a150918251806"
CIUDAD = "Quito"
ALTITUD_ESTIMADA = 2850

def obtener_datos_climaticos():
    try:
        url = f"https://api.weatherapi.com/v1/current.json?q={CIUDAD}&lang=es&key={WEATHER_API_KEY}"
        response = requests.get(url)
        data = response.json()

        temperatura = data["current"]["temp_c"]
        humedad = data["current"]["humidity"] / 100.0  # de 0 a 1
        presion = data["current"]["pressure_mb"] / 1000.0  # de mb a atm aprox

        return temperatura, humedad, presion
    except Exception as e:
        raise RuntimeError(f"Error al obtener clima: {str(e)}")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        campos = ["hora_dia", "stock", "dia_semana", "es_laboral", "demanda_comercial"]
        if not all(k in data for k in campos):
            return jsonify({"error": "Faltan campos en el JSON"}), 400

        # Obtener clima y altitud automáticamente
        temperatura, humedad, presion = obtener_datos_climaticos()
        altitud = ALTITUD_ESTIMADA

        # Variables numéricas
        X_num = np.array([[temperatura, data["hora_dia"], altitud,
                           data["stock"], humedad, presion]])

        # Variables categóricas
        X_cat = np.array([[str(data["dia_semana"]), str(data["es_laboral"]), str(data["demanda_comercial"])]])
        X_cat_encoded = encoder.transform(X_cat)

        # Concatenar y escalar
        X_input = np.hstack([X_num, X_cat_encoded])
        X_scaled = scaler.transform(X_input)

        # Predicción
        pred = modelo.predict(X_scaled)
        demanda_pred = pred[0]
        urgencia_pred = pred[1]

        return jsonify({
            "prediccion_demanda": float(demanda_pred[0][0]),
            "prediccion_urgencia": "URGENTE" if urgencia_pred[0][0] > 0.5 else "NORMAL",
            "probabilidad_urgencia": float(urgencia_pred[0][0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render usa PORT, por defecto 10000
    app.run(host="0.0.0.0", port=port)
