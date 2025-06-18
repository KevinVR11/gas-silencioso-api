from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np

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

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        campos = ["temperatura", "hora_dia", "altitud", "stock", "humedad", "presion",
                  "dia_semana", "es_laboral", "demanda_comercial"]
        if not all(k in data for k in campos):
            return jsonify({"error": "Faltan campos en el JSON"}), 400

        # Variables numéricas
        X_num = np.array([[data["temperatura"], data["hora_dia"], data["altitud"],
                           data["stock"], data["humedad"], data["presion"]]])

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
