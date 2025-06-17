from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import numpy as np

app = Flask(__name__)

# Cargar el modelo sin compilar para evitar errores de deserialización
modelo = load_model("modelo_gas_silencioso.h5", compile=False)
modelo.compile(loss=MeanSquaredError())  # Compilación manual por seguridad

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validación básica de los campos esperados
        campos_esperados = ["hora", "zona", "temperatura", "stock", "humedad", "presion"]
        if not all(campo in data for campo in campos_esperados):
            return jsonify({"error": "Faltan campos en el JSON"}), 400

        entrada = np.array([[data["hora"], data["zona"], data["temperatura"], data["stock"], data["humedad"], data["presion"]]])
        resultado = modelo.predict(entrada)

        return jsonify({"prediccion_demanda": float(resultado[0][0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
