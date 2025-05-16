from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pickle
import json
import re  # <- Añade esto para limpieza de texto

app = Flask(__name__)

# Cargar recursos
model = tf.keras.models.load_model("model/intent_classifier.keras")
with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("data/disease_db.json", "r") as f:
    disease_db = json.load(f)

# Limpiar y normalizar texto
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Elimina signos de puntuación
    return text

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = clean_text(request.json["message"])  # <- Limpia el input
        X = vectorizer.transform([user_input]).toarray()
        pred = model.predict(X, verbose=0)
        
        intents = ["diagnosis", "precautions", "description"]
        intent = intents[np.argmax(pred)]
        
        # Ejemplo: Buscar coincidencias de enfermedades
        matched_disease = None
        for disease in disease_db:
            if disease in user_input:
                matched_disease = disease
                break
        
        if intent == "diagnosis":
            if matched_disease:
                response = f"Posible {matched_disease}. Descripción: {disease_db[matched_disease]['description']}"
            else:
                response = "Por favor menciona síntomas específicos (ej: 'dolor de cabeza con fiebre')"
        
        elif intent == "precautions":
            if matched_disease:
                response = f"Precauciones para {matched_disease}: " + ", ".join(disease_db[matched_disease]["precautions"])
            else:
                response = "Di el nombre de una enfermedad (ej: 'precauciones para la gripe')"
        
        else:
            response = "¿Necesitas descripciones de enfermedades? Menciona una (ej: 'qué es la diabetes')"
        
        return jsonify({"response": response})
    
    except Exception as e:
        print("Error:", str(e))  # Debug en consola
        return jsonify({"response": "Hubo un error procesando tu solicitud"}), 500

if __name__ == "__main__":
    app.run(debug=True)