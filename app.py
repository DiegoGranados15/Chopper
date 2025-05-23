from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pickle
import json
import re
from datetime import datetime
from translate import Translator

app = Flask(__name__,
            static_url_path='',
            static_folder='static',
            template_folder='templates')

# Configuración de traducción
translator_en_es = Translator(to_lang="es")
translator_es_en = Translator(from_lang="es", to_lang="en")

# Cargar recursos
model = tf.keras.models.load_model("model/intent_classifier.keras")
with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("data/disease_db.json", "r", encoding='utf-8') as f:
    disease_db = json.load(f)

def traducir_a_esp(texto):
    """Traduce inglés->español con fallback elegante"""
    try:
        return translator_en_es.translate(texto) if texto else texto
    except Exception as e:
        print(f"Error traduciendo: {e}")
        return texto

def traducir_a_eng(texto):
    """Traduce español->inglés para el modelo"""
    try:
        return translator_es_en.translate(texto) if texto else texto
    except:
        return texto

def limpiar_texto(texto):
    texto = texto.lower().strip()
    texto = re.sub(r'[^\w\sáéíóúüñ]', '', texto)
    return texto

def buscar_enfermedad(texto):
    texto = limpiar_texto(texto)
    for enfermedad in disease_db:
        if enfermedad.lower() in texto:
            return enfermedad
    return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # 1. Recibir input
        user_input = request.json["message"]
        
        # 2. Preprocesar y traducir a inglés (para el modelo)
        cleaned_input = limpiar_texto(user_input)
        translated_input = traducir_a_eng(cleaned_input)
        
        # 3. Predecir intención
        X = vectorizer.transform([translated_input]).toarray()
        intents = ["diagnosis", "precautions", "description"]
        intent_prob = model.predict(X, verbose=0)[0]
        main_intent = intents[np.argmax(intent_prob)]
        
        # 4. Generar respuesta en inglés
        disease = buscar_enfermedad(cleaned_input)
        if main_intent == "diagnosis":
            response = (f"Possible {disease}. {disease_db[disease]['description']}" 
                       if disease else "Please describe specific symptoms")
        elif main_intent == "precautions":
            response = (f"Precautions for {disease}: " + 
                       ", ".join(disease_db[disease]["precautions"])
                       if disease else "Mention a disease name")
        else:
            response = (f"About {disease}: {disease_db[disease]['description']}" 
                       if disease else "What disease information do you need?")
        
        # 5. Traducir respuesta al español
        return jsonify({
            "response": traducir_a_esp(response),
            "intent": main_intent,
            "confidence": float(intent_prob.max()),
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
    except Exception as e:
        print(f"Error en /chat: {str(e)}")
        return jsonify({
            "response": traducir_a_esp("Sorry, an error occurred. Please try again."),
            "timestamp": datetime.now().strftime("%H:%M")
        }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)