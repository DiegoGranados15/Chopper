import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.feature_extraction.text import CountVectorizer
import json
import pickle  # ← Importación añadida

# Datos de ejemplo
intents = {
    "diagnosis": ["dolor de cabeza y fiebre", "tos persistente", "fatiga y náuseas"],
    "precautions": ["cómo prevenir la gripe", "consejos para la diabetes", "precauciones contra malaria"],
    "description": ["qué es la neumonía", "descripción de la diabetes", "información sobre alergias"]
}

# Preparar datos
texts = []
labels = []
label_map = {label: i for i, label in enumerate(intents.keys())}

for intent, examples in intents.items():
    texts.extend(examples)
    labels.extend([label_map[intent]] * len(examples))

# Vectorización
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()
y = np.array(labels)

# Modelo
model = Sequential([
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(intents), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento
model.fit(X, y, epochs=10)

# Guardar (versión corregida)
model.save("model/intent_classifier.keras")  # Formato moderno
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Modelo entrenado y guardado correctamente!")