import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import json
import pickle
import matplotlib.pyplot as plt
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
import re
import os

# --- 1. Configuración inicial ---
nltk.download('stopwords')
stemmer = SnowballStemmer("spanish")
stop_words = set(stopwords.words('spanish'))

# --- 2. Preprocesamiento mejorado ---
def preprocess_text(text):
    """Normalización robusta de texto en español"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower().strip()
    text = re.sub(r'[^\w\sáéíóúüñ]', '', text)
    tokens = [stemmer.stem(token) for token in text.split() if token not in stop_words]
    return ' '.join(tokens)

# --- 3. Datos de ejemplo mejorados ---
def load_data():
    intents = {
        "diagnosis": [
            "dolor de cabeza y fiebre", 
            "tos persistente con flema",
            "fatiga y náuseas",
            "dolor abdominal intenso",
            "mareos después de comer",
            "dolor en el pecho al respirar",
            "sangrado nasal frecuente",
            "pérdida de peso inexplicable"
        ],
        "precautions": [
            "cómo prevenir la gripe",
            "consejos para diabetes tipo 2",
            "precauciones contra malaria",
            "medidas para evitar resfriados",
            "higiene para prevenir infecciones",
            "cómo evitar contagios de covid",
            "prevención de enfermedades cardíacas"
        ],
        "description": [
            "qué es la neumonía bacteriana",
            "descripción de diabetes gestacional",
            "información sobre alergias al polen",
            "qué causa la hipertensión arterial",
            "síntomas de la migraña",
            "qué es el asma bronquial",
            "definición de artritis reumatoide"
        ]
    }
    return intents

# --- 4. Vectorización consistente ---
def create_vectorizer():
    """Configuración TF-IDF con parámetros optimizados"""
    return TfidfVectorizer(
        max_features=1024,  # Tamaño fijo para consistencia
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        stop_words=list(stop_words),
        dtype=np.float32
    )

# --- 5. Arquitectura del modelo ---
def build_model(input_dim, num_classes):
    """Modelo con dimensiones explícitas"""
    print(f"\n🔧 Construyendo modelo para input_dim={input_dim}, clases={num_classes}")
    
    model = Sequential([
        Input(shape=(input_dim,), name="input_layer"),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        Dense(num_classes, activation='softmax', name="output_layer")
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- 6. Pipeline de entrenamiento ---
def train():
    # 1. Preparar datos
    intents = load_data()
    texts, labels = [], []
    label_map = {label: i for i, label in enumerate(intents.keys())}
    
    for intent, examples in intents.items():
        texts.extend([preprocess_text(text) for text in examples])
        labels.extend([label_map[intent]] * len(examples))
    
    # 2. Vectorización
    vectorizer = create_vectorizer()
    X = vectorizer.fit_transform(texts).toarray()
    y = np.array(labels)
    
    print(f"\n📊 Shape de X: {X.shape}, Shape de y: {y.shape}")
    print(f"🧠 Tamaño del vocabulario: {len(vectorizer.vocabulary_)}")
    
    # 3. División train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Construir modelo
    model = build_model(X_train.shape[1], len(label_map))
    model.summary()
    
    # 5. Verificación dimensional
    sample = X_train[:1]
    try:
        pred = model.predict(sample)
        print(f"\n✅ Verificación dimensional exitosa. Shape de salida: {pred.shape}")
    except Exception as e:
        print(f"\n❌ Error en verificación dimensional: {str(e)}")
        print(f"Shape de entrada: {sample.shape}")
        print(f"Shape esperado: {model.input_shape}")
        return
    
    # 6. Entrenamiento
    print("\n🚀 Comenzando entrenamiento...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=500,
        batch_size=32,
        callbacks=[
            EarlyStopping(patience=8, restore_best_weights=True),
            ModelCheckpoint("model/best_model.keras", save_best_only=True)
        ],
        verbose=1
    )
    
    # 7. Evaluación
    print("\n📊 Evaluación final:")
    y_pred = model.predict(X_test).argmax(axis=1)
    print(classification_report(y_test, y_pred, target_names=intents.keys()))
    
    # 8. Guardar artefactos
    os.makedirs("model", exist_ok=True)
    model.save("model/intent_classifier.keras")
    with open("model/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    print("\n✅ Entrenamiento completado exitosamente!")

if __name__ == "__main__":
    train()