# **Documentación del Chatbot Médico con IA**  

**Nombre del Proyecto**: MediBot - Asistente Médico con Redes Neuronales  
**Objetivo**: Clasificar preguntas médicas y proporcionar información sobre síntomas, enfermedades y precauciones utilizando un modelo de red neuronal entrenado.  

---

## **📌 Tabla de Contenidos**  
1. [Requisitos](#-requisitos)  
2. [Estructura del Proyecto](#-estructura-del-proyecto)  
3. [Instalación](#-instalación)  
4. [Uso](#-uso)  
5. [Personalización](#-personalización)  
6. [Problemas Comunes](#-problemas-comunes)  
7. [Contribuciones](#-contribuciones)  

---

## **🧰 Requisitos**  
- Python 3.11 (TensorFlow no es compatible con Python 3.13+)  
- Bibliotecas:  
  ```bash
  pip install pandas tensorflow==2.15 scikit-learn flask flask-cors
  ```  
- Datasets en español (ejemplo en `data/symptom_description_es.csv` y `data/symptom_precaution_es.csv`).  

---

## **📂 Estructura del Proyecto**  
```
mediBot/
├── data/
│   ├── symptom_description_es.csv    # Descripciones de enfermedades (ES)
│   ├── symptom_precaution_es.csv     # Precauciones (ES)
│   └── disease_db.json              # Base de datos procesada
├── model/
│   ├── intent_classifier.keras      # Modelo entrenado
│   └── vectorizer.pkl               # Vectorizador de texto
├── templates/
│   └── index.html                   # Interfaz web del chatbot
├── app.py                           # Servidor Flask
├── preprocess.py                    # Procesamiento de datos
└── train_model.py                   # Entrenamiento del modelo
```

---

## **⚙️ Instalación**  
1. Clona el repositorio o descarga los archivos.  
2. Crea y activa un entorno virtual:  
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate # Mac/Linux
   ```  
3. Instala las dependencias:  
   ```bash
   pip install -r requirements.txt  # Si existe, o instala manualmente
   ```  

---

## **🚀 Uso**  
1. **Preprocesar los datos**:  
   ```bash
   python preprocess.py
   ```  
   *(Genera `disease_db.json` a partir de los CSVs)*.  

2. **Entrenar el modelo**:  
   ```bash
   python train_model.py
   ```  
   *(Crea `intent_classifier.keras` y `vectorizer.pkl`)*.  

3. **Iniciar el chatbot**:  
   ```bash
   python app.py
   ```  
   - Abre tu navegador en: `http://127.0.0.1:5000`.  

---

## **🎨 Personalización**  
### **1. Mejorar el modelo**  
- Añade más ejemplos de entrenamiento en `train_model.py`:  
  ```python
  intents = {
      "diagnosis": ["tengo fiebre", "dolor de cabeza intenso", ...],
      "precautions": ["cómo prevenir la gripe", ...],
  }
  ```  

### **2. Ampliar la base de datos**  
- Edita los archivos CSV en `data/` con nuevas enfermedades o precauciones.  

### **3. Modificar la interfaz**  
- Edita `templates/index.html` para cambiar el diseño (CSS/JS).  

---

## **⚠️ Problemas Comunes**  
| Error | Solución |  
|-------|----------|  
| `No matching distribution found for tensorflow` | Usa Python 3.11 y `pip install tensorflow==2.15`. |  
| `404 al acceder a /` | Verifica que `index.html` esté en `templates/`. |  
| `Error 500 en /chat` | Revisa que `disease_db.json` y el modelo estén en las rutas correctas. |  
| `El chatbot no entiende español` | Asegúrate de que los CSVs y las intenciones estén en español. |  

---

## **🤝 Contribuciones**  
1. Haz un fork del proyecto.  
2. Crea una rama: `git checkout -b mejora-interfaz`.  
3. Haz commit de tus cambios: `git commit -m "Añadí nuevas enfermedades"`.  
4. Haz push: `git push origin mejora-interfaz`.  
5. Abre un Pull Request.  

