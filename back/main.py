# ----------------------------- DEPENDENCIAS -----------------------------

from typing import Union
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os

import nltk
import pandas as pd
import numpy as np
from unidecode import unidecode
import re, string, unicodedata
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
nltk.download('punkt')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline

nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words("spanish"))
stopwords_list = list(stopwords) # Convierte el set de stopwords a una lista para poder ser pasado como parametro a TfidfVectorizer
snow_stemmer = SnowballStemmer(language="spanish")
import dill as pickle
#from pipeline_functions import df_to_array, preprocess

# ----------------------------- DEFINICION API -----------------------------

app = FastAPI()

origins = ["http://localhost:3000" , "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

## Definicion de clases para los endpoints
class Review(BaseModel):
    review: str
    
class DataFile(BaseModel):
    file_path: str

# ----------------------------- CARGA DE DATOS Y MODELO -----------------------------

df = pd.read_csv("../data/tipo2_entrenamiento_estudiantes.csv", sep=',', encoding = 'utf-8')

X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Class'], test_size=0.2, random_state=1)

classes = sorted(df['Class'].unique())  # Clases: 1, 2, 3, 4 y 5.

with open('logistical_optimal_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

# ----------------------------- API ENDPOINTS -----------------------------


# reentrena el modelo con un nuevo archivo de datos.
@app.post("/retrain_model_with_new_datafile")
async def retrain_model_with_new_datafile(file: UploadFile = File(...)):

    ##if file_path is None:
        ##raise HTTPException(status_code=404, detail="No file path provided")
    
    # Cargar el nuevo archivo de datos sobre el anterior.
    global df
    global model
    global classes
    global X_train, X_test, y_train, y_test
    
    try:
        ##file_path_relative = f"../data/{file_path.file_path}"
        ##print(file_path_relative)
        ##df = pd.read_csv(file_path_relative, sep=',')
        df = pd.read_csv(file.file, sep=',')
        
        # Verifica si el dataframe tiene exactamente dos columnas
        if len(df.columns) != 2:
            raise HTTPException(status_code=422, detail="CSV file must contain exactly two columns: 'Review' and 'Class'")
        
        df.columns = ['Review', 'Class']
        classes = sorted(df['Class'].unique())  # Clases: 1, 2, 3, 4 y 5.
    
    except pd.errors.ParserError:
        # Si ocurre algun error durante el parsing del archivo
        raise HTTPException(status_code=422, detail="Error parsing CSV file")

    except FileNotFoundError:
        # Si la ruta del archivo no es correcta
        raise HTTPException(status_code=422, detail="File not found. Example of correct path: 'new_file.csv'")

    except Exception as e:
        # Si ocurre algun otro error
        raise HTTPException(status_code=500, detail=str(e))
    
    X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Class'], test_size=0.2, random_state=1)
    model.fit(X_train, y_train)
    with open('logistical_optimal_pipeline.pkl', 'wb') as f:
        pickle.dump(model, f, recurse=True)

    return {"message": "Model Retrained Successfully"}
    
# ejecuta el modelo con los datos de prueba y devuelve las métricas de evaluación y los tokens más significativos de cada clase.
@app.get("/test_model")
async def test_model():
    global df 
    global model
    global classes
    global X_test, y_test

    y_test_pred = model.predict(X_test)
    
    # Metricas de interes:

    accuracy = round(accuracy_score(y_test, y_test_pred), 3)
    recall = round(recall_score(y_test, y_test_pred, average='weighted'), 3)
    precision = round(precision_score(y_test, y_test_pred, average='weighted'), 3)
    f1 = round(f1_score(y_test, y_test_pred, average='weighted'), 3)
    
    # Se accede al TfidfVectorizer y al objeto LogisticRegression del pipeline
    vectorizer_logistic = model.named_steps['vect']
    clf_logistic = model.named_steps['clf']

    # Coeficientes (pesos) del modelo de regresión logística
    coefficients = clf_logistic.coef_

    # Nombre de las características (tokens)
    feature_names_logistic = vectorizer_logistic.get_feature_names_out()

    # Estructura para almacenar los tokens más significativos de cada clase.
    significant_tokens_logistic = {}

    top_n = 10
    for i, class_name in enumerate(classes):
        class_coefficients = coefficients[i]
        top_indices = class_coefficients.argsort()[-top_n:][::-1]
        significant_tokens_logistic[class_name] = [feature_names_logistic[idx] for idx in top_indices]

    tokens_dict = {}
    for class_name, tokens in significant_tokens_logistic.items():
        tokens_dict[f"Class: {class_name}"] = "Top significant tokens: " + ", ".join(tokens)

    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        **tokens_dict
    }
    
# predice la clase de una nueva review.
@app.post("/predict")
async def predict(review: Union[Review, None] = None):
    global model
    global classes
    
    if review is None:
        raise HTTPException(status_code=422, detail="No review provided")
    
    try:
        data_dict = {"Review": review.review}
        review_df = pd.DataFrame(data_dict, index=[0])
        
        predicted_class = model.predict(review_df)[0]
        predicted_probs_class = model.predict_proba(review_df)[0].tolist()
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Invalid JSON format")
    
    except Exception as e:
        # Si ocurre algun otro error
        raise HTTPException(status_code=500, detail=str(e))
    
    predict_proba ={}
    for class_name in classes:
        predict_proba[f"Class: {class_name}"] = round(predicted_probs_class[classes.index(class_name)],3)
        
    return {
        "predicted class": str(predicted_class),
        "predicted probabilites for each class": predict_proba
        }
    
# predice la clase de un conjunto de reviews pasados como archivo .csv
@app.post("/predict_from_datafile")
async def predict_from_datafile(file: UploadFile = File(...)):
    global model
    
    try:
        df = pd.read_csv(file.file, sep=',')
        
        # Verifica si el dataframe tiene exactamente una columna
        if len(df.columns) != 1:
            raise HTTPException(status_code=422, detail="CSV file must contain exactly one column: 'Review'")
        
        df.columns = ['Review']
        
        prueba_classified = model.predict(df)
        df['Class'] = prueba_classified
        user_path = os.path.expanduser('~')
        df.to_csv(f'{user_path}/prueba_clasificados.csv', index=False)
        
    except pd.errors.ParserError:
        # Si ocurre algun error durante el parsing del archivo
        raise HTTPException(status_code=422, detail="Error parsing CSV file")
    
    except Exception as e:
        # Si ocurre algun otro error
        raise HTTPException(status_code=500, detail=str(e))
    
    return {
        "message": "Exito! Se ha creado un nuevo archivo 'prueba_clasificados.csv' con las predicciones. Encuentrelo en la carpeta data."
        }
