# ----------------------------- DEPENDENCIAS -----------------------------

from typing import Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
from pipeline_functions import df_to_array, preprocess


app = FastAPI()

origins = ["http://localhost:3000" , "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ----------------------------- CARGA DE DATOS Y MODELO -----------------------------

df = pd.read_csv("../data/tipo2_entrenamiento_estudiantes.csv", sep=',', encoding = 'utf-8')

with open('logistical_optimal_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

# ----------------------------- Validacion del funcionamiento del modelo: -----------------------------
# revs = df.copy()
# X_train, X_test, y_train, y_test = train_test_split(revs['Review'], revs['Class'], test_size=0.2, random_state=1)

# y_test_pred = model.predict(X_test)


# print("Metricas para el conjunto de test")
# print('Exactitud: %.2f' % accuracy_score(y_test, y_test_pred))
# print("Recall: {:.3f}".format(recall_score(y_test, y_test_pred, average='weighted')))
# print("Precisión: {:.3f}".format(precision_score(y_test, y_test_pred, average='weighted')))
# print("Puntuación F1: {:.3f}".format(f1_score(y_test, y_test_pred, average='weighted')))
# print()


# # Impresión de los tokens más significativos por clase después de ajustar el modelo

# # Se accede al TfidfVectorizer y al objeto LogisticRegression del pipeline
# vectorizer_logistic = model.named_steps['vect']
# clf_logistic = model.named_steps['clf']

# # Coeficientes (pesos) del modelo de regresión logística
# coefficients = clf_logistic.coef_

# # Nombre de las características (tokens)
# feature_names_logistic = vectorizer_logistic.get_feature_names_out()

# # Estructura para almacenar los tokens más significativos de cada clase.
# significant_tokens_logistic = {}

# top_n = 10
# classes = sorted(revs['Class'].unique())  # Clases: 1, 2, 3, 4 y 5.

# for i, class_name in enumerate(classes):
#     class_coefficients = coefficients[i]
#     top_indices = class_coefficients.argsort()[-top_n:][::-1]
#     significant_tokens_logistic[class_name] = [feature_names_logistic[idx] for idx in top_indices]

# for class_name, tokens in significant_tokens_logistic.items():
#     print(f"Class: {class_name}")
#     print("Top significant tokens:", tokens)
#     print()

# ----------------------------- API ENDPOINTS -----------------------------

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None, m: Union[str, None] = None):
    return {"item_id": item_id,
            "q": q,
            "m": m
           }
    
@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}


@app.post("/retrain_model_with_new_datafile")
async def retrain_model_with_new_datafile(file_path: Union[str, None] = None):

    if file_path is None:
        raise HTTPException(status_code=404, detail="No file path provided")
    
    # Cargar el nuevo archivo de datos sobre el anterior.
    global df
    global model
    try:
        file_path = f"../data/{file_path}"
        df = pd.read_csv(file_path, sep=',')
        
        # Verifica si el dataframe tiene exactamente dos columnas
        if len(df.columns) != 2:
            raise HTTPException(status_code=422, detail="CSV file must contain exactly two columns: 'Review' and 'Class'")
        
        df.columns = ['Review', 'Class']
    
    except pd.errors.ParserError:
        # Si ocurre algun error durante el parsing del archivo
        raise HTTPException(status_code=422, detail="Error parsing CSV file")

    except FileNotFoundError:
        # Si la ruta del archivo no es correcta
        raise HTTPException(status_code=422, detail="File not found. Example of correct path: 'new_file.csv'")

    except Exception as e:
        # Si ocurre algun otro error
        raise HTTPException(status_code=500, detail=str(e))
    
    model.fit(df['Review'], df['Class'])
    with open('logistical_optimal_pipeline.pkl', 'wb') as f:
        pickle.dump(model, f, recurse=True)

    return {"message": "Model Retrained Successfully"}
    
    
@app.get("/test_model")
async def test_model():
    X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Class'], test_size=0.3, random_state=1)

    y_test_pred = model.predict(X_test)
    
    # Metricas de interes:

    accuracy = accuracy_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred, average='weighted')
    precision = precision_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    
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
    classes = sorted(df['Class'].unique())  # Clases: 1, 2, 3, 4 y 5.

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
    
    