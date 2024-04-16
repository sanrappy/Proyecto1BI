
import re
from unidecode import unidecode
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
nltk.download('punkt')
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words("spanish"))
stopwords_list = list(stopwords) # Convierte el set de stopwords a una lista para poder ser pasado como parametro a TfidfVectorizer
snow_stemmer = SnowballStemmer(language="spanish")
import pandas as pd

# ----------------------------- PIPELINE -----------------------------

# Funciones de preprocesamiento (necesarias para que el pipeline funcione)

def df_to_array(X):
    return X.to_numpy()

def preprocess(X):

  # Quitar caracteres especiales, puntuaciones, marcas diacríticas, números y demás ruido del texto
  def preprepare(text):
      # Convierte el texto a minuscula
      processed_text = text.lower()

      # Reemplazar el carácter especial unicode '\xa0' con un espacio ' '
      processed_text = processed_text.replace(u'\xa0', u' ')

      # Reemplazar líneas vacías o líneas que contengan solo caracteres de espacio en blanco con un solo espacio
      processed_text = re.sub(r'^\s*$', ' ', str(processed_text))

      # Reemplazar '|' con un espacio
      processed_text = processed_text.replace('|', ' ')

      # Reemplazar varios caracteres especiales con espacios
      processed_text = processed_text.replace('ï', ' ')
      processed_text = processed_text.replace('»', ' ')
      processed_text = processed_text.replace('¿', '. ')
      processed_text = processed_text.replace('ï»¿', ' ')

      # Reemplazar comillas dobles y comillas simples con espacios
      processed_text = processed_text.replace('"', ' ')
      processed_text = processed_text.replace("'", " ")

      # Reemplazar signos de puntuación comunes con espacios
      processed_text = processed_text.replace('?', ' ')
      processed_text = processed_text.replace('!', ' ')
      processed_text = processed_text.replace(',', ' ')
      processed_text = processed_text.replace(';', ' ')
      processed_text = processed_text.replace('.', ' ')
      processed_text = processed_text.replace("(", " ")
      processed_text = processed_text.replace(")", " ")
      processed_text = processed_text.replace("{", " ")
      processed_text = processed_text.replace("}", " ")
      processed_text = processed_text.replace("[", " ")
      processed_text = processed_text.replace("]", " ")
      processed_text = processed_text.replace("~", " ")
      processed_text = processed_text.replace("@", " ")
      processed_text = processed_text.replace("#", " ")
      processed_text = processed_text.replace("$", " ")
      processed_text = processed_text.replace("%", " ")
      processed_text = processed_text.replace("^", " ")
      processed_text = processed_text.replace("&", " ")
      processed_text = processed_text.replace("*", " ")
      processed_text = processed_text.replace("<", " ")
      processed_text = processed_text.replace(">", " ")
      processed_text = processed_text.replace("/", " ")
      processed_text = processed_text.replace("\\", " ")
      processed_text = processed_text.replace("`", " ")
      processed_text = processed_text.replace("+", " ")
      processed_text = processed_text.replace("=", " ")
      processed_text = processed_text.replace("_", " ")
      processed_text = processed_text.replace("-", " ")
      processed_text = processed_text.replace(':', ' ')

      # Reemplazar marcas diacríticas en español con sus letras equivalentes sin marcas diacríticas
      processed_text = unidecode(processed_text)

      # Reemplazar caracteres de nueva línea con espacios
      processed_text = processed_text.replace('\n', ' ').replace('\r', ' ')

      # Reemplazar múltiples espacios consecutivos con un solo espacio
      processed_text = re.sub(" +", " ", processed_text)

      # Reemplazar cualquier carácter no alfabético con un espacio
      processed_text = re.sub('[^a-zA-Z]', ' ', processed_text)

      # Reemplazar múltiples espacios consecutivos con un solo espacio nuevamente
      processed_text = re.sub(' +', ' ', processed_text)

      # Eliminar espacios en blanco antes de los signos de puntuación
      processed_text = re.sub(r'\s([?.!"](?:\s|$))', r'\1', processed_text)

      # Tokeniza las palabras para aplicar SnowballStemmer y reducir las palabras a su raiz: chica, chico -> chic.
      tokens = word_tokenize(processed_text, language="spanish")
      stemmed_tokens = [snow_stemmer.stem(t) for t in tokens]
      processed_text = " ".join(stemmed_tokens) #devuelve la lista de tokens a texto.

      # Devolver la cadena preprocesada
      return processed_text

  df = pd.DataFrame(X)
  df[0] = df[0].apply(preprepare)
  # Extrae la columna como una serie
  column_series = df.iloc[:, 0]
  # Convierte a serie a una lista
  sentence_list = column_series.tolist()
  return sentence_list