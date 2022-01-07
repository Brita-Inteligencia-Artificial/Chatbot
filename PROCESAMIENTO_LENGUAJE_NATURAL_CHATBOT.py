import nltk
import numpy as np
import random
import string

print("1.- Definimos el corpus".center(150, "-"))

f = open(r'D:/BRITA INTELIGENCIA ARTIFICIAL/CURSO IA/Curso PLN/PROCESAMIENTO_LENGUAJE_NATURAL_CHAT-BOT/Corpus_crucero.txt', errors='ignore')  # Leemos la direccion del archivo o corpus
                                                                                                                        # ELIMINAMOS LOS ERRORES (CLARO QUE NO DEBERIA DE HABER ERRORES)
raw = f.read()   # Realizamos la lectura y guardamos el archivo en la variable raw
print(raw)       # Imprimimos y visualizamos. Si nos fijamos, existen los puntos (.) y los (\n\n) que nos serviran para que el algoritmo pueda partir las frases que
                 #    queremos devolver al cliente. Con los (.) y los (\n\n) los utlizaremos despues para partir las frases que queremos devolver con nuestro cliente
                 # En otro paso trataremos de hacer una coincidencia de lo que pegunta el cliente y cada una de las sentencias que hay en el corpus
                 # Trataremos de hacer una secuencia con lo que diga el cliente y con cada una de las sentencias que hay en el corpus

#########################################################################################################################################################################################

print("2.1.- Preprocemaniento del texto con NLTK CORPU".center(150, "-"))
# Primer paso, preprocesamiento del texto con NLTK del corpus

raw = raw.lower()# 1ro convertir el fichero de entrada en minúscula

nltk.download('punkt') # Instalar módulo punkt si no está ya instalado (solo ejecutar la primera vez)
nltk.download('wordnet') # Instalar módulo wordnet si no está ya instalado (solo ejecutar la primera vez),Wordnet diccionario semántico incluido en NLTK

sent_tokens = nltk.sent_tokenize(raw)# 2do, Convierte el CORPUS a una lista de sentencias por medio de la tokenizacion, con esto, crearemos una lista de las horaciones del corpus
word_tokens = nltk.word_tokenize(raw)#      Convierte el CORPUS a una lista de palabras por medio de la tokenizacion, con esto, creamos una lista de todas las palabras del corpus

lemmer = nltk.stem.WordNetLemmatizer()  # Utilizamos un lematizador de NLTK que lo utilizaremos dentro de 2 funciones, Wordnet diccionario semántico incluido en NLTK

def LemTokens(tokens):                                    # Una funcion, que seria "def LemTokens" que nos servira para lematizar token a token cada uno de los "word_tokens" y
    return [lemmer.lemmatize(token) for token in tokens]  #   vamos a remover los simbolos de puntuacion, para cada simbolo de puntuacion dentro de la libreria "string.punctuation"
                                                          #   vamos a eliminar todos ellos y el reultado lo vamos a optener en "remove_punct_dict"
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):                                                              # La segunda funcion es "def LemNormalize(text):" y lo que vamos a utilizar para "LemNormalize"
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))  #    es que le daremos un texto (test) y a ese texto tambien le va a eliminar todos los
                                                                                     #    simbolos de puntuacion y tambien lo va a tokenizar a traves del texto en minuscula.

########################################################################################################################################################################################

print("2.2  preprocesamiento del texto del usuario y 3.- Evaluar la similiud mensaje usuario - corpus".center(150, "-"))
# Preprocemaniento del texto del usuario y evaluar similitud del mensaje del usuario y el CORPUS (Evaluar la similitud entre el corpus y el mensaje de usuario)

from sklearn.feature_extraction.text import TfidfVectorizer
# Cuando tenemos muchas palabras fuertes como articulos de preposiciones pueden dominar dentro de un documento, entonces lo vamos a hacer es que todas las palabras
#         que no contienen la informacion base, vamos a hacer una agregacion de tal manera que esas palabras las escalemos y le demos un menor peso, eso lo hace la tecnica
#         TF-IDF (Frecuencia de documentos de frecuencia inversa a termino)
from sklearn.metrics.pairwise import cosine_similarity # Similitud de conseno
# Vamos a verificar la similitud de cosenos, que es una medida de similitud entre dos vectores distintos de cero, Se usara para encontrar la similitud entre las palabras
#         ingresadas por el usuario y las palabras en el corpus
from nltk.corpus import stopwords # Palabras de parada


# Función de respuesta que con base al mensaje que introduce el usuario "user_response" determina la similitud del texto insertado y el corpus
def response(user_response):     # Funcion de respuesta que en base a "user_response" o el mensaje de usuario vamos ha añadir esa
                                 #     respuesta de usuario dentro de todos lo tokens que ya definimos (corpus), lo vamos a lematizar "tokenizer=LemNormalize"
                                 #     y vamos ha eliminar las palabras de parada banzadonos en el español lemastop_words=stopwords.words('spanish')
    robo_response = ''   # Definimos la respuesta del robot como bacia
    sent_tokens.append(user_response)  # Añade al corpus la respuesta de usuario al final
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words=stopwords.words('spanish'))  # lematizamos y eliminamos las palabras de parada de español
                              # Como tokenizador usaremos "tokenizer=LemNormalize" que es la funcion que hemos creado arriba

    tfidf = TfidfVec.fit_transform(sent_tokens)   # Entrenaremos el objeto "tfidfVec", lo vamos a ajustar en base a listado de "sent_tokens", recordar que el
                                 # "sent_tokens" agregaba tanto las oraciones del corpus como el mensaje de usuario

    # EVALUAMOS LASOS DE COSENO ENTRE MENSAJE USUARIO (tfidf[-1] o ultimo elemento del tfidf, que justamente debe ser el ultimo ya que hemos añadido
    #     en "sent_tokens.append(user_response)" el mensaje que ha escrito el usuario) y el CORPUS (tfidf o resto de mensajes del CORPUS)
    vals = cosine_similarity(tfidf[-1], tfidf)  # Hacemos la similitud para que nos devuelva el valor del CORPUS que mas se aproxima al mensaje de usuario
    # Gracias  a la funcion "cosine_similarity" podemos ver similaridades entre cosenos
    idx = vals.argsort()[0][-2]  # --------------------|
    flat = vals.flatten()        #                     |_     Se tiene que hacer esa similitud para que nos devuelva el valor del corpus que mas que se
    flat.sort()                  #                     |      aproxima, el vector del corpus que mas se aproxima a nuestro mensaje de usuario
    req_tfidf = flat[-2]         # --------------------|
    # # Si el resultado es cero (que no hay ningun vector del corpus que este proximo al mensaje de usuario, el robot devolvera una respuesta vista en el if de abajo)
    if (req_tfidf == 0):
        robo_response = robo_response + "Lo siento, no te he entendido. Si no puedo responder a lo que buscas pongase en contacto con un agente de soporte"
        return robo_response

    else:  # Si el resultado no es cero o es casi cero la respuesta del robot cera justamente la sentencia de tokens con el indice que tienen la mayor similitud de conceno
           # Con esto tendriamos una respuesta del robot
        robo_response = robo_response + sent_tokens[idx]
        return robo_response

##############################################################################################################################################################################

print("4.- Definicion de coincidencias manuales, saludos, agradecimientos, despedida".center(150, "-"))

SALUDOS_INPUTS = ("hola", "buenas", "saludos", "qué tal", "que tal!", "hey", "buenos dias", "Que onda", "buen dia", "Buenas tardes", "que onda", "beuasn noches")
SALUDOS_OUTPUTS = ["Hola", "Hola, ¿Qué tal?, ¿en que te puedo ayudar?", "Hola, ¿Cómo te puedo ayudar?", "Hola, ¿En que te puedo ayudar?"]

def saludos(sentence):                      # Funcion que como entrada tiene una sentencia,
    for word in sentence.split():           # Esa sentencia la vamos a splitear (split) en cada una de las palabras
        if word.lower() in SALUDOS_INPUTS:  # para convertilas en minusculas y si esa palabra esta dentro del listado de SALUDOS_INPUTS:
            return random.choice(SALUDOS_OUTPUTS)  # Devolvera SALUDOS_OUTPUTS de forma aleatoria con la funcion "random." si coincide que hemos tenido como input cualquwera
                                                   #   de las palabras

################################################################################################################################################################################

print("5.- Generar la respuesta del chatbot".center(150, "-"))

flag = True
print("\n\nROBOT: \n¡Hola! Mi nombre es ROBOT. Contestare a tus preguntas acerca de sus vacaciones en el crucero. si quiere salir, escribe 'Salir'")
while (flag == True):        # Entraremos a un bucle infinito
    user_response = input()  # optenemos el mensaje del usuario como un input
    user_response = user_response.lower()  # Convertimos a minúscula el mensaje del usuario

    if (user_response != 'salir'):   # Si el mensaje de usuario es igual a salir, seguimos hasta el else de hasta abajo, si es salir entonces termina
        # Si el mensaje de usuario no es salir entonces pasa lo siguiente
        if (user_response == 'gracias' or user_response == 'muchas gracias' or user_response == 'gracias!'):
        # Si el usuario escibe gracias o muchas gracias el robot respondera No hay de que, se podria definir otra funcion de coincidencia
            flag = True
            print("ROBOT: \nNo hay de que")

        else:
            if (saludos(user_response) != None):  # Si la palabra insertada por el usuario es un saludo (Coincidencias manuales definidas previamente), evaluamos
                                                      #   si es un saludo y si lo es ocupamos la funcion en el punto 4 "def saludos(sentence):"
                print("ROBOT: \n" + saludos(user_response), "\n")  # Para mensaje de --->HOLA

            else:  # Si la palabra insertada no es un saludo --> CORPUS
                print("ROBOT: \n", end="")   # Para mensaje de ---> SALIDA
                print(response(user_response), "\n")     # funcion RESPUESTA DEL BOT response definida en el paso tres  ---> RESPUESTA BOT
                sent_tokens.remove(user_response)  # para eliminar del corpus la respuesta del usuario y volver a evaluar con el CORPUS limpio
    else:
        flag = False
        print("ROBOT: \nNos vemos pronto, ¡cuídate!")