import nltk

"""
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
"""

from nltk.stem import WordNetLemmatizer

#nltk.download('words')
#nltk.download('punkt')
#nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

import json
import pickle
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

#Importamos y cargamos el archivo JSON
randomwords=()
words=[]
classes=[]
documents=[]
ignore_words=['?','!','.',',']

data_file = open('intents.json', encoding='utf-8').read()
intents = json.loads(data_file)
print(intents)

#Preprocesamos los datos
# Creamos los tokens
#Iteramos a traves de los patrones y tokenizacion
# y agragamos a la lista de palabras y a la lista de clases
# nuestras etiquetas

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #agregamos el documento al corpus
        documents.append((w, intent['tag']))

        #agregamos a la lista de clases
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#Ahora lematizaremos cada palabra y eliminaremos palabras duplicadas en la lista
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
print(words)

#Valres de enuestro arreglo de palabras
pickle.dump(words, open('words.pkl', 'wb'))
#Valores de nuestro arreglo de clases
pickle.dump(classes, open('classes.pkl', 'wb'))

#Crear datos de entrenamiento y prueba
training = []

#Crear datos de entrenamiento y prueba
training = []
output_empty = [0] * len(classes)
for doc in documents:
    #Espacio para palabras
    bag = []
    #Lista de tokens
    pattern_words = doc[0]
    #Lematizamos cada palabra
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    #Se crea nuestra matriz de palabras con 1, si se encuentra una coincidencia
    #En el patron actual y en caso de lo contrario se pondra un 0
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
    print(training)

random.shuffle(training)
train_x = [t[0] for t in training]
train_y = [t[1] for t in training]

#Se crea el modelo
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)