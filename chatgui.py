import nltk, json, random, pickle
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import pickle
import numpy as np
from tensorflow import keras
import tkinter as tk
from tkinter import *
from tkinter import Entry

#Cargamos los archivos ecesarios para nuestro chatbot
model = keras.models.load_model('chatbot_model.h5')
intents = json.loads(open('intents.json', encoding='utf-8').read())
words  =pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

#Funci[on para tokenizar (separar) el texto definido por el usuario
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

#Se crea la bolsa de palabras
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return(np.array(bag))

#Función para calcular la predicción
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    #sort by strength of probability/medir la fuerza de la probabilidad
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

#Función para obtener una respuesta
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result

#Funcion para inicar chatboot
def being(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

#Se crear la interfaz de usuario
def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = being(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

base= Tk()
base.title("Hola")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Crear la ventana de chat
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)

scroll=Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scroll.set

senbutton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5, bd=0, bg="#32de97", fg='#ffffff', command=send)

#Crear la caja de texto
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
scroll.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
senbutton.place(x=6, y=401, height=90)
base.mainloop()