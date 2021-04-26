import json 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import os
import requests
from flask import Flask

dir = os.chdir(r"C:\Users\EI11609\Desktop\New folder")
with open('intents.json') as file:
    data = json.load(file)
    
training_sentences = []
training_labels = []
labels = []
responses = []


for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])
    
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
        
num_classes = len(labels)
#print(len(labels))
#print(labels)
#print(training_sentences)
#print(training_labels)
#print(len(training_labels))
#print(len(training_sentences))

lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)

vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])

model.summary()
epochs = 500
history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

# to save the trained model
model.save("chat_model")

app = Flask(__name__)
@app.route('/chatbot/', methods=['GET', 'POST'])
def chatbot():
    return "chat_model"
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)


class zomato:

    def __init__(self, key = r'a60df80ddb6a197f5e37a95238aa3432', base_url= 'https://developers.zomato.com/api/v2.1/'):
        self.key = key
        self.base_url = base_url

    def json_parse(self):
        json_data = json.loads(self.response)

        return json_data

    
    def getCategories(self):
        URL = self.base_url + "categories"
        header = {"User-agent": "curl/7.43.0", "Accept": "application/json", "user_key": self.key}

        response = requests.get(URL, headers=header)
        #pprint(response.json())
        return response.json()  

    def getCuisines(self, city_id, lat, lon):
        URL = [self.base_url, "cuisines?", "city_id=", str(city_id), "&lat=", str(lat), "&lon=", str(lon)]
        URL = "".join(URL)
        header = {"User-agent": "curl/7.43.0", "Accept": "application/json", "user_key": self.key}

        response = requests.get(URL, headers=header)
        #pprint(response.json())
        return response.json()
    

    def getLocation_Details(self, entity_id, entity_type):
        URL = [self.base_url, "location_details?", "entity_id=", str(entity_id), "&entity_type=", entity_type]
        URL = "".join(URL)
        header = {"User-agent": "curl/7.43.0", "Accept": "application/json", "user_key": self.key}

        response = requests.get(URL, headers=header)
        #pprint(response.json())
        return response.json()

zomato()

import pickle

# to save the fitted tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# to save the fitted label encoder
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
