import os
from keras.api import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
#nltk.download('all')
from nltk.stem   import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

#nltk.download('punkt')
#nltk.download('wordnet')

words=[]
print("words", words)

classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)
print(data_file)


#to get all tags from json file (load every thing)
for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        print("w:",w)
        #add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
        print("classes:", classes)
            

# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
print("words", words)
# sort classes
print ("classe:",classes)
classes = sorted(list(set(classes)))
print ("classes:",classes)
# documents = combination between patterns and intents
print (len(documents))
print("documents", documents)
# classes = intents
print (len(classes),"classes")
print("classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words")
print("words",words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    print ("bag is",bag)
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    print("pattern:", pattern_words)
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    print("pattern:", pattern_words)
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    print("Output",output_row)
    
    training.append([bag, output_row])
# shuffle our features and turn into np.array
#random.shuffle(training)
training = np.array(training, dtype=object)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
print ("training  x :",train_x)

train_y = list(training[:,1])
print ("training y :",train_y)
print("Training data created")


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
#2 to power 7 equal 128 and feature extractions
# 2 to power 6 equas 64
#try and error
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
#3adad el output
#train y = tags el 3andy zero's we one's 3ala hasb eloutput row
model.add(Dense(len(train_y[0]), activation='softmax'))
model.layers
#model summary
model.summary()
# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model, to get optimal parameter to get a2al error
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
print ("sgd",sgd)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print("categorical_crossentropy",model.compile)
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#fitting and saving the model 
history = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', history)
print("model created")
