from functools import partial
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import nltk
from nltk.stem import WordNetLemmatizer
import hyperlinkmanager
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
import smtplib
from email.mime.text import MIMEText
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    print(msg)
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


#Creating GUI with tkinter
import tkinter
from tkinter import *
import webbrowser
def callback(url):
    print (url)
    webbrowser.open_new(url)
def qs1(msg):
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
        ques = msg.split(",")
        print(ques)
        for x in ques:
            res = chatbot_response(x)
            if ("http" in res):
                resstr = str(res)
                hyperlink1 = hyperlinkmanager.HyperlinkManager(ChatLog)
                ChatLog.insert(END, "Bot: " )
                ChatLog.insert(END,"click here " + '\n\n',hyperlink1.add(partial(webbrowser.open,resstr)))
            else: 
                ChatLog.insert(END, "Bot: " + res + '\n\n')
            
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
        ques = msg.split(",")
        print(ques)
        for x in ques:
          res = chatbot_response(x)
          if ("http" in res):
            resstr = str(res)
            hyperlink1 = hyperlinkmanager.HyperlinkManager(ChatLog)
            ChatLog.insert(END, "Bot: " )
            ChatLog.insert(END,"click here " + '\n\n',hyperlink1.add(partial(webbrowser.open,resstr)))
          else:
            ChatLog.insert(END, "Bot: " + res + '\n\n')
            
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
 

base = Tk()
base.title("MIU CHATBOT")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#079CFF", activebackground="#0D87D8",fg='#FFFFFF',
                    command= send )
#question1 = Button(base, font=("Verdana",7), text="computer science", width=5, height=2,
# # bd=0, bg="#5C6266", activebackground="#0D87D8",fg='#FFFFFF',borderwidth = 0,
# command=lambda m="How much fees for computer science?": qs1(m))
#question2 = Button(base, font=("Verdana",7), text="How much fees for computer science?", width=5, height=2,
# bd=0, bg="#5C6266", activebackground="#0D87D8",fg='#FFFFFF',borderwidth = 0,
# command= qs1 )
#question3 = Button(base, font=("Verdana",7), text="Other", width=5, height=2,
#bd=0, bg="#5C6266", activebackground="#0D87D8",fg='#FFFFFF',borderwidth = 0,
#command= email)
##question4 = Button(base, font=("Verdana",7), text="How much fees for computer science?", width=5, height=2,
#bd=0, bg="#5C6266", activebackground="#0D87D8",fg='#FFFFFF',borderwidth = 0,
# command= qs1 )
#question5 = Button(base, font=("Verdana",7), text="Other", width=5, height=2,
# bd=0, bg="#5C6266", activebackground="#0D87D8",fg='#FFFFFF',borderwidth = 0,
# #command= qs1 )


#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
#question1.place(x = 6,y=350,height =25,width = 190)
##question2.place(x =207,y=376,height =25,width = 190)
#question3.place(x = 6,y=376,height =25,width = 190)


ChatLog.place(x=6,y=6, height=386, width=370)

EntryBox.place(x=6, y=401, height=90, width=265)
SendButton.place(x=271, y=401, height=90)

base.mainloop()
