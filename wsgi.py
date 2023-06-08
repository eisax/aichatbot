from flask import Flask, redirect, url_for, request,jsonify,json
from flask_cors import CORS
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
import wikipedia
import datetime
import time
app = Flask(__name__)
CORS(app)



def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# MESSAGE FROM THE USER

def getResponse(ints, intents_json,msg):
    list_of_intents = intents_json['intents']
    
    if (float(ints[0]['probability']) > 0.95):
        tag = ints[0]['intent']
        result = ""
        if (tag=="datetime"):
            result = time.strftime("%A") + time.strftime("%d %B %Y")+time.strftime("%H:%M:%S")
        else:
            for i in list_of_intents:
                if(i['tag']== tag):
                    result = random.choice(i['responses'])
                    break
    else:
        tag = "noanswer"
        try:
            content = wikipedia.page(msg).content
            try:
                first_paragraph = content.split("\n")[0]
                result = first_paragraph
            except AttributeError:
                first_paragraph = ''.join(content).split("\n")[0]
                result = first_paragraph
            
        except wikipedia.exceptions.DisambiguationError as err:
            try:
                content = wikipedia.search(msg, results=1, suggestion=False)
                try:
                    first_paragraph = content.split("\n")[0]
                    result = first_paragraph
                except AttributeError:
                    first_paragraph = ''.join(content).split("\n")[0]
                    result = first_paragraph
            except wikipedia.exceptions.PageError as eb:
                eb_options_list = ""
                for option in eb.options:
                    eb_options_list = option + "\n"
                    result = eb.title + " may refer to: \n" + eb_options_list
        except wikipedia.exceptions.PageError as e:
            result = "Sorry, I could not find any information on that topic."
            for i in list_of_intents:
                if(i['tag']== tag):
                    result = random.choice(i['responses'])
                    break        
    return result

def chatbot_response(msg):
    try:
        ints = predict_class(msg, model)
        res = getResponse(ints, intents, msg)
        return res
    except IndexError:
        return "I'm sorry, I didn't understand what you said. Please try again with a different message."

@app.route('/talktome',methods=["POST","GET"])
def talktome():
    # message = request.form['message']
    data = request.get_json()
    
    message = chatbot_response(data['message'])
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    response_dict = {
        "type": "bot",
        "time": current_time,
        "message": message
    }
    
    return jsonify(response_dict)

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')