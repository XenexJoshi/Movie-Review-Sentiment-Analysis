from flask import Flask, request, render_template, jsonify
import tensorflow.keras as keras
from joblib import load
import json
import string

MAX_LEN = 971
BATCH_SIZE = 32

def preprocess_text(text, vocab):
  tokens = text.split()
  translator = str.maketrans('', '', string.punctuation)
  tokens = [s.translate(translator) for s in tokens]
  tokens = [s for s in tokens if len(s) > 2]
  tokens = [s for s in tokens if s in vocab]
  return tokens

def predict_text(model, tokenizer, text, vocab):
  tokens = preprocess_text(text, vocab)
  encoded_tokens = tokenizer.texts_to_sequences([tokens])
  encoded_texts = keras.preprocessing.sequence.pad_sequences(encoded_tokens,
                                                            maxlen = MAX_LEN, 
                                                            padding = 'post')
  preds = model.predict([encoded_texts], batch_size = BATCH_SIZE)
  preds = model.predict([encoded_texts], batch_size = BATCH_SIZE)
  if (preds[0][0] >= 0.50):
    acc = round((preds[0][0] - 0.50) * 2 * 100, 2)
    tag = 'Negative'
  else:
    acc = round((0.50 - preds[0][0]) * 2 * 100, 2)
    tag = 'Positive'
  return tag, acc

app = Flask(__name__, static_folder = 'static')

model = load('model.joblib')

with open('../app/tokenizer.json', 'r') as json_file:
  tokenizer_json = json.load(json_file)

tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

@app.route("/", methods=["GET", "POST"])
def main():
  if request.method == 'GET':
    return render_template('index.html')
  else:
    if "text" not in request.form:
      return jsonify({"Error" : "No text file received"}), 400
    
    input_data = request.form["text"]
    with open('../model/vocab.txt', 'r') as file:
      vocab = file.read()
    label, confidence = predict_text(model = model, 
                         tokenizer = tokenizer, 
                         text = input_data, 
                         vocab = vocab)
    return jsonify({'prediction' : str(label), 
                    'confidence' : (str(confidence) + "%")}), 200
