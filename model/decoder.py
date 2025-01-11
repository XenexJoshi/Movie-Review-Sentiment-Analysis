import string
from preprocessing import *
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
  encoded_texts = pad_sequences(encoded_tokens, maxlen = MAX_LEN, padding = 'post')
  preds = model.predict([encoded_texts], batch_size = BATCH_SIZE)
  if (preds >= 0.50):
    return 'Negative'
  return 'Positive'
