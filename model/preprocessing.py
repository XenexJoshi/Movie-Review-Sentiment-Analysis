import os
import string
from nltk.corpus import stopwords
from collections import Counter

MIN_COUNT = 2
POS_PATH = './data/txt_sentoken/pos'
NEG_PATH = './data/txt_sentoken/neg'

def load_text(file_path):
  file = open(file_path, 'r')
  data = file.read()
  file.close()
  return data

def preprocess_text(text):
  tokens = text.split()
  translator = str.maketrans('', '', string.punctuation)
  tokens = [s.translate(translator) for s in tokens]
  tokens = [s for s in tokens if s.isalpha()]
  filter = set(stopwords.words('english'))
  tokens = [s for s in tokens if not s in filter]
  tokens = [s for s in tokens if len(s) > 2]
  return tokens

def increase_vocab(file_path, vocab):
  data = load_text(file_path)
  tokens = preprocess_text(data)
  vocab.update(tokens)

def load_data(dir, vocab):
  for filename in os.listdir(dir):
    path = dir + '/' + filename
    increase_vocab(path, vocab)

def save_tokens(lines, filename):
  data = '\n'.join(lines)
  file = open(filename, 'w')
  file.write(data)
  file.close()

if __name__ == '__main__':
  vocab = Counter()
  load_data(POS_PATH, vocab)
  print("Positive data loaded...")
  load_data(NEG_PATH, vocab)
  print("Negative data loaded...")
  tokens = [s for s, t in vocab.items() if t >= MIN_COUNT]
  print("Tokens filtered by occurance...")
  save_tokens(tokens,'model/vocab.txt')
  print('vocab.txt file generated...')