import tkinter as tk
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import ISRIStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tkinter import Tk, Label, Text, Button, Menu


w2v_model = Word2Vec.load('Models/word2vec.model')
model = load_model('Models/cnn_model.h5')

stemmer = ISRIStemmer()
tokenizer = nltk.RegexpTokenizer(r"\w+")
with open("Stopwords/list.txt", "r", encoding="utf-8") as f:
    arabic_stopwords = set(f.read().splitlines())

def tokenize_text(text):
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+', ' ', text)
    tokens = tokenizer.tokenize(text)
    tokens = [token for token in tokens if token not in arabic_stopwords]
    tokens = [stemmer.stem(token) for token in tokens]
    tokens = [token.translate(str.maketrans("", "", string.punctuation)) for token in tokens]
    tokens = [token for token in tokens if not token.isdigit()]
    tokens = [token for token in tokens if token]
    return tokens

def predict_category():
    article = text_box.get("1.0", tk.END)
    tokens = tokenize_text(article)
    sequence = [w2v_model.wv.key_to_index[token] if token in w2v_model.wv.key_to_index else 0 for token in tokens]
    sequence_padded = pad_sequences([sequence], maxlen=3843, padding="post")
    prediction = model.predict(sequence_padded)
    category = np.argmax(prediction)
    confidence = np.max(prediction)
    label = ['Politics', 'Entertainment', 'Economy', 'Sports'][category]
    result_label.config(text=f"Predicted Category: {label}\nConfidence Score: {confidence:.2f}")

root = tk.Tk()
root.title("Arabic News Article Classifier")

def copy():
    root.clipboard_clear()
    root.clipboard_append(text_box.selection_get())

def paste():
    text_box.insert(tk.INSERT, root.clipboard_get())

menu = Menu(root, tearoff=0)
menu.add_command(label="Copy", command=copy)
menu.add_command(label="Paste", command=paste)

label = Label(root, text="Enter the text of the article:")
label.pack()

text_box = Text(root, height=10, width=50)
text_box.pack()

def show_menu(event):
    menu.post(event.x_root, event.y_root)

text_box.bind("<Button-3>", show_menu)

button = Button(root, text="Predict", command=predict_category)
button.pack()

result_label = Label(root, text="")
result_label.pack()

root.mainloop()