from flask import Flask, render_template, request
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from bs4 import BeautifulSoup
import requests
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')

app = Flask(__name__)
nlp = spacy.load('en_core_web_sm')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    # Get the input URL from the form
    url = request.form['url']

    # Extract text from the URL
    text = extract_text(url)

    # Preprocess the text
    processed_text = preprocess_text(text)

    # Tokenize the sentences
    sentences = sent_tokenize(processed_text)

    # Perform summarization (example: extractive summarization using first 3 sentences)
    summary = ' '.join(sentences[:3])

    return render_template('index.html', summary=summary)

def extract_text(url):
    # Extract text from the URL using BeautifulSoup and requests
    r = requests.get(url, verify=False)
    soup = BeautifulSoup(r.text, 'html.parser')
    text = ' '.join([p.get_text() for p in soup.find_all('p')])

    return text

def preprocess_text(text):
    # Preprocess the text by removing stopwords and stemming
    stop_words_sklearn = list(ENGLISH_STOP_WORDS)
    doc = nlp(text)
    words = [token.lemma_ for token in doc if token.lemma_ not in stop_words_sklearn]

    return ' '.join(words)

if __name__ == '__main__':
    app.run(debug=True)
