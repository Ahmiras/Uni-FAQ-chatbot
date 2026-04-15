import json
import pickle
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) - {'not', 'no', 'when', 'where', 'what', 'how', 'who', 'which'}

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
    return ' '.join(tokens)

with open("intents.json") as f:
    data = json.load(f)

patterns = []
labels = []
responses = {}

for intent in data["intents"]:
    responses[intent["tag"]] = intent["responses"]
    for pattern in intent["patterns"]:
        patterns.append(preprocess(pattern))
        labels.append(intent["tag"])

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ('svm', SVC(kernel='linear', probability=True, C=1.0))
])

pipeline.fit(patterns, labels)

pickle.dump(pipeline, open("model.pkl", "wb"))
pickle.dump(responses, open("responses.pkl", "wb"))

print("Model trained successfully!")
print(f"Total intents: {len(responses)}")
print(f"Total training samples: {len(patterns)}")