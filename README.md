# UniBot - University FAQ Chatbot

A simple ML-powered chatbot that answers university-related questions (admissions, fees, courses, scholarships, etc.). Built with Streamlit and scikit-learn.

## Files

- `app.py` – Streamlit web app
- `train.py` – Train the SVM model
- `intents.json` – Questions and answers
- `model.pkl` / `responses.pkl` – Generated after training

## How to Run

1. Install dependencies:
pip install streamlit nltk scikit-learn numpy

2. Train the model:
python train.py

3. Run the app:
streamlit run app.py


## Customize

Edit `intents.json` to add or change questions/answers, then run `train.py` again.

## Requirements

- Python 3.7+
- Libraries: streamlit, nltk, scikit-learn, numpy
