from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import re
import emoji
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
try:
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    print("Error: Model or vectorizer file not found. Run model_building.py.")
    exit()

# Load product names and reviews
try:
    product_names = pd.read_csv('product_names.csv')['product_name'].tolist()
    reviews_df = pd.read_csv('cleaned_reviews.csv')
    # Filter reviews to only include valid sentiments
    valid_sentiments = {'positive', 'negative', 'neutral'}
    reviews_df = reviews_df[reviews_df['sentiment'].str.lower().isin(valid_sentiments)]
except FileNotFoundError:
    print("Error: 'product_names.csv' or 'cleaned_reviews.csv' not found. Run eda_and_preprocessing.py.")
    exit()

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"[^a-zA-Z\s']", '', text)
    text = text.lower()
    return text

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens) if tokens else "no review"

# Route for home page
@app.route('/')
def index():
    return render_template('index.html', products=product_names)

# Route for fetching predicted sentiment
@app.route('/get_sentiment', methods=['POST'])
def get_sentiment():
    product_name = request.form.get('product_name')
    if not product_name or product_name not in reviews_df['product_name'].values:
        return jsonify({'error': 'Product not found', 'sentiment': None})
    
    # Get the first review for the selected product
    product_review = reviews_df[reviews_df['product_name'] == product_name][['review_text']].head(1)
    if product_review.empty:
        return jsonify({'error': 'No reviews available for this product', 'sentiment': None})
    
    valid_sentiments = {'positive', 'negative', 'neutral'}
    review_text = product_review.iloc[0]['review_text']
    cleaned = clean_text(review_text)
    processed = preprocess_text(cleaned)
    vector = vectorizer.transform([processed])
    predicted_sentiment = model.predict(vector)[0]
    
    # Only return valid sentiments
    if predicted_sentiment in valid_sentiments:
        return jsonify({'sentiment': predicted_sentiment})
    else:
        return jsonify({'error': 'Invalid sentiment predicted', 'sentiment': None})

if __name__ == '__main__':
    app.run(debug=True)