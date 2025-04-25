
import pandas as pd
import re
import emoji
import matplotlib.pyplot as plt
from wordcloud import WordCloud
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

# Load datasets
datasets = ['product_review1.csv', 'product_review2.csv']
dfs = []
for dataset in datasets:
    try:
        df = pd.read_csv(dataset)
        print(f"Successfully loaded '{dataset}'.")
        dfs.append((df, dataset))
    except FileNotFoundError:
        print(f"Error: '{dataset}' not found.")
        continue

if not dfs:
    print("Error: No datasets loaded. Exiting.")
    exit()

# Function to normalize sentiment values
def normalize_sentiment(value):
    if not isinstance(value, str):
        try:
            rating = float(value)
            if rating <= 2:
                return 'negative'
            elif rating == 3:
                return 'neutral'
            elif rating >= 4:
                return 'positive'
        except (ValueError, TypeError):
            pass
    value = str(value).lower().strip()
    sentiment_map = {
        'positive': 'positive',
        'pos': 'positive',
        'good': 'positive',
        'great': 'positive',
        'excellent': 'positive',
        'negative': 'negative',
        'neg': 'negative',
        'bad': 'negative',
        'poor': 'negative',
        'neutral': 'neutral',
        'neu': 'neutral',
        'average': 'neutral'
    }
    return sentiment_map.get(value, None)

# Function to standardize columns
def standardize_columns(df, dataset_name):
    # Check for review text column
    text_column = None
    possible_text_columns = ['reviews.text', 'Text', 'review_body', 'reviewText', 'reviews.title', 'Summary', 'review_headline']
    for col in possible_text_columns:
        if col in df.columns:
            text_column = col
            break
    if not text_column:
        print(f"Error: No review text column found in {dataset_name}. Skipping dataset.")
        return None
    df = df.assign(review_text=df[text_column])
    df = df.drop(columns=[text_column])
    print(f"Renamed '{text_column}' to 'review_text' for {dataset_name}.")

    # Check for product_name column
    if 'product_name' not in df.columns:
        print(f"Error: 'product_name' column not found in {dataset_name}. Skipping dataset.")
        return None
    df = df.assign(product_name=df['product_name'].astype(str).replace('', 'Unknown Product').fillna('Unknown Product'))

    # Check for sentiment column
    sentiment_column = None
    possible_sentiment_columns = ['sentiment', 'label', 'polarity', 'division']
    for col in possible_sentiment_columns:
        if col in df.columns:
            sentiment_column = col
            break
    if not sentiment_column:
        print(f"Error: No sentiment column found in {dataset_name}. Skipping dataset.")
        return None
    
    # Normalize sentiment values
    df = df.assign(sentiment=df[sentiment_column].apply(normalize_sentiment))
    print(f"Using '{sentiment_column}' column for sentiment in {dataset_name}.")
    
    # Check for invalid sentiments
    valid_sentiments = {'positive', 'negative', 'neutral'}
    invalid_rows = df['sentiment'].isna() | ~df['sentiment'].isin(valid_sentiments)
    if invalid_rows.any():
        print(f"Warning: Found invalid sentiment values in {dataset_name}.")
        print(f"Invalid values (sample): {df[invalid_rows][sentiment_column].head(10).tolist()}")
        print(f"Total invalid rows: {invalid_rows.sum()}")
        df = df[~invalid_rows]
        if df.empty:
            print(f"Error: All rows filtered out in {dataset_name} due to invalid sentiments.")
            return None
    
    if sentiment_column != 'sentiment':
        df = df.drop(columns=[sentiment_column])

    # Select only required columns
    return df[['product_name', 'review_text', 'sentiment']]

# Standardize and merge datasets
valid_dfs = []
for df, dataset_name in dfs:
    processed_df = standardize_columns(df, dataset_name)
    if processed_df is not None:
        valid_dfs.append(processed_df)

if not valid_dfs:
    print("Error: No valid datasets after processing. Exiting.")
    exit()

df = pd.concat(valid_dfs, ignore_index=True)
print(f"Combined number of samples: {len(df)}")

if df.empty:
    print("Error: Combined dataset is empty after processing. Check sentiment values in datasets.")
    exit()

# Save product names for dropdown
product_counts = df.groupby('product_name').agg(
    review_count=('review_text', 'count'),
    positive_count=('sentiment', lambda x: (x == 'positive').sum())
).reset_index()
product_counts['positive_ratio'] = product_counts['positive_count'] / product_counts['review_count']
product_counts = product_counts.sort_values(['positive_ratio', 'review_count'], ascending=False)
product_counts = product_counts[product_counts['product_name'] != 'Unknown Product']
if product_counts.empty:
    print("Warning: No valid product names found. Using all unique product names.")
    product_counts = df[['product_name']].drop_duplicates().reset_index(drop=True)
product_counts[['product_name']].to_csv('product_names.csv', index=False)
print("Saved product names to 'product_names.csv'.")

# Handle missing values
df['review_text'] = df['review_text'].fillna("no review")
df['sentiment'] = df['sentiment'].fillna("neutral")
df['product_name'] = df['product_name'].fillna("Unknown Product")
df.dropna(subset=['review_text', 'sentiment', 'product_name'], inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Number of samples after handling missing values: {len(df)}")

# Validate sentiment labels
df['sentiment'] = df['sentiment'].str.lower()
valid_sentiments = {'positive', 'negative', 'neutral'}
if not df['sentiment'].isin(valid_sentiments).all():
    print("Warning: Invalid sentiment labels found after normalization.")
    print("Unique sentiment values:", df['sentiment'].unique())
    df = df[df['sentiment'].isin(valid_sentiments)]
    print(f"Number of samples after filtering invalid sentiments: {len(df)}")
else:
    print("Sentiment labels validated successfully.")

# Check sentiment distribution
print("\nSentiment distribution:")
print(df['sentiment'].value_counts())
if len(df) < 100:
    print("Warning: Combined dataset is too small (<100 samples). Consider using larger datasets.")
elif any(df['sentiment'].value_counts() < 10):
    print("Warning: Some sentiment classes have too few samples. Consider balancing the dataset.")

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

# Apply cleaning and preprocessing
df['clean_text'] = df['review_text'].apply(clean_text)
df['processed_text'] = df['clean_text'].apply(preprocess_text)

# Check for NaN values
print("\nNaN values after preprocessing:")
print(df[['clean_text', 'processed_text']].isna().sum())
df['clean_text'] = df['clean_text'].fillna("")
df['processed_text'] = df['processed_text'].fillna("")

# Sentiment distribution plot
sentiment_counts = df['sentiment'].value_counts()
if not sentiment_counts.empty:
    plt.figure(figsize=(8, 4))
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.savefig('sentiment_distribution.png')
    plt.close()
else:
    print("Skipping sentiment distribution plot: No data available.")

# Text length analysis
df['text_length'] = df['processed_text'].apply(lambda x: len(x.split()))
print("\nText Length Statistics:")
print(df['text_length'].describe())
if not df.empty:
    plt.figure(figsize=(8, 4))
    plt.hist(df['text_length'], bins=30, color='purple')
    plt.title('Distribution of Text Length (Words per Review)')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.savefig('text_length_distribution.png')
    plt.close()
else:
    print("Skipping text length plot: No data available.")

# Vocabulary size
all_words = ' '.join(df['processed_text']).split()
vocab_size = len(set(all_words))
print(f"Vocabulary Size: {vocab_size}")

# Word clouds by sentiment
def generate_wordcloud(sentiment):
    text = ' '.join(df[df['sentiment'] == sentiment]['processed_text'])
    if text.strip():
        wordcloud = WordCloud(width=800, height=400, background_color='white', min_word_length=3).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {sentiment} Reviews')
        plt.savefig(f'wordcloud_{sentiment}.png')
        plt.close()
    else:
        print(f"No text available for {sentiment} word cloud.")

for sentiment in ['positive', 'negative', 'neutral']:
    generate_wordcloud(sentiment)

# Save preprocessed data
df.to_csv('cleaned_reviews.csv', index=False)
print("Preprocessed data saved to 'cleaned_reviews.csv'.")
