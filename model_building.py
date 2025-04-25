
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")

# Load preprocessed data
try:
    df = pd.read_csv('cleaned_reviews.csv')
except FileNotFoundError:
    print("Error: 'cleaned_reviews.csv' not found.")
    exit()

# Check sentiment label distribution
print("Sentiment label distribution:")
print(df['sentiment'].value_counts(dropna=False))

# Filter valid sentiment labels
valid_sentiments = {'positive', 'negative', 'neutral'}
df = df[df['sentiment'].isin(valid_sentiments)]
print("\nSentiment distribution after filtering:")
print(df['sentiment'].value_counts(dropna=False))

# Handle NaN values
print("\nNaN values in key columns:")
print(df[['processed_text', 'sentiment']].isna().sum())
df = df.dropna(subset=['processed_text', 'sentiment'])
df['processed_text'] = df['processed_text'].astype(str)

# Check dataset size and class diversity
if len(df) < 100:
    print("Error: Dataset is too small (<100 samples). Please use a larger dataset.")
    exit()
if len(df['sentiment'].unique()) < 2:
    print("Error: Dataset contains only one class. Please ensure multiple sentiment classes (positive, negative, neutral).")
    exit()
if any(df['sentiment'].value_counts() < 10):
    print("Warning: Some sentiment classes have too few samples. Consider balancing the dataset.")

# Feature extraction with TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['processed_text'])
y = df['sentiment']

# Data splitting with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check train/test distribution
print("\nTrain set sentiment distribution:")
print(y_train.value_counts())
print("\nTest set sentiment distribution:")
print(y_test.value_counts())

# Initialize models
models = {
    'Naive Bayes': MultinomialNB(),
    'SVM': LinearSVC(),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'accuracy': accuracy,
        'report': classification_report(y_test, y_pred, zero_division=0, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred, labels=['positive', 'negative', 'neutral'])
    }
    print(f"\n{name} Accuracy: {accuracy:.4f}")
    print(f"{name} Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

# Plot confusion matrix for best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
cm = results[best_model_name]['confusion_matrix']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['positive', 'negative', 'neutral'], yticklabels=['positive', 'negative', 'neutral'])
plt.title(f'Confusion Matrix for {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.close()

# Hyperparameter tuning for best model
if best_model_name == 'Logistic Regression':
    param_grid = {'C': [0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
elif best_model_name == 'SVM':
    param_grid = {'C': [0.1, 1, 10]}
    grid = GridSearchCV(LinearSVC(), param_grid, cv=5)
else:
    param_grid = {'alpha': [0.1, 1, 10]}
    grid = GridSearchCV(MultinomialNB(), param_grid, cv=5)

grid.fit(X_train, y_train)
print(f"\nBest {best_model_name} Parameters: {grid.best_params_}")
print(f"Best {best_model_name} Accuracy: {grid.best_score_:.4f}")

# Save best model and vectorizer
with open('best_model.pkl', 'wb') as f:
    pickle.dump(grid.best_estimator_, f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("Best model and vectorizer saved.")
