# Sentiment Analysis Web App

This project is a Flask-based web application that performs sentiment analysis on Amazon product reviews from two datasets (`product_review1.csv` and `product_review2.csv`). Users can select a product from a dropdown menu and view the predicted sentiment (`positive`, `negative`, or `neutral`) for a single review, displayed in an attractive, modern UI. The app uses a machine learning model trained on preprocessed review data to predict sentiments.

## Features
- **Simple UI**: Select a product and click "Show Sentiment" to see the predicted sentiment for one review.
- **Sentiment Prediction**: Displays only `positive`, `negative`, or `neutral` sentiments, with vibrant styling.
- **Modern Design**: Clean layout, responsive design, and subtle animations for a professional look.
- **Data Processing**: Combines and preprocesses two datasets, handling text cleaning and sentiment normalization.
- **Machine Learning**: Trains and selects the best model (Naive Bayes, SVM, or Logistic Regression) for sentiment prediction.

## Project Structure
```
sentiment_analysis_project/
├── product_review1.csv           # First dataset with product reviews
├── product_review2.csv           # Second dataset with product reviews
├── eda_and_preprocessing.py      # Preprocesses datasets, generates cleaned_reviews.csv and product_names.csv
├── model_building.py             # Trains ML models, saves best_model.pkl and tfidf_vectorizer.pkl
├── app.py                        # Flask app for the web interface
├── templates/
│   └── index.html                # HTML template for the UI
├── static/
│   ├── style.css                 # CSS for styling the UI
│   └── script.js                 # JavaScript for fetching sentiments
├── cleaned_reviews.csv           # Preprocessed review data
├── product_names.csv             # List of product names for the dropdown
├── best_model.pkl                # Trained ML model
├── tfidf_vectorizer.pkl          # TF-IDF vectorizer for text processing
├── sentiment_distribution.png     # Visualization of sentiment distribution
├── text_length_distribution.png   # Visualization of review text lengths
├── wordcloud_positive.png        # Word cloud for positive reviews
├── wordcloud_negative.png        # Word cloud for negative reviews
├── wordcloud_neutral.png         # Word cloud for neutral reviews
├── confusion_matrix.png          # Confusion matrix for the best model
├── README.md                     # This file
```

## Prerequisites
- Python 3.8+
- pip for installing dependencies
- Datasets (`product_review1.csv` and `product_review2.csv`) with `product_name`, `sentiment`, and review text columns (e.g., `reviews.title` or `Summary`)

## Setup Instructions
1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd sentiment_analysis_project
   ```

2. **Install Dependencies**:
   ```bash
   pip install flask pandas scikit-learn nltk emoji matplotlib wordcloud
   ```

3. **Prepare Datasets**:
   - Ensure `product_review1.csv` and `product_review2.csv` are in the project root.
   - Verify they contain:
     - `product_name`: Product identifier
     - `sentiment`: Sentiment labels (e.g., `positive`, `negative`, `neutral`, or numerical ratings like 1–5)
     - Review text (e.g., `reviews.title` in `product_review1.csv`, `Summary` in `product_review2.csv`)
   - If `sentiment` values are invalid (e.g., `Positive`, `5`), fix them:
     ```python
     import pandas as pd
     for file in ['product_review1.csv', 'product_review2.csv']:
         df = pd.read_csv(file)
         if 'sentiment' in df.columns:
             df['sentiment'] = df['sentiment'].apply(
                 lambda x: 'positive' if str(x).lower() in ['positive', 'pos', 'good', '5', '4']
                 else 'negative' if str(x).lower() in ['negative', 'neg', 'bad', '1', '2']
                 else 'neutral' if str(x).lower() in ['neutral', 'neu', 'average', '3']
                 else 'neutral'
             )
             df.to_csv(file, index=False)
     ```

4. **Run Preprocessing**:
   ```bash
   python eda_and_preprocessing.py
   ```
   - Generates `cleaned_reviews.csv`, `product_names.csv`, and visualizations.
   - Check the output for errors (e.g., "Combined number of samples: 0") and fix invalid sentiments if needed.

5. **Train the Model**:
   ```bash
   python model_building.py
   ```
   - Generates `best_model.pkl`, `tfidf_vectorizer.pkl`, and `confusion_matrix.png`.

6. **Run the Flask App**:
   ```bash
   python app.py
   ```
   - Open `http://127.0.0.1:5000` in a browser.
   - Select a product and click "Show Sentiment" to view the predicted sentiment.

## Usage
1. **Access the Web App**:
   - Navigate to `http://127.0.0.1:5000`.
   - The UI features a dropdown menu with product names sorted by positive sentiment ratio.

2. **View Sentiment**:
   - Select a product from the dropdown.
   - Click "Show Sentiment" to display the predicted sentiment (`positive`, `negative`, or `neutral`) for one review.
   - The sentiment is styled with vibrant colors (green for positive, red for negative, yellow for neutral).

3. **Responsive Design**:
   - The UI adapts to mobile and desktop screens, with animations for a smooth experience.

## Troubleshooting
- **Empty Dropdown or No Sentiment**:
  - **Cause**: `cleaned_reviews.csv` or `product_names.csv` is empty due to invalid `sentiment` values in the datasets.
  - **Fix**: Inspect datasets:
    ```python
    import pandas as pd
    for file in ['product_review1.csv', 'product_review2.csv']:
        df = pd.read_csv(file)
        print(f"\nColumns in {file}:")
        print(df.columns.tolist())
        print(f"Unique sentiment values:")
        print(df.get('sentiment', pd.Series()).unique())
    ```
    - Fix invalid sentiments (see "Prepare Datasets" above) and rerun `eda_and_preprocessing.py`.
    - Check outputs:
      ```python
      print(pd.read_csv('cleaned_reviews.csv').head())
      print(pd.read_csv('cleaned_reviews.csv')['sentiment'].value_counts())
      print(pd.read_csv('product_names.csv'))
      ```

- **Invalid Sentiments Displayed**:
  - **Cause**: The model predicts values other than `positive`, `negative`, `neutral`.
  - **Fix**: Verify `cleaned_reviews.csv`:
    ```python
    print(pd.read_csv('cleaned_reviews.csv')['sentiment'].unique())
    ```
    - Retrain the model:
      ```bash
      python model_building.py
      ```

- **UI Issues**:
  - **Cause**: Browser cache or incorrect `style.css`.
  - **Fix**: Clear browser cache or use incognito mode. Ensure `style.css` is updated.

- **Flask Errors**:
  - **Cause**: Missing files (`best_model.pkl`, `tfidf_vectorizer.pkl`, etc.).
  - **Fix**: Rerun `eda_and_preprocessing.py` and `model_building.py`.

## Notes
- The app assumes `product_name` and `sentiment` columns exist in both datasets. If missing, update the datasets or modify `eda_and_preprocessing.py`.
- The UI is designed to be minimalistic, displaying only the sentiment with no review text, as per requirements.
- If the datasets are small or unbalanced, consider adding more data or balancing sentiments for better model performance.

## Future Improvements
- Add a feature to display sentiment distribution for each product.
- Allow users to input custom reviews for sentiment prediction.
- Enhance the UI with additional animations or dark mode.

## License
This project is for educational purposes and not licensed for commercial use.

## Contact
For issues or suggestions, please open an issue on the repository or contact the project maintainer.
