import pandas as pd
from data_preprocess import load_and_preprocess_data
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def main():
    print("Loading and preprocessing data...")
    # We only need train_df to fit the vectorizer
    train_df, _ = load_and_preprocess_data('archive/train.csv', 'archive/test.csv')
    
    print("Fitting vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000)
    vectorizer.fit(train_df['Processed_Text'])
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    print("Vectorizer saved to models/tfidf_vectorizer.pkl successfully.")

if __name__ == "__main__":
    main()
