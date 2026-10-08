import joblib
import pandas as pd
from data_preprocess import preprocess_text

def predict(text, model, vectorizer):
    """
    Preprocesses the text, vectorizes it, and predicts the class.
    """
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]
    
    class_names = {
        1: "World",
        2: "Sports",
        3: "Business",
        4: "Sci-Tech"
    }
    return class_names.get(prediction, f"Unknown Class {prediction}")

def main():
    print("Loading model and vectorizer...")
    try:
        model = joblib.load('models/ag_news_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Please ensure models are trained and saved.")
        return

    print("Sampling 5 random rows from the test dataset...")
    # Load test data
    test_df = pd.read_csv('archive/test.csv')
    
    # Sample a few rows
    sample_df = test_df.sample(n=5, random_state=42)
    
    print("\n" + "="*80)
    for index, row in sample_df.iterrows():
        # Safely combine Title and Description
        title = str(row['Title']) if pd.notna(row['Title']) else ""
        desc = str(row['Description']) if pd.notna(row['Description']) else ""
        full_text = (title + " " + desc).strip()
        
        # Get prediction
        predicted_category = predict(full_text, model, vectorizer)
        
        # Print input alongside prediction
        print(f"INPUT TEXT: {full_text}")
        print(f"=> PREDICTION: {predicted_category}")
        print("-" * 80)

if __name__ == "__main__":
    main()
