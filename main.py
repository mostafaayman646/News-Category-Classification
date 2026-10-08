import argparse
from data_preprocess import load_and_preprocess_data
from train import train_model
from plots import plot_frequent_words, plot_training_loss
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_args():
    parser = argparse.ArgumentParser(description="Train News Category Classification Model")
    parser.add_argument('--train_path', type=str, default='archive/train.csv', help='Path to training data')
    parser.add_argument('--test_path', type=str, default='archive/test.csv', help='Path to testing data')
    parser.add_argument('--epochs', type=int, default=10, help='Max iterations for training')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--max_words', type=int, default=5000, help='Maximum number of words for TF-IDF')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load and Preprocess Data
    train_df, test_df = load_and_preprocess_data(args.train_path, args.test_path)
    
    # 2. Visualize frequent words before training
    # Generating word clouds for training data
    plot_frequent_words(train_df, text_col='Processed_Text', label_col='Class Index')
    
    # 3. Vectorize text using TF-IDF
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=args.max_words)
    
    # Fit on training data and transform both training and testing data
    X_train = vectorizer.fit_transform(train_df['Processed_Text'])
    X_test = vectorizer.transform(test_df['Processed_Text'])
    
    y_train = train_df['Class Index'].values
    y_test = test_df['Class Index'].values
    
    print(f"TF-IDF X_train shape: {X_train.shape}")
    print(f"TF-IDF X_test shape: {X_test.shape}")
    
    # 4. Train Model
    model = train_model(X_train, y_train, X_test, y_test, args)
    
    # Save the vectorizer
    import joblib
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    print("Vectorizer saved to models/tfidf_vectorizer.pkl")
    
    # 5. Visualize training loss
    # MLPClassifier exposes the loss curve
    if hasattr(model, 'loss_curve_'):
        plot_training_loss(model.loss_curve_)

if __name__ == "__main__":
    main()
