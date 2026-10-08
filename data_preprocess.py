import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def preprocess_text(text):
    if pd.isna(text):
        return ""
    
    # Lowercase
    text = str(text).lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove punctuation and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(lemmatized)

def load_and_preprocess_data(train_path, test_path):
    print("Loading data...")
    # Load AG News dataset
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Combine Title and Description
    train_df['Text'] = train_df['Title'].fillna('') + " " + train_df['Description'].fillna('')
    test_df['Text'] = test_df['Title'].fillna('') + " " + test_df['Description'].fillna('')
    
    print("Preprocessing training text data... (this will take some time)")
    train_df['Processed_Text'] = train_df['Text'].apply(preprocess_text)
    
    print("Preprocessing testing text data...")
    test_df['Processed_Text'] = test_df['Text'].apply(preprocess_text)
    
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = load_and_preprocess_data('archive/train.csv', 'archive/test.csv')
    print("Data preprocessing completed successfully!")
