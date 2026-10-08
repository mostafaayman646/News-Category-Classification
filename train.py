from sklearn.neural_network import MLPClassifier
import joblib
import os

def build_model(args):
    """
    Builds a simple Feedforward Neural Network using Scikit-Learn.
    This avoids the TensorFlow 'Illegal instruction' error on unsupported CPUs.
    """
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        batch_size=args.batch_size,
        max_iter=args.epochs,
        early_stopping=True,
        verbose=True,
        random_state=42
    )
    return model

def train_model(X_train, y_train, X_test, y_test, args):
    """
    Initializes and trains the model, returning the trained model.
    """
    print("Building model (MLP Classifier)...")
    model = build_model(args)
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    print(f"Training Accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Validation Accuracy: {model.score(X_test, y_test):.4f}")
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/ag_news_model.pkl')
    print("Model saved to models/ag_news_model.pkl")
    
    return model
