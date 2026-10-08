import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

def plot_frequent_words(df, text_col='Processed_Text', label_col='Class Index'):
    """
    Generates and saves a word cloud for each category.
    """
    print("Generating word clouds...")
    os.makedirs('plots', exist_ok=True)
    
    classes = df[label_col].unique()
    
    # Class names for AG News
    class_names = {
        1: "World",
        2: "Sports",
        3: "Business",
        4: "Sci-Tech"
    }
    
    for cls in classes:
        # Get all text for the current class
        class_text = " ".join(df[df[label_col] == cls][text_col].dropna().values)
        
        if not class_text.strip():
            continue
            
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(class_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        title = class_names.get(cls, f"Class {cls}")
        plt.title(f'Most Frequent Words - {title}')
        
        save_path = f'plots/wordcloud_class_{cls}.png'
        plt.savefig(save_path)
        plt.close()
        print(f"Saved word cloud for {title} to {save_path}")

def plot_training_loss(loss_curve):
    """
    Plots and saves the training loss from sklearn MLPClassifier.
    """
    print("Generating training plots...")
    os.makedirs('plots', exist_ok=True)
    
    # Plot Loss
    plt.figure(figsize=(8, 6))
    plt.plot(loss_curve, label='Training Loss')
    plt.title('Model Loss over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plots/training_loss.png')
    plt.close()
    
    print("Saved training loss plot to plots/")
