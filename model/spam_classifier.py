#!/usr/bin/env python3

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_data():
    print("Loading SMS Spam Collection dataset...")
    try:
        df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'text'])
        print(f"Dataset loaded successfully!")
        print(f"Total samples: {len(df)}")
        print(f"Ham samples: {len(df[df['label'] == 'ham'])}")
        print(f"Spam samples: {len(df[df['label'] == 'spam'])}")
        print(f"Class distribution: {df['label'].value_counts().to_dict()}")
        print("\nSample data:")
        print(df.head(10))
        print(f"\nMissing values: {df.isnull().sum().sum()}")
        return df
    except FileNotFoundError:
        print("Error: SMSSpamCollection file not found!")
        print("\nPlease download the dataset from:")
        print("https://archive.ics.uci.edu/dataset/228/sms+spam+collection")
        print("\nInstructions:")
        print("1. Download the 'sms+spam+collection.zip' file")
        print("2. Extract it and place 'SMSSpamCollection' file in the same directory as this script")
        print("3. The file should be tab-separated with format: [label]\\t[message]")
        print("\nCreating sample dataset for demonstration...")
        sample_data = {
            'label': ['ham', 'spam', 'ham', 'spam', 'ham'],
            'text': [
                'Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...',
                'Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&Cs apply 08452810075over18\'s',
                'U dun say so early hor... U c already then say...',
                'FreeMsg Hey there darling it\'s been 3 week\'s now and no word back! I\'d like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv',
                'Even my brother is not like to speak with me. They treat me like aids patent.'
            ]
        }
        df = pd.DataFrame(sample_data)
        print("Using sample data for demonstration.")
        return df
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    df = load_data()
    print("\n" + "="*50)
    print("STEP 2: Preprocessing data...")
    print("="*50)
    df['processed_text'] = df['text'].apply(preprocess_text)
    df['label_binary'] = df['label'].map({'ham': 0, 'spam': 1})
    print("Text preprocessing completed!")
    print("\nSample preprocessed text:")
    for i in range(3):
        print(f"Original: {df['text'].iloc[i]}")
        print(f"Processed: {df['processed_text'].iloc[i]}")
        print(f"Label: {df['label'].iloc[i]} -> {df['label_binary'].iloc[i]}")
        print("-" * 30)
    X = df['processed_text']
    y = df['label_binary']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nData split completed:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print("\n" + "="*50)
    print("STEP 3: Vectorizing text data...")
    print("="*50)
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    print(f"Vectorization completed!")
    print(f"Feature matrix shape: {X_train_vectorized.shape}")
    print(f"Number of features: {X_train_vectorized.shape[1]}")
    print("\n" + "="*50)
    print("STEP 4: Training Logistic Regression model...")
    print("="*50)
    model = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    )
    model.fit(X_train_vectorized, y_train)
    print("Model training completed!")
    y_pred = model.predict(X_test_vectorized)
    y_pred_proba = model.predict_proba(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1-Score: {f1:.4f}")
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    print("\n" + "="*50)
    print("STEP 6: Saving model and vectorizer...")
    print("="*50)
    joblib.dump(model, 'spam_model.joblib')
    print("Model saved as 'spam_model.joblib'")
    joblib.dump(vectorizer, 'vectorizer.joblib')
    print("Vectorizer saved as 'vectorizer.joblib'")
    print("\nTraining pipeline completed successfully!")
    print("\n" + "="*50)
    print("TESTING PREDICTION FUNCTION")
    print("="*50)
    test_texts = [
        "Go until jurong point, crazy.. Available only in bugis n great world la e buffet...",
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121",
        "U dun say so early hor... U c already then say...",
        "FreeMsg Hey darling it's been 3 weeks now! Text me back! £1.50 to receive",
        "Sorry, I'll call you later in meeting",
        "URGENT! You have won a £1000 cash prize! Call 09061701461",
        "Can you pick me up at the airport tomorrow at 3pm?",
        "Congratulations! You've won a FREE holiday! Text WIN to 12345 now!"
    ]
    for text in test_texts:
        result, confidence = predict_spam(text)
        print(f"Text: '{text}'")
        print(f"Prediction: {result} (Confidence: {confidence:.1f}%)")
        print("-" * 50)

def predict_spam(text):
    try:
        model = joblib.load('spam_model.joblib')
        vectorizer = joblib.load('vectorizer.joblib')
        processed_text = preprocess_text(text)
        text_vectorized = vectorizer.transform([processed_text])
        probabilities = model.predict_proba(text_vectorized)[0]
        prediction = model.predict(text_vectorized)[0]
        if prediction == 1:
            result = "Spam"
            confidence = probabilities[1] * 100
        else:
            result = "Ham"
            confidence = probabilities[0] * 100
        return result, confidence
    except FileNotFoundError:
        print("Error: Model files not found. Please run the training script first.")
        return "Error", 0.0
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return "Error", 0.0

if __name__ == "__main__":
    print("="*60)
    print("SPAM CLASSIFIER TRAINING SCRIPT")
    print("="*60)
    main()
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print("\nYou can now use the predict_spam() function to classify new texts.")
    print("Example usage:")
    print("result, confidence = predict_spam('Your text here')")
    print(f"print(f'Prediction: {{result}} ({{confidence:.1f}}% confidence)')")
