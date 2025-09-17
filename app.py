
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)


model = None
vectorizer = None
model_trained = False

def preprocess_text(text):
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and numbers using regex
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_and_train_model():
    """Load data and train the model if not already trained"""
    global model, vectorizer, model_trained

    # Build paths
    model_dir = 'model'
    dataset_path = os.path.join(model_dir, 'SMSSpamCollection')
    model_path = os.path.join(model_dir, 'spam_model.joblib')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.joblib')

    # Check if model files already exist
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        try:
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            model_trained = True
            print(f"Loaded existing model and vectorizer from {model_dir}/")
            return True
        except:
            print("Error loading existing model, will retrain...")

    # Load dataset and train model
    try:
        # Load dataset from model/ folder
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path, sep='\t', header=None, names=['label', 'text'])
            print(f"Loaded dataset from {model_dir}/ with {len(df)} samples.")
        else:
            # Fallback sample data
            print(f"Dataset not found in {model_dir}/, using sample data...")
            sample_data = {
                'label': ['ham', 'spam', 'ham', 'spam', 'ham'] * 400,
                'text': [
                    'Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...',
                    'Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question',
                    'U dun say so early hor... U c already then say...',
                    'FreeMsg Hey there darling it has been 3 week now and no word back! I would like some fun you up for it still',
                    'Even my brother is not like to speak with me. They treat me like aids patent.'
                ] * 400
            }
            df = pd.DataFrame(sample_data)

        print(f"Training with {len(df)} samples...")

        # Preprocess data
        df['processed_text'] = df['text'].apply(preprocess_text)
        df['label_binary'] = df['label'].map({'ham': 0, 'spam': 1})

        # Split data
        X = df['processed_text']
        y = df['label_binary']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Vectorize
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

        # Train model
        model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        model.fit(X_train_vectorized, y_train)

        # Evaluate
        y_pred = model.predict(X_test_vectorized)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Model trained! Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

        # Save to model/
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        print(f"Saved model/vectorizer to {model_dir}/")

        model_trained = True
        return True

    except Exception as e:
        print(f"Error training model: {str(e)}")
        return False

def predict_spam(text):
    """Predict if text is spam or ham"""
    global model, vectorizer, model_trained
    
    if not model_trained:
        return "Error: Model not trained", 0.0
    
    try:
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Vectorize
        text_vectorized = vectorizer.transform([processed_text])
        
        # Predict
        probabilities = model.predict_proba(text_vectorized)[0]
        prediction = model.predict(text_vectorized)[0]
        
        if prediction == 1:
            result = "Spam"
            confidence = probabilities[1] * 100
        else:
            result = "Ham"
            confidence = probabilities[0] * 100
            
        return result, confidence
        
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Please enter a message'}), 400
        
        if len(message) > 1000:
            return jsonify({'error': 'Message too long (max 1000 characters)'}), 400
        
        # Get prediction
        result, confidence = predict_spam(message)
        
        if result.startswith('Error'):
            return jsonify({'error': result}), 500
        
        return jsonify({
            'prediction': result,
            'confidence': round(confidence, 1),
            'message': message[:100] + ('...' if len(message) > 100 else '')
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/model-status')
def model_status():
    return jsonify({
        'trained': model_trained,
        'model_exists': os.path.exists('spam_model.joblib'),
        'vectorizer_exists': os.path.exists('vectorizer.joblib')
    })

if __name__ == '__main__':
    print("Starting Spam Detector Flask App...")
    print("Loading and training model...")
    
    success = load_and_train_model()
    if success:
        print("Model ready!")
        print("Starting Flask server on http://127.0.0.1:5000")
        app.run(debug=True, host='127.0.0.1', port=5000)
    else:
        print("Failed to load/train model. Please check your setup.")   