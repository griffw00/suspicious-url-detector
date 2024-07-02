from flask import Flask, render_template, request, flash
import joblib
import pandas as pd
from urllib.parse import urlparse

app = Flask(__name__)

# Load your machine learning model
model = joblib.load('model/bootstrap_model.pkl')  # Update with your actual model path

# Function to extract features from a URL
def extract_url_features(url):
    parsed_url = urlparse(url)
    features = {
        'url_length': len(url),
        'domain_length': len(parsed_url.hostname) if parsed_url.hostname else 0,
        'path_length': len(parsed_url.path),
        'num_dots': url.count('.'),
        'num_special_chars': sum([url.count(char) for char in '/-_?=']),
        'is_ip_address': int(parsed_url.hostname.replace('.', '').isdigit()) if parsed_url.hostname else 0,
        'num_subdomains': len(parsed_url.hostname.split('.')) - 2 if parsed_url.hostname else 0,
    }
    return pd.Series(features)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get URL input from the form
    url = request.form['url']

    # Check if URL is empty
    if not url.strip():  
        error_message = "Please enter a URL."
        return render_template('index.html', error_message=error_message)

    # Extract features from the URL
    features = extract_url_features(url)

    # Prepare the features for prediction
    feature_vector = features.values.reshape(1, -1)  # Reshape for single sample prediction

    # Make prediction using your model
    prediction = model.predict_proba(feature_vector)[0][1] * 100  # Assuming binary classification

    # Round prediction to two decimal places
    prediction = round(prediction, 2)

    # Pass URL and prediction to the template
    return render_template('index.html', url=url, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
