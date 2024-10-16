# app.py
from flask import Flask, render_template, request
import numpy as np
import pickle
from model import load_model

app = Flask(__name__)

# Load the trained model
model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    size = float(request.form.get('size'))
    bedrooms = int(request.form.get('bedrooms'))
    location = int(request.form.get('location'))

    # Prepare the input for prediction
    features = np.array([[size, bedrooms, location]])

    # Predict using the loaded model
    predicted_price = model.predict(features)[0]

    return render_template('index.html', prediction=f"Predicted House Price: ${predicted_price:.2f}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
