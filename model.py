# model.py
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# Define and train a more robust model with expanded data
def train_model():
    # Expanded dataset (Size in sq ft, Number of bedrooms, Location index)
    X = np.array([
        [1500, 3, 1], [2000, 4, 2], [2500, 4, 3], [3000, 5, 3], [3500, 5, 4],
        [1200, 2, 1], [1800, 3, 2], [2400, 4, 3], [3200, 5, 4], [4000, 6, 5],
        [1300, 2, 2], [2100, 3, 3], [2800, 4, 4], [3600, 5, 5], [4200, 6, 5],
        [1100, 2, 1], [1900, 3, 2], [2600, 4, 3], [3400, 5, 4], [3700, 5, 4]
    ])
    
    y = np.array([
        300000, 400000, 500000, 600000, 700000,
        220000, 340000, 480000, 620000, 780000,
        250000, 390000, 520000, 670000, 820000,
        200000, 350000, 470000, 630000, 700000
    ])  # Prices

    model = LinearRegression()
    model.fit(X, y)

    # Save the model to disk
    with open('house_price_model.pkl', 'wb') as f:
        pickle.dump(model, f)

def load_model():
    # Load the model from disk
    with open('house_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model
