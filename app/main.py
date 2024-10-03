import streamlit as st
import pickle
import tensorflow as tf
import xgboost as xgb
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime, timedelta

# Set up Flask app for API
app = Flask(__name__)

# Load models
@st.cache_resource
def load_lstm_model():
    model = tf.keras.models.load_model('models/LSTM_model.h5')
    return model

@st.cache_resource
def load_xgboost_model():
    with open('models/xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

lstm_model = load_lstm_model()
xgboost_model = load_xgboost_model()

# Function to predict using LSTM
def predict_sales_lstm(data):
    return lstm_model.predict(data)

# Function to predict using XGBoost
def predict_sales_xgboost(data):
    dmatrix = xgb.DMatrix(data)
    return xgboost_model.predict(dmatrix)

# --- Streamlit App ---
st.title('Sales Prediction API')

# Select model for prediction
model_type = st.selectbox('Choose Model', ['LSTM', 'XGBoost'])

# Input features for prediction
store_id = st.number_input('Store ID', min_value=1)
item_id = st.number_input('Item ID', min_value=1)
other_features = st.text_area('Other Features (comma separated)', '')

if st.button('Predict'):
    # Prepare input data
    input_data = np.array([store_id, item_id] + list(map(float, other_features.split(',')))).reshape(1, -1)
    
    if model_type == 'LSTM':
        prediction = predict_sales_lstm(input_data)
    else:
        prediction = predict_sales_xgboost(input_data)
    
    st.write(f"Predicted Sales: {prediction}")

# --- Flask API Endpoints ---
# Project description
@app.route('/')
def home():
    project_description = {
        "project": "Sales Prediction API",
        "description": "This API provides sales forecasts for both national and store/item levels.",
        "endpoints": {
            "/": "Display project description and list of endpoints",
            "/health/": "API health check",
            "/sales/national/": {
                "GET": {
                    "description": "Predict next 7 days' national sales",
                    "parameters": {
                        "date": "Starting date (YYYY-MM-DD)"
                    },
                    "output": {
                        "2016-01-01": 10000.01,
                        "2016-01-02": 10001.12
                    }
                }
            },
            "/sales/stores/items/": {
                "GET": {
                    "description": "Predict sales for specific store and item on a given date",
                    "parameters": {
                        "date": "Date (YYYY-MM-DD)",
                        "store_id": "Store identifier",
                        "item_id": "Item identifier"
                    },
                    "output": {
                        "prediction": 19.72
                    }
                }
            }
        },
        "github_repo": "https://github.com/yourrepo"
    }
    return jsonify(project_description)

# Health check
@app.route('/health/')
def health():
    return "API is running smoothly!", 200

# National sales forecast
@app.route('/sales/national/', methods=['GET'])
def predict_national_sales():
    date_str = request.args.get('date')
    
    try:
        start_date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
    
    # Mock input data for LSTM (replace with actual preprocessing logic)
    input_data = np.random.rand(1, 10)  # example input
    
    # Get 7-day forecast from LSTM model
    predictions = lstm_model.predict(input_data)[0]  # assuming this gives 7-day predictions
    
    forecast = {}
    for i in range(7):
        day = start_date + timedelta(days=i+1)
        forecast[day.strftime('%Y-%m-%d')] = round(predictions[i], 2)
    
    return jsonify(forecast)

# Store-item sales prediction
@app.route('/sales/stores/items/', methods=['GET'])
def predict_store_item_sales():
    date_str = request.args.get('date')
    store_id = request.args.get('store_id')
    item_id = request.args.get('item_id')
    
    try:
        _ = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
    
    if not store_id or not item_id:
        return jsonify({"error": "Missing store_id or item_id"}), 400
    
    # Mock input data for XGBoost (replace with actual preprocessing logic)
    input_data = np.array([[store_id, item_id, np.random.rand()]]).reshape(1, -1)
    dmatrix = xgb.DMatrix(input_data)
    
    # Predict sales for the given store and item
    prediction = xgboost_model.predict(dmatrix)[0]
    
    return jsonify({"prediction": round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)
