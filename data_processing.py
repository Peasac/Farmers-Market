import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess data
def load_and_preprocess_data():
    # Load data
    data = pd.read_csv('model/location.csv')

    # Drop unwanted columns
    columns_to_remove = ['modal_price', 'min_price', 'max_price', 'variety', 'arrival_date']
    data.drop(columns=columns_to_remove, inplace=True)

    # Convert columns to lowercase
    data['commodity'] = data['commodity'].astype(str).str.lower()
    data['district'] = data['district'].astype(str).str.lower()
    data['state'] = data['state'].astype(str).str.lower()
    data['market'] = data['market'].astype(str).str.lower()

    # Label encode categorical data
    global commodity_encoder, district_encoder, state_encoder, market_encoder, scaler, model
    commodity_encoder = LabelEncoder()
    district_encoder = LabelEncoder()
    state_encoder = LabelEncoder()
    market_encoder = LabelEncoder()

    data['commodity'] = commodity_encoder.fit_transform(data['commodity'])
    data['district'] = district_encoder.fit_transform(data['district'])
    data['state'] = state_encoder.fit_transform(data['state'])
    data['market'] = market_encoder.fit_transform(data['market'])

    # Split features and target
    X = data[['commodity', 'district', 'state']]
    y = data['market']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

# Load and preprocess data on module import
load_and_preprocess_data()


