import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Ensure the directory for saving models exists
os.makedirs("model/saved_models", exist_ok=True)

# Load and preprocess data
data = pd.read_csv('location.csv')

# Drop unwanted columns
columns_to_remove = ['modal_price', 'min_price', 'max_price', 'variety', 'arrival_date']
data.drop(columns=columns_to_remove, inplace=True)

# Convert columns to lowercase for consistency
data['commodity'] = data['commodity'].astype(str).str.lower()
data['district'] = data['district'].astype(str).str.lower()
data['state'] = data['state'].astype(str).str.lower()
data['market'] = data['market'].astype(str).str.lower()

# Initialize label encoders
commodity_encoder = LabelEncoder()
district_encoder = LabelEncoder()
state_encoder = LabelEncoder()
market_encoder = LabelEncoder()

# Fit and transform encoders
data['commodity'] = commodity_encoder.fit_transform(data['commodity'])
data['district'] = district_encoder.fit_transform(data['district'])
data['state'] = state_encoder.fit_transform(data['state'])
data['market'] = market_encoder.fit_transform(data['market'])

# Split data into features and target
X = data[['commodity', 'district', 'state']]
y = data['market']

# Initialize and fit scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_scaled, y)

# Save the trained model, scaler, and encoders
joblib.dump(model, 'model/saved_models/model.pkl', compress=('gzip', 3))
joblib.dump(scaler, 'model/saved_models/scaler.pkl', compress=('gzip', 3))
joblib.dump(commodity_encoder, 'model/saved_models/commodity_encoder.pkl', compress=('gzip', 3))
joblib.dump(district_encoder, 'model/saved_models/district_encoder.pkl', compress=('gzip', 3))
joblib.dump(state_encoder, 'model/saved_models/state_encoder.pkl', compress=('gzip', 3))
joblib.dump(market_encoder, 'model/saved_models/market_encoder.pkl', compress=('gzip', 3))

print("Model, scaler, and encoders saved successfully!")


