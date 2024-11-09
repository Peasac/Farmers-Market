import streamlit as st
import joblib
import numpy as np
import base64

# Load the saved model, scaler, and encoders
model = joblib.load('model/saved_models/model.pkl')
scaler = joblib.load('model/saved_models/scaler.pkl')
commodity_encoder = joblib.load('model/saved_models/commodity_encoder.pkl')
district_encoder = joblib.load('model/saved_models/district_encoder.pkl')
state_encoder = joblib.load('model/saved_models/state_encoder.pkl')
market_encoder = joblib.load('model/saved_models/market_encoder.pkl')

# Function to encode image to Base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Get base64 of the background image
img_file = 'nightcrop.jpg'  # Ensure this path is correct
base64_img = get_base64_of_bin_file(img_file)

# Custom CSS for background image, footer, button styles, and output text
st.markdown(
    f"""
    <style>
    /* Apply background image */
    .stApp {{
        background: url("data:image/jpg;base64,{base64_img}") no-repeat center center fixed;
        background-size: cover;
        color: white;
        padding-bottom: 120px;  /* Adds space above footer */
    }}
    /* Ensure all text is white in both light and dark modes */
    .stText, .stTitle, .stHeader, .stSubheader {{
        color: white !important;
    }}
    /* Ensure all text labels for inputs are clearly visible */
    label {{
        color: #000000 !important;  /* Dark color for readability */
        font-weight: bold;
        background-color: rgba(255, 255, 255, 0.7);  /* Semi-transparent background */
        padding: 5px;
        border-radius: 5px;
    }}
    /* Make the input fields clearly visible */
    .stNumberInput, .stTextInput {{
        background-color: rgba(255, 255, 255, 0.8);  /* Semi-transparent background for inputs */
        color: black !important;  /* Black text inside the input fields */
        border-radius: 5px;
    }}
    /* Ensure the buttons are visible */
    .stButton button {{
        color: black !important;  /* Black text for buttons */
        background-color: #F0F0F0 !important;  /* Light background for buttons */
        border-radius: 5px;
        font-weight: bold;
    }}
    /* Style the output text to make it white and visible */
    .stSuccess, .stError, .stWarning, .stAlert {{
        color: white !important;  /* White color for output text */
        font-weight: bold !important;
        background-color: rgba(0, 0, 0, 0.5) !important; /* Darken the background */
        border-radius: 5px; /* Rounded edges for better UI */
    }}
    .footer {{
        width: 100%;
        background-color: #1F2937;
        color: white;
        text-align: center;
        padding: 10px;
        position: relative;
        margin-top: 20px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Prediction function
def predict_market(commodity, district, state):
    # Encode inputs
    commodity_encoded = commodity_encoder.transform([commodity.lower()])[0]
    district_encoded = district_encoder.transform([district.lower()])[0]
    state_encoded = state_encoder.transform([state.lower()])[0]
    
    # Prepare input data
    input_data = np.array([[commodity_encoded, district_encoded, state_encoded]])
    input_data_scaled = scaler.transform(input_data)
    
    # Predict the market and decode it back
    predicted_market = model.predict(input_data_scaled)
    return market_encoder.inverse_transform(predicted_market)[0]

# Streamlit UI
st.title("Market Prediction App")
st.write("Enter details to find the best market for your commodity.")

# User input fields
commodity = st.selectbox("Commodity", commodity_encoder.classes_)
district = st.selectbox("District", district_encoder.classes_)
state = st.selectbox("State", state_encoder.classes_)

# Predict button
if st.button("Predict Market"):
    try:
        market = predict_market(commodity, district, state)
        st.success(f"The recommended market for {commodity} is: {market}")
    except ValueError as e:
        st.error(f"Error: {e}")

# Footer section
st.markdown(
    """
    <div class="footer">
        Copyright © SmartSow2024
    </div>
    """,
    unsafe_allow_html=True
)
