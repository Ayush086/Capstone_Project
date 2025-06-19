
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Property Price Predictor", 
    page_icon="💰",
    layout="wide"
)

# Custom CSS for cleaner styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .form-container {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .prediction-container {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        margin: 1.5rem 0;
        text-align: center;
    }
    .prediction-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #0D47A1;
    }
    .prediction-range {
        font-size: 1.2rem;
        color: #616161;
    }
</style>
""", unsafe_allow_html=True)

# Page title
st.markdown("<h1 class='main-header'>Property Price Predictor</h1>", unsafe_allow_html=True)

# Load data and model
@st.cache_resource
def load_data_and_model():
    try:
        with open('D:\Projects\capstone-project\models\df.pkl', 'rb') as file:
            df = pickle.load(file)
        
        with open('D:\Projects\capstone-project\models\pipeline.pkl', 'rb') as file:
            pipeline = pickle.load(file)
        
        return df, pipeline
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        return None, None

df, pipeline = load_data_and_model()

if df is None or pipeline is None:
    st.error("Failed to load necessary files. Please check the file paths and try again.")
    st.stop()

# Create two-column layout
col1, col2 = st.columns([2, 1])

with col1:
    # Form for property details
    st.subheader("Enter Property Details")
    
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    
    # Basic property details
    property_type = st.selectbox('Property Type', options=df['property_type'].unique())
    sector = st.selectbox('Sector', options=sorted(df['sector'].unique().tolist()))
    
    # Room configuration
    col_bed, col_bath = st.columns(2)
    with col_bed:
        bedroom = float(st.selectbox('Bedrooms', options=sorted(df['bedRoom'].unique().tolist())))
    with col_bath:
        bathroom = float(st.selectbox('Bathrooms', options=sorted(df['bathroom'].unique().tolist())))
    
    # Size and features
    builtup_area = st.number_input('Built-up Area (sq.ft)', min_value=100, max_value=10000, value=1500)
    
    col_balcony, col_age = st.columns(2)
    with col_balcony:
        balcony = st.selectbox('Balconies', options=sorted(df['balcony'].unique().tolist()))
    with col_age:
        age_possession = st.selectbox('Age of Possession', options=df['agePossession'].unique().tolist())
    
    # Additional rooms
    col_servant, col_store = st.columns(2)
    with col_servant:
        servant_room = float(st.selectbox('Servant Room', options=sorted(df['servant room'].unique().tolist())))
    with col_store:
        store_room = float(st.selectbox('Store Room', options=sorted(df['store room'].unique().tolist())))
    
    # Additional features in expandable section
    with st.expander("Additional Features"):
        furnishing_type = st.selectbox('Furnishing Type', options=df['furnishing_type'].unique())
        facility_category = st.selectbox('Facility Category', options=df['facility_category'].unique())
        floor_category = st.selectbox('Floor Category', options=df['floor_category'].unique())
    
    predict_button = st.button('Predict Price', use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Prediction results
    if predict_button:
        # Create input dataframe
        input_data = [property_type, sector, bedroom, bathroom, balcony, age_possession, 
                    builtup_area, servant_room, store_room, furnishing_type, 
                    facility_category, floor_category]
        
        cols = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony', 'agePossession', 
                'builtup_area', 'servant room', 'store room', 'furnishing_type', 
                'facility_category', 'floor_category']
        
        input_df = pd.DataFrame([input_data], columns=cols)
        
        # Make prediction
        try:
            with st.spinner("Calculating property price..."):
                # Make the prediction
                result = np.expm1(pipeline.predict(input_df))
                
                # Calculate price range
                lower = result - .22  # 22% lower
                upper = result + .22  # 22% higher
                
            
            # Display prediction
            st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
            
            st.markdown("<h2>Estimated Property Value</h2>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="prediction-value">₹{result[0]:.3f} Cr</div>
            <div class="prediction-range">Range: ₹{lower[0]:.3f} Cr - ₹{upper[0]:.3f} Cr</div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.error("Please check your input values and try again.")

# # Simple footer
st.markdown("""
---
**Note:** Predictions are estimates only. Actual property values may vary based on market conditions.
""")
