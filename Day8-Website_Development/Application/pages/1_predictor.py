import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Estate Price Predictor")

st.title("page 2")


# building a form to take user input

	# property_type	sector	bedRoom	bathroom	balcony	agePossession	builtup_area	servant room	store room	furnishing_type	facility_category	floor_category

with open('../../models/df.pkl', 'rb') as file:
    df = pickle.load(file)
    
# pipeline
with open('../../models/pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)
    
# st.dataframe(df)

# form
st.header('Enter Property Details')
property_type = st.selectbox('Property Type', options=df['property_type'].unique(), key='property_type')
sector = st.selectbox('Sector', options=sorted(df['sector'].unique().tolist()), key='sector')
bedroom = float(st.selectbox('Bedrooms Count', options=sorted(df['bedRoom'].unique().tolist()), key='bedroom'))
bathroom = float(st.selectbox('Bathrooms Count', options=sorted(df['bathroom'].unique().tolist()), key='bathroom'))
balcony = st.selectbox('Balconies', options=sorted(df['balcony'].unique().tolist()), key='balcony')
age_possession = st.selectbox('Age of Possession', options=df['agePossession'].unique().tolist(), key='age_possession')
builtup_area = st.number_input('Built-up Area (in sq. ft.)', key='builtup_area')
servant_room = float(st.selectbox('Servant Room', options=sorted(df['servant room'].unique().tolist()), key='servant_room'))
store_room = float(st.selectbox('Store Room', options=sorted(df['store room'].unique().tolist()), key='store_room'))
# optional fields
furnishing_type = st.selectbox('Furnishing Type', options=df['furnishing_type'].unique(), key='furnishing_type')
facility_category = st.selectbox('Facility Category', options=df['facility_category'].unique(), key='facility_category')
floor_category = st.selectbox('Floor Category', options=df['floor_category'].unique(), key='floor_category')

if st.button('Predict Price'):
    # form dataframe
    input_data = [property_type, sector, bedroom, bathroom, balcony, age_possession, builtup_area, servant_room, store_room, furnishing_type, facility_category, floor_category]
    cols = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony', 'agePossession', 'builtup_area', 'servant room', 'store room', 'furnishing_type', 'facility_category', 'floor_category']
    input_df = pd.DataFrame([input_data], columns=cols)
    # st.dataframe(input_df)
    result = np.expm1(pipeline.predict(input_df))
    lower = result - .22
    upper = result + .22
    st.success(f'Predicted Price: ₹{result[0]:,.3f} cr. (Price Range: ₹{lower[0]:,.3f} cr. - ₹{upper[0]:,.3f} cr.)')
    # st.success(f'Predicted Price: ₹{result[0]:,.4f} cr.')