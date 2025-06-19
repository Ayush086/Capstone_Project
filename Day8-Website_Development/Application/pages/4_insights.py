import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Page title and description
st.title("Real Estate Insights")


# Prepare data and model
@st.cache_resource
def load_model_and_data():
    # Load the dataset
    df = pd.read_csv('../datasets/final_dataset_missing_value_imputed_v5.csv')
    df = df.drop(columns=['store room', 'society', 'price_per_sqft', 'balcony', 
                         'property_id', 'study room', 'pooja room', 'others'])
    
    # Data preparation (similar to notebook)
    # Age possession mapping
    df['agePossession'].replace({
        'Relatively New': 'new',
        'Moderately Old': 'old',
        'New Property': 'new',
        'Old Property': 'old',
        'Under Construction': 'under_construction',
    }, inplace=True)
    
    # Property type mapping
    df['property_type'].replace({
        'flat': 0,
        'house': 1
    }, inplace=True)
    
    # Facility score categorization
    def categorize_facility_score(score):
        if(0 <= score < 50):
            return "Low"
        elif(50 <= score < 150):
            return "Medium"
        elif 150 <= score < 175:
            return "High"
        else:
            return None

    df['facility_category'] = df['facility_score'].apply(categorize_facility_score)
    df.drop(columns=['facility_score'], inplace=True)
    df['facility_category'].replace({
        'Low': 0,
        'Medium': 1,
        'High': 2
    }, inplace=True)
    
    # One-hot encoding
    new_df = pd.get_dummies(df, columns=['sector', 'agePossession'], drop_first=True)
    
    # Prepare features and target
    X = new_df.drop(columns=['price'])
    y = new_df['price']
    y_log = np.log1p(y)  # Log transformation
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    lr = LinearRegression()
    lr.fit(X_scaled, y_log)
    
    return df, new_df, X, y, y_log, scaler, lr

# Load model and data
df, new_df, X, y, y_log, scaler, lr = load_model_and_data()

X_orig_mean = X.mean().to_frame().T  # mean for base values

# Feature impact calculation function
def calculate_feature_impact_fixed(feature_name, feature_value, model, scaler, X_orig_mean, is_categorical=False):
    """
    Calculate the impact of changing a feature value on the predicted price.
    """
    # Make a copy of average values
    X_base = X_orig_mean.copy()
    
    # Get prediction with average values (baseline)
    X_base_scaled = scaler.transform(X_base)
    base_pred = model.predict(X_base_scaled)[0]
    base_price = np.expm1(base_pred)
    
    # Create sample with new feature value
    X_new = X_orig_mean.copy()
    
    if is_categorical:
        # For one-hot encoded features
        if feature_name == 'sector':
            # Reset all sector columns to 0
            sector_cols = [col for col in X_new.columns if col.startswith('sector_')]
            for col in sector_cols:
                X_new[col] = 0
            
            # Set the specified sector to 1
            target_col = f'sector_{feature_value}'
            if target_col in X_new.columns:
                X_new[target_col] = 1
        
        elif feature_name == 'agePossession':
            # Reset all age columns to 0
            age_cols = [col for col in X_new.columns if col.startswith('agePossession_')]
            for col in age_cols:
                X_new[col] = 0
            
            # Set the specified age to 1
            target_col = f'agePossession_{feature_value}'
            if target_col in X_new.columns:
                X_new[target_col] = 1
    else:
        # For continuous features
        X_new[feature_name] = feature_value
    
    # Get prediction with new feature value
    X_new_scaled = scaler.transform(X_new)
    new_pred = model.predict(X_new_scaled)[0]
    new_price = np.expm1(new_pred)
    
    # Calculate percentage change
    percent_change = ((new_price - base_price) / base_price) * 100
    
    return base_price, new_price, percent_change

# Compare sectors function
def compare_sectors_fixed(model, scaler, X_orig_mean):
    """Compare how different sectors affect property prices"""
    # all sector columns
    sector_cols = [col for col in X_orig_mean.columns if col.startswith('sector_')]
    sector_names = [col.replace('sector_', '') for col in sector_cols]
    
    results = []
    
    #base price with all sector values set to 0 (reference sector)
    X_base = X_orig_mean.copy()
    for col in sector_cols:
        X_base[col] = 0
    
    X_base_scaled = scaler.transform(X_base)
    base_pred = model.predict(X_base_scaled)[0]
    base_price = np.expm1(base_pred)
    
    # impact for each sector
    for sector in sector_names:
        _, new_price, percent_change = calculate_feature_impact_fixed(
            'sector', sector, model, scaler, X_base, is_categorical=True)
        
        if percent_change != 0:  # Only add if there's an impact
            results.append({
                'sector': sector,
                'base_price': round(base_price/100, 2),  # Convert to crores
                'new_price': round(new_price/100, 2),    
                'price_change': round((new_price - base_price)/100, 2),
                'percent_change': round(percent_change, 1)
            })
    
    result_df = pd.DataFrame(results).sort_values('percent_change', ascending=False)
    return result_df

X_orig_mean = X.mean().to_frame().T






# Streamlit interface
st.header("Feature Impact Analysis")

# Create tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs(["Location Impact", "Property Features", "Interactive Analysis", "Model Details"])

with tab1:
    st.subheader("Location Impact on Property Prices")
    st.write("How different sectors affect property prices compared to the baseline")
    
    # sector comparison
    sector_comparison = compare_sectors_fixed(lr, scaler, X_orig_mean)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    top_sectors = sector_comparison.head(15)
    sns.barplot(x='percent_change', y='sector', data=top_sectors, ax=ax)
    ax.set_title('Top 15 Sectors by Price Impact', fontsize=16)
    ax.set_xlabel('Price Impact (%)', fontsize=12)
    ax.set_ylabel('Sector', fontsize=12)
    st.pyplot(fig)
    
    # Display detailed table
    st.write("Detailed sector price impact:")
    st.dataframe(sector_comparison)

with tab2:
    st.subheader("Property Features Impact")
    
    # Create columns for different property features
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Impact of Bedrooms")
        
        # Plot bedroom impact
        bedroom_results = []
        for bedrooms in range(1, 6):
            base_price, new_price, percent_change = calculate_feature_impact_fixed(
                'bedRoom', bedrooms, lr, scaler, X_orig_mean)
            bedroom_results.append({
                'bedrooms': bedrooms,
                'price': new_price/100,  # Convert to crores
                'percent_change': percent_change
            })
        
        bedroom_df = pd.DataFrame(bedroom_results)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(bedroom_df['bedrooms'], bedroom_df['percent_change'])
        ax.set_xlabel('Number of Bedrooms')
        ax.set_ylabel('Percentage Change in Price')
        ax.set_title('Impact of Number of Bedrooms on Price')
        ax.set_xticks(bedroom_df['bedrooms'])
        st.pyplot(fig)

    with col2:
        st.write("#### Impact of Built-up Area")
        
        # Plot area impact
        area_values = [1000, 1500, 2000, 2500, 3000, 3500]
        area_results = []
        
        for area in area_values:
            base_price, new_price, percent_change = calculate_feature_impact_fixed(
                'builtup_area', area, lr, scaler, X_orig_mean)
            area_results.append({
                'area': area,
                'price': new_price/100,
                'percent_change': percent_change
            })
        
        area_df = pd.DataFrame(area_results)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(area_df['area'], area_df['percent_change'], marker='o')
        ax.set_xlabel('Built-up Area (sq.ft.)')
        ax.set_ylabel('Percentage Change in Price')
        ax.set_title('Impact of Built-up Area on Price')
        st.pyplot(fig)
    
    # Another row of features
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Impact of Bathrooms")
        
        bathroom_results = []
        for bathrooms in range(1, 6):
            base_price, new_price, percent_change = calculate_feature_impact_fixed(
                'bathroom', bathrooms, lr, scaler, X_orig_mean)
            bathroom_results.append({
                'bathrooms': bathrooms,
                'price': new_price/100,
                'percent_change': percent_change
            })
        
        bathroom_df = pd.DataFrame(bathroom_results)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(bathroom_df['bathrooms'], bathroom_df['percent_change'])
        ax.set_xlabel('Number of Bathrooms')
        ax.set_ylabel('Percentage Change in Price')
        ax.set_title('Impact of Number of Bathrooms on Price')
        ax.set_xticks(bathroom_df['bathrooms'])
        st.pyplot(fig)

    with col2:
        st.write("#### Impact of Property Age")
        
        # Get age possession columns
        age_cols = [col for col in X_orig_mean.columns if col.startswith('agePossession_')]
        age_types = [col.replace('agePossession_', '') for col in age_cols]
        
        age_results = []
        for age_type in age_types:
            base_price, new_price, percent_change = calculate_feature_impact_fixed(
                'agePossession', age_type, lr, scaler, X_orig_mean, is_categorical=True)
            age_results.append({
                'age_type': age_type,
                'price': new_price/100,
                'percent_change': percent_change
            })
        
        age_df = pd.DataFrame(age_results)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='age_type', y='percent_change', data=age_df, ax=ax)
        ax.set_xlabel('Property Age')
        ax.set_ylabel('Percentage Change in Price')
        ax.set_title('Impact of Property Age on Price')
        plt.xticks(rotation=45)
        st.pyplot(fig)
with tab3:
    st.subheader("Interactive Feature Analysis")
    st.write("Explore how changing different features affects property price")
    
    # Feature selection
    feature_type = st.selectbox(
        "Select feature type", 
        ["Continuous Features", "Categorical Features"]
    )
    
    if feature_type == "Continuous Features":
        # Continuous features
        continuous_features = ['bedRoom', 'bathroom', 'builtup_area', 'servant_room']
        feature = st.selectbox("Select feature", continuous_features)
        
        if feature == 'bedRoom':
            min_val, max_val = 1, 10
            step = 1
            default = 3
            format = "%d"
        elif feature == 'bathroom':
            min_val, max_val = 1, 8
            step = 1
            default = 2
            format = "%d"
        elif feature == 'builtup_area':
            min_val, max_val = 500, 5000
            step = 100
            default = 1500
            format = "%d"
        elif feature == 'servant_room':
            min_val, max_val = 0, 3
            step = 1
            default = 0
            format = "%d"
        
        value = st.slider(
            f"Select {feature} value", 
            min_value=min_val, 
            max_value=max_val, 
            value=default,
            step=step,
            format=format
        )
        
        is_categorical = False
        
    else:
        # Categorical features
        categorical_type = st.selectbox(
            "Select category", 
            ["Location (Sector)", "Property Age"]
        )
        
        if categorical_type == "Location (Sector)":
            feature = 'sector'
            # Get all sector names
            sector_cols = [col for col in X_orig_mean.columns if col.startswith('sector_')]
            sector_names = [col.replace('sector_', '') for col in sector_cols]
            value = st.selectbox("Select sector", sector_names)
            
        else:  # Property Age
            feature = 'agePossession'
            age_types = ['new', 'old', 'under_construction']
            value = st.selectbox("Select property age", age_types)
            
        is_categorical = True
    
    # Calculate impact
    base_price, new_price, percent_change = calculate_feature_impact_fixed(
        feature, value, lr, scaler, X_orig_mean, is_categorical)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Base Price", f"₹{base_price/100:.2f} Cr")
    
    with col2:
        st.metric("New Price", f"₹{new_price/100:.2f} Cr")
    
    with col3:
        st.metric("Price Change", f"{percent_change:.1f}%", 
                  delta=f"{(new_price-base_price)/100:.2f} Cr")

with tab4:
    st.subheader("Model Details")
    
    st.write("""
    #### Model Performance
    The model uses Linear Regression on log-transformed price data. Features are standardized
    for better comparability of their impacts.
    """)
    
    # Display feature importance
    coef_df = pd.DataFrame(
        lr.coef_.reshape(1, len(X.columns)), 
        columns=X.columns
    ).stack().reset_index()
    
    coef_df = coef_df.drop(columns='level_0').rename(columns={'level_1': 'feature', 0: 'coefficient'})
    coef_df = coef_df.sort_values('coefficient', ascending=False)
    
    st.write("#### Top 10 Features by Impact:")
    st.dataframe(coef_df.head(10))
    
    st.write("#### Bottom 10 Features by Impact:")
    st.dataframe(coef_df.tail(10))
    
    # Model interpretation
    st.write("""
    #### Interpretation
    
    - **Positive coefficients** indicate features that increase property prices
    - **Negative coefficients** indicate features that decrease property prices
    - The values represent the relative importance of each feature
    
    The analysis shows that location (sector) and built-up area are the strongest determinants
    of property price in this market.
    """)
