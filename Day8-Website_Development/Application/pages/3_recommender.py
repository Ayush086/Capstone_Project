
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Property Recommender", 
    page_icon="🏘️",
    layout="wide"
) 

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .recommendation-card {
        
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .recommendation-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
    }
    .metric-container {
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .filter-section {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .map-container {
        border: 1px solid #ddd;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }
    .property-details {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .highlight {
        padding: 10px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_resource
def load_data():
    try:
        base_path = Path('D:/Projects/capstone-project/Application/assets')
        
        # Load location data
        location_df = pickle.load(open(base_path / 'location_dist.pkl', 'rb'))
        
        # Load similarity matrices
        cosine_sim1 = pickle.load(open(base_path / 'cosine_sim.pkl', 'rb'))
        cosine_sim2 = pickle.load(open(base_path / 'cosine_sim2.pkl', 'rb'))
        cosine_sim3 = pickle.load(open(base_path / 'cosine_sim3.pkl', 'rb'))
        
        # Load property details
        try:
            property_details = pd.read_csv(base_path / 'property_details.csv', index_col=0)
        except:
            # If detailed property data is not available, create a dummy dataframe
            property_details = pd.DataFrame(index=location_df.index)
            property_details['price'] = np.random.randint(5000000, 20000000, size=len(property_details))
            property_details['bedRoom'] = np.random.randint(1, 5, size=len(property_details))
            property_details['bathroom'] = np.random.randint(1, 4, size=len(property_details))
            property_details['builtup_area'] = np.random.randint(800, 3000, size=len(property_details))
            property_details['sector'] = [f"sector {np.random.randint(1, 120)}" for _ in range(len(property_details))]
        
        return location_df, cosine_sim1, cosine_sim2, cosine_sim3, property_details
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

# Load the data
location_df, cosine_sim1, cosine_sim2, cosine_sim3, property_details = load_data()

# Check if data is loaded successfully
if location_df is None:
    st.error("Failed to load necessary data. Please check data files and try again.")
    st.stop()

# Function to recommend similar properties
def recommender(property_name, top_n=10, weight_location=0.5, weight_features=0.8, weight_additional=1.0):
    """
    Recommend similar properties based on weighted similarity scores
    
    Parameters:
    -----------
    property_name : str
        Name of the reference property
    top_n : int
        Number of recommendations to return
    weight_location : float
        Weight for location-based similarity
    weight_features : float
        Weight for feature-based similarity
    weight_additional : float
        Weight for additional similarity metrics
    """
    # Combine similarity matrices with weights
    cosine_sim_matrix = weight_location * cosine_sim1 + weight_features * cosine_sim2 + weight_additional * cosine_sim3
    
    # Get the property index
    try:
        property_idx = location_df.index.get_loc(property_name)
    except KeyError:
        st.error(f"Property '{property_name}' not found in the database.")
        return None
    
    # Get pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim_matrix[property_idx]))
    
    # Sort properties based on similarity scores (excluding the reference property)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [score for score in sim_scores if score[0] != property_idx]
    
    # Get top N similar properties
    top_indices = [i[0] for i in sim_scores[:top_n]]
    top_scores = [i[1] for i in sim_scores[:top_n]]
    
    # Get property names
    top_properties = location_df.index[top_indices].tolist()
    
    # Create recommendations dataframe
    recommendations_df = pd.DataFrame({
        'property_name': top_properties,
        'similarity_score': top_scores
    })
    
    # Add property details if available
    if property_details is not None:
        recommendations_df = recommendations_df.merge(
            property_details, 
            left_on='property_name', 
            right_index=True, 
            how='left'
        )
    
    return recommendations_df

# Function to filter properties by location distance
def find_nearby_properties(location, radius_km=5):
    """Find properties within specified radius of a location"""
    if location not in location_df.columns:
        st.error(f"Location '{location}' not found in the database.")
        return None
    
    # Calculate distance and filter
    nearby = location_df[location_df[location] <= radius_km*1000]
    
    # Sort by distance and create result dataframe
    result_df = pd.DataFrame({
        'property_name': nearby.index,
        'distance_km': nearby[location]/1000
    }).sort_values('distance_km')
    
    # Add property details if available
    if property_details is not None:
        result_df = result_df.merge(
            property_details, 
            left_on='property_name', 
            right_index=True, 
            how='left'
        )
    
    return result_df

# Function to create a map visualization of properties
def create_property_map(properties_df, center_location=None):
    """Create a folium map with property locations"""
    # Set map center (if center_location is None, use mean of properties)
    if center_location:
        map_center = [center_location[0], center_location[1]]
    else:
        # Use default coordinates if lat/lon not available
        map_center = [28.4595, 77.0266]  # Default: Gurgaon coordinates
    
    # Create map
    m = folium.Map(location=map_center, zoom_start=12, tiles="OpenStreetMap")
    
    # Add marker cluster
    marker_cluster = MarkerCluster().add_to(m)
    
    # Add markers for each property
    for idx, row in properties_df.iterrows():
        # Get property name
        name = row['property_name'] if 'property_name' in row else idx
        
        # Get property details for popup
        popup_text = f"<b>{name}</b><br>"
        
        # Add available property details to popup
        if 'price' in row:
            popup_text += f"Price: ₹{row['price']/100:.2f} Cr<br>"
        if 'bedRoom' in row:
            popup_text += f"Bedrooms: {row['bedRoom']}<br>"
        if 'builtup_area' in row:
            popup_text += f"Area: {row['builtup_area']} sq.ft<br>"
        if 'distance_km' in row:
            popup_text += f"Distance: {row['distance_km']:.2f} km<br>"
        if 'similarity_score' in row:
            popup_text += f"Similarity: {row['similarity_score']:.2f}<br>"
        
        # Create popup
        popup = folium.Popup(popup_text, max_width=300)
        
        # Create marker with custom icon
        icon = folium.Icon(
            icon="home",
            prefix="fa",
            color="blue" if 'similarity_score' not in row else "green"
        )
        
        # Add marker - use random coordinates around map center for demo
        # In a real implementation, you would use actual property coordinates
        lat_offset = np.random.uniform(-0.05, 0.05)
        lng_offset = np.random.uniform(-0.05, 0.05)
        folium.Marker(
            location=[map_center[0] + lat_offset, map_center[1] + lng_offset],
            popup=popup,
            icon=icon,
            tooltip=name
        ).add_to(marker_cluster)
    
    return m

# Check for session state to handle page interactions
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = {}

# Main interface
st.markdown("<h1 class='main-header'>Property Recommender</h1>", unsafe_allow_html=True)
st.markdown("""
<div class="highlight">
Find properties that match your preferences based on location, features, and similarity to properties you like.
Use the tools below to discover your perfect property match.
</div>
""", unsafe_allow_html=True)

# Create tabs for different recommendation methods
tab1, tab2, tab3 = st.tabs([
    "🔍 Find by Location",
    "🏠 Similar Properties",
    "🔢 Advanced Filters"
])

with tab1:
    st.markdown("<h2 class='sub-header'>Find Properties by Location</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class="filter-section">
    Select a location and specify how far you're willing to travel. We'll find properties within that radius.
    </div>
    """, unsafe_allow_html=True)
    
    # Location search interface
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        location = st.selectbox(
            'Select Location', 
            sorted(location_df.columns.to_list()), 
            key='location',
            help="Choose a landmark or area as your reference point"
        )
    
    with col2:
        radius = st.slider(
            'Search Radius (km)', 
            min_value=1, 
            max_value=30, 
            value=5, 
            step=1, 
            key='radius',
            help="Maximum distance from your chosen location"
        )
    
    with col3:
        st.write("")  # Spacing
        st.write("")  # Spacing
        search_button = st.button('Search Nearby', use_container_width=True)
    
    # Search results
    if search_button or 'location_results' in st.session_state:
        with st.spinner('Finding nearby properties...'):
            # Find nearby properties
            nearby_properties = find_nearby_properties(location, radius_km=radius)
            
            if nearby_properties is not None and not nearby_properties.empty:
                st.session_state.location_results = nearby_properties
                
                # Summary stats
                num_properties = len(nearby_properties)
                avg_price = nearby_properties['price'].mean()/100 if 'price' in nearby_properties else 0
                
                # Display summary metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Properties Found", f"{num_properties}")
                col2.metric("Avg. Distance", f"{nearby_properties['distance_km'].mean():.2f} km")
                col3.metric("Avg. Price", f"₹{avg_price:.2f} Cr" if avg_price > 0 else "N/A")
                
                # Map view
                st.markdown("<h3>Map View</h3>", unsafe_allow_html=True)
                property_map = create_property_map(nearby_properties)
                
                with st.container():
                    st.markdown('', unsafe_allow_html=True)
                    folium_static(property_map, width=800)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Table view with filters
                st.markdown("<h3>Property List</h3>", unsafe_allow_html=True)
                
                # Filtering options for the table
                col1, col2, col3 = st.columns(3)
                with col1:
                    max_distance = st.slider(
                        "Max Distance (km)", 
                        min_value=0.1,
                        max_value=float(nearby_properties['distance_km'].max()),
                        value=float(nearby_properties['distance_km'].max()),
                        step=0.1
                    )
                
                if 'bedRoom' in nearby_properties:
                    with col2:
                        bedroom_filter = st.multiselect(
                            "Bedrooms",
                            options=sorted(nearby_properties['bedRoom'].unique()),
                            default=sorted(nearby_properties['bedRoom'].unique())
                        )
                else:
                    bedroom_filter = None
                
                if 'price' in nearby_properties:
                    with col3:
                        max_price = st.slider(
                            "Max Price (₹ Cr)",
                            min_value=float(nearby_properties['price'].min()/100),
                            max_value=float(nearby_properties['price'].max()/100),
                            value=float(nearby_properties['price'].max()/100),
                            step=0.1
                        )
                else:
                    max_price = None
                
                # Apply filters
                filtered_nearby = nearby_properties[nearby_properties['distance_km'] <= max_distance]
                
                if bedroom_filter is not None:
                    filtered_nearby = filtered_nearby[filtered_nearby['bedRoom'].isin(bedroom_filter)]
                
                if max_price is not None:
                    filtered_nearby = filtered_nearby[filtered_nearby['price']/100 <= max_price]
                
                # Display the filtered table
                if len(filtered_nearby) > 0:
                    # Format the dataframe for display
                    display_df = filtered_nearby.copy()
                    
                    if 'price' in display_df:
                        display_df['price'] = display_df['price'].apply(lambda x: f"₹{x/100:.2f} Cr")
                    
                    if 'builtup_area' in display_df:
                        display_df['builtup_area'] = display_df['builtup_area'].apply(lambda x: f"{x} sq.ft")
                    
                    if 'distance_km' in display_df:
                        display_df['distance_km'] = display_df['distance_km'].apply(lambda x: f"{x:.2f} km")
                    
                    # Rename columns for display
                    column_map = {
                        'property_name': 'Property',
                        'distance_km': 'Distance',
                        'price': 'Price',
                        'bedRoom': 'Beds',
                        'bathroom': 'Baths',
                        'builtup_area': 'Area',
                        'sector': 'Sector'
                    }
                    display_df = display_df.rename(columns=column_map)
                    
                    # Reorder columns
                    display_cols = [col for col in ['Property', 'Distance', 'Price', 'Beds', 'Baths', 'Area', 'Sector'] if col in display_df.columns]
                    
                    st.dataframe(
                        display_df[display_cols],
                        use_container_width=True,
                        height=400
                    )
                    
                    # Option to select a property for recommendations
                    if st.button("Find similar properties to these results"):
                        # Store first property for tab 2
                        st.session_state.selected_property = filtered_nearby['property_name'].iloc[0]
                        # Switch to tab 2
                        st.experimental_set_query_params(tab="similar-properties")
                else:
                    st.warning("No properties match your filters. Try adjusting your criteria.")
            else:
                st.warning(f"No properties found within {radius}km of {location}.")

with tab2:
    st.markdown("<h2 class='sub-header'>Find Similar Properties</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class="filter-section">
    Select a property you like, and we'll recommend similar properties based on location, features, and other factors.
    </div>
    """, unsafe_allow_html=True)
    
    # Property selection 
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Use the property from session state if available, otherwise default to the first one
        default_idx = 0
        if 'selected_property' in st.session_state and st.session_state.selected_property in location_df.index:
            default_idx = location_df.index.get_loc(st.session_state.selected_property)
        
        selected_apartment = st.selectbox(
            'Select a reference property', 
            sorted(location_df.index.to_list()), 
            index=default_idx,
            key='apartment'
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        recommend_button = st.button('Find Similar Properties', use_container_width=True)
    
    # Advanced options in expander
    with st.expander("Advanced Options"):
        st.write("Adjust the importance of different factors in the recommendation algorithm")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            weight_location = st.slider(
                "Location Importance", 
                min_value=0.1, 
                max_value=1.0, 
                value=0.5, 
                step=0.1,
                help="How important is the location in finding similar properties"
            )
        
        with col2:
            weight_features = st.slider(
                "Features Importance", 
                min_value=0.1, 
                max_value=1.0, 
                value=0.8, 
                step=0.1,
                help="How important are property features (bedrooms, bathrooms, etc.)"
            )
        
        with col3:
            num_recommendations = st.slider(
                "Number of Recommendations", 
                min_value=3, 
                max_value=20,
                value=5,
                step=1
            )
    
    # Reference property details
    if selected_apartment:
        property_info = property_details.loc[selected_apartment] if selected_apartment in property_details.index else None
        
        if property_info is not None:
            st.markdown("<h3>Reference Property</h3>", unsafe_allow_html=True)
            
            # Property details card
            st.markdown('<div class="property-details">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.markdown(f"<h4>{selected_apartment}</h4>", unsafe_allow_html=True)
                if 'sector' in property_info:
                    st.write(f"**Location:** {property_info['sector']}")
            
            with col2:
                if 'bedRoom' in property_info:
                    st.metric("Bedrooms", f"{property_info['bedRoom']}")
            
            with col3:
                if 'bathroom' in property_info:
                    st.metric("Bathrooms", f"{property_info['bathroom']}")
            
            with col4:
                if 'price' in property_info:
                    st.metric("Price", f"₹{property_info['price']/100:.2f} Cr")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Get recommendations when button is clicked or recommendations already in session state
    if recommend_button or 'recommendations' in st.session_state:
        with st.spinner('Finding similar properties...'):
            # Get recommendations
            recommendations = recommender(
                selected_apartment, 
                top_n=num_recommendations,
                weight_location=weight_location,
                weight_features=weight_features
            )
            
            if recommendations is not None and not recommendations.empty:
                st.session_state.recommendations = recommendations
                
                st.markdown("<h3>Recommended Properties</h3>", unsafe_allow_html=True)
                
                # Display each recommendation as a card
                for i, (idx, row) in enumerate(recommendations.iterrows()):
                    similarity = row['similarity_score'] * 100  # Convert to percentage
                    
                    # Start card
                    st.markdown(f'<div class="recommendation-card">', unsafe_allow_html=True)
                    
                    # Property header with similarity score
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"<h4>{row['property_name']}</h4>", unsafe_allow_html=True)
                        
                        if 'sector' in row:
                            st.write(f"**Location:** {row['sector']}")
                    
                    with col2:
                        st.markdown(
                            f"""
                            <div style="background-color: {'#e6f7ff' if similarity >= 80 else '#fff7e6'}; 
                                        padding: 10px; border-radius: 5px; text-align: center;">
                                <p style="margin: 0; font-size: 0.9rem;">Match Score</p>
                                <p style="margin: 0; font-size: 1.5rem; font-weight: bold; color: {'#0077b6' if similarity >= 80 else '#ff9500'};">
                                    {similarity:.1f}%
                                </p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    
                    # Property details
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if 'bedRoom' in row:
                            st.metric("Bedrooms", f"{row['bedRoom']}")
                    
                    with col2:
                        if 'bathroom' in row:
                            st.metric("Bathrooms", f"{row['bathroom']}")
                    
                    with col3:
                        if 'builtup_area' in row:
                            st.metric("Area", f"{row['builtup_area']} sq.ft")
                    
                    with col4:
                        if 'price' in row:
                            st.metric("Price", f"₹{row['price']/100:.2f} Cr")
                    
                    # End card
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Visualization of recommendations
                if len(recommendations) >= 3 and 'price' in recommendations.columns:
                    st.markdown("<h3>Price Comparison</h3>", unsafe_allow_html=True)
                    
                    # Add reference property to comparison
                    reference_price = property_details.loc[selected_apartment]['price']/100 if selected_apartment in property_details.index else 0
                    
                    fig = go.Figure()
                    
                    # Reference property line
                    fig.add_shape(
                        type="line",
                        x0=-0.5,
                        y0=reference_price,
                        x1=len(recommendations)-0.5,
                        y1=reference_price,
                        line=dict(
                            color="red",
                            width=2,
                            dash="dash",
                        )
                    )
                    
                    # Add price bars
                    fig.add_trace(go.Bar(
                        x=recommendations['property_name'],
                        y=recommendations['price']/100,
                        marker_color='royalblue',
                        name='Property Price'
                    ))
                    
                    fig.update_layout(
                        title="Price Comparison with Reference Property",
                        xaxis_title="Recommended Properties",
                        yaxis_title="Price (₹ Cr)",
                        annotations=[
                            dict(
                                x=len(recommendations)/2,
                                y=reference_price,
                                xref="x",
                                yref="y",
                                text=f"Reference: ₹{reference_price:.2f} Cr",
                                showarrow=False,
                                font=dict(
                                    size=12,
                                    color="red"
                                ),
                                bgcolor="white",
                                bordercolor="red",
                                borderwidth=1,
                                borderpad=4
                            )
                        ]
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Unable to generate recommendations for this property.")

with tab3:
    st.markdown("<h2 class='sub-header'>Advanced Property Search</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class="filter-section">
    Use multiple criteria to find properties that match your specific requirements.
    </div>
    """, unsafe_allow_html=True)
    
    # Define ranges/options for different property attributes
    if property_details is not None:
        # Price range
        if 'price' in property_details.columns:
            min_price = float(property_details['price'].min() / 100)
            max_price = float(property_details['price'].max() / 100)
            
            col1, col2 = st.columns(2)
            with col1:
                price_min = st.number_input(
                    "Minimum Price (₹ Cr)",
                    min_value=min_price,
                    max_value=max_price,
                    value=min_price,
                    step=0.1
                )
            
            with col2:
                price_max = st.number_input(
                    "Maximum Price (₹ Cr)",
                    min_value=min_price,
                    max_value=max_price,
                    value=max_price,
                    step=0.1
                )
        
        # Bedrooms
        if 'bedRoom' in property_details.columns:
            bed_options = sorted(property_details['bedRoom'].unique())
            bedrooms = st.multiselect(
                "Number of Bedrooms",
                options=bed_options,
                default=[]
            )
        
        # Location (sector)
        if 'sector' in property_details.columns:
            sector_options = sorted(property_details['sector'].unique())
            sectors = st.multiselect(
                "Preferred Sectors",
                options=sector_options,
                default=[]
            )
        
        # Area range
        if 'builtup_area' in property_details.columns:
            min_area = int(property_details['builtup_area'].min())
            max_area = int(property_details['builtup_area'].max())
            
            col1, col2 = st.columns(2)
            with col1:
                area_min = st.number_input(
                    "Minimum Area (sq.ft)",
                    min_value=min_area,
                    max_value=max_area,
                    value=min_area,
                    step=100
                )
            
            with col2:
                area_max = st.number_input(
                    "Maximum Area (sq.ft)",
                    min_value=min_area,
                    max_value=max_area,
                    value=max_area,
                    step=100
                )
        
        # Search button
        if st.button("Search Properties", use_container_width=True):
            with st.spinner('Searching for matching properties...'):
                # Apply filters to property details
                filtered_df = property_details.copy()
                
                # Price filter
                if 'price' in filtered_df.columns:
                    filtered_df = filtered_df[(filtered_df['price']/100 >= price_min) & 
                                              (filtered_df['price']/100 <= price_max)]
                
                # Bedroom filter
                if 'bedRoom' in filtered_df.columns and bedrooms:
                    filtered_df = filtered_df[filtered_df['bedRoom'].isin(bedrooms)]
                
                # Sector filter
                if 'sector' in filtered_df.columns and sectors:
                    filtered_df = filtered_df[filtered_df['sector'].isin(sectors)]
                
                # Area filter
                if 'builtup_area' in filtered_df.columns:
                    filtered_df = filtered_df[(filtered_df['builtup_area'] >= area_min) & 
                                              (filtered_df['builtup_area'] <= area_max)]
                
                # Add property name as column
                filtered_df = filtered_df.reset_index().rename(columns={'index': 'property_name'})
                
                # Display results
                if len(filtered_df) > 0:
                    st.success(f"Found {len(filtered_df)} matching properties")
                    
                    # Format the dataframe for display
                    display_df = filtered_df.copy()
                    
                    if 'price' in display_df.columns:
                        display_df['price'] = display_df['price'].apply(lambda x: f"₹{x/100:.2f} Cr")
                    
                    if 'builtup_area' in display_df.columns:
                        display_df['builtup_area'] = display_df['builtup_area'].apply(lambda x: f"{x} sq.ft")
                    
                    # Rename columns
                    column_map = {
                        'property_name': 'Property',
                        'price': 'Price',
                        'bedRoom': 'Beds',
                        'bathroom': 'Baths',
                        'builtup_area': 'Area',
                        'sector': 'Sector'
                    }
                    display_df = display_df.rename(columns=column_map)
                    
                    # Order columns
                    display_cols = [col for col in ['Property', 'Price', 'Beds', 'Baths', 'Area', 'Sector'] 
                                   if col in display_df.columns]
                    
                    st.dataframe(
                        display_df[display_cols],
                        use_container_width=True,
                        height=400
                    )
                    
                    # Option to find similar properties
                    if st.button("Find properties similar to these results"):
                        # Store first property for tab 2
                        st.session_state.selected_property = filtered_df['property_name'].iloc[0]
                        # Switch to tab 2
                        st.experimental_set_query_params(tab="similar-properties")
                else:
                    st.warning("No properties match your search criteria. Try adjusting your filters.")
    else:
        st.error("Property details data is not available for advanced filtering.")

# Footer with additional information
st.markdown("""
<div style="margin-top: 50px; padding: 20px; background-color: #f5f5f5; border-radius: 10px;">
    <h3 style="text-align: center;">About Our Recommendation System</h3>
    <p style="text-align: center;">
    Our property recommender uses advanced algorithms to find properties that match your preferences based on location, 
    features, and similarity to properties you've shown interest in.
    </p>
    <p style="text-align: center;">
    <i>Note: This recommender system is intended to help narrow down options and may not account for all property details.</i>
    </p>
</div>
""", unsafe_allow_html=True)
