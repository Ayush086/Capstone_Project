import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Market Analytics", 
    page_icon="📊",
    layout="wide"
)

# Custom CSS for consistent styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e0e0e0;
    }
    .chart-container {
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .insight-box {
        padding: 1rem;
        border-left: 5px solid #1E88E5;
        margin-bottom: 1rem;
    }
    .filter-container {
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Page title with custom styling
st.markdown("<h1 class='main-header'>Real Estate Market Analytics</h1>", unsafe_allow_html=True)

# Description of the analytics page
st.markdown("""
<div class="insight-box">
Explore comprehensive data visualizations to understand the real estate market trends, 
property distributions, and price patterns across different sectors and property types.
</div>
""", unsafe_allow_html=True)

# Load data with caching to improve performance
@st.cache_data
def load_data():
    try:
        # Load the main dataset
        df = pd.read_csv('D:/Projects/capstone-project/Application/assets/final_dataset_with_latlong.csv')
        
        # Load the feature text for word cloud
        feature_text = pickle.load(open('D:/Projects/capstone-project/Application/assets/feature_text.pkl', 'rb'))
        
        return df, feature_text
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Load the data
new_df, feature_text = load_data()

if new_df is None:
    st.error("Failed to load data. Please check the data files and try again.")
    st.stop()

# Create the grouped dataframe for map visualization
group_df = new_df.groupby('sector').mean(numeric_only=True)[['price', 'price_per_sqft', 'builtup_area', 'latitude', 'longitude']]

# Create tabs for organization
tab1, tab2, tab3 = st.tabs(["📍 Location Analysis", "💰 Price Analysis", "🏠 Property Features"])

# ====================== TAB 1: LOCATION ANALYSIS ======================
with tab1:
    st.markdown("<h2 class='section-header'>Location-based Market Analysis</h2>", unsafe_allow_html=True)
    
    # Description/insight about the map
    st.markdown("""
    <div class="insight-box">
    This map visualizes average property prices across different sectors. Larger circles indicate 
    areas with larger properties, while the color intensity represents price per square foot.
    </div>
    """, unsafe_allow_html=True)
    
    # Map visualization container
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Sector-wise Price Map with improved aesthetics
    fig_map = px.scatter_mapbox(
        group_df, 
        lat='latitude', 
        lon='longitude', 
        size='builtup_area',
        color='price_per_sqft',
        color_continuous_scale="Viridis",
        zoom=10, 
        mapbox_style="open-street-map", 
        width=1200, 
        height=700,
        hover_name=group_df.index,
        hover_data={
            'price': True,
            'price_per_sqft': True,
            'builtup_area': True,
            'latitude': False,
            'longitude': False
        },
        labels={
            'price': 'Avg. Price (₹)',
            'price_per_sqft': 'Price/sqft (₹)',
            'builtup_area': 'Avg. Area (sqft)'
        }
    )
    
    # Improve the layout
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            title="Price/sqft (₹)",
            thicknessmode="pixels",
            thickness=20,
            lenmode="pixels",
            len=300,
        )
    )
    
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sector-wise comparison
    st.markdown("<h3>Sector Comparison</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 most expensive sectors
        top_sectors = group_df.sort_values('price', ascending=False).head(10)
        
        fig_top_sectors = px.bar(
            top_sectors,
            y=top_sectors.index,
            x='price',
            orientation='h',
            title='Top 10 Most Expensive Sectors (Avg. Price)',
            labels={'price': 'Average Price (₹)', 'y': 'Sector'},
            color='price',
            color_continuous_scale="Viridis",
            height=500
        )
        
        fig_top_sectors.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_top_sectors, use_container_width=True)
    
    with col2:
        # Top 10 sectors by price per square foot
        top_price_per_sqft = group_df.sort_values('price_per_sqft', ascending=False).head(10)
        
        fig_top_price_per_sqft = px.bar(
            top_price_per_sqft,
            y=top_price_per_sqft.index,
            x='price_per_sqft',
            orientation='h',
            title='Top 10 Sectors by Price per Square Foot',
            labels={'price_per_sqft': 'Price per Square Foot (₹)', 'y': 'Sector'},
            color='price_per_sqft',
            color_continuous_scale="Viridis",
            height=500
        )
        
        fig_top_price_per_sqft.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_top_price_per_sqft, use_container_width=True)

# ====================== TAB 2: PRICE ANALYSIS ======================
with tab2:
    st.markdown("<h2 class='section-header'>Price Analysis & Trends</h2>", unsafe_allow_html=True)
    
    # Area vs Price visualization
    st.markdown("<h3>Price vs. Built-up Area Analysis</h3>", unsafe_allow_html=True)
    
    # Filter container
    st.markdown('<div class="filter-container">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        property_type = st.selectbox('Property Type', ['flat', 'house', 'both'])
    
    with col2:
        bedroom_filter = st.multiselect(
            'Number of Bedrooms', 
            options=sorted(new_df['bedRoom'].unique()),
            default=sorted(new_df['bedRoom'].unique())
        )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Filter data based on user selection
    if property_type == 'both':
        filtered_df = new_df[new_df['bedRoom'].isin(bedroom_filter)]
    else:
        filtered_df = new_df[(new_df['property_type'] == property_type) & 
                             (new_df['bedRoom'].isin(bedroom_filter))]
    
    # Create scatter plot with improved aesthetics
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    fig_scatter = px.scatter(
        filtered_df,
        x='builtup_area',
        y='price',
        color='bedRoom',
        size='price_per_sqft',
        hover_name='sector',
        hover_data=['property_type', 'price_per_sqft'],
        labels={
            'builtup_area': 'Built-up Area (sq.ft)',
            'price': 'Price (₹)',
            'bedRoom': 'Bedrooms',
            'price_per_sqft': 'Price/sq.ft'
        },
        color_continuous_scale="Viridis",
        title=f"Price vs. Built-up Area for {property_type.capitalize() if property_type != 'both' else 'All'} Properties"
    )
    
    # Add trend line
    fig_scatter.update_layout(
        height=600,
        xaxis=dict(title_font=dict(size=14)),
        yaxis=dict(title_font=dict(size=14))
    )
    
    fig_scatter.update_traces(
        marker=dict(line=dict(width=1, color='DarkSlateGrey')),
        selector=dict(mode='markers')
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Box plot for price distribution by bedroom
    st.markdown("<h3>Price Distribution by Number of Bedrooms</h3>", unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig_box = px.box(
        filtered_df,
        x='bedRoom',
        y='price',
        color='property_type',
        labels={
            'bedRoom': 'Number of Bedrooms',
            'price': 'Price (₹)',
            'property_type': 'Property Type'
        },
        title="Price Distribution by Number of Bedrooms",
        color_discrete_map={'flat': '#1E88E5', 'house': '#FFC107'}
    )
    
    fig_box.update_layout(height=500)
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Price distribution of property types
    st.markdown("<h3>Price Distribution by Property Type</h3>", unsafe_allow_html=True)
    
    # Use plotly instead of matplotlib for consistent theme
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Create data for the price distribution
    fig_dist = make_subplots(specs=[[{"secondary_y": False}]])
    
    # Add traces for house and flat
    for prop_type, color in zip(['house', 'flat'], ['#FFC107', '#1E88E5']):
        prop_data = new_df[new_df['property_type'] == prop_type]['price']
        
        # Create histogram with KDE
        fig_dist.add_trace(
            go.Histogram(
                x=prop_data,
                name=prop_type.capitalize(),
                marker_color=color,
                opacity=0.6,
                nbinsx=30
            )
        )
    
    # Update layout for better appearance
    fig_dist.update_layout(
        title="Price Distribution by Property Type",
        xaxis_title="Price (₹)",
        yaxis_title="Count",
        bargap=0.1,
        bargroupgap=0.2,
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ====================== TAB 3: PROPERTY FEATURES ======================
with tab3:
    st.markdown("<h2 class='section-header'>Property Features Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    # Word Cloud of Property Features
    with col1:
        st.markdown("<h3>Common Property Features</h3>", unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Create word cloud with custom colors matching the theme
        wordcloud = WordCloud(
            width=800,
            height=500,
            background_color='white',
            colormap='viridis',
            stopwords=set(['s']),
            min_font_size=10
        ).generate(feature_text)
        
        # Use matplotlib for the word cloud (kept for compatibility)
        plt.figure(figsize=(10, 6), facecolor=None)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(plt)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Bedroom Distribution
    with col2:
        st.markdown("<h3>Bedroom Distribution</h3>", unsafe_allow_html=True)
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        
        # Filter for sector selection
        options = new_df['sector'].unique().tolist()
        options.insert(0, 'All Sectors')
        sector_filter = st.selectbox('Select Sector', options)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Filter data based on sector selection
        if sector_filter == 'All Sectors':
            chart_df = new_df
        else:
            chart_df = new_df[new_df['sector'] == sector_filter]
        
        # Create pie chart with improved aesthetics
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        bedroom_counts = chart_df['bedRoom'].value_counts().reset_index()
        bedroom_counts.columns = ['bedRoom', 'count']
        
        fig_pie = px.pie(
            bedroom_counts,
            values='count',
            names='bedRoom',
            title=f"Bedroom Distribution in {sector_filter}",
            color_discrete_sequence=px.colors.sequential.Viridis,
            hole=0.4
        )
        
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            marker=dict(line=dict(color='#FFFFFF', width=1))
        )
        
        fig_pie.update_layout(
            height=400,
            legend_title="Number of Bedrooms"
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Property Type Distribution Across Sectors
    st.markdown("<h3>Property Type Distribution Across Top Sectors</h3>", unsafe_allow_html=True)
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Get top 10 sectors by property count
    top_sectors_by_count = new_df['sector'].value_counts().head(10).index.tolist()
    top_sectors_df = new_df[new_df['sector'].isin(top_sectors_by_count)]
    
    # Create stacked bar chart for property types
    property_type_counts = pd.crosstab(top_sectors_df['sector'], top_sectors_df['property_type'])
    
    fig_stacked = px.bar(
        property_type_counts,
        barmode='stack',
        labels={'sector': 'Sector', 'value': 'Number of Properties'},
        color_discrete_map={'flat': '#1E88E5', 'house': '#FFC107'},
        title="Property Type Distribution in Top 10 Sectors"
    )
    
    fig_stacked.update_layout(
        legend_title="Property Type",
        xaxis={'categoryorder': 'total descending'},
        height=500
    )
    
    st.plotly_chart(fig_stacked, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Property size analysis
    st.markdown("<h3>Property Size Analysis</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Create histogram of built-up areas by property type
        fig_size = px.histogram(
            new_df,
            x='builtup_area',
            color='property_type',
            nbins=30,
            opacity=0.7,
            labels={
                'builtup_area': 'Built-up Area (sq.ft)',
                'property_type': 'Property Type'
            },
            title="Distribution of Property Sizes",
            color_discrete_map={'flat': '#1E88E5', 'house': '#FFC107'}
        )
        
        fig_size.update_layout(bargap=0.1)
        st.plotly_chart(fig_size, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Average built-up area by number of bedrooms
        avg_area_by_bedroom = new_df.groupby('bedRoom')['builtup_area'].mean().reset_index()
        
        fig_avg_area = px.bar(
            avg_area_by_bedroom,
            x='bedRoom',
            y='builtup_area',
            color='builtup_area',
            color_continuous_scale="Viridis",
            labels={
                'bedRoom': 'Number of Bedrooms',
                'builtup_area': 'Avg. Built-up Area (sq.ft)'
            },
            title="Average Property Size by Number of Bedrooms"
        )
        
        fig_avg_area.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_avg_area, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer with insights summary
st.markdown("""
<div style="; padding: 20px; border-radius: 10px; margin-top: 30px;">
    <h3 style="text-align: center; color: #0D47A1;">Key Market Insights</h3>
    <ul>
        <li><strong>Location Impact:</strong> Property prices vary significantly across sectors, with premium locations commanding substantially higher prices.</li>
        <li><strong>Size-Price Relationship:</strong> While price generally increases with property size, the rate of increase varies by location and property type.</li>
        <li><strong>Bedroom Premium:</strong> Properties with 3-4 bedrooms tend to offer the best value per square foot in most areas.</li>
        <li><strong>Property Type Differences:</strong> Houses generally have higher price variability compared to flats, reflecting greater diversity in size and features.</li>
    </ul>
    <p style="text-align: center; font-style: italic; margin-top: 15px;">
        This analysis provides a comprehensive overview of the real estate market to help inform your property decisions.
    </p>
</div>
""", unsafe_allow_html=True)
