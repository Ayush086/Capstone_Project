import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Plotting Demo")

st.title("Analytics")

new_df = pd.read_csv('../assets/final_dataset_with_latlong.csv')

group_df = new_df.groupby('sector').mean(numeric_only=True)[['price', 'price_per_sqft', 'builtup_area', 'latitude', 'longitude']]

# geo map
st.header('Sector-wise Price Map')
fig = px.scatter_mapbox(group_df, lat='latitude', lon='longitude', size='builtup_area', color='price_per_sqft', zoom=10, mapbox_style="open-street-map", width=1200, height=700, hover_name=group_df.index)
st.plotly_chart(fig, use_container_width=True)


# word cloud
st.header('Features Word Cloud')
feature_text = pickle.load(open('../assets/feature_text.pkl', 'rb'))
wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=set(['s']),
                      min_font_size=10).generate(feature_text)

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
st.pyplot(plt)


# scatter plot
fig2_df = pd.read_csv('../../datasets/final_dataset_feature_selection_v6.csv')
st.header('Area vs Price')

property_type = st.selectbox('Property Type', ['flat', 'house'])
if property_type == 'house':
    fig2 = px.scatter(new_df[new_df['property_type'] == 'house'], x='builtup_area', y='price', color='bedRoom')
else:
    fig2 = px.scatter(new_df[new_df['property_type'] == 'flat'], x='builtup_area', y='price', color='bedRoom')
st.plotly_chart(fig2)


# pie chart
st.header('Bedroom Distribution')

options = new_df['sector'].unique().tolist()
options.insert(0, 'All Sectors')
filter = st.selectbox('Sector', options)

if filter == 'All Sectors':
    chart_df = new_df
else:
    chart_df = new_df[new_df['sector'] == filter]
fig3 = px.pie(chart_df, names='bedRoom')
st.plotly_chart(fig3, use_container_width=True)


# box plot
st.header('')

fig4 = px.box(new_df, x='bedRoom', y='price')
st.plotly_chart(fig4, use_container_width=True)

# dist plot
st.header("Property Type Distribution")
fig5 = plt.figure(figsize=(10, 4))
sns.distplot(new_df[new_df['property_type'] == 'house']['price'])
sns.distplot(new_df[new_df['property_type'] == 'flat']['price'])
plt.legend(['House', 'Flat'])
st.pyplot(fig5)
