import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# Set page configuration
st.set_page_config(page_title='GHG Emissions Dashboard',
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# Custom CSS to incorporate the provided theme and design style
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv', encoding='ISO-8859-1')
    df['Total Emissions'] = pd.to_numeric(df['Total Emissions'], errors='coerce')
    df.dropna(subset=['Total Emissions', 'Latitude', 'Longitude'], inplace=True)
    return df

data = load_data()

# Dashboard title
st.title('Advanced Greenhouse Gas Emissions Dashboard')

# Sidebar for filtering
st.sidebar.header('Filter Options')

# Year range filter
selected_years = st.sidebar.slider(
    'Select Year Range',
    int(data['Year'].min()),
    int(data['Year'].max()),
    (int(data['Year'].min()), int(data['Year'].max()))
)

# Province filter with "Select All" option
all_provinces = ['All'] + sorted(data['Province'].unique().tolist())
selected_provinces = st.sidebar.multiselect(
    'Select Provinces',
    options=all_provinces,
    default=all_provinces[0]
)

# Update selected provinces if 'All' is selected
if selected_provinces == ['All'] or not selected_provinces:
    selected_provinces = all_provinces[1:]  # Exclude 'All' from the list

# Facility Type filter based on selected provinces
available_types = data[data['Province'].isin(selected_provinces)]['Facility Type'].unique().tolist()
selected_facility_types = st.sidebar.multiselect(
    'Select Facility Types',
    options=['All'] + available_types,
    default=['All']
)

# Update selected facility types if 'All' is selected
if selected_facility_types == ['All'] or not selected_facility_types:
    selected_facility_types = available_types

# Apply filters
filtered_data = data[
    (data['Year'].between(selected_years[0], selected_years[1])) &
    (data['Province'].isin(selected_provinces)) &
    (data['Facility Type'].isin(selected_facility_types))
]

# KPI calculations based on filtered data
total_emissions = int(filtered_data['Total Emissions'].sum())
average_emission_by_type = filtered_data.groupby('Facility Type')['Total Emissions'].mean().mean()
num_unique_facility_type = filtered_data['Facility Type'].nunique()

# Display KPIs
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Emissions", value=f"{total_emissions:,}")
with col2:
    st.metric(label="Avg Emissions by Facility Type", value=f"{average_emission_by_type:,.2f}")
with col3:
    st.metric(label="Number of Unique Facility Types", value=num_unique_facility_type)

# Display filtered data
if st.checkbox('Show Filtered Data'):
    st.write(filtered_data)


st.header('Map of GHG Emissions by Province in Canada')
# Create a custom hover text
filtered_data['custom_hover'] = filtered_data.apply(
    lambda row: f"Province: {row['Province']}<br>City: {row['Facility City']}<br>Facility Type: {row['Facility Type']}",
    axis=1
)

fig_map = px.scatter_geo(
    filtered_data,
    lat='Latitude',
    lon='Longitude',
    color='Total Emissions',
    size='Total Emissions',  # Adjust or remove the size parameter as needed
    hover_name='custom_hover',
    projection='natural earth',
    title='Geospatial Distribution of Emissions in Canada',
    color_continuous_scale='Viridis'
)

# Customize the map's geo layout
fig_map.update_geos(
    landcolor='grey',
    lakecolor='LightBlue',
    showland=True,
    showocean=True,
    showlakes=True,
    showcountries=True,
    countrycolor='White'
)

# Customize the figure's layout
fig_map.update_layout(
    paper_bgcolor='grey',
    plot_bgcolor='grey',
    margin={"r":0,"t":0,"l":0,"b":0}
)

# Add the visualization to the Streamlit app
st.plotly_chart(fig_map)



# Emissions by City
st.header('Emissions by City')
emissions_by_city = filtered_data.groupby('Facility City')['Total Emissions'].sum().sort_values(ascending=False).reset_index()
fig_city_emissions = px.bar(
    emissions_by_city,
    y='Facility City',  # Set the y-axis as the City
    x='Total Emissions',  # Set the x-axis as the Total Emissions
    orientation='h',  # Make the bar chart horizontal
    title='Total GHG Emissions by City'
)

# Reverse the order of cities to display the highest emissions on top
fig_city_emissions.update_yaxes(autorange="reversed")

st.plotly_chart(fig_city_emissions)



# Emissions by Province
st.header('Emissions by Province')
emissions_by_province = filtered_data.groupby('Province')['Total Emissions'].sum().sort_values(ascending=False).reset_index()
fig_province = px.bar(
    emissions_by_province,
    x='Province',
    y='Total Emissions',
    title='Total GHG Emissions by Province'
)
st.plotly_chart(fig_province)

# Total Emissions Over Time by Province
st.header('Total Emissions Over Time by Province')
province_year_data = filtered_data.groupby(['Year', 'Province'])['Total Emissions'].sum().reset_index()
fig_province_year = px.line(
    province_year_data,
    x='Year',
    y='Total Emissions',
    color='Province',
    title='Total Emissions Over Time by Province'
)
st.plotly_chart(fig_province_year)


# Line chart for Total Emissions by Facility City over the Years
st.header('Total Emissions by Facility City Over the Years')
city_year_data = filtered_data.groupby(['Year', 'Facility City'])['Total Emissions'].sum().reset_index()
fig_city_year = px.line(
    city_year_data,
    x='Year',
    y='Total Emissions',
    color='Facility City',
    title='Total Emissions Over Time by Facility City'
)
st.plotly_chart(fig_city_year)

# Total Emissions Over Time by Province with facility type filter
st.header('Total Emissions Over Time by Facility Type')
province_facility_year_data = filtered_data.groupby(['Year', 'Province', 'Facility Type'])['Total Emissions'].sum().reset_index()
fig_province_facility_year = px.line(
    province_facility_year_data,
    x='Year',
    y='Total Emissions',
    color='Facility Type',
    line_group='Facility Type',
    hover_name='Facility Type',
    title='Total Emissions Over Time by Province and Facility Type'
)
st.plotly_chart(fig_province_facility_year)

# Donut Chart Visualization - Emissions by Emission Factors / Coefficients d'émission
st.header('Total Emissions by Emission Factors')
# Assuming 'Emission Factor' is a column in your dataset
emission_factors = filtered_data['Emission Factors'].value_counts().reset_index()
emission_factors.columns = ['Emission Factors', 'Total Emissions']
fig_donut = px.pie(
    emission_factors,
    names='Emission Factors',
    values='Total Emissions',
    hole=0.3,  # Creates the donut hole
    title='Total Emissions by Emission Factors')
st.plotly_chart(fig_donut)


st.header('Predictive Emissions Forecast by Province')
model = LinearRegression()

# Prepare a figure for plotting predictions for each province
fig_forecast = px.line()

# Create predictions for each province
for province in selected_provinces:
    province_data = filtered_data[filtered_data['Province'] == province]
    if not province_data.empty:
        # Fit the model on all available years
        model_years = np.array(province_data['Year']).reshape(-1, 1)
        model_emissions = np.array(province_data['Total Emissions'])
        model.fit(model_years, model_emissions)
        
        # Predict for future years
        future_years = np.array([[y] for y in range(selected_years[1] + 1, selected_years[1] + 11)])
        predictions = model.predict(future_years)
        
        # Create a DataFrame for the forecast
        forecast_data = pd.DataFrame({
            'Year': future_years.flatten(), 
            'Predicted Emissions': predictions,
            'Province': province
        })
        
        # Add the prediction line to the plot
        fig_forecast.add_scatter(
            x=forecast_data['Year'], 
            y=forecast_data['Predicted Emissions'], 
            mode='lines', 
            name=province
        )

# Update plot layout
fig_forecast.update_layout(
    title='Predictive Emissions Forecast by Province', 
    xaxis_title='Year', 
    yaxis_title='Predicted Emissions',
    legend_title='Province'
)

# Display forecast plot
st.plotly_chart(fig_forecast)


# Download data feature
st.sidebar.header('Download Filtered Data')
if st.sidebar.button('Download Data as CSV'):
    csv = filtered_data.to_csv(index=False)
    st.sidebar.download_button(label='Download CSV', data=csv, file_name='filtered_data.csv', mime='text/csv')

# Instructions for further analysis
st.markdown("""
## Further Analysis
- **Geospatial Analysis**: Map visualization helps in understanding the geographic distribution of emissions.
- **Linear Regression**: Used for basic trend forecasting.
""", unsafe_allow_html=True)
