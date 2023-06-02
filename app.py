# %% Importations
import streamlit as st
import streamlit_folium as st_folium
import pandas as pd
from datetime import datetime
from datetime import timedelta
import datetime
import plotly
import numpy as np
import plotly.express as px
import folium
import plotly.offline as pyoff
import plotly.graph_objs as go
from prophet import Prophet

from plotly.subplots import make_subplots

# Définition des variables globales et paramètres
# Configuration des paramètres globaux des pages du streamlit
height_graph = 675
st.set_page_config(layout="wide")
# Supression de la barre blanche en haut de page
st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)


# Importation des données
airports = ("LGW-AMS", "LGW-BCN", "LIS-OPO", "LIS-ORY", "LYS-PIS", "NTE-FUE", "PNH-NGB", "POP-JFK", "SCL-LHR", "SSA-GRU")
baseline_model = Prophet()
df = pd.read_parquet('data/traffic_10lines.parquet')
airports_data = {
'code': ['LGW', 'BCN', 'AMS', 'LIS', 'ORY', 'OPO', 'SSA', 'GRU', 'NTE', 'FUE', 'LYS', 'PIS', 'PNH', 'NGB', 'POP', 'JFK', 'SCL', 'LHR'],
'latitude': [51.1537, 41.2976, 52.3086, 38.7742, 48.7262, 41.237, -12.9086, -23.4356, 47.1569, 28.4527, 45.7256, 46.5871, 11.5466, 6.3201, 19.7579, 40.6413, -33.3928, 51.4700],
'longitude': [-0.1821, 2.0834, 4.7639, -9.1354, 2.3669, -8.6741, -38.3229, -46.4731, -1.6114, -13.8649, 5.0908, 0.3076, 104.8441, 3.2221, -70.5700, -73.7781, -70.7944, -0.4543]
}
coord = pd.DataFrame(airports_data)

# Définition de la fonction draw_ts_multiple

def prophet_process(df, home_airport, paired_airport, nb_days):
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df[df["date"]<forecast_date]
    baseline_model.fit(generate_route_df(df, home_airport, paired_airport).rename(columns={'date': 'ds', 'pax_total': 'y'}))
    forecast_df = baseline_model.predict(baseline_model.make_future_dataframe(periods=nb_days))
    forecast_df = forecast_df[["ds", "yhat"]]
    forecast_df["train"] = np.nan
    forecast_df["prediction"] = np.nan
    forecast_df.loc[0:len(forecast_df)-nb_days,"train"] = forecast_df.loc[0-nb_days:len(forecast_df)-nb_days,"yhat"]
    forecast_df.loc[len(forecast_df)-nb_days:len(forecast_df),"prediction"] = forecast_df.loc[len(forecast_df)-nb_days:len(forecast_df),"yhat"]
    fig = px.line(forecast_df, x="ds", y=["prediction","train"])
    fig.update_xaxes(rangeslider_visible=True)
    return fig
    
def generate_route_df(traffic_df: pd.DataFrame, homeAirport: str, pairedAirport: str) -> pd.DataFrame:
    _df = (traffic_df
         .query('home_airport == "{home}" and paired_airport == "{paired}"'.format(home=homeAirport, paired=pairedAirport))
         .groupby(['home_airport', 'paired_airport', 'date'])
         .agg(pax_total=('pax', 'sum'))
         .reset_index()
         )
    return _df
    
def generate_map():
    coords_home = coord.loc[coord['code'] == home_airport, ['latitude', 'longitude']].values[0]
    coords_arrival = coord.loc[coord['code'] == paired_airport, ['latitude', 'longitude']].values[0]
    airport_map = folium.Map(location=[coords_home[0], coords_home[1]], zoom_start=5)
    folium.Marker(
        location=[coords_home[0], coords_home[1]],
        popup=home_airport,
        icon=folium.Icon(icon='plane', color='green')
    ).add_to(airport_map)
    folium.Marker(
        location=[coords_arrival[0], coords_arrival[1]],
        popup=paired_airport,
        icon=folium.Icon(icon='plane', color='red')
    ).add_to(airport_map)
    folium.PolyLine([coords_home, coords_arrival], color='blue', weight=3).add_to(airport_map)
    return airport_map


st.title('Traffic Forecaster')


with st.sidebar:
    airline = st.selectbox(
        'Airline', airports)
    forecast_date = st.date_input('Forecast Start Date')
    nb_days = st.slider('Days of forecast', 30, 150, 30)
    run_forecast = st.button('Forecast')

home_airport = airline[:3]
paired_airport = airline[4:]


# Affichage du Plot
col1, col2 = st.columns(2)

with col1:
    st.write('Home Airport selected:', home_airport)
    st.write('Paired Airport selected:', paired_airport)
    st.write('Days of forecast:', nb_days)
    st.write('Date selected:', forecast_date)
with col2:
    st.dataframe(data=df, width=600, height=300)
    
col1, col2 = st.columns(2)

with col1:
    m = generate_map()
    st_folium.folium_static(m, height=300, width=450)
with col2:
    st.plotly_chart(prophet_process(df, home_airport, paired_airport, nb_days))
    


