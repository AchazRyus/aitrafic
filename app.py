# %% Importations
import streamlit as st
import pandas as pd
from datetime import datetime
from datetime import timedelta
import datetime
import plotly
import numpy as np
import plotly.express as px

import plotly.offline as pyoff
import plotly.graph_objs as go
from prophet import Prophet

from plotly.subplots import make_subplots

# Définition des variables globales et paramètres
# Configuration des paramètres globaux des pages du streamlit
height_graph = 675
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
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
AIRPORTS = ("LGW-AMS", "LGW-BCN", "LIS-OPO", "LIS-ORY", "LYS-PIS", "NTE-FUE", "PNH-NGB", "POP-JFK" "SCL-LHR" "SSA-GRU")
baseline_model = Prophet()
df = pd.read_parquet('data/traffic_10lines.parquet')

# Définition de la fonction draw_ts_multiple

def prophet_process(df, home_airport, paired_airport, nb_days):
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
    """Extract route dataframe from traffic dataframe for route from home airport to paired airport

    Args:
    - traffic_df (pd.DataFrame): traffic dataframe
    - homeAirport (str): IATA Code for home airport
    - pairedAirport (str): IATA Code for paired airport

    Returns:
    - pd.DataFrame: aggregated daily PAX traffic on route (home-paired)
    """
    _df = (traffic_df
         .query('home_airport == "{home}" and paired_airport == "{paired}"'.format(home=homeAirport, paired=pairedAirport))
         .groupby(['home_airport', 'paired_airport', 'date'])
         .agg(pax_total=('pax', 'sum'))
         .reset_index()
         )
    return _df

st.title('Traffic Forecaster')


with st.sidebar:
    airport = st.selectbox(
        'Route', AIRPORTS)
    forecast_date = st.date_input('Forecast Start Date')
    nb_days = st.slider('Days of forecast', 7, 30, 1)
    run_forecast = st.button('Forecast')

home_airport = airport[:3]
paired_airport = airport[4:]

st.write('Home Airport selected:', home_airport)
st.write('Paired Airport selected:', paired_airport)
st.write('Days of forecast:', nb_days)
st.write('Date selected:', forecast_date)


# Affichage du Plot

row1_1, row1_2 = st.columns((1, 1))
with row1_1:
    st.dataframe(data=df, width=600, height=300)
with row1_2:
    st.plotly_chart(prophet_process(df, home_airport, paired_airport, nb_days))

