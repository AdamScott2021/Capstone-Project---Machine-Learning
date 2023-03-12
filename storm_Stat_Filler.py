import pandas as pd
from Predict import RandStormIDs


def storm_Stat_Filler():
    df = pd.read_csv('cleanedStormInfo.csv')
    df_ml = df[df['key'].isin(RandStormIDs)][['lat', 'lon', 'windSpd', 'pressure']]
