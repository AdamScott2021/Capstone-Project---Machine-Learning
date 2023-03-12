import pandas as pd


def averages():
    # Load the hurricane data
    df = pd.read_csv('cleanedStormInfo.csv')

    # Group the data by row number and find the mean for each column
    df_mean = df.groupby(df.groupby('key').cumcount())[['key', 'lat', 'lon', 'windSpd', 'pressure']].mean()
    df_mean.fillna(-9999, inplace=True)
    df_mean.insert(0, 'key', 765)
    df_mean.to_csv("Averages.csv")
    # Print/return the dataframe
    return df_mean
