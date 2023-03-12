import webbrowser
import pandas as pd
import random
import folium
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import warnings
import pickle
from csv import writer
import os

# Load the cleaned data
newDataFrame = pd.read_csv('cleanedStormInfo.csv')
RandStormIDs = random.sample(newDataFrame['key'].unique().tolist(), 1)
# Create a new dataframe with only the selected attributes, this dataframe contains historical data
PredictDataFrame = newDataFrame[newDataFrame['key'].isin(RandStormIDs)][['key', 'lat', 'lon', 'windSpd', 'pressure']]
# Takes the new dataframe and calculates how many rows is half the current data frame to use below
HalfVal = len(PredictDataFrame) // 2
# creates a 2nd data frame that hold on the first half of the unique storms attributes
HalfDataFrame = PredictDataFrame.iloc[:HalfVal]
# Select 4 storms to plot
storm_ids = random.sample(newDataFrame['key'].unique().tolist(), 4)

# Create a new dataframe with only the selected attributes
df_ml = newDataFrame[newDataFrame['key'].isin(storm_ids)][['key', 'lat', 'lon', 'windSpd', 'pressure']]
le_x = LabelEncoder()
df_ml['key'] = le_x.fit_transform(df_ml['key'])
HalfDataFrame['key'] = le_x.fit_transform(HalfDataFrame['key'])
# list items you want to predict
labels = ['lat', 'lon', 'windSpd', 'pressure']
accuracyList = []
for label in labels:
    predict = '{}'.format(label)
    # create numpy array where label column is dropped
    X = np.array(df_ml.drop(columns='{}'.format(predict)))
    # create numpy array containing only the column to be predicted
    y = np.array(df_ml[predict])
    hi_accuracy = 0

    for i in range(10):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
        # Train a decision tree regressor on the training set
        regressor = DecisionTreeRegressor()
        regressor.fit(X_train, y_train)
        # Determine accuracy
        accuracy = regressor.score(X_test, y_test)
        if accuracy > hi_accuracy:
            # Each time new high accuracy is achieved replace previous high
            hi_accuracy = accuracy
            # Overwrite the previous and save to pickle file
            with open('{}.pickle'.format(label), 'wb') as f:
                pickle.dump(regressor, f)
    accuracyList.append(hi_accuracy)


'''
# Create csv for prediction recording
with open('storm_predictions.csv', 'a') as f:
    # truncate old file with each use
    f.truncate()
    w = writer(f)
    # Write headers
    w.writerow(['lat', 'lon', 'windSpd', 'pressure'])
    f.close()
'''

for i in range(1):
    # Create csv for prediction recording
    with open('storm{}_prediction.csv'.format(i), 'a') as f:
        # truncate old file with each use
        # f.truncate()
        w = writer(f)
        # Write headers
        w.writerow(['lat', 'lon', 'windSpd', 'pressure'])
        f.close()
    RandStormIDs = random.sample(newDataFrame['key'].unique().tolist(), 1)
    PredictDataFrame = newDataFrame[newDataFrame['key'].isin(RandStormIDs)][
        ['key', 'lat', 'lon', 'windSpd', 'pressure']]
    HalfVal = len(PredictDataFrame) // 2
    HalfDataFrame['key'] = le_x.fit_transform(HalfDataFrame['key'])
    HalfDataFrame = PredictDataFrame.iloc[:HalfVal]
    HalfDataFrame['key'] = le_x.fit_transform(HalfDataFrame['key'])
    # Run stat filler function to fill unknown variables for prediction
    stormAvgDf = HalfDataFrame
    # Using above values input into DF and run prediction models
    # attributes = ['lat', 'lon', 'windSpd', 'pressure']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        for index, row in stormAvgDf.iterrows():
            # Create new empty array for each row
            predictions = []
            for label in labels:
                # Determine column index to be dropped
                if label == 'lat':
                    idx = 1
                elif label == 'lon':
                    idx = 2
                elif label == 'windSpd':
                    idx = 3
                elif label == 'pressure':
                    idx = 4
                # Convert DF row to numpy array
                newX = row.to_numpy()
                # Drop label column index
                newX = np.delete(newX, idx)
                # Reshape numpy array
                newX = newX.reshape(1, -1)
                # Select variable to predict
                input_model = open('{}.pickle'.format(label), 'rb')
                model = pickle.load(input_model)
                try:
                    prediction = model.predict(newX)
                    # append prediction for label to list
                    predictions.append(prediction[0])
                    # Once list is populated with 4 label predictions add to csv row
                    if len(predictions) == 4:
                        with open('storm{}_prediction.csv'.format(i), 'a') as f:
                            w = writer(f)
                            w.writerow(predictions)
                            f.close()
                except ValueError:
                    print('Prediction Error')
                # add row to predictions csv file
    stormAvgDf.to_csv('storm{}.csv'.format(i))
    predictionDF = pd.read_csv('storm{}_prediction.csv'.format(i))
    predictionDF.dropna(inplace=True)
    predictionDF.to_csv('storm{}_prediction.csv'.format(i))

# Load the storm0 and storm0_prediction CSV files into pandas DataFrames
df_storm0 = pd.read_csv("storm0.csv")
df_storm0_pred = pd.read_csv("storm0_prediction.csv")


# Extract latitude and longitude columns from the DataFrames
latitudes_storm0 = df_storm0['lat']
longitudes_storm0 = df_storm0['lon']
windSpd_storm0 = df_storm0['windSpd']
pressure_storm0 = df_storm0['pressure']
latitudes_storm0_pred = df_storm0_pred['lat']
longitudes_storm0_pred = df_storm0_pred['lon']
windSpd_storm0_pred = df_storm0_pred['windSpd']
pressure_storm0_pred = df_storm0_pred['pressure']

# Create a map object centered at the mean of the latitudes and longitudes of storm0 CSV
m = folium.Map(location=[latitudes_storm0.mean(), longitudes_storm0.mean()], zoom_start=10)

# Add a marker for each point of storm0 CSV to the map
for lat, lon, windSpd, pressure in zip(latitudes_storm0, longitudes_storm0, windSpd_storm0, pressure_storm0):
    popup_text = f"Wind speed: {windSpd} km/h, Pressure: {pressure} hPa"
    folium.Marker(location=[lat, lon], icon=folium.Icon(color='blue'), popup=popup_text).add_to(m)

# Add a marker for each point of storm0_prediction CSV to the map
for lat, lon, windSpd, pressure in zip(latitudes_storm0_pred, longitudes_storm0_pred, windSpd_storm0_pred,
                                       pressure_storm0_pred):
    popup_text = f"Wind speed: {windSpd} km/h, Pressure: {pressure} hPa"
    folium.Marker(location=[lat, lon], icon=folium.Icon(color='red'), popup=popup_text).add_to(m)

points_storm0_pred = list(zip(latitudes_storm0_pred, longitudes_storm0_pred))
folium.PolyLine(points_storm0_pred, color='red').add_to(m)
points_storm0 = list(zip(latitudes_storm0, longitudes_storm0))
folium.PolyLine(points_storm0, color='blue').add_to(m)

m.save("Storms.html")

file_path = "storm0_prediction.csv"

file_paths = ["storm0.csv", "storm0_prediction.csv"]

for file_path in file_paths:
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File {file_path} deleted.")
    else:
        print(f"File {file_path} not found.")

webbrowser.open_new_tab("Storms.html")

plt.scatter(df_ml["lat"], df_ml["lon"])
plt.xlabel("Real Latitude")
plt.ylabel("Real Longitude")
plt.show()
plt.scatter(df_ml["windSpd"], df_ml["pressure"])
plt.xlabel(" Real Wind Speed")
plt.ylabel("Real Pressure")
plt.show()

plt.scatter(df_storm0_pred["lat"], df_storm0_pred["lon"])
plt.xlabel("Predicted Latitude")
plt.ylabel("Predicted Longitude")
plt.show()

plt.scatter(df_storm0_pred["windSpd"], df_storm0_pred["pressure"])
plt.xlabel("Predicted Wind Speed")
plt.ylabel("Predicted Pressure")
plt.show()

print(accuracyList)
# Define the labels for the x-axis and the colors for each data series
labels = ["Latitude", "Longitude", "Wind Speed", "Pressure"]
colors = ["blue", "orange", "green", "red"]

# Plot the bars
plt.bar(np.arange(len(accuracyList)), accuracyList, color=colors)

# Add labels for the x-axis and y-axis
plt.xticks(np.arange(len(accuracyList)), labels)
plt.ylabel("Accuracy")

# Set the y-axis limits to 0 and 100
plt.ylim([0, 1])

# Show the plot
plt.show()
