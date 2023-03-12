import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

# Read the data from a CSV file
data = pd.read_csv('StormInfo.csv', header=0, names=['index', 'key', 'name', 'dateTime', 'stmType', 'lat',
                                                     'lon', 'windSpd', 'pressure'])


# Define a function to plot the hurricane data for a specific year
def plot_hurricanes(year):
    plt.clf()

    # Filter the data by year
    year_data = data[data['dateTime'].str.startswith(str(year))]

    # Group the data by hurricane name
    grouped_data = year_data.groupby('key')

    # Create a Basemap instance
    mapData = Basemap(projection='merc', resolution='l', llcrnrlat=20, urcrnrlat=50, llcrnrlon=-130, urcrnrlon=-60)

    # Draw the map boundaries and coastlines
    mapData.drawmapboundary(fill_color='aqua')
    mapData.fillcontinents(color='coral', lake_color='aqua')
    mapData.drawcoastlines()

    # Loop through the hurricane groups and plot the data for each hurricane
    for name, group in grouped_data:
        # Extract the longitude and latitude columns for the current hurricane
        lon = group['lon'].tolist()
        lat = group['lat'].tolist()

        # Convert the latitude and longitude coordinates to map coordinates
        x, y = mapData(lon, lat)

        # Plot the data points with lines connecting them
        mapData.plot(x, y, marker='o', linewidth=2, markersize=10)

    # Show the plot
    plt.show()


# Define a function to handle the selection of a year from the dropdown menu
def handle_selection():
    # Create a GUI window
    window = tk.Toplevel()
    window.title("Select Year")

    # Create a label and a dropdown menu
    label = ttk.Label(window, text="Please select a year:")
    label.pack(pady=10)

    years = list(range(1969, 2017))
    year_var = tk.StringVar(window)
    year_var.set(str(years[-1]))  # set the default value to the last year in the list
    dropdown = ttk.OptionMenu(window, year_var, *years)
    dropdown.pack()

    # Create a button to plot the hurricanes for the selected year
    plot_button = ttk.Button(window, text="Plot Hurricanes", command=lambda: plot_hurricanes(int(year_var.get())))
    plot_button.pack(pady=10)

    # Show the window
    window.mainloop()
