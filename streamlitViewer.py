import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pydeck as pdk
from collections import Counter
import googlemaps
import random
import plotly.express as px

# link to the data stored in google cloud
link= "https://storage.googleapis.com/all_divvy_data_with_weather/all_data_shortened_updated.csv"

# function for loading data. This function returns a random subset of data from the link with the n rows, where n is the
# argument passed to the function
@st.cache(persist=True, allow_output_mutation=True)
def load_data(nrows):
    num_lines = 3000000

    skip_idx = random.sample(range(1, num_lines), num_lines - nrows)
    data = pd.read_csv(link, skiprows=skip_idx)
    data = data.dropna()
    return data

st.sidebar.title("Filters")
st.sidebar.write("(Try refreshing if error is thrown initially)")
num_rows = st.sidebar.number_input('How many datapoints would you like?', value=100000)

df = load_data(num_rows).reset_index()


# converting to datetime
df['started_at'] = pd.to_datetime(df['started_at'])
df['ended_at'] = pd.to_datetime(df['ended_at'])
# title
st.title('Divvy Data Visualizer')
st.write(
    """
    Author: Dylan Andrews\n
    Github: https://github.com/dandrews19\n
    LinkedIn: https://www.linkedin.com/in/dylan-m-andrews/\n
    Email: dmandrew@usc.edu\n
    A tool that allows the user to easily visualize publicly accessible Divvy data alongside factors including weather
    and time of year. 
    """)


# creating columns for visualizing starting and ending locations
col1, col2 = st.columns(2)


# setting up filters
from_date = st.sidebar.date_input("From Day", value=df['started_at'][0], min_value=df['started_at'][0],
                                  max_value=df['ended_at'][len(df) - 1])


to_date = st.sidebar.date_input("To Day", value=df['ended_at'][len(df) - 1], min_value=from_date, max_value=df['ended_at'][len(df) - 1])

from_time = st.sidebar.time_input("From Time", value= datetime.time(0, 0, 0))

to_time = st.sidebar.time_input("To Time", value= datetime.time(23, 59, 59))

from_temperature = st.sidebar.slider(label="Minimum Temperature (F)", min_value=-50, max_value=130, value=-50)

to_temperature = st.sidebar.slider(label="Maximum Temperature (F)", min_value=from_temperature, max_value=130, value=130)

from_windspeed = st.sidebar.slider(label="Minimum Wind Speed (MPH)", min_value=0, max_value=100, value=0)

to_windspeed = st.sidebar.slider(label="Maximum Wind Speed (MPH)", min_value=from_windspeed, max_value=100, value=100)

wind_direction = st.sidebar.multiselect(label= 'Wind Directions', options=['All', 'NE','N', 'NW', 'W', 'SW', 'S', 'SE','E'], default='All')

from_duration = st.sidebar.number_input('Minimum Trip Duration (Minutes)', min_value=0, value=0)

to_duration = st.sidebar.number_input('Maximum Trip Duration (Minutes)', min_value=from_duration, value=99999999)
# changing the index to the ended_at column in order to sort by time
df['ended_at_index'] = df['ended_at']
df.set_index('ended_at_index',inplace=True)
df_new = df.between_time(start_time=from_time, end_time=to_time)
# condition if wind direction is specified
if 'All' in wind_direction:
    wind_direction = ['N', 'NW', 'NE', 'E', 'W', 'S', 'SE', 'SW']

# filtering based on user inputs from above
df_new = df_new[(df_new['started_at'].dt.date >= from_date) & (df_new['ended_at'].dt.date <= to_date) &
                (df_new['tmpf'] >= from_temperature) & (df_new['tmpf'] <= to_temperature) &
                (df_new['wind_speed_mph'] >= from_windspeed) & (df_new['wind_speed_mph'] <= to_windspeed) &
                (df_new['trip_duration_minutes'] >= from_duration) & (df_new['trip_duration_minutes'] <= to_duration) &
                (df_new['wind_direction_string']).isin(wind_direction)]



# dropping null values to avoid errors in displaying data
df_new = df_new.dropna()

# creating a bootstrapped sample of data to display using Pydeck to increase efficiency
if len(df_new) > 10000:
    sampled_df = df_new.sample(10000, replace=True)
else:
    sampled_df = df_new





st.title("Most Popular Routes")


# displaying the most popular starting and ending locations and their frequency of use with Pydeck
col1.write(
    """ 
    **Most Popular Starting Stations**\n
    Click and drag map to load properly
    """)
col1.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v9",
        initial_view_state={
            "latitude": 41.878166,
            "longitude": -87.631929,
            "zoom": 12,
            "pitch": 100.5,
            "bearing":-27.36,
        },
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=sampled_df,
                get_position=["start_longitude", "start_latitude"],
                radius=100,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
                coverage = 1,
            ),
        ]
    ))

col2.write(
    """ 
    **Most Popular Ending Stations**\n
    Click and drag map to load properly
    """)
col2.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v9",
        initial_view_state={
            "latitude": 41.878166,
            "longitude": -87.631929,
            "zoom": 12,
            "pitch": 100.5,
            "bearing":-27.36,
        },
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=sampled_df,
                get_position=["end_longitude", "end_latitude"],
                radius=100,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
                coverage = 1,
            ),
        ]
    ))


# google maps portion

API_KEY = 'AIzaSyAXSrhT8lBW6mJbBoW6LRXVyPu1RUwOA1g'
# connecting to google maps api
gmaps = googlemaps.Client(key = API_KEY)

# a function that takes in a description of the path taken, and whether or not we want the start or the end coordinates
# and returns the latitude and longitude of the specified start or end as a tuple
def get_coordinates(start_to_end, location):
    index = list(df_new['start_to_end']).index(start_to_end)
    latitude = df_new[location + '_latitude'][index]
    longitude = df_new[location + '_longitude'][index]

    return (latitude, longitude)

# using the counter moudule to find the most popular paths taken
start_to_end_counter = Counter(df_new['start_to_end']).most_common(22)


name_list = []
path_list = []
# looping thru the 22 most popular routes
for i in start_to_end_counter:
    # calling the get coordinates function to get the longitude and latitude of the start and end stations
    start_coordinates = get_coordinates(i[0], 'start')
    end_coordinates = get_coordinates(i[0], 'end')
    # calling the google maps api to get directions from the start to the end station, using "bicycling" as a mode
    # of transportation
    directions = gmaps.directions(origin=start_coordinates, destination=end_coordinates, mode="bicycling")

    # creating a dictionary to describe each step of the path
    start_dict = {'lat': start_coordinates[0], 'lng': start_coordinates[1]}
    # creating a list that stores each step
    list_of_directions = [start_dict]
    # going thru each step and appending the end location based on data received from google maps api
    for c in directions[0]['legs'][0]['steps']:
        list_of_directions.append(c['end_location'])

    path = []
    # making another list to store path so it is readable by Pydeck
    for h in list_of_directions:
        path.append([h['lng'], h['lat']])

    name_list.append(i[0])
    path_list.append(path)

# a set of distinct colors to properly display paths with no overlapping colors
colors = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (67, 99, 216), (245, 130, 49), (145, 30, 180), (70, 240, 240),
          (240, 50, 230), (188, 246, 12), (250, 190, 190), (0, 128, 128), (230, 190, 255), (154, 99, 36),
          (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 216, 177), (0, 0, 117), (128, 128, 128),
          (255, 255, 255), (0, 0, 0)]
# if there's less than 22 paths filtered, this line ensures it will still display a map
colors = colors[:len(name_list)]

# creating a dataframe containing path information tol be read by Pydeck
paths_df = pd.DataFrame({"path": path_list, "names": name_list, 'color': colors})

# using Pydeck to visualize the 22 most popular paths
view_state = pdk.ViewState(latitude=41.878166, longitude=-87.631929, zoom=11)

layer = pdk.Layer(
    type="PathLayer",
    data=paths_df,
    pickable=True,
    width_scale=20,
    width_min_pixels=2,
    get_path="path",
    get_width=2,
    get_color='color'
)

st.write(pdk.Deck(map_style="mapbox://styles/mapbox/dark-v9", layers=[layer], initial_view_state=view_state, tooltip={"text": "{names}"}))


# displaying summary statistics
st.title("Summary Statistics")

st.write("Rides in sample matching criteria: ", len(df_new))
st.write("Median Ride Duration: ", np.median(df_new['trip_duration_minutes']), " minutes")
st.write("Average Trip Duration: ", np.mean(df_new['trip_duration_minutes']), " minutes")
st.write("Average Temperature: ", np.mean(df_new['tmpf']), " degrees F")
# most popular starting and ending stations

starting_station = Counter(df_new['start_station_name']).most_common(1)
ending_station = Counter(df_new['end_station_name']).most_common(1)
st.write("Most Popular Starting Station: ", starting_station[0][0], ", ", starting_station[0][1], "rides in sample started here")
st.write("Most Popular Ending Station: ", ending_station[0][0], ", ", ending_station[0][1], "rides in sample ended here")


# displaying correlation
st.title("What Variables are Correlated With Ride Duration?")
st.write("(with given filters)")
options = ["Temperature (F)", "Wind Speed (MPH)", "Visibility", "Humidity", "Hour and Minute"]

second_variable= st.selectbox("Choose Second Variable", options=options, index=0)
method = st.selectbox("Choose Method", options= ['Pearson', 'Kendall', 'Spearman'], index=0)
method = method.lower()
name_to_variable_dict = {
    "Ride Duration": "trip_duration_minutes",
    "Temperature (F)": "tmpf",
    "Wind Speed (MPH)": "wind_speed_mph",
    "Visibility": "visibility",
    "Humidity": "humidity%",
    "Hour and Minute": "hour and minute"
}

if second_variable == "Hour and Minute":
    correlation_col = (df_new['started_at'].dt.hour * 360) + (df_new['started_at'].dt.minute * 60) + (df_new['started_at'].dt.second)
    column2 = pd.to_datetime(df_new['started_at'].dt.time.astype(str))
else:
    column2 = df_new[name_to_variable_dict[second_variable]]
    correlation_col = df_new[name_to_variable_dict[second_variable]]


correlation = df_new['trip_duration_minutes'].corr(correlation_col, method=method)

fig = px.scatter(y=df_new['trip_duration_minutes'], x=column2, trendline="ols", title= "Correlation between " + second_variable + " and Trip Duration (minutes)", labels={"x": second_variable, "y": "Trip Duration (minutes)"
})


st.write(fig)

st.write("Correlation: ", correlation)






