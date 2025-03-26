import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

geolocator = Nominatim(user_agent="geoapiExercises")

def get_lat_lon(location):
    """Fetches latitude and longitude for a given location string."""
    try:
        geo = geolocator.geocode(location, timeout=10)
        if geo:
            return geo.latitude, geo.longitude
    except GeocoderTimedOut:
        time.sleep(1)
    return None, None

def enrich_location_data(df, location_column='location'):
    """Adds latitude and longitude columns to the dataset."""
    df[['latitude', 'longitude']] = df[location_column].apply(
        lambda loc: pd.Series(get_lat_lon(loc))
    )
    return df