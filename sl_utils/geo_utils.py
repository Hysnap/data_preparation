# sl_utils/geo_utils.py — Extended and modularized

import os
import time
import pandas as pd
from flashtext import KeywordProcessor
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import GoogleV3
from google.cloud import api_keys_v2
from google.cloud.api_keys_v2 import Key
from tqdm.auto import tqdm
import spacy
from sl_utils.logger import datapipeline_logger as logger, log_function_call


# Attach tqdm to pandas
tqdm.pandas()

# Load SpaCy NLP
nlp = spacy.load("en_core_web_sm")

# Initialize GoogleV3 geolocator
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
geolocator = GoogleV3(api_key=api_key)


@log_function_call(logger)
def build_world_location_set(worldcities_df, usstates_df, countrycontinent_df):
    loc_set = set()

    for series in [
        worldcities_df['city_ascii'],
        worldcities_df['country'],
        worldcities_df['iso2'],
        worldcities_df['iso3'],
        usstates_df['Place Name'],
        usstates_df['State Capital'],
        countrycontinent_df['country'],
        countrycontinent_df['sub_region'],
        countrycontinent_df['continent'],
    ]:
        loc_set.update(
            s.lower().strip()
            for s in series.dropna()
            if isinstance(s, str)
        )

    return loc_set


@log_function_call(logger)
def init_keyword_processor(location_set):
    kp = KeywordProcessor()
    for loc in location_set:
        kp.add_keyword(loc)
    return kp


@log_function_call(logger)
def extract_locations_column(df,
                             text_col,
                             keyword_processor,
                             nlp,
                             useprelocationenricheddata=False,
                             valid_locations_set=None
                             ):
    df['locationsfromarticle'] = df[text_col].progress_apply(
        lambda text: extract_locations(str(text),
                                       keyword_processor,
                                       nlp,
                                       valid_locations_set
                                       )
    )
    return df


@log_function_call(logger)
def extract_locations(text, keyword_processor, nlp, valid_locations_set=None):
    text = str(text).lower()
    doc = nlp(text)
    nlp_locations = {ent.text.lower()
                     for ent in doc.ents
                     if ent.label_ == "GPE"}
    keywords = set(keyword_processor.extract_keywords(text))

    # combine but filer through known locations
    combined = keywords | nlp_locations
    if valid_locations_set:
        filtered = [loc for loc in combined if loc in valid_locations_set]
    else:
        filtered = list(combined)
        
    return filtered


@log_function_call(logger)
def enrich_unique_locations(unique_locations_df, worldcities_df, usstates_df, continent_df):
    enriched_rows = []

    worldcities_df = worldcities_df.rename(columns={
        col: col.strip() for col in worldcities_df.columns  # Strip any weird whitespace
        })

    expected_cols = ['city_ascii', 'admin_name', 'country', 'iso2', 'iso3', 'lat', 'lng']
    missing = [col for col in expected_cols if col not in worldcities_df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in worldcities_df: {missing}")

    city_lookup = worldcities_df[['city_ascii', 'admin_name', 'country', 'iso2', 'iso3', 'lat', 'lng']].dropna()
    city_lookup.columns = ['city', 'state', 'country', 'iso2', 'iso3', 'latitude', 'longitude']
    city_lookup = city_lookup.map(lambda x: x.lower() if isinstance(x, str) else x)

    usstates_df = usstates_df.rename(columns={
        col: col.strip() for col in usstates_df.columns  # Strip any weird whitespace
        })

    # Include both state names and capitals for matching

    expected_cols = ["Place Name", "State Capital", "Country", "Latitude", "Longitude"]
    missing = [col for col in expected_cols if col not in usstates_df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in usstates_df: {missing}")
    
    usstates_df = usstates_df.rename(columns={"Place Name": "state",
                                              "State Capital": "capital",
                                              "Country": "country",
                                              "Latitude": "latitude",
                                              "Longitude": "longitude"})
    state_lookup = usstates_df[['state', 'country', 'capital', 'latitude', 'longitude']].dropna().map(lambda x: x.lower() if isinstance(x, str) else x)

    continent_df = continent_df.rename(columns={
        col: col.strip() for col in continent_df.columns  # Strip any weird whitespace
        })

    expected_cols = ['country', 'sub_region', 'continent', 'code_2', 'code_3']
    missing = [col for col in expected_cols if col not in continent_df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in worldcities_df: {missing}")

    continent_lookup = continent_df[['country', 'sub_region', 'continent', 'code_2', 'code_3']]
    continent_lookup.columns = ['country', 'subcontinent', 'continent', 'iso2', 'iso3']
    continent_lookup = continent_lookup.dropna().map(lambda x: x.lower() if isinstance(x, str) else x)

    for _, row in unique_locations_df.iterrows():
        loc = row['location'].lower()
        result = {
            'location': loc,
            'ignore': row.get('ignore', 1),
            'city': None,
            'state': None,
            'country': None,
            'subcontinent': None,
            'continent': None,
            'latitude': None,
            'longitude': None,
            'iso2': None,
            'iso3': None
        }

        city_match = city_lookup[city_lookup['city'] == loc]
        if not city_match.empty:
            best = city_match.iloc[0]
            result.update({
                'city': best['city'],
                'state': best['state'],
                'country': best['country'],
                'latitude': best['latitude'],
                'longitude': best['longitude'],
                'iso2': best['iso2'],
                'iso3': best['iso3']
            })
            region_match = continent_lookup[continent_lookup['country'] == best['country']]
            if not region_match.empty:
                result.update(region_match.iloc[0].to_dict())

        else:
            # Match by state name or capital
            state_match = state_lookup[(state_lookup['state'] == loc) | (state_lookup['capital'] == loc)]
            if not state_match.empty:
                best = state_match.iloc[0]
                result.update({
                    'city': best['capital'],
                    'state': best['state'],
                    'country': best['country']
                })
                region_match = continent_lookup[continent_lookup['country'] == best['country']]
                if not region_match.empty:
                    result.update(region_match.iloc[0].to_dict())
                subset = city_lookup[city_lookup['state'] == best['state']]
                if not subset.empty:
                    result['latitude'] = subset['latitude'].mean()
                    result['longitude'] = subset['longitude'].mean()
            else:
                # Match by country name
                country_match = continent_lookup[continent_lookup['country'] == loc]
                if not country_match.empty:
                    best = country_match.iloc[0]
                    result.update({
                        'city': 'Country level only',
                        'country': best['country'],
                        'subcontinent': best['subcontinent'],
                        'continent': best['continent'],
                        'iso2': best['iso2'],
                        'iso3': best['iso3']
                    })
                    subset = city_lookup[city_lookup['country'] == best['country']]
                    if not subset.empty:
                        result['latitude'] = subset['latitude'].mean()
                        result['longitude'] = subset['longitude'].mean()
                else:
                    # Match by ISO2 code
                    iso2_match = continent_lookup[continent_lookup['iso2'] == loc]
                    if not iso2_match.empty:
                        best = iso2_match.iloc[0]
                        result.update({
                            'city': 'Country level only',
                            'country': best['country'],
                            'subcontinent': best['subcontinent'],
                            'continent': best['continent'],
                            'iso2': best['iso2'],
                            'iso3': best['iso3']
                        })
                        subset = city_lookup[city_lookup['iso2'] == best['iso2']]
                        if not subset.empty:
                            result['latitude'] = subset['latitude'].mean()
                            result['longitude'] = subset['longitude'].mean()
                    else:
                        # Match by ISO3 code
                        iso3_match = continent_lookup[continent_lookup['iso3'] == loc]
                        if not iso3_match.empty:
                            best = iso3_match.iloc[0]
                            result.update({
                                'city': 'Country level only',
                                'country': best['country'],
                                'subcontinent': best['subcontinent'],
                                'continent': best['continent'],
                                'iso2': best['iso2'],
                                'iso3': best['iso3']
                            })
                            subset = city_lookup[city_lookup['iso3'] == best['iso3']]
                            if not subset.empty:
                                result['latitude'] = subset['latitude'].mean()
                                result['longitude'] = subset['longitude'].mean()

        enriched_rows.append(result)

    return pd.DataFrame(enriched_rows)


@log_function_call(logger)
def get_geolocation_info(location):
    usetimedelay = True
    try:
        location_info = geolocator.geocode(location, timeout=10)
        if location_info:
            if usetimedelay:
                time.sleep(1)
            return {
                'latitude': location_info.latitude,
                'longitude': location_info.longitude,
                'address': location_info.address
            }
        else:
            if usetimedelay:
                time.sleep(1)
            return {
                'latitude': None,
                'longitude': None,
                'address': None
            }
    except GeocoderTimedOut:
        print(location)
        if usetimedelay:
            time.sleep(1)
        return {
            'latitude': None,
            'longitude': None,
            'address': None
        }
    except Exception as e:
        print(f"Geocoding error: {e}")
        if usetimedelay:
            time.sleep(1)
        return {
            'latitude': None,
            'longitude': None,
            'address': None
        }


@log_function_call(logger)
# Function to extract geolocation details
def extract_geolocation_details(address):
    if address:
        address_parts = address.split(', ')
        country = address_parts[-1] if len(address_parts) > 0 else None
        state = address_parts[-2] if len(address_parts) > 1 else None
        continent = address_parts[-3] if len(address_parts) > 2 else None
        return continent, country, state
    else:
        return None, None, None
