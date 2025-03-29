import pandas as pd

# load combined_data_pre_location_enriched.zip

df = pd.read_csv('data/combined_data_pre_location_enriched.zip',
                    low_memory=False)


df