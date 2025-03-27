from c_data_extract_combine.ETL import dataload

def load_raw_data(flags: dict):
    if not flags.get("useprecombineddata"):
        return dataload()  # fake_df, true_df, test_df
    return None, None, None