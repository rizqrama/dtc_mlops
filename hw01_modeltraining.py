# %% Load Libraries
import duckdb as db
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error


# %% Load Data
def get_taxi_data(year: int, month: int) -> pd.DataFrame:
    """
    Fetch NYC yellow taxi data for a specific year and month and calculate trip duration

    Args:
        year: The year of the data (e.g., 2023)
        month: The month of the data (1-12)

    Returns:
        pd.DataFrame: Taxi trip data for the specified period
        with duration calculated
    """
    month_str = str(month).zfill(2)  # Convert 1 to '01', 2 to '02', etc.
    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    url = f"{base_url}/yellow_tripdata_{year}-{month_str}.parquet"

    return con.execute(f"""
        SELECT *,
            EXTRACT(
                EPOCH FROM (tpep_dropoff_datetime - tpep_pickup_datetime)
                ) / 60 AS duration
        FROM read_parquet('{url}')
    """).df()


# Connect to DuckDB
con = db.connect("taxi_data.duckdb")

# Get data for January through March 2023
months_to_fetch = [1, 2, 3]
taxi_data = {}

for month in months_to_fetch:
    taxi_data[f'df_{month:02d}'] = get_taxi_data(2023, month)

# %%
df_jan = taxi_data['df_01']
df_feb = taxi_data['df_02']

# %%
df_jan.shape

# %%
df_jan.duration.describe()
# %%
df_jan_clean = con.execute("""
    FROM df_jan
    WHERE
        duration >= 1 AND duration <= 60
""").df()

df_jan_clean.shape
# %%
round(len(df_jan_clean) / len(df_jan) * 100, 2)

# %%
