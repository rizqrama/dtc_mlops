# %% Load Libraries
import duckdb as db
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error


# %% Load Data
def get_taxi_data(year: int, month: int) -> pd.DataFrame:
    """
    Fetch NYC yellow taxi data for a specific year and month
    and calculate trip duration

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

    # Create table name based on month
    table_name = f"taxi_data_{year}_{month_str}"

    # Create table with the data and duration calculation
    return con.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT *,
            EXTRACT(
                EPOCH FROM (tpep_dropoff_datetime - tpep_pickup_datetime)
                ) / 60 AS duration
        FROM read_parquet('{url}')
    """)


# %%
# Connect to DuckDB
con = db.connect("taxi_data.duckdb")
# %%
# Get data for January through March 2023
months_to_fetch = [1, 2, 3]
taxi_data = {}

for month in months_to_fetch:
    taxi_data[f'df_{month:02d}'] = get_taxi_data(2023, month)

# %%
df_jan = con.sql("SELECT * FROM taxi_data_2023_01").df()
df_feb = con.sql("SELECT * FROM taxi_data_2023_02").df()

# %%
df_jan.duration.describe()


# %%
def df_prep(dataset):
    df = dataset[(dataset.duration >= 1) & (dataset.duration <= 60)]
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    return df


df_train = df_prep(df_jan)
df_val = df_prep(df_feb)

len(df_train), len(df_val)

# %%
round(len(df_train) / len(df_jan) * 100, 2)

# %%
df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
df_val['PU_DO'] = df_val['PULocationID'] + '_' + df_val['DOLocationID']

# %%
categorical = categorical = ['PULocationID', 'DOLocationID']
numerical = ['trip_distance']

dv = DictVectorizer()

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)

# %%
target = 'duration'
y_train = df_train[target].values
y_val = df_val[target].values

# %%
print(f"Number of features: {X_train.shape[1]}")

# %%
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)

root_mean_squared_error(y_train, y_pred)

# %%
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_val)

root_mean_squared_error(y_val, y_pred)

# %%
lr = Lasso(0.01)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_val)

root_mean_squared_error(y_val, y_pred)

# %%
sns.histplot(y_pred, kde=True, stat="density", label='prediction')
sns.histplot(y_val, kde=True, stat="density", label='actual')

plt.legend()

# %%
