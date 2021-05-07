import pandas as pd
import requests
import json
import time
import numpy as np

dataframe_path = "dataframes//Airbnb_Norway.csv"
web_app_link = "https://airbnb-norway-predictor.herokuapp.com/"

df_data = pd.read_csv(dataframe_path).drop(["title", "url", "price"], axis=1)

# Check post response with one input
time_start = time.time()
sample_a = df_data.loc[[0]].to_dict("records")
resp = requests.post(f"{web_app_link}/predict", data=json.dumps({"inputs": sample_a}))
print(json.loads(resp.text))
print(f"Post single request is done! Time elapsed: {time.time()-time_start} seconds")

# Check post response with batch input
time_start = time.time()
sample_b = df_data.loc[[0, 2]].to_dict("records")
resp = requests.post(f"{web_app_link}/predict", data=json.dumps({"inputs": sample_b}))
print(json.loads(resp.text))
print(f"Post batch request is done! Time elapsed: {time.time()-time_start} seconds")

# Check get response
time_start = time.time()
resp = requests.get(f"{web_app_link}/history")
print(json.loads(resp.text))
print(f"Get request is done! Time elapsed: {time.time()-time_start} seconds")
