import pickle
import pandas as pd
import numpy as np

# load model
model = pickle.load(open("../model/house_model.pkl", "rb"))
columns = pickle.load(open("../model/columns.pkl", "rb"))

# example input (you can change values)
input_data = {
    'bedrooms': 3,
    'bathrooms': 2,
    'sqft_living': 1800,
    'sqft_lot': 5000,
    'floors': 1,
    'waterfront': 0,
    'view': 0,
    'condition': 3,
    'sqft_above': 1500,
    'sqft_basement': 300,
    'yr_built': 2000
}

# convert to dataframe
input_df = pd.DataFrame([input_data])

# match training columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# predict (log scale)
log_pred = model.predict(input_df)

# convert back to real price
real_price = np.exp(log_pred)

print("Predicted Price:", real_price[0])