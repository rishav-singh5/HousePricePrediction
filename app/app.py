from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__, template_folder='templates', static_folder='static')


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model", "house_model.pkl")
columns_path = os.path.join(BASE_DIR, "model", "columns.pkl")


model = pickle.load(open(model_path, "rb"))
columns = pickle.load(open(columns_path, "rb"))

@app.route('/')
def home():
    return render_template('index.html', form_data={})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # get values from form
        input_data = {
            'bedrooms': float(request.form['bedrooms']),
            'bathrooms': float(request.form['bathrooms']),
            'sqft_living': float(request.form['sqft_living']),
            'sqft_lot': float(request.form['sqft_lot']),
            'floors': float(request.form['floors']),
            'waterfront': float(request.form['waterfront']),
            'view': float(request.form['view']),
            'condition': float(request.form['condition']),
            'sqft_above': float(request.form['sqft_above']),
            'sqft_basement': float(request.form['sqft_basement']),
            'yr_built': float(request.form['yr_built'])
        }

        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=columns, fill_value=0)

        log_pred = model.predict(input_df)
        real_price = np.exp(log_pred)

        return render_template(
            'index.html',
            prediction=round(real_price[0], 2),
            form_data=input_data
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction="Error: " + str(e),
            form_data={}
        )

if __name__ == "__main__":
    app.run()
