from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.preprocessing import StandardScaler
import pickle
from elmz import elm  # Ensure this matches the actual module name and path
from fwi import FWICLASS
import requests

app = Flask(__name__)

# Custom Unpickler to handle 'elm' attribute
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "elmz"
        return super().find_class(module, name)

# Load model using Custom Unpickler
with open('elmV2.pkl', 'rb') as file:
    model = CustomUnpickler(file).load()

# Database connection and other functions
def get_db_connection():
    return psycopg2.connect(user="ffp-indonesia",
                            password="forestfire123",
                            host="34.101.120.221",
                            port="5432",
                            database="ffp-indonesia",
                            cursor_factory=RealDictCursor)

def fetch_data(selected_provinsi, selected_kabupaten):
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT provinsi, kabupaten, date, windspeed, humidity, rainfall, temperature FROM posts WHERE provinsi = %s AND kabupaten = %s", (selected_provinsi, selected_kabupaten))
    data = cursor.fetchall()
    cursor.close()
    connection.close()
    return data

def initialize_scalers(data):
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    
    scalers = {}
    
    X_humidity = df[['temperature', 'rainfall', 'windspeed']].values
    y_humidity = df[['humidity']].values
    scalers['humidity'] = StandardScaler().fit(X_humidity)
    scalers['target_humidity'] = StandardScaler().fit(y_humidity)

    X_temp = df[['humidity', 'rainfall', 'windspeed']].values
    y_temp = df[['temperature']].values
    scalers['temp'] = StandardScaler().fit(X_temp)
    scalers['target_temp'] = StandardScaler().fit(y_temp)

    X_rain = df[['temperature', 'humidity', 'windspeed']].values
    y_rain = df[['rainfall']].values
    scalers['rain'] = StandardScaler().fit(X_rain)
    scalers['target_rain'] = StandardScaler().fit(y_rain)

    X_wind = df[['humidity', 'rainfall', 'temperature']].values
    y_wind = df[['windspeed']].values
    scalers['wind'] = StandardScaler().fit(X_wind)
    scalers['target_wind'] = StandardScaler().fit(y_wind)

    return scalers, df

@app.route('/forecast/temperature', methods=['GET'])
def forecast_temperature():
    selected_provinsi = request.args.get('selectedProvinsi')
    selected_kabupaten = request.args.get('selectedKabupaten')
    data = fetch_data(selected_provinsi, selected_kabupaten)
    scalers, df = initialize_scalers(data)
    
    num_days = 7
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_days)

    sampled_humidity = np.random.choice(df['humidity'], num_days, replace=True)
    sampled_rainfall = np.random.choice(df['rainfall'], num_days, replace=True)
    sampled_windspeed = np.random.choice(df['windspeed'], num_days, replace=True)

    forecast_features = pd.DataFrame({
        'date': future_dates,
        'humidity': sampled_humidity,
        'rainfall': sampled_rainfall,
        'windspeed': sampled_windspeed
    })

    forecast_features_scaled = scalers['temp'].transform(forecast_features[['humidity', 'rainfall', 'windspeed']])
    predicted_forecast_temperature = model.predict(forecast_features_scaled)
    predicted_forecast_temperature_denormalized = scalers['target_temp'].inverse_transform(predicted_forecast_temperature)

    forecast_features['temperature'] = predicted_forecast_temperature_denormalized
    result = forecast_features[['date', 'temperature']].round(1)

    return jsonify(result.to_dict(orient='records'))

@app.route('/forecast/humidity', methods=['GET'])
def forecast_humidity():
    selected_provinsi = request.args.get('selectedProvinsi')
    selected_kabupaten = request.args.get('selectedKabupaten')
    data = fetch_data(selected_provinsi, selected_kabupaten)
    scalers, df = initialize_scalers(data)
    
    num_days = 7
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_days)

    sampled_temperature = np.random.choice(df['temperature'], num_days, replace=True)
    sampled_rainfall = np.random.choice(df['rainfall'], num_days, replace=True)
    sampled_windspeed = np.random.choice(df['windspeed'], num_days, replace=True)

    forecast_features = pd.DataFrame({
        'date': future_dates,
        'temperature': sampled_temperature,
        'rainfall': sampled_rainfall,
        'windspeed': sampled_windspeed
    })

    forecast_features_scaled = scalers['humidity'].transform(forecast_features[['temperature', 'rainfall', 'windspeed']])
    predicted_forecast_humidity = model.predict(forecast_features_scaled)
    predicted_forecast_humidity_denormalized = scalers['target_humidity'].inverse_transform(predicted_forecast_humidity)

    forecast_features['humidity'] = predicted_forecast_humidity_denormalized
    result = forecast_features[['date', 'humidity']].round(1)

    return jsonify(result.to_dict(orient='records'))

@app.route('/forecast/rainfall', methods=['GET'])
def forecast_rainfall():
    selected_provinsi = request.args.get('selectedProvinsi')
    selected_kabupaten = request.args.get('selectedKabupaten')
    data = fetch_data(selected_provinsi, selected_kabupaten)
    scalers, df = initialize_scalers(data)
    
    num_days = 7
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_days)

    sampled_temperature = np.random.choice(df['temperature'], num_days, replace=True)
    sampled_humidity = np.random.choice(df['humidity'], num_days, replace=True)
    sampled_windspeed = np.random.choice(df['windspeed'], num_days, replace=True)

    forecast_features = pd.DataFrame({
        'date': future_dates,
        'temperature': sampled_temperature,
        'humidity': sampled_humidity,
        'windspeed': sampled_windspeed
    })

    forecast_features_scaled = scalers['rain'].transform(forecast_features[['temperature', 'humidity', 'windspeed']])
    predicted_forecast_rain = model.predict(forecast_features_scaled)
    predicted_forecast_rain_denormalized = scalers['target_rain'].inverse_transform(predicted_forecast_rain)

    forecast_features['rainfall'] = predicted_forecast_rain_denormalized
    result = forecast_features[['date', 'rainfall']].round(1)

    return jsonify(result.to_dict(orient='records'))

@app.route('/forecast/windspeed', methods=['GET'])
def forecast_windspeed():
    selected_provinsi = request.args.get('selectedProvinsi')
    selected_kabupaten = request.args.get('selectedKabupaten')
    data = fetch_data(selected_provinsi, selected_kabupaten)
    scalers, df = initialize_scalers(data)
    
    num_days = 7
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_days)

    sampled_humidity = np.random.choice(df['humidity'], num_days, replace=True)
    sampled_rainfall = np.random.choice(df['rainfall'], num_days, replace=True)
    sampled_temperature = np.random.choice(df['temperature'], num_days, replace=True)

    forecast_features = pd.DataFrame({
        'date': future_dates,
        'humidity': sampled_humidity,
        'rainfall': sampled_rainfall,
        'temperature': sampled_temperature
    })

    forecast_features_scaled = scalers['wind'].transform(forecast_features[['humidity', 'rainfall', 'temperature']])
    predicted_forecast_wind = model.predict(forecast_features_scaled)
    predicted_forecast_wind_denormalized = scalers['target_wind'].inverse_transform(predicted_forecast_wind)

    forecast_features['windspeed'] = predicted_forecast_wind_denormalized
    result = forecast_features[['date', 'windspeed']].round(1)

    return jsonify(result.to_dict(orient='records'))

@app.route('/forecast', methods=['POST'])
def forecast():
    selected_provinsi = request.json.get('selectedProvinsi')
    selected_kabupaten = request.json.get('selectedKabupaten')

    temperature_forecast = requests.get(f'http://127.0.0.1:8888/forecast/temperature?selectedProvinsi={selected_provinsi}&selectedKabupaten={selected_kabupaten}').json()
    humidity_forecast = requests.get(f'http://127.0.0.1:8888/forecast/humidity?selectedProvinsi={selected_provinsi}&selectedKabupaten={selected_kabupaten}').json()
    rainfall_forecast = requests.get(f'http://127.0.0.1:8888/forecast/rainfall?selectedProvinsi={selected_provinsi}&selectedKabupaten={selected_kabupaten}').json()
    windspeed_forecast = requests.get(f'http://127.0.0.1:8888/forecast/windspeed?selectedProvinsi={selected_provinsi}&selectedKabupaten={selected_kabupaten}').json()

    forecast_combined = []

    ffmc0 = 85.0  
    dmc0 = 6.0    
    dc0 = 15.0

    for temp, hum, rain, wind in zip(temperature_forecast, humidity_forecast, rainfall_forecast, windspeed_forecast):
        future_date = pd.to_datetime(temp['date'])

        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM post_predicts WHERE provinsi = %s AND kabupaten = %s AND date = %s", (selected_provinsi, selected_kabupaten, future_date))
        existing_data = cursor.fetchone()
        cursor.close()
        connection.close()

        month = future_date.month

        if existing_data:
            fwi_instance = FWICLASS(temp=existing_data['temperature_predict'], rhum=existing_data['humidity_predict'], wind=existing_data['windspeed_predict'], prcp=existing_data['rainfall_predict'])
        else:
            fwi_instance = FWICLASS(temp=temp['temperature'], rhum=hum['humidity'], wind=wind['windspeed'], prcp=rain['rainfall'])

        ffmc = fwi_instance.FFMCcalc(ffmc0)
        dmc = fwi_instance.DMCcalc(dmc0, month)
        dc = fwi_instance.DCcalc(dc0, month)
        isi = fwi_instance.ISIcalc(ffmc)
        bui = fwi_instance.BUIcalc(dmc, dc)
        fwi = fwi_instance.FWIcalc(isi, bui)

        if existing_data:
            forecast_combined.append({
                "date": temp['date'],
                "temperature_predict": existing_data['temperature_predict'],
                "humidity_predict": existing_data['humidity_predict'],
                "rainfall_predict": existing_data['rainfall_predict'],
                "windspeed_predict": existing_data['windspeed_predict']
            })
        else:
            connection = get_db_connection()
            cursor = connection.cursor()
            cursor.execute("INSERT INTO post_predicts (provinsi, kabupaten, date, temperature_predict, humidity_predict, rainfall_predict, windspeed_predict) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                           (selected_provinsi, selected_kabupaten, future_date, temp['temperature'], hum['humidity'], rain['rainfall'], wind['windspeed']))
            cursor.execute("INSERT INTO fwis (provinsi, kabupaten, date, ffmc, dmc, dc, isi, bui, fwi) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                           (selected_provinsi, selected_kabupaten, future_date, ffmc, dmc, dc, isi, bui, fwi))
            connection.commit()
            cursor.close()
            connection.close()

            forecast_combined.append({
                "date": temp['date'],
                "temperature_predict": temp['temperature'],
                "humidity_predict": hum['humidity'],
                "rainfall_predict": rain['rainfall'],
                "windspeed_predict": wind['windspeed']
            })

        ffmc0 = ffmc
        dmc0 = dmc
        dc0 = dc

    sorted_forecast_combined = sorted(forecast_combined, key=lambda x: pd.to_datetime(x['date']))
    return jsonify(sorted_forecast_combined)

if __name__ == '__main__':
    app.run(debug=True, port=8888)
