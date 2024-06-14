from flask import Flask, request, jsonify
import pickle
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import pandas as pd
from dateutil.parser import parse as parse_date
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from elmz import elm
from fwi import FWICLASS
from datetime import datetime

app = Flask(__name__)

def format_date(date):
    return date.strftime('%Y-%m-%d')

with open('modelsz.pkl', 'rb') as file:
    model = pickle.load(file)

connection = psycopg2.connect(user="ffp-indonesia",
                              password="forestfire123",
                              host="34.101.120.221",
                              port="5432",
                              database="ffp-indonesia",
                              cursor_factory=RealDictCursor)

def get_db_connection():
    return psycopg2.connect(user="ffp-indonesia",
                            password="forestfire123",
                            host="34.101.120.221",
                            port="5432",
                            database="ffp-indonesia",
                            cursor_factory=RealDictCursor)

# Define global scaler and dataframe
scaler_X = StandardScaler()
scaler_y = StandardScaler()
df = pd.DataFrame()

def fetch_data(selected_provinsi, selected_kabupaten):
    cursor = connection.cursor()
    query = "SELECT provinsi, date, windspeed, humidity, rainfall, temperature FROM posts WHERE kabupaten = %s AND provinsi = %s"
    app.logger.info(f"Executing query: {query} with values {selected_kabupaten}, {selected_provinsi}")
    cursor.execute(query, (selected_kabupaten, selected_provinsi))
    data = cursor.fetchall()
    cursor.close()
    if not data:
        app.logger.error(f"No data found for Province ID: {selected_provinsi}, Regency ID: {selected_kabupaten}")
        return None
    return pd.DataFrame(data)

@app.route('/train/temperature', methods=['GET'])
def train_temp():
    global df, scaler_X, scaler_y, model
    selected_provinsi = request.args.get('provinsi')
    selected_kabupaten = request.args.get('kabupaten')
    df = fetch_data(selected_provinsi, selected_kabupaten)
    if df is None:
        return jsonify({"error": "No data found"}), 404

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    train_data = df[df['date'] <= '2022-02-28']
    test_data = df[df['date'] > '2022-02-28']

    X_train = train_data[['humidity', 'windspeed', 'rainfall']].values
    y_train = train_data[['temperature']].values
    X_test = test_data[['humidity', 'windspeed', 'rainfall']].values
    y_test = test_data[['temperature']].values

    dates = test_data['date'].tolist()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    model = elm(hidden_units=20, activation_function='sigmoid', x=X_train_scaled, y=y_train_scaled, random_type='normal')
    model.fit()

    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test = scaler_y.inverse_transform(y_test_scaled)

    result_df = pd.DataFrame({
        'date': dates,
        'Actual Temperature': y_test.flatten(),
        'Predicted Temperature': y_pred.flatten()
    })

    response = {
        'result': result_df.to_dict(orient='records')
    }
    return jsonify(response)

@app.route('/train/humidity', methods=['GET'])
def train_humid():
    global df, scaler_X, scaler_y, model
    selected_provinsi = request.args.get('provinsi')
    selected_kabupaten = request.args.get('kabupaten')
    df = fetch_data(selected_provinsi, selected_kabupaten)
    if df is None:
        return jsonify({"error": "No data found"}), 404

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    train_data = df[df['date'] <= '2022-02-28']
    test_data = df[df['date'] > '2022-02-28']

    X_train = train_data[['temperature', 'windspeed', 'rainfall']].values
    y_train = train_data[['humidity']].values
    X_test = test_data[['temperature', 'windspeed', 'rainfall']].values
    y_test = test_data[['humidity']].values

    dates = test_data['date'].tolist()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    model = elm(hidden_units=20, activation_function='sigmoid', x=X_train_scaled, y=y_train_scaled, random_type='normal')
    model.fit()

    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test = scaler_y.inverse_transform(y_test_scaled)

    result_df = pd.DataFrame({
        'date': dates,
        'Actual Humidity': y_test.flatten(),
        'Predicted Humidity': y_pred.flatten()
    })

    response = {
        'result': result_df.to_dict(orient='records')
    }
    return jsonify(response)

@app.route('/train/rainfall', methods=['GET'])
def train_rain():
    global df, scaler_X, scaler_y, model
    selected_provinsi = request.args.get('provinsi')
    selected_kabupaten = request.args.get('kabupaten')
    df = fetch_data(selected_provinsi, selected_kabupaten)
    if df is None:
        return jsonify({"error": "No data found"}), 404

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    train_data = df[df['date'] <= '2022-02-28']
    test_data = df[df['date'] > '2022-02-28']

    X_train = train_data[['temperature', 'humidity', 'windspeed']].values
    y_train = train_data[['rainfall']].values
    X_test = test_data[['temperature', 'humidity', 'windspeed']].values
    y_test = test_data[['rainfall']].values

    dates = test_data['date'].tolist()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    model = elm(hidden_units=20, activation_function='sigmoid', x=X_train_scaled, y=y_train_scaled, random_type='normal')
    model.fit()

    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test = scaler_y.inverse_transform(y_test_scaled)

    result_df = pd.DataFrame({
        'date': dates,
        'Actual Rainfall': y_test.flatten(),
        'Predicted Rainfall': y_pred.flatten()
    })

    response = {
        'result': result_df.to_dict(orient='records')
    }
    return jsonify(response)

@app.route('/train/windspeed', methods=['GET'])
def train_wind():
    global df, scaler_X, scaler_y, model
    selected_provinsi = request.args.get('provinsi')
    selected_kabupaten = request.args.get('kabupaten')
    df = fetch_data(selected_provinsi, selected_kabupaten)
    if df is None:
        return jsonify({"error": "No data found"}), 404

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    train_data = df[df['date'] <= '2022-02-28']
    test_data = df[df['date'] > '2022-02-28']

    X_train = train_data[['temperature', 'humidity', 'rainfall']].values
    y_train = train_data[['windspeed']].values
    X_test = test_data[['temperature', 'humidity', 'rainfall']].values
    y_test = test_data[['windspeed']].values

    dates = test_data['date'].tolist()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    model = elm(hidden_units=20, activation_function='sigmoid', x=X_train_scaled, y=y_train_scaled, random_type='normal')
    model.fit()

    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test = scaler_y.inverse_transform(y_test_scaled)

    result_df = pd.DataFrame({
        'date': dates,
        'Actual Windspeed': y_test.flatten(),
        'Predicted Windspeed': y_pred.flatten()
    })

    response = {
        'result': result_df.to_dict(orient='records')
    }
    return jsonify(response)

@app.route('/train', methods=['POST', 'GET'])
def final_data():
    global df, scaler_X, scaler_y, model
    data = request.json
    selected_provinsi = data.get('selectedProvinsi')
    selected_kabupaten = data.get('selectedKabupaten')

    app.logger.info(f'Received Province ID: {selected_provinsi}, Regency ID: {selected_kabupaten}')
    df = fetch_data(selected_provinsi, selected_kabupaten)
    if df is None:
        return jsonify({"error": "No data found for training"}), 404

    response_temp = app.test_client().get('/train/temperature', query_string={'provinsi': selected_provinsi, 'kabupaten': selected_kabupaten})
    response_humid = app.test_client().get('/train/humidity', query_string={'provinsi': selected_provinsi, 'kabupaten': selected_kabupaten})
    response_rain = app.test_client().get('/train/rainfall', query_string={'provinsi': selected_provinsi, 'kabupaten': selected_kabupaten})
    response_wind = app.test_client().get('/train/windspeed', query_string={'provinsi': selected_provinsi, 'kabupaten': selected_kabupaten})

    if response_temp.status_code != 200:
        app.logger.error('Error in temperature prediction: ' + response_temp.get_json().get('error', 'Unknown error'))
    if response_humid.status_code != 200:
        app.logger.error('Error in humidity prediction: ' + response_humid.get_json().get('error', 'Unknown error'))
    if response_rain.status_code != 200:
        app.logger.error('Error in rainfall prediction: ' + response_rain.get_json().get('error', 'Unknown error'))
    if response_wind.status_code != 200:
        app.logger.error('Error in windspeed prediction: ' + response_wind.get_json().get('error', 'Unknown error'))

    result_temp = response_temp.get_json().get('result', [])
    result_humid = response_humid.get_json().get('result', [])
    result_rain = response_rain.get_json().get('result', [])
    result_wind = response_wind.get_json().get('result', [])

    ffmc0 = 85.0  
    dmc0 = 6.0    
    dc0 = 15.0

    predict_data = {}
    for temp, humid, rain, wind in zip(result_temp, result_humid, result_rain, result_wind):
        date_str = temp['date']
        date = parse_date(date_str)
        formatted_date = format_date(date)

        cursor = connection.cursor()
        check_query = "SELECT * FROM post_predicts WHERE provinsi = %s AND kabupaten = %s AND date = %s"
        cursor.execute(check_query, (selected_provinsi, selected_kabupaten, formatted_date))
        existing_record = cursor.fetchone()
        month = date.month

        fwi_instance = FWICLASS(temp=temp['Actual Temperature'], rhum=humid['Actual Humidity'], wind=wind['Actual Windspeed'], prcp=rain['Actual Rainfall'])
        
        ffmc = fwi_instance.FFMCcalc(ffmc0)
        dmc = fwi_instance.DMCcalc(dmc0, month)
        dc = fwi_instance.DCcalc(dc0, month)
        isi = fwi_instance.ISIcalc(ffmc)
        bui = fwi_instance.BUIcalc(dmc, dc)
        fwi = fwi_instance.FWIcalc(isi, bui)

        if existing_record:
            update_query = """
            UPDATE post_predicts
            SET temperature_predict = %s, rainfall_predict = %s, humidity_predict = %s, windspeed_predict = %s
            WHERE provinsi = %s AND kabupaten = %s AND date = %s
            """
            cursor.execute(update_query, (
                temp['Predicted Temperature'], rain['Predicted Rainfall'], humid['Predicted Humidity'], wind['Predicted Windspeed'],
                selected_provinsi, selected_kabupaten, formatted_date
            ))
        else:
            insert_query = """
            INSERT INTO post_predicts (provinsi, kabupaten, temperature_predict, rainfall_predict, humidity_predict, windspeed_predict, date)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (
                selected_provinsi, selected_kabupaten, temp['Predicted Temperature'], rain['Predicted Rainfall'], humid['Predicted Humidity'], wind['Predicted Windspeed'], formatted_date
            ))

        connection.commit()
        cursor.close()

        cursor = connection.cursor()
        check_query_fwi = "SELECT * FROM fwis WHERE provinsi = %s AND kabupaten = %s AND date = %s"
        cursor.execute(check_query_fwi, (selected_provinsi, selected_kabupaten, formatted_date))
        existing_record_fwi = cursor.fetchone()

        if existing_record_fwi:
            update_query = """
            UPDATE fwis
            SET ffmc = %s, dmc = %s, dc = %s, isi = %s, bui = %s, fwi = %s
            WHERE provinsi = %s AND kabupaten = %s AND date = %s
            """
            cursor.execute(update_query, (
                ffmc, dmc, dc, isi, bui, fwi,
                selected_provinsi, selected_kabupaten, formatted_date
            ))
        else:
            insert_query = """
            INSERT INTO fwis (provinsi, kabupaten, ffmc, dmc, dc, isi, bui, fwi, date)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (
                selected_provinsi, selected_kabupaten, ffmc, dmc, dc, isi, bui, fwi, formatted_date
            ))

        predict_data[formatted_date] = {
            'Date': formatted_date,
            'Actual Temperature': temp['Actual Temperature'],
            'Predicted Temperature': temp['Predicted Temperature'],
            'Actual Humidity': humid['Actual Humidity'],
            'Predicted Humidity': humid['Predicted Humidity'],
            'Actual Rainfall': rain['Actual Rainfall'],
            'Predicted Rainfall': rain['Predicted Rainfall'],
            'Actual Windspeed': wind['Actual Windspeed'],
            'Predicted Windspeed': wind['Predicted Windspeed'],
            'FFMC': ffmc,
            'DMC': dmc,
            'DC': dc,
            'ISI': isi,
            'BUI': bui,
            'FWI': fwi
        }

        ffmc0 = ffmc
        dmc0 = dmc
        dc0 = dc

    return jsonify(predict_data)

@app.route('/api/fwi-data-map', methods=['GET', 'POST'])
def fwi_data():
    data = request.json
    date = data['date']
    
    cursor = connection.cursor()
    q = ("""
            SELECT date, temperature_predict, humidity_predict, rainfall_predict, windspeed_predict, provinsi, kabupaten
            FROM post_predicts
            WHERE date = %s
            """)
    cursor.execute(q, (date,))
    data = cursor.fetchall()

    results = []

    ffmc0 = 85.0  
    dmc0 = 6.0    
    dc0 = 15.0

    for entry in data:
        kabupaten = entry['kabupaten']
        provinsi = entry['provinsi']
        date = entry['date']
        date_out = format_date(date)
        windspeed = entry['windspeed']
        humidity = entry['humidity']
        rainfall = entry['rainfall']
        temperature = entry['temperature']
        
        cursor = connection.cursor()
        q1 = ("SELECT name FROM regencies WHERE id = %s AND province_id = %s")
        cursor.execute(q1, (kabupaten, provinsi))
        name_data = cursor.fetchall()

        for entry_name in name_data:
            name = entry_name['name']
        
        month = date.month
        
        fwi_instance = FWICLASS(temp=temperature, rhum=humidity, wind=windspeed, prcp=rainfall)
        
        ffmc = fwi_instance.FFMCcalc(ffmc0)
        dmc = fwi_instance.DMCcalc(dmc0, month)
        dc = fwi_instance.DCcalc(dc0, month)
        isi = fwi_instance.ISIcalc(ffmc)
        bui = fwi_instance.BUIcalc(dmc, dc)
        fwi = fwi_instance.FWIcalc(isi, bui)
        
        results.append({
            "kabupaten_id": kabupaten,
            "name": name,
            "date": date_out,
            "FFMC": ffmc,
            "DMC": dmc,
            "DC": dc,
            "ISI": isi,
            "BUI": bui,
            "FWI": fwi
        })

        ffmc0 = ffmc
        dmc0 = dmc
        dc0 = dc
    return jsonify(results)

@app.route('/api/fwi-data-0', methods=['GET'])
def fwi_data1():
    data = request.json
    date = data['date']

    cursor = connection.cursor()
    q = ("""
            SELECT date, temperature, humidity, rainfall, windspeed, provinsi, kabupaten
            FROM post
            WHERE date = %s
            """)
    cursor.execute(q, (date))
    data = cursor.fetchall()

    if not data:
        return jsonify({"error": "No data found"}), 404

    results = []

    ffmc0 = 85.0  
    dmc0 = 6.0    
    dc0 = 15.0

    for entry in data:
        kabupaten = entry['kabupaten']
        provinsi = entry['provinsi']
        date = entry['date']
        date_out = format_date(date)
        windspeed = entry['windspeed']
        humidity = entry['humidity']
        rainfall = entry['rainfall']
        temperature = entry['temperature']

        q1 = ("SELECT name FROM regencies WHERE id = %s AND province_id = %s")
        cursor.execute(q1, (kabupaten, provinsi))
        name_data = cursor.fetchall()

        for entry_name in name_data:
            name = entry_name['name']

        month = date.month

        fwi_instance = FWICLASS(temp=temperature, rhum=humidity, wind=windspeed, prcp=rainfall)

        ffmc = fwi_instance.FFMCcalc(ffmc0)
        dmc = fwi_instance.DMCcalc(dmc0, month)
        dc = fwi_instance.DCcalc(dc0, month)
        isi = fwi_instance.ISIcalc(ffmc)
        bui = fwi_instance.BUIcalc(dmc, dc)
        fwi = fwi_instance.FWIcalc(isi, bui)

        results.append({
            "name": name,
            "date": date_out,
            "FFMC": ffmc,
            "DMC": dmc,
            "DC": dc,
            "ISI": isi,
            "BUI": bui,
            "FWI": fwi
        })

        ffmc0 = ffmc
        dmc0 = dmc
        dc0 = dc

    print("FWI Data:", results)
    return jsonify(results)

@app.route('/fwi-upload_db', methods=['GET', 'POST'])
def fwi_data_db():

    cursor = connection.cursor()
    q = ("""
            SELECT date, temperature, humidity, rainfall, windspeed, provinsi, kabupaten
            FROM posts
            WHERE kabupaten = '3671' AND date BETWEEN '2019-01-01' AND '2024-01-29'
            """)
    cursor.execute(q)
    data = cursor.fetchall()
    
    if not data:
        return jsonify({"error": "No data found"}), 404
    
    results = []

    ffmc0 = 85.0  
    dmc0 = 6.0    
    dc0 = 15.0

    for entry in data:
        kabupaten = entry['kabupaten']
        provinsi = entry['provinsi']
        date = entry['date']
        date_out = format_date(date)
        windspeed = entry['windspeed']
        humidity = entry['humidity']
        rainfall = entry['rainfall']
        temperature = entry['temperature']
        
        month = date.month
        
        fwi_instance = FWICLASS(temp=temperature, rhum=humidity, wind=windspeed, prcp=rainfall)
        
        ffmc = fwi_instance.FFMCcalc(ffmc0)
        dmc = fwi_instance.DMCcalc(dmc0, month)
        dc = fwi_instance.DCcalc(dc0, month)
        isi = fwi_instance.ISIcalc(ffmc)
        bui = fwi_instance.BUIcalc(dmc, dc)
        fwi = fwi_instance.FWIcalc(isi, bui)
        
        insert_query = "INSERT INTO fwis (provinsi, kabupaten, ffmc, dmc, dc, isi, bui, fwi, date) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        cursor.execute(insert_query, (provinsi, kabupaten, ffmc, dmc, dc, isi, bui, fwi, date))
        connection.commit()
        results.append({
            "date": date_out,
            "FFMC": ffmc,
            "DMC": dmc,
            "DC": dc,
            "ISI": isi,
            "BUI": bui,
            "FWI": fwi
        })

        ffmc0 = ffmc
        dmc0 = dmc
        dc0 = dc
    return jsonify(results)

@app.route('/api/fwi-data-all', methods=['POST'])
def fwi_data_xx():
    data = request.json

    start_date = data['start_date']
    end_date = data['end_date']
    province = data['provinsi']
    kabupaten = data['kabupaten']
    
    connection = get_db_connection()
    cursor = connection.cursor()
    q = """
        SELECT date, ffmc, dmc, dc, isi, bui, fwi, provinsi, kabupaten
        FROM fwis
        WHERE provinsi = %s AND kabupaten = %s AND date BETWEEN %s AND %s
    """
    cursor.execute(q, (province, kabupaten, start_date, end_date))
    data = cursor.fetchall()
    cursor.close()
    connection.close()

    if not data:
        return jsonify({"error": "No data found"}), 404

    results = []

    for entry in data:
        results.append({
            "date": entry['date'].strftime('%Y-%m-%d'),
            "FFMC": entry['ffmc'],
            "DMC": entry['dmc'],
            "DC": entry['dc'],
            "ISI": entry['isi'],
            "BUI": entry['bui'],
            "FWI": entry['fwi'],
            "provinsi": entry['provinsi'],
            "kabupaten": entry['kabupaten']
        })

    return jsonify(results)

@app.route('/api/fwi-data-current', methods=['POST'])
def fwi_data_current():
    data = request.json
    date = data.get('date', datetime.now().strftime('%Y-%m-%d'))

    connection = get_db_connection()
    cursor = connection.cursor()
    query = """
        SELECT f.date, f.ffmc, f.dmc, f.dc, f.isi, f.bui, f.fwi, f.provinsi, f.kabupaten, r.name
        FROM fwis f
        JOIN regencies r ON f.kabupaten = r.id AND f.provinsi = r.province_id
        WHERE f.date = %s
    """
    cursor.execute(query, (date,))
    data = cursor.fetchall()
    cursor.close()
    connection.close()

    if not data:
        return jsonify({"error": "No data found"}), 404

    results = []
    for entry in data:
        results.append({
            "kabupaten_id": entry['kabupaten'],
            "name": entry['name'],
            "date": entry['date'].strftime('%Y-%m-%d'),
            "FFMC": entry['ffmc'],
            "DMC": entry['dmc'],
            "DC": entry['dc'],
            "ISI": entry['isi'],
            "BUI": entry['bui'],
            "FWI": entry['fwi']
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)