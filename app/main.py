from flask import Flask, request, render_template
import psycopg2
import psycopg2.extras as extras

from datetime import datetime
import pandas as pd
import numpy as np
import pickle

from typing import Optional
import json
from dotenv import load_dotenv
import os

load_dotenv()

SAVED_MODEL_PATH = "model/regressor.pkl"
regressor = pickle.load(open(SAVED_MODEL_PATH, "rb"))

SAVED_ENCODER_PATH = "model//encoder.pkl"
encoder = pickle.load(open(SAVED_ENCODER_PATH, "rb"))

APP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(APP_PATH, "templates/")

app = Flask(__name__, template_folder=TEMPLATE_PATH)


def __connect() -> psycopg2.extensions.connection:
    """Connects to database and returns psycopg connection object"""
    db_connection = psycopg2.connect(
        database=os.environ["DATABASE"],
        user=os.environ["USER"],
        password=os.environ["PASSWORD"],
        host=os.environ["HOST"],
        port="5432",
    )

    db_connection.autocommit = True
    return db_connection


@app.route("/")
def home_page() -> None:
    """Open index.html page with default specified parameters"""
    return render_template(
        "index.html",
        city="Oslo",
        location="Grünerløkka",
        property_type="Private room",
        latitude="59.92572",
        longitude="10.76314",
        rating="4.55",
        reviews="225",
        guests="2",
        studio="0.0",
        bedrooms="1",
        beds="1.0",
        baths="1.0",
        shared_bath="1.0",
        kitchen="1.0",
        wifi="1.0",
        washer="1.0",
        tv="1.0",
        parking="0.0",
        refrigerator="1.0",
    )


@app.route("/history", methods=["GET"])
def history() -> dict:
    """Connects to database and gets 10 most recent entries"""
    try:
        db_connection = __connect()
        cur = db_connection.cursor()
        cur.execute("""SELECT * FROM history ORDER BY id DESC LIMIT 10;""")
        rows = cur.fetchall()

        data_fetched = []
        for row in rows:
            data_input = dict.fromkeys(["id", "datetime", "features", "prediction"])
            data_input["id"] = row[0]
            data_input["datetime"] = row[1].strftime("%m/%d/%Y")
            data_input["features"] = row[2]
            data_input["prediction"] = float(row[3])

            data_fetched.append(data_input)
        return json.dumps(data_fetched)
    except:
        return json.dumps({"error": "CONNECTION TO DATABASE FAILED"}), 500


def __process_input(request_data: dict) -> pd.DataFrame:
    """
    Takes post request json data, encodes it and exports it as pandas Dataframe.

    Parameters:
        request_data : dict
            data in json format
    """
    inputs = json.loads(request_data)["inputs"]
    assert type(inputs) == list, "'inputs' value must be a list"
    assert type(inputs[0]) == dict, "'inputs' value must contain dictionaries"
    assert len(inputs[0]) == 19, "'inputs' must contain 19 different features"

    df_inputs = pd.concat(
        [pd.DataFrame([input], columns=inputs[0].keys()) for input in inputs],
        ignore_index=True,
    )
    df_encoded = pd.DataFrame(
        encoder.transform(df_inputs[["city", "property_type", "location"]]).toarray()
    )
    df_joined = pd.concat([df_inputs, df_encoded], axis=1).drop(
        ["city", "property_type", "location"], axis=1
    )

    return df_joined


def __process_form_input(request_data: dict) -> dict:
    """
    Takes post request form data from HTML, converts it to dictinory.

    Parameters:
        request_data : werkzeug.MultiDict
            data in werkzeug MultiDict format
    """
    keys = [i for i in request_data.keys()][:-1]
    values = [i for i in request_data.values()][:-1]

    values_float = [float(val) for val in values[3:]]
    values_categorical = values[:3]

    values_converted = values_categorical + values_float
    input_dict = dict(zip(keys, values_converted))

    return input_dict


def __insert_into_database(request_data: list, predictions: list) -> None:
    """
    Takes post request data, list of predictions, connects to database and insert data into it.

    Parameters:
        request_data : werkzeug.MultiDict
            data in werkzeug MultiDict format
    """
    try:
        db_connection = __connect()
        cur = db_connection.cursor()
        try:
            date = datetime.now()
            data_joined = []

            # Joining data as tuples
            for input, predict in zip(request_data, predictions):
                row_data = (date, f"{input}", predict)
                data_joined.append(row_data)

            # Inserting data as a batch into database
            insert_query = "insert into history (date,features,prediction) values %s"
            psycopg2.extras.execute_values(
                cur, insert_query, data_joined, template=None, page_size=100
            )
        except:
            print("Couldn't insert values")
        db_connection.close()
    except:
        print("Couldn't connect to database")


@app.route("/predict", methods=["POST"])
def predict() -> str:
    """
    App route for model prediction. Checks POST request input, uses data preprocess functions
    and outputs JSON with predicted values.
    """
    try:
        # Interact with index.html
        if request.form:
            try:
                if request.form["history"] == "History":
                    input_dict = __process_form_input(request.form)
                    data_history = json.loads(history())
                    # Collect database history entries for html insertions
                    data_entries = {}
                    for k in range(10):
                        if 0 <= k < len(data_history):
                            data_entries[f"entry_{k}"] = data_history[k]
                        else:
                            data_entries[f"entry_{k}"] = ""

                    return render_template(
                        "index.html",
                        entry_00=data_entries["entry_0"],
                        entry_01=data_entries["entry_1"],
                        entry_02=data_entries["entry_2"],
                        entry_03=data_entries["entry_3"],
                        entry_04=data_entries["entry_4"],
                        entry_05=data_entries["entry_5"],
                        entry_06=data_entries["entry_6"],
                        entry_07=data_entries["entry_7"],
                        entry_08=data_entries["entry_8"],
                        entry_09=data_entries["entry_9"],
                        city=input_dict["city"],
                        location=input_dict["location"],
                        property_type=input_dict["property_type"],
                        latitude=input_dict["latitude"],
                        longitude=input_dict["longitude"],
                        rating=input_dict["rating"],
                        reviews=input_dict["reviews"],
                        guests=input_dict["guests"],
                        studio=input_dict["studio"],
                        bedrooms=input_dict["bedrooms"],
                        beds=input_dict["beds"],
                        baths=input_dict["baths"],
                        shared_bath=input_dict["shared_bath"],
                        kitchen=input_dict["kitchen"],
                        wifi=input_dict["wifi"],
                        washer=input_dict["washer"],
                        tv=input_dict["tv"],
                        parking=input_dict["parking"],
                        refrigerator=input_dict["refrigerator"],
                    )
            except:
                if request.form["prediction"] == "Predict":
                    input_dict = __process_form_input(request.form)

                    input_data = __process_input(json.dumps({"inputs": [input_dict]}))
                    predictions = regressor.predict(input_data)
                    # Connect to database and insert input
                    __insert_into_database([input_dict], predictions)

                    return render_template(
                        "index.html",
                        result=f"Predicted Price: {round(predictions[0], 2)} €",
                        city=input_dict["city"],
                        location=input_dict["location"],
                        property_type=input_dict["property_type"],
                        latitude=input_dict["latitude"],
                        longitude=input_dict["longitude"],
                        rating=input_dict["rating"],
                        reviews=input_dict["reviews"],
                        guests=input_dict["guests"],
                        studio=input_dict["studio"],
                        bedrooms=input_dict["bedrooms"],
                        beds=input_dict["beds"],
                        baths=input_dict["baths"],
                        shared_bath=input_dict["shared_bath"],
                        kitchen=input_dict["kitchen"],
                        wifi=input_dict["wifi"],
                        washer=input_dict["washer"],
                        tv=input_dict["tv"],
                        parking=input_dict["parking"],
                        refrigerator=input_dict["refrigerator"],
                    )
        # Get raw output
        else:
            input_data = __process_input(request.data)
            predictions = regressor.predict(input_data)
            # Connect to database and insert input
            __insert_into_database(json.loads(request.data)["inputs"], predictions)
            return json.dumps({"predicted": predictions.tolist()})
    except (KeyError, json.JSONDecodeError, AssertionError):
        if request.form:
            return render_template("index.html", result="CHECK INPUT")
        else:
            return json.dumps({"error": "CHECK INPUT"}), 400
    except:
        if request.form:
            return render_template("index.html", result="ERROR - PREDICTION FAILED")
        else:
            return json.dumps({"error": "PREDICTION FAILED"}), 500
