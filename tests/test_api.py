import requests
import json


web_app_url = "https://airbnb-norway-predictor.herokuapp.com/"


def test_get_request() -> None:
    """
    Check if /history get request method respons correctly
    """
    resp = requests.get(f"{web_app_url}/history")
    assert resp.status_code == 200


def test_post_request_prediction_value() -> None:
    """
    Check if web app prediction value is correct
    """
    request_input = json.dumps(
        {
            "inputs": [
                {
                    "city": "Oslo",
                    "location": "Grünerløkka",
                    "property_type": "Private room",
                    "latitude": 59.92572,
                    "longitude": 10.76314,
                    "rating": 4.55,
                    "reviews": 225.0,
                    "guests": 2,
                    "studio": 0,
                    "bedrooms": 1,
                    "beds": 1,
                    "baths": 1.0,
                    "shared_bath": 1.0,
                    "kitchen": 1.0,
                    "wifi": 1.0,
                    "washer": 1.0,
                    "tv": 1.0,
                    "parking": 0.0,
                    "refrigerator": 1.0,
                }
            ]
        }
    )
    resp = requests.post(f"{web_app_url}/predict", data=request_input)
    assert json.loads(resp.text) == {"predicted": [40.206530634452676]}


def test_post_request_input_failure() -> None:
    """
    Check if /predict post method gives input error with incorrect input
    """
    request_input = json.dumps({"inputs": [{"test": "test"}]})
    resp = requests.post(f"{web_app_url}/predict", data=request_input)
    assert resp.status_code == 400


def test_post_request_predict_failure() -> None:
    """
    Check if /predict post method gives prediction error with incorrect input
    """
    request_input = json.dumps(
        {
            "inputs": [
                {
                    "city": "test",
                    "location": "test",
                    "property_type": "test",
                    "latitude": 59.92572,
                    "longitude": 10.76314,
                    "rating": 4.55,
                    "reviews": 225.0,
                    "guests": 2,
                    "studio": 0,
                    "bedrooms": 1,
                    "beds": 1,
                    "baths": 1.0,
                    "shared_bath": 1.0,
                    "kitchen": 1.0,
                    "wifi": 1.0,
                    "washer": 1.0,
                    "tv": 1.0,
                    "parking": 0.0,
                    "refrigerator": "1.0",
                }
            ]
        }
    )
    resp = requests.post(f"{web_app_url}/predict", data=request_input)
    assert resp.status_code == 500
