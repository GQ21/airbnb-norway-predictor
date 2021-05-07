import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import main

import pytest
import json
import pandas as pd


def test_preprocess_input_list() -> None:
    """
    Check if preprocess input function raises assertion error with wrong input that is not a list
    """
    input = json.dumps({"inputs": "test"})
    with pytest.raises(AssertionError):
        main.__process_input(input)


def test_preprocess_input_dict() -> None:
    """
    Check if preprocess input function raises assertion error with wrong input that contains not dictionaries
    """
    input = json.dumps({"inputs": ["test"]})
    with pytest.raises(AssertionError):
        main.__process_input(input)


def test_preprocess_input_values_count() -> None:
    """
    Check if preprocess input function raises assertion error with wrong less than 19 features input
    """
    input = json.dumps({"inputs": [{"test": 1, "test": 2}]})
    with pytest.raises(AssertionError):
        main.__process_input(input)


def test_preprocess_input_dataframe_output_type() -> None:
    """
    Check if preprocess input function outputs dataframe
    """
    input = json.dumps(
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
    assert type(main.__process_input(input)) == pd.DataFrame


def test_preprocess_input_dataframe_output_len() -> None:
    """
    Check if preprocess input function outputs dataframe with 219 columns
    """
    input = json.dumps(
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
    df = main.__process_input(input)
    assert len(df.columns) == 212
