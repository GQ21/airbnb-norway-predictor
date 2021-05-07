# airbnb-norway-predictor

<img src="https://i.ibb.co/GcKNRWh/norway-landscape.png" alt="norway-landscape" border="0">

## Introduction
Norway is the country which has to offer many spectacular wilderness places in Europe – what makes it a perfect destination for hiking in the nature and the mountains. Planning budget for a stay in Norway can be exhausting. Scrolling through airbnb and deciding which option is overpriced and which is perfect match could take quite some time. To speed up planning Airbnb Norway Predictor comes for a help. It is flask based web application where you can check what stay fits best in your budget and what price range you shoud consider as a normal price.

Check it out! https://airbnb-norway-predictor.herokuapp.com/

## Table of contents
* [Technologies](#technologies)
* [Setup](#setup)
* [Requests](#requests)
* [Model](#model)
* [License](#license)

## Technologies
App was created with:
* Python 3.6.8
* pandas 1.2.3 
* numpy 1.19.5
* flask 1.1.2
* gunicorn 20.1.0
* scikit-learn 0.24.1
* requests 2.25.1
* Heroku

## Setup
To deploy Airbnb Norway Predictor recommended to use Heroku. Simpliest way is to create copied version of this repository to your github. Go to heroku and choose deployment method - github where you will be able to connect your copied repository and make cloud based web app.

## Requests
To make batch request you can use your preferable program like postman or by using preferable IDE with libraries like requests. For this case use app_requests.py located in request directory as an example.

Web app has two main routes:
 * **/predict** takes post request and spits out predicted values.
 * **/history** takes get request and retrieves 10 most recent requests.

This is how batch post input that contains two inputs should look like:

```
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
        },
        {
            "city": "Tønsberg",
            "location": "Tønsberg",
            "property_type": "Entire apartment",
            "latitude": 59.27723,
            "longitude": 10.45184,
            "rating": 4.67,
            "reviews": 275.0,
            "guests": 2,
            "studio": 1,
            "bedrooms": 1,
            "beds": 1,
            "baths": 1.0,
            "shared_bath": 0.0,
            "kitchen": 1.0,
            "wifi": 1.0,
            "washer": 1.0,
            "tv": 1.0,
            "parking": 0.0,
            "refrigerator": 1.0,
        },
    ]
}
```

Although for single request and history checking go to deployed web app link and play with request inputs. Like for example https://airbnb-norway-predictor.herokuapp.com/

## Model
Modeling part was done with Gradient Boosting Regressor and can be seen in `model_GBR.ipynb` notebook located in `model` directory. Model is not very accurate, it's coefficient of determination R squared is `0.4011240714203681` and mean absolute error is `45.21328136294829`. Therefore deep learning/computer vision algorithms can be used for improvement.

## License
This project is licensed under [MIT license](https://tldrlegal.com/license/mit-license)
