import pickle
from enum import Enum
from typing import Optional
from contextlib import asynccontextmanager
from datetime import date, timedelta
import pandas as pd
import fastapi
from pydantic import BaseModel, FutureDate


def load_model():
    model = pickle.load(open('models/ml_model.pkl', 'rb'))
    return model


def load_preprocessing_pipeline():
    preprocessing_pipeline = pickle.load(open('pipelines/ml_model_pipeline.pkl', 'rb'))
    return preprocessing_pipeline


model = {}


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    model['model'] = load_model()
    model['preprocessing_pipeline'] = load_preprocessing_pipeline()
    yield
    # Clean up the ML models and release the resources
    model.clear()


class Manufacturer(str, Enum):
    acura = 'acura'
    alfa_romeo = 'alfa-romeo'
    aston_martin = 'aston-martin'
    audi = 'audi'
    bmw = 'bmw'
    buick = 'buick'
    cadillac = 'cadillac'
    chevrolet = 'chevrolet'
    chrysler = 'chrysler'
    dodge = 'dodge'
    ferrari = 'ferrari'
    fiat = 'fiat'
    ford = 'ford'
    genesis_motor = 'genesis motor'
    gm_daewoo = 'gm-daewoo'
    gm_daewoo_gm_korea = 'gm-daewoo/gm-korea'
    gmc = 'gmc'
    harley_davidson = 'harley-davidson'
    hino = 'hino'
    honda = 'honda'
    hyundai = 'hyundai'
    infiniti = 'infiniti'
    isuzu = 'isuzu'
    jaguar = 'jaguar'
    jeep = 'jeep'
    kawasaki = 'kawasaki'
    kia = 'kia'
    land_rover = 'land rover'
    lexus = 'lexus'
    lincoln = 'lincoln'
    mazda = 'mazda'
    merato_motorcycle_taizhou_zhongneng_motorcycle = 'merato motorcycle taizhou zhongneng motorcycle co. ltd. (znen)'
    mercedes_benz = 'mercedes-benz'
    mercury = 'mercury'
    mini = 'mini'
    mitsubishi = 'mitsubishi'
    morgan = 'morgan'
    nissan = 'nissan'
    pontiac = 'pontiac'
    porsche = 'porsche'
    pt_yamaha_indonesia_motor_mfg = 'pt yamaha indonesia motor mfg.'
    ram = 'ram'
    rover = 'rover'
    saab = 'saab'
    saturn = 'saturn'
    subaru = 'subaru'
    suzuki = 'suzuki'
    tesla = 'tesla'
    toyota = 'toyota'
    volkswagen = 'volkswagen'
    volvo = 'volvo'
    yamaha = 'yamaha'


class Fuel(str, Enum):
    diesel = 'diesel'
    electric = 'electric'
    gas = 'gas'
    hybrid = 'hybrid'
    other = 'other'


class Type(str, Enum):
    suv = 'SUV'
    bus = 'bus'
    convertible = 'convertible'
    coupe = 'coupe'
    hatchback = 'hatchback'
    mini_van = 'mini-van'
    offroad = 'offroad'
    other = 'other'
    pickup = 'pickup'
    sedan = 'sedan'
    truck = 'truck'
    van = 'van'
    wagon = 'wagon'


class Drive(str, Enum):
    four_wd = '4wd'
    fwd = 'fwd'
    rwd = 'rwd'


class PaintColor(str, Enum):
    black = 'black'
    blue = 'blue'
    brown = 'brown'
    custom = 'custom'
    green = 'green'
    grey = 'grey'
    orange = 'orange'
    purple = 'purple'
    red = 'red'
    silver = 'silver'
    white = 'white'
    yellow = 'yellow'


class Cylinder(str, Enum):
    ten_cylinders = '10 cylinders'
    twelve_cylinders = '12 cylinders'
    three_cylinders = '3 cylinders'
    four_cylinders = '4 cylinders'
    five_cylinders = '5 cylinders'
    six_cylinders = '6 cylinders'
    eight_cylinders = '8 cylinders'
    other = 'other'


class Transmission(str, Enum):
    automatic = 'automatic'
    manual = 'manual'
    other = 'other'


class InputData(BaseModel):
    manufacturer: Optional[Manufacturer] = None
    odometer: int
    year: int
    posting_date: FutureDate = date.today() + timedelta(days=1)
    cylinders: Optional[Cylinder] = None
    model: Optional[str] = None
    type: Optional[Type] = None
    fuel: Optional[Fuel] = None
    transmission: Optional[Transmission] = None
    drive: Optional[Drive] = None
    paint_color: Optional[PaintColor] = None
    VIN: Optional[str] = None

input_data = InputData(
    manufacturer='toyota',
    odometer=100000,
    year=2010,
    posting_date=date.today() + timedelta(days=1),
    cylinders='6 cylinders', 
    model='camry',
    type='sedan',
    fuel='gas', 
    transmission='automatic',
    drive='fwd',
    paint_color='black',
    VIN='1234567890'
)

app = fastapi.FastAPI(lifespan=lifespan)


@app.get("/health")
async def check_health():
    return {"Server": "I am healthy!"}


@app.post("/v1/predict")
async def make_predictions(input_data: InputData):
    # Preprocess the data
    input_dict = input_data.model_dump(mode='json')
    df_inference = pd.DataFrame([input_dict])

    try:
        df_inference = model['preprocessing_pipeline'].transform(df_inference)
    except Exception as e:
        return {"error": f"Error during preprocessing: {e}"}

    # Perform predictions
    try:
        predictions = model['model'].predict(df_inference)
        return {"predictions": predictions[0]}
    except Exception as e:
        return {"error": f"Error during prediction: {e}"}
