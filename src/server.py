import json
import pickle
import time

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, create_model
from sklearn.pipeline import Pipeline


class HousingInputModel(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int


class HousingOutputModel(BaseModel):
    prediction: float
    metadata: dict


app = FastAPI(title="Housing Pricing Model",
              version="1.0.0",
              docs_url="/docs",
              redoc_url="/redoc",
              openapi_url="/openapi.json")

model: Pipeline = pickle.load(open("./data/model/model.pkl", "rb"))

with open("./data/model/model_features.json", "r", encoding="utf-8") as f:
    model_features = json.load(f)

config = {v: (float, ...) for v in model_features}

HousingInputRequiredModel = create_model("HousingInputRequiredModel", **config)


demographics = pd.read_csv("./data/zipcode_demographics.csv", dtype={'zipcode': int})


def filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[[col for col in model_features if col in df.columns]]


@app.get("/health")
async def health_check():
    """
    Check the service health
    :return:
    """
    return {"status": "ok"}


@app.post("/predict")
async def predict(data: HousingInputModel, with_metadata: bool = False) -> HousingOutputModel:
    """
    This endpoint accepts input data for housing prediction and optionally returns metadata related to the prediction.

    :param data: A `HousingInputModel` object containing the input features for housing prediction.

    :param with_metadata: A boolean flag to determine whether to include metadata in the response.
                           Default is `False`.

    :return: A `HousingOutputModel` object containing the predicted housing price.
             If `with_metadata` is `True`, the output will also include metadata regarding the prediction process.

    Example:
    ------------
    Input:
        data = {
            "bedrooms": 5,
            "bathrooms": 1.75,
            "sqft_living": 2330,
            "sqft_lot": 3800,
            "floors": 1.5,
            "waterfront": 0,
            "view": 0,
            "condition": 3,
            "grade": 7,
            "sqft_above": 1360,
            "sqft_basement": 970,
            "yr_built": 1927,
            "yr_renovated": 0,
            "zipcode": 98115,
            "lat": 47.6835,
            "long": -122.308,
            "sqft_living15": 2100,
            "sqft_lot15": 3800
        }

    Output:
        HousingOutputModel = {
            "predicted_price": 525000.0,
            "metadata": {  # Only if 'with_metadata' is True
                "latency": 0.002555,
            }
        }
    """

    start = time.time()

    dataframe = pd.DataFrame([data.model_dump()])

    dataframe = dataframe.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")

    dataframe = filter_columns(dataframe)

    prediction = model.predict(dataframe)

    metadata = {}

    if with_metadata:
        end = time.time()

        metadata = {
            "latency": end - start
        }

    return HousingOutputModel(prediction=prediction, metadata=metadata)


@app.post("/predict_required")
async def predict_required(data: HousingInputRequiredModel, with_metadata: bool = False) -> HousingOutputModel:
    """
    This endpoint accepts only the required input data for housing prediction and optionally returns metadata related to the prediction.

    :param data: A `HousingInputRequiredModel` object containing the input features for housing prediction.

    :param with_metadata: A boolean flag to determine whether to include metadata in the response.
                           Default is `False`.

    :return: A `HousingOutputModel` object containing the predicted housing price.
             If `with_metadata` is `True`, the output will also include metadata regarding the prediction process.

    Example:
    ------------
    Input:
        data = {
            "bedrooms": 5,
            "bathrooms": 1.75,
            "sqft_living": 2330,
            "sqft_lot": 3800,
            "floors": 1.5,
            "sqft_above": 1360,
            "sqft_basement": 970,
            "ppltn_qty": 100000,
            "urbn_ppltn_qty": 60000,
            "sbrbn_ppltn_qty": 30000,
            "farm_ppltn_qty": 10000,
            "non_farm_qty": 90000,
            "medn_hshld_incm_amt": 75000,
            "medn_incm_per_prsn_amt": 35000,
            "hous_val_amt": 450000,
            "edctn_less_than_9_qty": 5000,
            "edctn_9_12_qty": 8000,
            "edctn_high_schl_qty": 20000,
            "edctn_some_clg_qty": 15000,
            "edctn_assoc_dgre_qty": 10000,
            "edctn_bchlr_dgre_qty": 25000,
            "edctn_prfsnl_qty": 7000,
            "per_urbn": 0.6,
            "per_sbrbn": 0.3,
            "per_farm": 0.1,
            "per_non_farm": 0.9,
            "per_less_than_9": 0.05,
            "per_9_to_12": 0.08,
            "per_hsd": 0.2,
            "per_some_clg": 0.15,
            "per_assoc": 0.1,
            "per_bchlr": 0.25,
            "per_prfsnl": 0.07
        }

    Output:
        HousingOutputModel = {
            "predicted_price": 525000.0,
            "metadata": {  # Only if 'with_metadata' is True
                "latency": 0.002555,
            }
        }
    """

    start = time.time()

    dataframe = pd.DataFrame([data.model_dump()])

    prediction = model.predict(dataframe)

    metadata = {}

    if with_metadata:
        end = time.time()

        metadata = {
            "latency": end - start
        }

    return HousingOutputModel(prediction=prediction, metadata=metadata)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
