import uvicorn
from fastapi import FastAPI
from pydantic import create_model, BaseModel
import sklearn
import pickle
import json
import pandas as pd

app = FastAPI(title="Housing Pricing Model",
              version="1.0.0",
              docs_url="/docs",
              redoc_url="/redoc",
              openapi_url="/openapi.json")

model: sklearn.base.BaseEstimator = pickle.load(open("/data/model/model.pkl", "rb"))

with open("/data/model/model_features.json", "r", encoding="utf-8") as f:
    model_features = json.load(f)

features_config = {v: (float, ...) for v in model_features}

demographics = pd.read_csv("/data/zipcode_demographics.csv", dtype={'zipcode': str})

HousingInputModel = create_model("HousingModel", **features_config)


class HousingOutputModel(BaseModel):
    prediction: float
    metadata: dict


@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/predict_raw")
async def predict_raw(data: dict) -> HousingOutputModel:

    data |= HousingInputModel.model_dump()

    data = HousingInputModel(**data)

    print(data)

    return await predict(data)

@app.post("/predict")
async def predict(data: HousingInputModel) -> HousingOutputModel:

    dataframe = pd.DataFrame(data.dict())

    dataframe = dataframe.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")

    prediction = model.predict(dataframe.dict())

    metadata = {
        "mse_error": 0
    }

    return HousingOutputModel(prediction=prediction, metadata=metadata)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=80, reload=True)