import csv
from typing import Any
import requests
import json


def post(payload: str, url: str = "http://localhost:8080/predict_raw") -> Any | None:

    headers = {
        "Content-Type": "application/json"
    }

    try:

        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            return response.json()

    except requests.exceptions.RequestException:
        return None

def main():

    with open("../data/future_unseen_examples.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for line in reader:

            payload = json.dumps({k: float(v) for k, v in line.items()})

            print(f"Requesting: {payload}")
            response = post(payload)
            print(f"Response: {response}\n")


if __name__ == "__main__":
    main()