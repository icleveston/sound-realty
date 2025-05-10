import csv
from typing import Any
import requests


def post(payload: dict, url: str = "http://192.168.49.2:32622/predict?with_metadata=true") -> Any | None:
    """
    Send the payload to the Api.
    :param payload: a dictionary containing the payload
    :param url: API url
    :return: the prediction and metadata
    """

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(response.json())

def main():

    while True:

        with open("./data/future_unseen_examples.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for line in reader:

                payload = {k: float(v) for k, v in line.items()}

                print(f"Request: {payload}")
                response = post(payload)
                print(f"Response: {response}\n")


if __name__ == "__main__":
    main()