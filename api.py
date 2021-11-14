import sys
import traceback

import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/predict", methods=['POST',"GET"])
def predict():
    if model:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(model(query))

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return 'No model here to use'


if __name__ == '__main__':

    model = joblib.load("model.pkl")  # Load "model.pkl"
    print('Model loaded')

    model_columns = joblib.load("model_columns.pkl")  # Load "model_columns.pkl"
    print('Model columns loaded')

    app.run(debug=True)
