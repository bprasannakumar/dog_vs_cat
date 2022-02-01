import json

import joblib
import numpy as np
from flask import Flask, json, render_template, request

app = Flask(__name__)
model = joblib.load("models/xgb_model3")


@app.route("/")
def home():
    return render_template("index.html")


# @app.route("/predict", methods=["POST"])
# def predict():
#     """
#     For rendering results on HTML GUI
#     """
#     int_features = [int(x) for x in request.form.values()]
#     t = int_features[0]
#     s = int_features[1]
#     prediction = preprocess_and_predict([t], [s])
#     input_data = [s, t]
#     output = prediction[0]
#     return render_template("index.html", input_value="Input Value: " + str(input_data), prediction_value=output)


# @app.route("/predict_json", methods=["POST"])
# def predict_json():
#     input_data = request.form['input_data']
#     input_data = input_data.strip()
#     warning_message = "Please provide input s and t in JSON format"
#     if input_data is None or len(input_data) == 0:
#         return render_template("index.html", message=warning_message)
#     try:
#         input_data = json.loads(input_data)
#     except:
#         return render_template("index.html", message=warning_message)
#     t = input_data.get("t")
#     s = input_data.get("s")
#     if t is None or s is None:
#         return render_template("index.html", message=warning_message)
#     predictions = preprocess_and_predict(t, s)
#     return render_template("index.html", input_values="Input Values: " + str(input_data), predicted_values=predictions)


# def preprocess_and_predict(t, s):
#     test_data = list(zip(t, s))
#     final_test_data = []
#     for data in test_data:
#         final_test_data.append(
#             [data[0], data[1], data[1]*data[1], data[1]*data[1]*data[1]])
#     minMaxScaler = joblib.load("models/minMaxScaler")
#     final_test_data_trans = minMaxScaler.transform(np.array(final_test_data))
#     predictions = model.predict(final_test_data_trans)
#     return [round(val, 10) for val in predictions.tolist()]


if __name__ == "__main__":
    app.run(debug=True)
