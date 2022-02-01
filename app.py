import json

import joblib
import numpy as np
from flask import Flask, json, render_template, request

app = Flask(__name__)
# model = joblib.load("models/xgb_model3")


@app.route("/")
def home():
    return "Home page"
    # return render_template("index.html")


@app.route("/predict")
def predict():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
