import json
from re import M

import numpy as np
from flask import Flask, render_template, request, redirect
from PIL import Image
from tensorflow import keras
from tensorflow.keras.utils import img_to_array
import os
from werkzeug.utils import secure_filename
from configs import config
from logs.log_config import logging
from training import train

app = Flask(__name__)
# model = keras.models.load_model("models/dog_vs_cat_cnn_v1.h5")
model = keras.models.load_model("models/dog_vs_cat_inception_transfer_learning_v1.h5")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/to_train", methods=["GET", "POST"])
def to_train():
    if request.method == "GET":
        print("get method")
        return render_template("train.html")
    else:
        print("post method")
        return_message = train.train_model()
        return render_template("train.html", message=return_message)


@app.route("/train", methods=["POST"])
def train_model():
    return_message = train.train_model()
    return render_template("train.html", message=return_message)


@app.route("/upload_image", methods=["POST"])
def upload_image():
    if "cat_images" not in request.files or "dog_images" not in request.files:
        return redirect(request.url)
    cat_images = request.files.getlist("cat_images")
    dog_images = request.files.getlist("dog_images")
    for cat_img in cat_images:
        cat_filename = cat_img.filename
        folder_path = os.path.join(config.BASE_UPLOAD_FOLDER_PATH, "cat")
        cat_img.save(os.path.join(folder_path, secure_filename(cat_filename)))
    for dog_img in dog_images:
        dog_filename = dog_img.filename
        folder_path = os.path.join(config.BASE_UPLOAD_FOLDER_PATH, "dog")
        dog_img.save(os.path.join(folder_path, secure_filename(dog_filename)))
    return render_template(
        "train.html", message=str("Images uploaded successfully. Please start training")
    )


@app.route("/to_predict")
def to_predict():
    return render_template("predict.html")


@app.route("/predict", methods=["POST"])
def predict():
    img_file = request.files["image_1"]
    if not img_file:
        return render_template("predict.html", error_message="Please select an image")

    org_img = Image.open(img_file.stream)
    img_array = img_to_array(org_img)
    input_img_shape = img_array.shape[0]
    if input_img_shape < config.TRANSFER_LEARNING_IMAGE_HEIGHT:
        return render_template("predict.html", error_message=str("invalid img size"))

    resized_img = np.resize(
        img_array,
        (
            config.TRANSFER_LEARNING_IMAGE_HEIGHT,
            config.TRANSFER_LEARNING_IMAGE_WIDTH,
            3,
        ),
    )
    resized_img /= 255
    resized_img = np.expand_dims(resized_img, axis=0)
    # thumbnail_img = np.resize(org_img, (10, 10))
    # input_data = org_img.thumbnail((10, 10))

    pred = model.predict(resized_img, batch_size=1)
    output_category = ""
    pred_val = pred[0][0]
    if pred_val >= config.THRESHOLD:
        output_category = "dog"
    else:
        output_category = "cat"
    logging.info(f"pred: {pred}")
    logging.info(f"pred_val: {pred_val}")
    return render_template(
        "predict.html", input_value=org_img, prediction_value=str(output_category)
    )


if __name__ == "__main__":
    app.run(debug=True)
