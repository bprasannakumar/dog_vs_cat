import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array

from PIL import Image
from datetime import datetime
from configs import config

# find out number of images for train and validation
train_cat_image_path = "training/data/train/cat"
train_dog_image_path = "training/data/train/dog"
val_cat_image_path = "training/data/val/cat"
val_dog_image_path = "training/data/val/dog"

train_cat_image_names = os.listdir(path=train_cat_image_path)
train_dog_image_names = os.listdir(path=train_dog_image_path)
val_cat_image_names = os.listdir(path=val_cat_image_path)
val_dog_image_names = os.listdir(path=val_dog_image_path)
total_train_images = len(train_cat_image_names) + len(train_dog_image_names)
total_val_images = len(val_cat_image_names) + len(val_dog_image_names)

# keeping image hight and weight small, so that I can train model quickly because of system configuration
IMAGE_HEIGHT = 150
IMAGE_WIDTH = 150
IMAGE_CHANNEL = 3
BATCH_SIZE = 32

TRAIN_STEPS_PER_EPOCHS = int(total_train_images / BATCH_SIZE)
VAL_STEPS_PER_EPOCHS = int(total_val_images / BATCH_SIZE)
print(
    f"TRAIN_STEPS_PER_EPOCHS: {TRAIN_STEPS_PER_EPOCHS}, VAL_STEPS_PER_EPOCHS: {VAL_STEPS_PER_EPOCHS}"
)

# using previously tuned hyperparameter values along with Dropout.  # {'units': 128, 'kernel_size': 3, 'learning_rate': 0.0001}
# can also add dropout to hyperparameter list and find best value using kerastuner
# also adding data_augmentation
def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=3, activation=activations.relu))
    model.add(
        layers.MaxPool2D(
            pool_size=(2, 2), strides=1, padding="valid", data_format="channels_last"
        )
    )
    model.add(layers.Conv2D(filters=128, kernel_size=3, activation=activations.relu))
    model.add(
        layers.MaxPool2D(
            pool_size=(2, 2), strides=1, padding="valid", data_format="channels_last"
        )
    )
    model.add(layers.Flatten())
    model.add(layers.Dense(units=128, activation=activations.relu))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=1, activation=activations.sigmoid))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.01),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model():
    print("Model training started")
    # create traning and validation generators
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # load data from train and validation directory
    train_generator = train_datagen.flow_from_directory(
        "training/data/train",
        target_size=(
            config.TRANSFER_LEARNING_IMAGE_HEIGHT,
            config.TRANSFER_LEARNING_IMAGE_WIDTH,
        ),
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )
    val_generator = val_datagen.flow_from_directory(
        "training/data/val",
        target_size=(
            config.TRANSFER_LEARNING_IMAGE_HEIGHT,
            config.TRANSFER_LEARNING_IMAGE_WIDTH,
        ),
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )

    # keeping epochs small, again to train the model quickly
    model = build_model()
    model_object = model.fit(
        train_generator,
        steps_per_epoch=TRAIN_STEPS_PER_EPOCHS,
        epochs=5,
        validation_data=val_generator,
        validation_steps=VAL_STEPS_PER_EPOCHS,
    )
    model_name = f"models/trained_model_{datetime.utcnow()}.h5"
    model_object.save(model_name)
    return f"Trained and saved model as {model_name}"

