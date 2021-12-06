import tensorflow as tf
import numpy as np
import glob
from preprocessing import preprocess


print("TODO")
model = tf.keras.models.load_model(
    "siamese_model.h5",
    custom_objects={
        "BinaryCrossentropy": tf.losses.BinaryCrossentropy,
    },
)

paths = glob.glob("application_data/verification_images/*.jpg")

list_data = np.array([preprocess(path) for path in paths])

embedding = model.layers[2]
features = embedding.predict(list_data)

np.savetxt("features.csv", features, delimiter=",")
