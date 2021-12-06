import tensorflow as tf
import cv2
import os
import numpy as np
from preprocessing.preprocessing import preprocess


def check_predictions(model, detection_threshold, verification_threshold):
    output_image = preprocess(
        os.path.join("output", "output_image", "output_image.jpg")
    )

    predict_feature = model.layers[2].predict(np.expand_dims(output_image, axis=0))
    preds_feat = np.tile(predict_feature, (len(features), 1))
    ans = model.layers[3](preds_feat, features)
    results = model.layers[4](ans)
    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(features)
    verified = verification > verification_threshold

    return results, verified, detection, verification


model = tf.keras.models.load_model(
    "siamesemodel.h5",
    custom_objects={
        "BinaryCrossentropy": tf.losses.BinaryCrossentropy,
    },
)


features = np.genfromtxt("features.csv", delimiter=",")

cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    frame = frame[120 : 120 + 250, 200 : 200 + 250, :]

    cv2.imshow("Verification", frame)
    if cv2.waitKey(10) & 0xFF == ord("v"):
        cv2.imwrite(os.path.join("output", "output_image", "output_image.jpg"), frame)
        results, verified, _, _ = check_predictions(model, 0.9, 0.9)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
