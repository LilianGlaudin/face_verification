import os

import numpy as np

from preprocessing.preprocessing import preprocess


def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(os.path.join("application_data", "verification_images")):
        input_img = preprocess(
            os.path.join("application_data", "input_image", "input_image.jpg")
        )
        validation_img = preprocess(
            os.path.join("application_data", "verification_images", image)
        )

        # Make Predictions
        result = model.predict(
            list(np.expand_dims([input_img, validation_img], axis=1))
        )
        results.append(result)

    # Detection Threshold: Metric above which a prediciton is considered positive
    detection = np.sum(np.array(results) > detection_threshold)

    # Verification Threshold: Proportion of positive predictions / total positive samples
    verification = detection / len(
        os.listdir(os.path.join("application_data", "verification_images"))
    )
    verified = verification > verification_threshold

    return results, verified
