from tensorflow.keras.metrics import Recall, Accuracy, Precision


def compare_batch(preds, true_labels):
    comparison = [1 if prediction > 0.5 else 0 for prediction in preds]
    rec = Recall()
    rec.update_state(true_labels, preds)
    rec_result = rec.result().numpy()

    prec = Precision()
    prec.update_state(true_labels, preds)
    prec_result = prec.result().numpy()
    return {"recall": rec_result, "precision": prec_result}
