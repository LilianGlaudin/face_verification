import tensorflow as tf
from preprocessing.preprocessing import preprocess_twin


def load_data(anc_path, pos_path, neg_path):
    anchor = tf.data.Dataset.list_files(anc_path + "/*.jpg").take(300)
    positive = tf.data.Dataset.list_files(pos_path + "/*.jpg").take(300)
    negative = tf.data.Dataset.list_files(neg_path + "/*.jpg").take(300)
    return anchor, positive, negative


def build_data(anchor, positive, negative):
    positives = tf.data.Dataset.zip(
        (anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor))))
    )
    negatives = tf.data.Dataset.zip(
        (anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor))))
    )
    data = positives.concatenate(negatives)
    return data


def prepare_data(data):
    data = data.map(preprocess_twin)
    data = data.cache()
    data = data.shuffle(1024)
    return data


def prepare_train(data, split=0.7, batch_size=16):
    train_data = data.take(round(len(data) * split))
    train_data = train_data.batch(batch_size)
    train_data = train_data.prefetch(8)
    return train_data


def prepare_test(data, split=0.7, batch_size=16):
    test_data = data.skip(round(len(data) * split))
    test_data = test_data.take(round(len(data) * 1 - split))
    test_data = test_data.batch(batch_size)
    test_data = test_data.prefetch(8)
    return test_data


def prepare(anc_path, pos_path, neg_path):
    anchor, positive, negative = load_data(anc_path, pos_path, neg_path)
    data = build_data(anchor, positive, negative)
    data = prepare_data(data)
    ds_train = prepare_train(data)
    ds_test = prepare_test(data)
    return ds_train, ds_test
