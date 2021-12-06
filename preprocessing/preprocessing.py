import tensorflow as tf


def preprocess(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0

    return img


def preprocess_twin(anchor_img, input_img, label):
    return (preprocess(anchor_img), preprocess(input_img), label)
