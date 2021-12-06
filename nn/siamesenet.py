from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from utils import dist_l1


def conv_maxpool(inp, filters, kernel_size, pool_size, strides, activation="relu"):
    conv = Conv2D(filters, kernel_size, activation=activation)(inp)
    pool = MaxPooling2D(pool_size, strides, padding="same")(conv)
    return pool


def build_embedding(nb_embeddings=1024):
    inp = Input(shape=(100, 100, 3), name="input_image")

    b1 = conv_maxpool(inp, 64, (10, 10), 64, (2, 2))
    b2 = conv_maxpool(b1, 128, (7, 7), 64, (2, 2))
    b3 = conv_maxpool(b2, 128, (4, 4), 64, (2, 2))

    c4 = Conv2D(256, (4, 4), activation="relu")(b3)

    f1 = Flatten()(c4)
    d1 = Dense(nb_embeddings, activation="sigmoid")(f1)

    return Model(inputs=[inp], outputs=[d1], name="embedding")


def build_siamese_model(nb_embeddings):
    embedding = build_embedding(nb_embeddings)

    anchor = Input(name="anchor_img", shape=(100, 100, 3))
    input = Input(name="testing_img", shape=(100, 100, 3))

    dist = Lambda(dist_l1, name="distance")([embedding(anchor), embedding(input)])

    output = Dense(1, activation="sigmoid")(dist)

    return Model(
        inputs=[anchor, input],
        outputs=output,
        name="Siamese_Network",
    )
