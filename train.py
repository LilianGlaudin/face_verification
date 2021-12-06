import tensorflow as tf
import os

from training import train
from dataloader import prepare
from nn import build_siamese_model

POS_PATH = os.path.join("dataset", "positive")
NEG_PATH = os.path.join("dataset", "negative")
ANC_PATH = os.path.join("dataset", "anchor")

train_data, test_data = prepare(ANC_PATH, POS_PATH, NEG_PATH)

model = build_siamese_model(nb_embeddings=4096)

binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)
checkpoint_path = "./training_checkpoints"

EPOCHS = 50

train(train_data, EPOCHS, model, opt, binary_cross_loss, checkpoint_path)

model.save("siamese_model.h5")
