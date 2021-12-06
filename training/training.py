import tensorflow as tf
import os


@tf.function
def train_step(batch, siamese_model, opt, binary_cross_loss):
    with tf.GradientTape() as tape:
        X = batch[:2]
        y = batch[2]

        preds = siamese_model(X, training=True)
        loss = binary_cross_loss(y, preds)
    print(loss)

    grad = tape.gradient(loss, siamese_model.trainable_variables)
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    return loss


def build_checkpoint(path, siamese_model, opt):
    checkpoint_dir = path
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)
    return checkpoint_prefix, checkpoint


def train(data, EPOCHS, siamese_model, opt, loss, checkpoint_path, checkpoint_every=10):
    print(f">>> Start training for {EPOCHS} epochs")
    checkpoint_prefix, checkpoint = build_checkpoint(
        checkpoint_path, siamese_model, opt
    )

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS} \n")
        progbar = tf.keras.utils.Progbar(len(data))

        for idx, batch in enumerate(data):
            train_step(batch, siamese_model, opt, loss)
            progbar.update(idx + 1)
        if epoch % checkpoint_every == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
