import tensorflow as tf


class KL_divergence:

    def __init__(self, from_logits=True):
        self.kl = tf.keras.losses.KLDivergence()
        self.from_logits = from_logits

    def __call__(self, target, pred):
        if self.from_logits:
            pred = tf.nn.softmax(pred, axis=-1)
            target = tf.nn.softmax(target, axis=-1)
        loss = self.kl(target, pred)
        return loss

