import tensorflow as tf
from tensorflow.keras.layers import Softmax

from hanser.train.callbacks import Callback


class ViewTrainableParams(Callback):

    def __init__(self, eval_freq=10, print_fn=print):
        super().__init__()
        self.print_fn = print_fn
        self.eval_freq = eval_freq
        self.softmax = Softmax(axis=-1)

    def after_epoch(self, state):
        learner = self.learner
        if (self.learner.epoch + 1) % self.eval_freq == 0:
            self.print_fn("[sp weights]:", learner.model.da.sp_weights)
            self.print_fn("[sp softmax weights]:", self.softmax(learner.model.da.sp_weights))
            self.print_fn("[sp magnitudes mean]:", learner.model.da.sp_magnitudes_mean)
            self.print_fn('[sp magnitudes std]:', learner.model.da.sp_magnitudes_std)
