import tensorflow as tf
from hanser.train.learner import Learner, cast

from loss.KLD import KL_divergence


class OGCDDALearner(Learner):

    def __init__(self, model, criterion, optimizer_arch, optimizer_model,
                 grad_clip_norm=0.0, train_arch=False, kl_ratio=True,
                 **kwargs):
        self.grad_clip_norm = grad_clip_norm
        self.train_arch = train_arch
        self.kl_ratio = kl_ratio
        self.kl = KL_divergence()
        super().__init__(model, criterion, (optimizer_arch, optimizer_model), **kwargs)

    @tf.function(experimental_compile=False)
    def _train_arch(self, batch):
        model = self.model
        arch_params = model.trainable_variables[model.param_splits()[1]]
        optimizer_arch = self.optimizers[0]

        input_search, target_search = batch
        with tf.GradientTape() as tape:
            input_search = cast(input_search, self.dtype)
            logits_search, ori_search = self.model(input_search, training=True)
            logits_search = cast(logits_search, tf.float32)

            if self.kl_ratio > 0:
                kl_loss = self.kl_ratio * self.kl(ori_search, logits_search)
            else:
                kl_loss = 0.

            per_example_loss = self.criterion(target_search, logits_search) + kl_loss
            loss_search = self.reduce_loss(per_example_loss)
        grads = tape.gradient(loss_search, arch_params)
        self.apply_gradients(optimizer_arch, grads, arch_params)

    @tf.function(experimental_compile=False)
    def _train_model(self, batch):
        model = self.model
        model_params = model.trainable_variables[model.param_splits()[0]]
        optimizer_model = self.optimizers[1]

        input, target = batch
        with tf.GradientTape() as tape:
            input = cast(input, self.dtype)
            logits = model(input, training=True)[0]
            logits = cast(logits, tf.float32)

            per_example_loss = self.criterion(target, logits)
            loss = self.reduce_loss(per_example_loss)

        grads = tape.gradient(loss, model_params)
        self.apply_gradients(optimizer_model, grads, model_params, self.grad_clip_norm)
        self.update_metrics(self.train_metrics, target, logits, per_example_loss)

    def train_batch(self, batch):
        if self.train_arch:
            self._train_arch(batch[1])
        self._train_model(batch[0])

    def eval_batch(self, batch):
        inputs, target = batch
        inputs = cast(inputs, self.dtype)
        preds = self.model(inputs, training=False)[0]
        preds = cast(preds, tf.float32)
        self.update_metrics(self.eval_metrics, target, preds)