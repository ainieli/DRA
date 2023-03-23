import random
import collections
import itertools

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant

from hanser.ops import gumbel_softmax, sample_relaxed_bernoulli

from auto_augment.operations import ShearX, ShearY, TranslateX, TranslateY, Rotate, Brightness, Color, \
    Sharpness, Contrast, Cutout, Solarize, Posterize, Equalize, AutoContrast, Invert, SolarizeAdd, Identity


CADIDATE_OPS_DICT_32 = collections.OrderedDict({
            'ShearX': ShearX(-0.3, 0.3),    # xla not supported
            'ShearY': ShearY(-0.3, 0.3),    # xla not supported
            'TranslateX': TranslateX(-10, 10),  # xla not supported, CIFAR RANGE
            'TranslateY': TranslateY(-10, 10),  # xla not supported, CIFAR RANGE
            'Rotate': Rotate(-30, 30),  # xla not supported
            'Brightness': Brightness(0.1, 1.9),
            'Color': Color(0.1, 1.9),
            'Sharpness': Sharpness(0.1, 1.9),
            'Contrast': Contrast(0.1, 1.9),  # xla not supported
            'Cutout': Cutout(0, 0.2),
            'Solarize': Solarize(0, 256),
            'Posterize': Posterize(0, 4),
            'Equalize': Equalize(0, 1),     # xla not supported
            'AutoContrast': AutoContrast(0, 1),   # xla not supported
            'Invert': Invert(0, 1),
            # Identity(0, 1),
            'SolarizeAdd': SolarizeAdd(0, 110),
        })

CADIDATE_OPS_DICT_224 = collections.OrderedDict({
            'ShearX': ShearX(-0.3, 0.3),    # xla not supported
            'ShearY': ShearY(-0.3, 0.3),    # xla not supported
            'TranslateX': TranslateX(-100, 100),  # xla not supported
            'TranslateY': TranslateY(-100, 100),  # xla not supported
            'Rotate': Rotate(-30, 30),  # xla not supported
            'Brightness': Brightness(0.1, 1.9),
            'Color': Color(0.1, 1.9),
            'Sharpness': Sharpness(0.1, 1.9),
            'Contrast': Contrast(0.1, 1.9),  # xla not supported
            'Cutout': Cutout(0, 0.2),
            'Solarize': Solarize(0, 256),
            'Posterize': Posterize(0, 4),
            'Equalize': Equalize(0, 1),     # xla not supported
            'AutoContrast': AutoContrast(0, 1),   # xla not supported
            'Invert': Invert(0, 1),
            # Identity(0, 1),
            'SolarizeAdd': SolarizeAdd(0, 110),
        })

CADIDATE_OPS_DICT_160 = collections.OrderedDict({
            'ShearX': ShearX(-0.3, 0.3),    # xla not supported
            'ShearY': ShearY(-0.3, 0.3),    # xla not supported
            'TranslateX': TranslateX(-72, 72),  # xla not supported
            'TranslateY': TranslateY(-72, 72),  # xla not supported
            'Rotate': Rotate(-30, 30),  # xla not supported
            'Brightness': Brightness(0.1, 1.9),
            'Color': Color(0.1, 1.9),
            'Sharpness': Sharpness(0.1, 1.9),
            'Contrast': Contrast(0.1, 1.9),  # xla not supported
            'Cutout': Cutout(0, 0.2),
            'Solarize': Solarize(0, 256),
            'Posterize': Posterize(0, 4),
            'Equalize': Equalize(0, 1),     # xla not supported
            'AutoContrast': AutoContrast(0, 1),   # xla not supported
            'Invert': Invert(0, 1),
            # Identity(0, 1),
            'SolarizeAdd': SolarizeAdd(0, 110),
        })


class DifferentiableRandAugment(Layer):

    def __init__(self, num_ops_per_sub_policy=2, tau=1.0, learnable_w=False, fix_mean=None, fix_std=None, scale=1.0,
                 resolution=224, augment_style='RA', p_min_t=0.2, p_max_t=0.8):
        super().__init__()
        assert augment_style in ['RA']

        if augment_style == 'RA':
            # if fix_mean is None: learnable mean
            # else: fixed value
            if resolution == 224:
                CADIDATE_OPS_LIST = [op for op in CADIDATE_OPS_DICT_224.values()]
            elif resolution == 160:
                CADIDATE_OPS_LIST = [op for op in CADIDATE_OPS_DICT_160.values()]
            elif resolution == 32:
                CADIDATE_OPS_LIST = [op for op in CADIDATE_OPS_DICT_32.values()]
        else:
            raise NotImplementedError()

        self.cadidate_ops = CADIDATE_OPS_LIST
        num_candidate_ops = len(self.cadidate_ops)
        self.sp_weights = self.add_weight(
            name="sp_weights", shape=(num_ops_per_sub_policy, num_candidate_ops), dtype=tf.float32, trainable=True,
            initializer=Constant(1e-3), experimental_autocast=False)
        self.sp_magnitudes_mean = self.add_weight(
            name="sp_magnitudes_mean", shape=(num_ops_per_sub_policy, num_candidate_ops), dtype=tf.float32, trainable=True,
            initializer=Constant(0.5), experimental_autocast=False)
        self.sp_magnitudes_std = self.add_weight(
            name="sp_magnitudes_std", shape=(num_ops_per_sub_policy, num_candidate_ops), dtype=tf.float32, trainable=True,
            initializer=Constant(0.1), experimental_autocast=False)
        self.num_ops_per_sub_policy = num_ops_per_sub_policy
        self.tau = tau
        self.num_candidate_ops = num_candidate_ops
        self.learnable_w = learnable_w
        self.scale = scale
        self.p_min_t = p_min_t
        self.p_max_t = p_max_t

        self.fix_mean = fix_mean
        self.fix_std = fix_std
        self.augment_style = augment_style

    def clip_value(self):
        # Use "assign" instead of "=" to make sp_probs/sp_magnitudes differentiable
        self.sp_magnitudes_mean.assign(tf.clip_by_value(self.sp_magnitudes_mean, 0.0, 1.0 * self.scale))
        self.sp_magnitudes_std.assign(tf.clip_by_value(self.sp_magnitudes_std, 0.0, 1.0))

    def call(self, x):
        self.clip_value()
        # x: (N, H, W, 3), float32, 0-1
        if len(tf.shape(x)) == 3:
            x = x[None]
        if tf.reduce_max(x) <= 1:
            x = x * 255
        x = tf.cast(x, tf.float32)

        for i in range(self.num_ops_per_sub_policy):
            if self.learnable_w:
                hardwts = gumbel_softmax(self.sp_weights[i], tau=self.tau, hard=True)
            else:
                hardwts = tf.one_hot(
                    tf.random.uniform((), maxval=self.num_candidate_ops, dtype=tf.int32),
                    depth=self.num_candidate_ops
                )
            p = tf.random.uniform((), minval=self.p_min_t, maxval=self.p_max_t, dtype=tf.float32)

            xs = [
                tf.cond(hardwts[j] == 1,
                    lambda: apply_op(
                        x, self.cadidate_ops[j], p,
                        self.sp_magnitudes_mean[i, j],
                        self.sp_magnitudes_std[i, j],
                        self.fix_mean,
                        self.fix_std,
                        self.scale,
                        self.augment_style) * hardwts[j],
                    lambda: tf.ones_like(x) * hardwts[j])
                for j in range(len(self.cadidate_ops))
            ]
            x = tf.add_n(xs)

        x = x / 255.
        return x


def resample_normal(sp_m_mean, sp_m_std):
    eta = tf.random.normal((), mean=0., stddev=1., dtype=tf.float32)
    mag = sp_m_mean + eta * sp_m_std
    return mag


def apply_op(x, op, prob, sp_m_mean, sp_m_std, fix_mean, fix_std, scale, augment_style):
    if fix_mean is not None:
        if augment_style == 'RA':
            mean = fix_mean
        else:
            mean = tf.cast(tf.random.uniform((), maxval=31, dtype=tf.int32), dtype=tf.float32) / 30.
    else:
        mean = sp_m_mean

    if fix_std is not None:
        std = fix_std
    else:
        std = sp_m_std
    magnitude = resample_normal(mean, std)
    magnitude = tf.clip_by_value(magnitude, 0., 1. * scale)
    x_m = x + magnitude - tf.stop_gradient(magnitude)
    x = tf.cond(
        tf.random.uniform((), dtype=tf.float32) < prob,
        lambda: op(x_m, magnitude),
        lambda: x
    )
    return x



