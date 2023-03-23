import tensorflow as tf
from tensorflow.keras import Model

from hanser.models.layers import NormAct, Conv2d, GlobalAvgPool, Linear
from hanser.models.common.modules import make_layer
from hanser.models.common.preactresnet import BasicBlock
from hanser.transform import normalize

from auto_augment.dra import DifferentiableRandAugment as DRA
from auto_augment.operations import cutout


class _ResNet(Model):

    def __init__(self, depth, block, num_classes=10, channels=(16, 16, 32, 64), dropout=0,
                 learnable_w=False, fix_mean=None, fix_std=None, scale=1.0,
                 tau=1.0, num_ops_per_sub_policy=2,
                 norm_mean=[0.491, 0.482, 0.447], norm_std=[0.247, 0.243, 0.262],
                 cutout_len=16, resolution=32, augment_style='RA', p_min_t=0.2, p_max_t=0.8):
        super().__init__()
        layers = [(depth - 4) // 6] * 3

        stem_channels, *channels = channels

        self.stem = Conv2d(3, stem_channels, kernel_size=3)
        c_in = stem_channels

        strides = [1, 2, 2]
        for i, (c, n, s) in enumerate(zip(channels, layers, strides)):
            layer = make_layer(block, c_in, c, n, s,
                               dropout=dropout)
            c_in = c * block.expansion
            setattr(self, "layer" + str(i + 1), layer)

        self.norm_act = NormAct(c_in)
        self.avgpool = GlobalAvgPool()
        self.fc = Linear(c_in, num_classes)

        self.da = DRA(num_ops_per_sub_policy=num_ops_per_sub_policy, tau=tau,
                        learnable_w=learnable_w, fix_mean=fix_mean, fix_std=fix_std, scale=scale,
                        resolution=resolution, augment_style=augment_style, p_min_t=p_min_t, p_max_t=p_max_t)
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.cutout_len = cutout_len

    def param_splits(self):
        return slice(None, -3), slice(-3, None)

    def net_forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.norm_act(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

    def call(self, x, training=True):
        ori = x
        if training:
            n, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
            x = tf.map_fn(self.da, x)
            x = tf.reshape(x, [n, h, w, c])
            if self.cutout_len > 0:
                x = cutout(x, self.cutout_len)

        x = normalize(x, self.norm_mean, self.norm_std)
        x = self.net_forward(x)

        ori = normalize(ori, self.norm_mean, self.norm_std)
        ori = self.net_forward(ori)

        return x, ori


class ResNet(_ResNet):

    def __init__(self,
                 # network params
                 depth, k, block=BasicBlock, num_classes=10, channels=(16, 16, 32, 64),
                 dropout=0,
                 # augment params
                 learnable_w=False, fix_mean=None, fix_std=None, scale=1.0,
                 tau=1.0, num_ops_per_sub_policy=2,
                 norm_mean=[0.491, 0.482, 0.447], norm_std=[0.247, 0.243, 0.262],
                 cutout_len=16, resolution=32, augment_style='RA', p_min_t=0.2, p_max_t=0.8):
        channels = (channels[0],) + tuple(c * k for c in channels[1:])
        super().__init__(depth, block, num_classes, channels, dropout,
                         learnable_w, fix_mean, fix_std, scale,
                         tau, num_ops_per_sub_policy, norm_mean, norm_std,
                         cutout_len, resolution, augment_style, p_min_t, p_max_t)


def WRN_40_2(**kwargs):
    return ResNet(depth=40, k=2, block=BasicBlock, **kwargs)


def WRN_28_10(**kwargs):
    return ResNet(depth=28, k=10, block=BasicBlock, **kwargs)
