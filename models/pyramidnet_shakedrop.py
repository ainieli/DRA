import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer

from hanser.models.layers import Pool2d, Conv2d, Norm, Act, GlobalAvgPool, Linear
from hanser.models.modules import PadChannel
from hanser.models.cifar.shakedrop.layers import ShakeDrop
from hanser.transform import normalize

from auto_augment.dra import DifferentiableRandAugment as DRA
from auto_augment.operations import cutout

__all__ = [
    "PyramidNet"
]


class Shortcut(Sequential):
    def __init__(self, in_channels, out_channels, stride):
        layers = []
        if stride == 2:
            layers.append(Pool2d(2, 2, type='avg'))
        if in_channels != out_channels:
            layers.append((PadChannel(out_channels - in_channels)))
        super().__init__(layers)


class BasicBlock(Layer):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, p_shakedrop=0):
        super().__init__()
        branch1 = [
            Norm(in_channels),
            Conv2d(in_channels, channels, kernel_size=3, stride=stride,
                   norm='def', act='def'),
            Conv2d(channels, channels, kernel_size=3, norm='def'),
        ]
        if p_shakedrop:
            branch1.append(ShakeDrop(p_shakedrop, (-1, 1), (0, 1)))
        self.branch1 = Sequential(branch1)
        self.branch2 = Shortcut(in_channels, channels, stride)

    def call(self, x):
        return self.branch1(x) + self.branch2(x)


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, p_shakedrop=0):
        super().__init__()
        out_channels = channels * self.expansion
        branch1 = [
            Norm(in_channels),
            Conv2d(in_channels, channels, kernel_size=1,
                   norm='def', act='def'),
            Conv2d(channels, channels, kernel_size=3, stride=stride,
                   norm='def', act='def'),
            Conv2d(channels, out_channels, kernel_size=1,
                   norm='def'),
        ]
        if p_shakedrop:
            branch1.append(ShakeDrop(p_shakedrop, (-1, 1), (0, 1)))
        self.branch1 = Sequential(branch1)
        self.branch2 = Shortcut(in_channels, out_channels, stride)

    def call(self, x):
        return self.branch1(x) + self.branch2(x)


def rd(c):
    return int(round(c, 2))


class PyramidNet(Model):
    def __init__(self, start_channels, alpha, depth, block='bottleneck', p_shakedrop=0.5, num_classes=10,
                 learnable_w=False, fix_mean=None, fix_std=None, scale=1.0,
                 tau=1.0, num_ops_per_sub_policy=2,
                 norm_mean=[0.491, 0.482, 0.447], norm_std=[0.247, 0.243, 0.262],
                 cutout_len=16, resolution=32, augment_style='RA', p_min_t=0.2, p_max_t=0.8):
        super().__init__()

        if block == 'basic':
            num_layers = [(depth - 2) // 6] * 3
            block = BasicBlock
        elif block == 'bottleneck':
            num_layers = [(depth - 2) // 9] * 3
            block = Bottleneck
        else:
            raise ValueError("block must be `basic` or `bottleneck`, got %s" % block)

        self.num_layers = num_layers

        strides = [1, 2, 2]

        add_channel = alpha / sum(num_layers)
        in_channels = start_channels

        self.init_block = Conv2d(3, start_channels, kernel_size=3, norm='def')

        channels = start_channels
        k = 1
        units = []
        for n, s in zip(num_layers, strides):
            for i in range(n):
                stride = s if i == 0 else 1
                channels = channels + add_channel
                units.append(block(in_channels, rd(channels), stride=stride,
                                   p_shakedrop=k / sum(num_layers) * p_shakedrop))
                in_channels = rd(channels) * block.expansion
                k += 1

        self.units = units
        self.post_activ = Sequential([
            Norm(in_channels),
            Act(),
        ])

        assert (start_channels + alpha) * block.expansion == in_channels

        self.final_pool = GlobalAvgPool()
        self.fc = Linear(in_channels, num_classes)

        self.da = DRA(num_ops_per_sub_policy=num_ops_per_sub_policy, tau=tau,
                        learnable_w=learnable_w, fix_mean=fix_mean, fix_std=fix_std, scale=scale,
                        resolution=resolution, augment_style=augment_style, p_min_t=p_min_t, p_max_t=p_max_t)
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.cutout_len = cutout_len

    def param_splits(self):
        return slice(None, -3), slice(-3, None)

    def net_forward(self, x):
        x = self.init_block(x)
        for unit in self.units:
            x = unit(x)
        x = self.post_activ(x)
        x = self.final_pool(x)
        x = self.fc(x)
        return x

    def call(self, x, training=True):
        ori = x
        if training:
            # Apply augment to each image
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

