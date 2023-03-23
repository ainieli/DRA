import tensorflow as tf
from tensorflow.keras import Model

from hanser.models.layers import GlobalAvgPool, Linear, Dropout
from hanser.models.common.resnet import BasicBlock, Bottleneck
from hanser.models.imagenet.stem import ResNetStem
from hanser.models.common.modules import make_layer

from hanser.transform import normalize

from auto_augment.dra import DifferentiableRandAugment as DRA


def _get_kwargs(kwargs, i):
    d = {}
    for k, v in kwargs.items():
        if isinstance(v, tuple) and len(v) == 4:
            d[k] = v[i]
        else:
            d[k] = v
    return d


class _ResNet(Model):

    def __init__(self, stem, block, layers, num_classes=1000, channels=(64, 128, 256, 512),
                 strides=(1, 2, 2, 2), dropout=0,
                 learnable_w=False, fix_mean=None, fix_std=None, scale=1.0,
                 tau=1.0, num_ops_per_sub_policy=2,
                 norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225], resolution=224, augment_style='RA',
                 p_min_t=0.2, p_max_t=0.8,
                 **kwargs):
        super().__init__()

        self.stem = stem
        c_in = stem.out_channels

        blocks = (block,) * 4 if not isinstance(block, tuple) else block
        for i, (block, c, n, s) in enumerate(zip(blocks, channels, layers, strides)):
            layer = make_layer(
                block, c_in, c, n, s, **_get_kwargs(kwargs, i))
            c_in = c * block.expansion
            setattr(self, "layer" + str(i + 1), layer)

        self.avgpool = GlobalAvgPool()
        self.dropout = Dropout(dropout) if dropout else None
        self.fc = Linear(c_in, num_classes)

        self.da = DRA(num_ops_per_sub_policy=num_ops_per_sub_policy, tau=tau,
                        learnable_w=learnable_w, fix_mean=fix_mean, fix_std=fix_std, scale=scale,
                        resolution=resolution, augment_style=augment_style, p_min_t=p_min_t, p_max_t=p_max_t)
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def param_splits(self):
        return slice(None, -3), slice(-3, None)

    def net_forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        return x

    def call(self, x, training=True):
        ori = x
        if training:
            # Apply augment to each image
            n, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
            x = tf.map_fn(self.da, x)
            x = tf.reshape(x, [n, h, w, c])

        x = normalize(x, self.norm_mean, self.norm_std)
        x = self.net_forward(x)

        ori = normalize(ori, self.norm_mean, self.norm_std)
        ori = self.net_forward(ori)

        return x, ori


class ResNet(_ResNet):

    def __init__(self, block, layers, num_classes=1000, channels=(64, 64, 128, 256, 512),
                 dropout=0, zero_init_residual=False,
                 learnable_w=False, fix_mean=None, fix_std=None, scale=1.0,
                 tau=1.0, num_ops_per_sub_policy=2,
                 norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225],
                 resolution=224, augment_style='RA', p_min_t=0.2, p_max_t=0.8):
        stem_channels, *channels = channels
        stem = ResNetStem(stem_channels)
        super().__init__(stem, block, layers, num_classes, channels,
                         dropout=dropout, zero_init_residual=zero_init_residual,
                         learnable_w=learnable_w, fix_mean=fix_mean, fix_std=fix_std, scale=scale,
                         tau=tau, num_ops_per_sub_policy=num_ops_per_sub_policy,
                         norm_mean=norm_mean, norm_std=norm_std,
                         resolution=resolution, augment_style=augment_style,
                         p_min_t=p_min_t, p_max_t=p_max_t)


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

def resnet200(**kwargs):
    return ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)