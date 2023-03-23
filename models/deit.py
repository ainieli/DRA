import tensorflow as tf
from tensorflow.keras import Model

from hanser.models.transformer.vision.deit import deit_ti
from hanser.transform import normalize

from auto_augment.dra import DifferentiableRandAugment as DRA


class DeiT(Model):
    def __init__(self, model_name='tiny', patch_size=16, drop_rate=0.0, drop_path=0.1, num_classes=1000,
                 learnable_w=False, fix_mean=None, fix_std=None, scale=1.0,
                 tau=1.0, num_ops_per_sub_policy=2,
                 norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225],
                 resolution=224, augment_style='RA', p_min_t=0.2, p_max_t=0.8):
        super().__init__()

        if model_name == 'tiny':
            self.model = deit_ti(patch_size=patch_size, drop_rate=drop_rate, drop_path=drop_path, num_classes=num_classes)
        else:
            raise NotImplementedError('Support Tiny only')

        self.da = DRA(num_ops_per_sub_policy=num_ops_per_sub_policy, tau=tau,
                        learnable_w=learnable_w, fix_mean=fix_mean, fix_std=fix_std, scale=scale,
                        resolution=resolution, augment_style=augment_style, p_min_t=p_min_t, p_max_t=p_max_t)
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def param_splits(self):
        return slice(None, -3), slice(-3, None)

    def net_forward(self, x):
        x = self.model(x)
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
