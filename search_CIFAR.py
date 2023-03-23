# Applied on GPU device
from toolz import curry

import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, CategoricalCrossentropy
from tensorflow_addons.optimizers import AdamW


from hanser.distribute import setup_runtime, distribute_datasets
from hanser.transform import random_crop, normalize, to_tensor, cutout, mixup
from hanser.train.optimizers import SGD
from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy
from hanser.models.layers import set_defaults

from models.preact_resnet import ResNet
from models.pyramidnet_shakedrop import PyramidNet
from metrics.callback import ViewTrainableParams
from learner import OGCDDALearner
from datasets.split import make_cifar10_dataset, make_cifar100_dataset
from datasets import CIFAR_MEAN, CIFAR_STD


RES = 32
eval_batch_size = 128
epochs = 20
arch_lr = 5e-3
arch_wd = 0
tau = 0.5
ra_depth = 2
cutout_len = 16
kl_ratio = 1.
dropout = 0.3
shakedrop = 0.5
augment_style = 'RA'
p_min_t = 0.2
p_max_t = 0.8
grad_clip_norm = 5.0

model_name = 'WRN_28_10'
if model_name == 'WRN_28_10' or model_name == 'WRN_40_2':
    batch_size = 128
    base_lr = batch_size / 128 * 0.1
    wd = 5e-4
elif model_name == 'PyramidNet+ShakeDrop':
    batch_size = 32
    base_lr = batch_size / 256 * 0.1
    wd = 5e-5
else:
    raise NotImplementedError('Require hyperparameter customization for model %s' % model_name)

DATASET = 'CIFAR10'
if DATASET == 'CIFAR10':
    NUM_CLASS = 10
elif DATASET == 'CIFAR100':
    NUM_CLASS = 100
else:
    raise NotImplementedError('CIFAR dataset only!')

learnable_w = True
scale = 1.
fix_mean = None
fix_std = None

work_dir = './%s' % DATASET

set_defaults({
    'bn': {
        'affine': False,
    },
    'fixed_padding': True,
})

@curry
def transform(image, label, training):
    if training:
        image = random_crop(image, (RES, RES), (RES // 8, RES // 8))
        image = tf.image.random_flip_left_right(image)
    image, label = to_tensor(image, label)
    label = tf.one_hot(label, NUM_CLASS)
    # DRA/RA is applied as layers in the model
    return image, label


if DATASET == 'CIFAR10':
    ds_train, ds_test, steps_per_epoch, test_steps = make_cifar10_dataset(
        batch_size, eval_batch_size, transform, sub_ratio=0.08)
else:
    ds_train, ds_test, steps_per_epoch, test_steps = make_cifar100_dataset(
        batch_size, eval_batch_size, transform, sub_ratio=0.08)


setup_runtime(fp16=True)
ds_train, ds_test = distribute_datasets(ds_train, ds_test)
if model_name == 'WRN_28_10':
    model = ResNet(depth=28, k=10, num_classes=NUM_CLASS, dropout=dropout,
               learnable_w=learnable_w,
               fix_mean=fix_mean, fix_std=fix_std, scale=scale,
               tau=tau, num_ops_per_sub_policy=ra_depth,
               norm_mean=CIFAR_MEAN, norm_std=CIFAR_STD,
               cutout_len=cutout_len, resolution=RES, augment_style=augment_style,
               p_min_t=p_min_t, p_max_t=p_max_t)
elif model_name == 'WRN_40_2':
    model = ResNet(depth=40, k=2, num_classes=NUM_CLASS, dropout=dropout,
                   learnable_w=learnable_w,
                   fix_mean=fix_mean, fix_std=fix_std, scale=scale,
                   tau=tau, num_ops_per_sub_policy=ra_depth,
                   norm_mean=CIFAR_MEAN, norm_std=CIFAR_STD,
                   cutout_len=cutout_len, resolution=RES, augment_style=augment_style,
                   p_min_t=p_min_t, p_max_t=p_max_t)
elif model_name == 'PyramidNet_ShakeDrop':
    model = PyramidNet(16, depth=272, alpha=200, block='bottleneck', p_shakedrop=shakedrop, num_classes=NUM_CLASS,
               learnable_w=learnable_w,
               fix_mean=fix_mean, fix_std=fix_std, scale=scale,
               tau=tau, num_ops_per_sub_policy=ra_depth,
               norm_mean=CIFAR_MEAN, norm_std=CIFAR_STD,
               cutout_len=cutout_len, resolution=RES, augment_style=augment_style,
               p_min_t=p_min_t, p_max_t=p_max_t)
else:
    raise NotImplementedError('Not implemented model!')
model.build((None, RES, RES, 3))
model.summary()


criterion = CrossEntropy()
lr_schedule = CosineLR(base_lr, steps_per_epoch, epochs=epochs, min_lr=0)
optimizer = SGD(lr_schedule, momentum=0.9, weight_decay=wd, nesterov=True)
arch_optimizer = AdamW(arch_wd, arch_lr, beta_1=0.5)


train_metrics = {
    'loss': Mean(),
    'acc': CategoricalAccuracy(),
}
eval_metrics = {
    'loss': CategoricalCrossentropy(from_logits=True),
    'acc': CategoricalAccuracy(),
}
learner = OGCDDALearner(
    model, criterion, arch_optimizer, optimizer,
    grad_clip_norm=grad_clip_norm,
    train_arch=True,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir=work_dir,
    jit_compile=False,  # some ops don't support jit compile
    kl_ratio=kl_ratio)

hist = learner.fit(ds_train, epochs, ds_test, val_freq=1,
    steps_per_epoch=steps_per_epoch, val_steps=test_steps,
    callbacks=[ViewTrainableParams(eval_freq=1)])
