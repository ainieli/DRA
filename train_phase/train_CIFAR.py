# Applied on TPU
from toolz import curry

import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, CategoricalCrossentropy

from hanser.distribute import setup_runtime, distribute_datasets
from hanser.datasets.classification.cifar import make_cifar100_dataset, make_cifar10_dataset

from hanser.transform import random_crop, normalize, to_tensor, cutout
from hanser.models.cifar.preactresnet import ResNet
from hanser.models.cifar.shakedrop.pyramidnet import PyramidNet
from hanser.train.optimizers import SGD
from hanser.train.cls import SuperLearner
from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy

from auto_augment.augment import apply_autoaugment_from_distribution_ra_mc


# policy parameters for WRN-28-10 on CIFAR-100
W = [[-0.0009, 0.2242, -0.1763, 0.2105,
      0.0685, -0.0262, 0.0500, 0.1379,
      -0.1412, 0.0258, 0.1639, 0.0242,
      -0.1400, 0.0894, -0.2989, -0.0550],
     [0.1769, 0.0300, 0.0633, 0.0656,
      0.0724, -0.0949, -0.0155, 0.0901,
      0.1075, 0.0916, 0.1079, 0.1531,
      -0.3673, -0.0273, -0.2288, -0.1339]]

M_MEAN = [[0.5872, 0.7351, 0.3695, 0.8067,
     0.6578, 0.0000, 0.0030, 0.1558,
     0.0185, 0.3823, 0.3003, 0.3681,
     0.5405, 0.5000, 0.9534, 0.4526],
    [0.5182, 0.6929, 0.4262, 0.9703,
     0.4355, 0.0381, 0.0000, 0.2863,
     0.0172, 0.5804, 0.1860, 0.6051,
     0.4063, 0.5000, 0.7953, 0.4938]]

M_STD = [[0.0402, 0.3393, 0.0046, 0.2675,
     0.1660, 0.0173, 0.0067, 0.0441,
     0.0255, 0.2503, 0.0527, 0.1339,
     0.0240, 0.1001, 0.1261, 0.0085],
    [0.2145, 0.0012, 0.2877, 0.0995,
     0.0654, 0.0224, 0.0000, 0.0175,
     0.0163, 0.1065, 0.0539, 0.0917,
     0.0117, 0.1000, 0.1768, 0.2105]]

scale = 1.0
augment_style = 'RA'
ra_depth = 2
p_min_t = 0.2
p_max_t = 0.8

RES = 32
eval_batch_size = 128
cutout_len = 16

model_name = 'WRN_28_10'
if model_name == 'WRN_28_10':
    batch_size = 128
    base_lr = 0.1 * batch_size / 128
    epochs = 200
    wd = 5e-4
    dropout = 0.3
elif model_name == 'PyramidNet+ShakeDrop':
    batch_size = 128
    base_lr = 0.1 * batch_size / 256
    epochs = 1800
    wd = 5e-4
    shakedrop = 0.5
else:
    raise NotImplementedError('Require customized hyperparameter settings!')

DATASET = 'CIFAR100'
if DATASET == 'CIFAR10':
    NUM_CLASS = 10
elif DATASET == 'CIFAR100':
    NUM_CLASS = 100
else:
    raise NotImplementedError('Only support CIFAR dataset')

work_dir = "./%s" % DATASET

@curry
def transform(image, label, training):
    if training:
        image = random_crop(image, (RES, RES), (RES // 8, RES // 8))
        image = tf.image.random_flip_left_right(image)
        image = apply_autoaugment_from_distribution_ra_mc(image,
            W, M_MEAN, M_STD, scale,
            ra_depth=ra_depth, augment_style=augment_style,
            p_min_t=p_min_t, p_max_t=p_max_t
        )
    image, label = to_tensor(image, label)

    if training and cutout_len > 0:
        image = cutout(image, cutout_len)

    image = normalize(image, [0.491, 0.482, 0.447], [0.247, 0.243, 0.262])
    label = tf.one_hot(label, NUM_CLASS)

    return image, label

if DATASET == 'CIFAR10':
    ds_train, ds_test, steps_per_epoch, test_steps = make_cifar10_dataset(
        batch_size, eval_batch_size, transform)
elif DATASET == 'CIFAR100':
    ds_train, ds_test, steps_per_epoch, test_steps = make_cifar100_dataset(
        batch_size, eval_batch_size, transform)
else:
    raise NotImplementedError('Only support CIFAR dataset!')

setup_runtime(fp16=True)
ds_train, ds_test = distribute_datasets(ds_train, ds_test)

if model_name == 'WRN_28_10':
    model = ResNet(depth=28, k=10, num_classes=NUM_CLASS, dropout=dropout)
elif model_name == 'PyramidNet+ShakeDrop':
    model = PyramidNet(16, depth=272, alpha=200, block='bottleneck', p_shakedrop=shakedrop, num_classes=NUM_CLASS)
else:
    raise NotImplementedError("Not implemented model!")
model.build((None, RES, RES, 3))
model.summary()

criterion = CrossEntropy()
lr_schedule = CosineLR(base_lr, steps_per_epoch, epochs=epochs, min_lr=0)
optimizer = SGD(lr_schedule, momentum=0.9, weight_decay=wd, nesterov=True)

train_metrics = {
    'loss': Mean(),
    'acc': CategoricalAccuracy(),
}
eval_metrics = {
    'loss': CategoricalCrossentropy(from_logits=True),
    'acc': CategoricalAccuracy(),
}

learner = SuperLearner(
    model, criterion, optimizer,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir=work_dir)

learner.fit(ds_train, epochs, ds_test, val_freq=1,
                   steps_per_epoch=steps_per_epoch, val_steps=test_steps,
                   callbacks=[])


