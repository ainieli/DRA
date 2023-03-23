# Applied on TPU environment

import os
import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, CategoricalCrossentropy, TopKCategoricalAccuracy

from hanser.distribute import setup_runtime, distribute_datasets
from hanser.datasets.imagenet import make_imagenet_dataset
from hanser.transform import random_resized_crop, resize, center_crop, normalize, to_tensor

from hanser.train.optimizers import SGD, AdamW
from hanser.models.imagenet.resnet import resnet50, ResNet, Bottleneck
from hanser.models.transformer.vision.deit import deit_ti
from hanser.train.cls import SuperLearner
from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy
from hanser.models.layers import set_defaults

from auto_augment.augment import apply_autoaugment_from_distribution_ra_mc


"""
Settings for RA/TA/DRA:
    DRA:
        W: List of List
        M_MEAN: List of List
        M_STD: List of List
        scale: 1.0/3.0 (for over-range)
        augment_style: RA/None(for custom candidate ops)
    RA:
        W: None
        M_MEAN: float/int
        M_STD: 0/None
        scale: 1.0/3.0 (for over-range)
        augment_style: RA
"""


def resnet200(**kwargs):
    return ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)


def train_fn(_i):

    TASK_NAME = None    # modify this before training
    assert TASK_NAME is not None
    work_dir = f'./ImageNet/{TASK_NAME}'

    # Policy parameters for DeiT-Tiny-16-224
    W = [[ 0.2113,  0.1599, -0.1565,  0.0485, -0.3173, -0.0658, -0.0770, -0.0759,
         -0.0797, -0.1753, -0.2511, -0.0685, -0.1375, -0.1803,  0.6805,  0.2452],
        [ 0.0375, -0.1515, -0.0313, -0.3504,  0.0261, -0.1358, -0.2391, -0.0907,
         -0.1701, -0.0406, -0.1552,  0.1671, -0.0024, -0.2321,  0.6099,  0.5412]]
    M_MEAN = [[1.0000, 0.9013, 0.8414, 0.9984, 0.7495, 0.2688, 0.0347, 0.8177, 0.0404,
         0.7966, 0.4674, 1.0000, 0.6276, 0.4998, 0.0000, 0.6095],
        [0.9947, 0.9529, 1.0000, 1.0000, 1.0000, 0.3204, 0.2385, 0.6959, 0.0738,
         0.9831, 0.4816, 1.0000, 0.7614, 0.4539, 0.0000, 0.4893]]
    M_STD = [[0.0000, 0.0900, 0.1873, 0.0484, 0.0690, 0.3308, 0.0084, 0.1074, 0.0380,
         0.0536, 0.0416, 0.1261, 0.2518, 0.0999, 0.0000, 0.1724],
        [0.0353, 0.0967, 0.0187, 0.0169, 0.0000, 0.2317, 0.0456, 0.0000, 0.0357,
         0.1045, 0.0134, 0.0784, 0.3898, 0.1462, 0.0000, 0.2220]]

    ra_depth = 2
    p_min_t = 0.2
    p_max_t = 0.8
    augment_style = 'RA'  # None, RA

    eval_batch_size = 256
    model_name = 'deit_tiny_16_224'
    if 'deit' in model_name:
        batch_size = 1024
        lr = 5e-4 * batch_size / 512
        wd = 1e-4
        label_smoothing = 0.1
        scale = 1.0  # indicate the max range of magnitude
        epochs = 300
        dropout = 0.0
        drop_path = 0.1
    elif '_50' in model_name:
        batch_size = 512
        lr = 0.1 * batch_size / 256
        wd = 1e-4
        label_smoothing = 0.0
        scale = 1.0
        epochs = 270
    elif '_200' in model_name:
        batch_size = 1024
        lr = 0.1 * batch_size / 256
        wd = 0.05
        label_smoothing = 0.0
        scale = 3.0
        epochs = 270
    else:
        raise NotImplementedError('Require customized model hyperparameter setting!')

    TRAIN_RES = 224
    VAL_RESIZE_RES = 256
    if TRAIN_RES == 224:
        TRANSLATE_MAX = 100
    elif TRAIN_RES == 160:
        TRANSLATE_MAX = 72
    else:
        raise NotImplementedError('Require customized transform range!')

    set_defaults({
        'fixed_padding': True,
        'inplace_abn': {
            'enabled': True,
        }
    })

    def transform(image, label, training):
        if training:
            image = random_resized_crop(image, TRAIN_RES, scale=(0.05, 1.0), ratio=(0.75, 1.33))
            image = tf.image.random_flip_left_right(image)
            image = apply_autoaugment_from_distribution_ra_mc(
                image, W, M_MEAN, M_STD, scale,
                ra_depth=ra_depth,
                hparams={'translate_max': TRANSLATE_MAX},
                augment_style=augment_style, p_min_t=p_min_t, p_max_t=p_max_t
            )
        else:
            image = resize(image, VAL_RESIZE_RES)
            image = center_crop(image, TRAIN_RES)

        image, label = to_tensor(image, label, label_offset=1)
        image = normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        label = tf.one_hot(label, 1000)
        return image, label

    ######################################################################################################
    # Add your own paths of ImageNet dataset in train_files and search_files.
    # It should be of format TFRecord
    # We do not offer the paths for ImageNet dataset due to the charge of external downloading.
    # You can create your own dataset to run this template.
    # Ref: https://www.tensorflow.org/tutorials/load_data/tfrecord
    #      https://cloud.google.com/tpu/docs/imagenet-setup
    train_files = []
    eval_files = []
    ds_train, ds_eval, steps_per_epoch, eval_steps = make_imagenet_dataset(
        batch_size, eval_batch_size, transform, train_files=train_files, eval_files=eval_files,
        n_batches_per_step=8)
    #######################################################################################################

    setup_runtime(fp16=True)
    ds_train, ds_eval = distribute_datasets(ds_train, ds_eval)

    if model_name == 'resnet_50':
        model = resnet50()
    elif model_name == 'resnet_200':
        model = resnet200()
    elif model_name == 'deit_tiny_16_224':
        model = deit_ti(patch_size=16, drop_rate=dropout, drop_path=drop_path)
    else:
        NotImplementedError('Require customized model!')
    model.build((None, TRAIN_RES, TRAIN_RES, 3))
    model.summary()

    criterion = CrossEntropy(label_smoothing=label_smoothing)
    lr_schedule = CosineLR(lr, steps_per_epoch, epochs=epochs, min_lr=0,
                           warmup_epoch=5, warmup_min_lr=0)
    if 'deit' in model_name:
        optimizer = AdamW(lr_schedule, weight_decay=wd)
    elif 'resnet' in model_name:
        optimizer = SGD(lr_schedule, momentum=0.9, weight_decay=wd, nesterov=True)
    else:
        raise NotImplementedError('Require customized optimizer!')

    train_metrics = {
        'loss': Mean(),
        'acc': CategoricalAccuracy(),
    }
    eval_metrics = {
        'loss': CategoricalCrossentropy(from_logits=True),
        'acc': CategoricalAccuracy(),
        'acc5': TopKCategoricalAccuracy(k=5),
    }

    learner = SuperLearner(
        model, criterion, optimizer, n_batches_per_step=8,
        train_metrics=train_metrics, eval_metrics=eval_metrics,
        work_dir=work_dir)

    if learner.load(miss_ok=True):
        learner.recover_log()

    learner.fit(ds_train, epochs, ds_eval, val_freq=1,
                steps_per_epoch=steps_per_epoch, val_steps=eval_steps,
                save_freq=5)


train_fn(None)