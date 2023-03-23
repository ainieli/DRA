# Applied on GPU device
from toolz import curry

import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, CategoricalCrossentropy, TopKCategoricalAccuracy
# from tensorflow_addons.optimizers import AdamW


from hanser.distribute import setup_runtime, distribute_datasets
from hanser.transform import random_resized_crop, resize, center_crop, normalize, to_tensor
from hanser.train.optimizers import SGD, AdamW
from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy
from hanser.models.layers import set_defaults
from hanser.datasets.imagenet_c120 import make_imagenet_dataset_split

from models.resnet import resnet50
from models.deit import DeiT
from metrics.callback import ViewTrainableParams
from learner import OGCDDALearner
from datasets import IMAGENET_MEAN, IMAGENET_STD


RES = 224
val_resize_RES = 256
eval_batch_size = 128
batch_size = 32
epochs = 20
arch_lr = 5e-3
arch_wd = 0
tau = 0.5
ra_depth = 2
kl_ratio = 1.
augment_style = 'RA'
p_min_t = 0.2       # 1.0
p_max_t = 0.8       # 1.0
grad_clip_norm = 5.0

model_name = 'resnet_50'
if model_name == 'resnet_50' or model_name == 'resnet_200':
    base_lr = batch_size / 256 * 0.1
    wd = 1e-4
    dropout = None
    droppath = None
    if '_50' in model_name:
        scale = 1.0
    else:
        scale = 3.0
        # need extra manually tuning the initial value of \mu for DRA
elif model_name == 'deit_tiny_16_224':
    base_lr = batch_size / 512 * 5e-4
    wd = 5e-3
    dropout = 0.0
    droppath = 0.1
else:
    raise NotImplementedError('Require hyperparameter customization for model %s' % model_name)

NUM_CLASS = 120     # a proxy of ImageNet

learnable_w = True
scale = 1.
fix_mean = None
fix_std = None

work_dir = './ImageNet'

set_defaults({
    'bn': {
        'affine': False,
    },
    'fixed_padding': True,
})


@curry
def transform(image, label, training):
    if training:
        image = random_resized_crop(image, RES, scale=(0.05, 1.0), ratio=(0.75, 1.33))
        image = tf.image.random_flip_left_right(image)
    else:
        image = resize(image, val_resize_RES)
        image = center_crop(image, RES)
    image, label = to_tensor(image, label, label_offset=1)
    label = tf.one_hot(label, NUM_CLASS)
    # DRA/RA is applied as layers in the model
    return image, label


######################################################################################################
# Add your own paths of ImageNet dataset in train_files and search_files.
# It should be of format TFRecord
# We do not offer the paths for ImageNet dataset due to the charge of external downloading.
# You can create your own dataset to run this template.
# Ref: https://www.tensorflow.org/tutorials/load_data/tfrecord
#      https://cloud.google.com/tpu/docs/imagenet-setup
train_files = []
search_files = []
ds_train, steps_per_epoch = make_imagenet_dataset_split(
    batch_size, transform(training=True), train_files, split='train', training=True)
ds_search, test_steps = make_imagenet_dataset_split(
    batch_size, transform(training=False), search_files, split='train', training=True)
ds_train = tf.data.Dataset.zip((ds_train, ds_search))
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
#######################################################################################################


setup_runtime(fp16=True)
ds_train, _ = distribute_datasets(ds_train, None)

if model_name == 'resnet_50':
    model = resnet50(num_classes=NUM_CLASS,
               learnable_w=learnable_w, fix_mean=fix_mean, fix_std=fix_std, scale=scale,
               tau=tau, num_ops_per_sub_policy=ra_depth,
               norm_mean=IMAGENET_MEAN, norm_std=IMAGENET_STD,
               resolution=RES, augment_style=augment_style,
               p_min_t=p_min_t, p_max_t=p_max_t)
elif model_name == 'resnet_200':
    model = resnet50(num_classes=NUM_CLASS,
               learnable_w=learnable_w, fix_mean=fix_mean, fix_std=fix_std, scale=scale,
               tau=tau, num_ops_per_sub_policy=ra_depth,
               norm_mean=IMAGENET_MEAN, norm_std=IMAGENET_STD,
               resolution=RES, augment_style=augment_style,
               p_min_t=p_min_t, p_max_t=p_max_t)
elif model_name == 'deit_tiny_16_224':
    model = DeiT(model_name='tiny', patch_size=16, drop_rate=dropout, drop_path=droppath, num_classes=NUM_CLASS,
                 learnable_w=learnable_w, fix_mean=fix_mean, fix_std=fix_std, scale=scale,
                 tau=tau, num_ops_per_sub_policy=ra_depth,
                 norm_mean=IMAGENET_MEAN, norm_std=IMAGENET_STD,
                 resolution=RES, augment_style=augment_style,
                 p_min_t=p_min_t, p_max_t=p_max_t)
else:
    raise NotImplementedError('Model not implemented')


model.build((None, RES, RES, 3))
model.summary()


criterion = CrossEntropy()
lr_schedule = CosineLR(base_lr, steps_per_epoch, epochs=epochs, min_lr=0)
if 'resnet' in model_name:
    optimizer = SGD(lr_schedule, momentum=0.9, weight_decay=wd, nesterov=True)
elif 'deit' in model_name:
    optimizer = AdamW(lr_schedule, weight_decay=wd)
else:
    raise NotImplementedError()
arch_optimizer = AdamW(arch_lr, weight_decay=arch_wd, beta_1=0.5)


train_metrics = {
    'loss': Mean(),
    'acc': CategoricalAccuracy(),
}
eval_metrics = {
    'loss': CategoricalCrossentropy(from_logits=True),
    'acc': CategoricalAccuracy(),
    'acc5': TopKCategoricalAccuracy(k=5),
}
learner = OGCDDALearner(
    model, criterion, arch_optimizer, optimizer,
    grad_clip_norm=grad_clip_norm,
    train_arch=True,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir=work_dir,
    jit_compile=False,
    kl_ratio=kl_ratio)

hist = learner.fit(ds_train, epochs, None, val_freq=1,
    steps_per_epoch=steps_per_epoch,
    # val_steps=test_steps,
    callbacks=[ViewTrainableParams(eval_freq=1)])
