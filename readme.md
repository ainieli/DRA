# Differentiable RandAugment

---

The official implementation of DRA on TensorFlow 2.

## Environment

+ Python 3.7

+ TensorFlow 2.3.4 (CUDA 10.1)

+ Search Device: GPU

+ Train Device: TPU (e.g., v2 (8 Cores))

## Install Requirements

```shell
git clone https://github.com/sbl1996/hanser.git
pip install -U tensorflow==2.3.4
pip install -U tensorflow_probability==0.11.1
pip install -U tensorflow_addons==0.13.0
cd hanser && pip install -e .
```

## Search (on GPU)

```shell
# Experiments on CIFAR-10/100 + WRN-28-10/PyramidNet+ShakeDrop
# Change settings in file search_CIFAR.py before search
python search_CIFAR.py

# Experiments on ImageNet
# Need customed TFRecords for proxy ImageNet before search
# Change settings in file search_ImageNet.py before search
python search_ImageNet.py
```

## Train (on TPU)

```shell
# CIFAR-10/100 + WRN-28-10/PyramidNet+ShakeDrop
# Change settings in file train_phase/train_CIFAR.py
cd train_phase && python train_CIFAR.py

# ResNet-50/200, DeiT-Tiny on ImageNet (need TFRecords for ImageNet)
# Change settings in file train_phase/train_ImageNet.py
cd train_phase && python train_ImageNet.py
```

## Use ImageNet

TFRecords Ref: https://www.tensorflow.org/tutorials/load_data/tfrecord

ImageNet on TPU Ref: https://cloud.google.com/tpu/docs/imagenet-setup

## Policy Parameters for Transfer Learning

See policy.py.

## Reference

Coming soon :)
