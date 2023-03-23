import math
import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.layers import Layer

from tensorflow_addons.image.transform_ops import angles_to_projective_transforms
from tensorflow_addons.image import equalize as equalize_original

from hanser.transform import _fill_region


def blend(image1, image2, factor):
    """Blend image1 and image2 using 'factor'.
    Factor can be above 0.0.  A value of 0.0 means only image1 is used.
    A value of 1.0 means only image2 is used.  A value between 0.0 and
    1.0 means we linearly interpolate the pixel values between the two
    images.  A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values
    between 0 and 255.
    Args:
      image1: An image Tensor of float32.
      image2: An image Tensor of float32.
      factor: A floating point value above 0.0.
    Returns:
      A blended image Tensor of type float32.
    """
    image1 = tf.cast(image1, tf.float32)
    image2 = tf.cast(image2, tf.float32)

    if factor == 0.0:
        return image1
    if factor == 1.0:
        return image2

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = image1 + scaled

    # Interpolate
    # noinspection PyChainedComparisons
    if factor > 0.0 and factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return temp

    # Extrapolate:
    # We need to clip and then cast.
    return tf.clip_by_value(temp, 0.0, 255.0)


def aug_transform(x, transforms):
    transform_or_transforms = tf.convert_to_tensor(transforms, dtype=tf.float32)
    if len(transform_or_transforms.shape) == 1:
        transforms = transform_or_transforms[None]
    x = tf.raw_ops.ImageProjectiveTransformV2(
        images=x,
        transforms=transforms,
        output_shape=tf.shape(x)[1:3],
        interpolation='BILINEAR'
    )
    return x


def shear_x(images, magnitude):
    return aug_transform(
        images, [1., magnitude, 0., 0., 1., 0., 0., 0.])


def shear_y(images, magnitude):
    return aug_transform(
        images, [1., 0., 0., magnitude, 1., 0., 0., 0.])


def translate_x(images, pixels):
    return aug_transform(
        images, [1, 0, -pixels, 0, 1, 0, 0, 0])


def translate_y(images, pixels):
    return aug_transform(
        images, [1, 0, 0, 0, 1, -pixels, 0, 0])


def rotate(images, angle):
    radians = angle * math.pi / 180.0
    image_height = tf.cast(tf.shape(images)[1], tf.float32)
    image_width = tf.cast(tf.shape(images)[2], tf.float32)
    transforms = angles_to_projective_transforms(
        radians, image_height, image_width)
    return aug_transform(
        images, transforms)


@tf.custom_gradient
def posterize(images, bits):
    images = tf.cast(images, tf.int32)
    shift = tf.cast(tf.ones_like(images) * bits, images.dtype)
    images = tf.raw_ops.RightShift(x=images, y=shift)
    images = tf.raw_ops.LeftShift(x=images, y=shift)
    images = tf.cast(images, tf.float32)
    def grad(upstream):
        return upstream, None
    return images, grad


def brightness(images, factor):
    degenerate = tf.zeros_like(images)
    images = blend(degenerate, images, factor)
    return images


def color(images, factor):
    degenerate = tf.map_fn(tf.image.rgb_to_grayscale, images)
    degenerate = tf.map_fn(tf.image.grayscale_to_rgb, degenerate)
    images = blend(degenerate, images, factor)
    return images


def solarize(images, threshold):
    threshold = tf.cast(threshold, images.dtype)
    images = tf.where(images < threshold, images, 255.0 - images)
    return images


def solarize_add(images, addition=0., threshold=128.):
    # For each pixel in the image less than threshold
    # we add 'addition' amount to it and then clip the
    # pixel value to be between 0 and 255. The value
    # of 'addition' is between -128 and 128.
    added_image = images + tf.cast(addition, images.dtype)
    added_image = tf.cast(tf.clip_by_value(added_image, 0., 255.), tf.float32)
    return tf.where(images < threshold, added_image, images)


def sharpness(images, factor):
    kernel = tf.constant(
        [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32,
        shape=[3, 3, 1, 1]) / 13.
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    strides = [1, 1, 1, 1]
    degenerate = tf.nn.depthwise_conv2d(
        images, kernel, strides, padding='VALID', dilations=[1, 1])
    degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
    mask = tf.ones_like(degenerate)
    padded_mask = tf.pad(mask, [[0, 0], [1, 1], [1, 1], [0, 0]])
    padded_degenerate = tf.pad(degenerate, [[0, 0], [1, 1], [1, 1], [0, 0]])
    result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, images)
    images = blend(result, images, factor)
    return images


def contrast(images, factor):
    def _contrast(x):
        degenerate = tf.image.rgb_to_grayscale(x)
        mean = tf.reduce_mean(degenerate)
        degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
        degenerate = tf.image.grayscale_to_rgb(degenerate)
        x = blend(degenerate, x, factor)
        return x
    images = tf.map_fn(_contrast, images)
    return images


def cutout(images, length):
    shape = tf.shape(images)
    bs, h, w, c = shape[0], shape[1], shape[2], shape[3]

    cy = tf.random.uniform((bs,), 0, h, dtype=tf.int32)
    cx = tf.random.uniform((bs,), 0, w, dtype=tf.int32)

    half_len = length // 2
    t = tf.maximum(0, cy - half_len)
    b = tf.minimum(h, cy + (length - half_len))
    l = tf.maximum(0, cx - half_len)
    r = tf.minimum(w, cx + (length - half_len))

    hi = tf.range(h)[None, :, None, None]
    mh = (hi >= t[:, None, None, None]) & (hi < b[:, None, None, None])
    wi = tf.range(w)[None, None, :, None]
    mw = (wi >= l[:, None, None, None]) & (wi < r[:, None, None, None])
    masks = tf.cast(tf.logical_not(mh & mw), images.dtype)

    fill = _fill_region((bs, h, w, c), 0, images.dtype)
    images = tf.where(tf.equal(masks, 0), fill, images)
    return images


def autocontrast(images):
    def _scale_channel(img):
        """Scale the 2D image using the autocontrast rule."""
        lo = tf.cast(tf.reduce_min(img), tf.float32)
        hi = tf.cast(tf.reduce_max(img), tf.float32)
        # Scale the image, making the lowest value 0 and the highest value 255.
        def _scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = tf.cast(im, tf.float32) * scale + offset
            im = tf.clip_by_value(im, 0.0, 255.0)
            return im
        result = tf.cond(hi > lo, lambda: _scale_values(img), lambda: img)
        return result
    def _scale_RGB(x):
        s1 = _scale_channel(x[:, :, 0])
        s2 = _scale_channel(x[:, :, 1])
        s3 = _scale_channel(x[:, :, 2])
        x = tf.stack([s1, s2, s3], 2)
        return x
    images = tf.map_fn(_scale_RGB, images)
    return images


@tf.custom_gradient
def equalize(images):
    images = equalize_original(images)
    def grad(upstream):
        return upstream
    return images, grad


class ShearX(Layer):

    def __init__(self, min_val, max_val):
        super().__init__()
        assert min_val + max_val == 0
        self.min_val = min_val
        self.max_val = max_val
        self.op_name = 'ShearX'

    def _magnitude_to_arg(self, magnitude):
        if tf.random.uniform(()) < 0.5:
            magnitude = -magnitude
        return self.max_val * magnitude

    def call(self, x, magnitude):
        x = shear_x(x, self._magnitude_to_arg(magnitude))
        return x


class ShearY(Layer):

    def __init__(self, min_val, max_val):
        super().__init__()
        assert min_val + max_val == 0
        self.min_val = min_val
        self.max_val = max_val
        self.op_name = 'ShearY'

    def _magnitude_to_arg(self, magnitude):
        if tf.random.uniform(()) < 0.5:
            magnitude = -magnitude
        return self.max_val * magnitude

    def call(self, x, magnitude):
        x = shear_y(x, self._magnitude_to_arg(magnitude))
        return x


class TranslateX(Layer):

    def __init__(self, min_val, max_val):
        super().__init__()
        assert min_val + max_val == 0
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.op_name = 'TranslateX'

    def _magnitude_to_arg(self, magnitude):
        if tf.random.uniform(()) < 0.5:
            magnitude = -magnitude
        return self.max_val * magnitude

    def call(self, x, magnitude):
        x = translate_x(x, self._magnitude_to_arg(magnitude))
        return x


class TranslateY(Layer):

    def __init__(self, min_val, max_val):
        super().__init__()
        assert min_val + max_val == 0
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.op_name = 'TranslateY'

    def _magnitude_to_arg(self, magnitude):
        if tf.random.uniform(()) < 0.5:
            magnitude = -magnitude
        return self.max_val * magnitude

    def call(self, x, magnitude):
        x = translate_y(x, self._magnitude_to_arg(magnitude))
        return x


class Rotate(Layer):

    def __init__(self, min_val, max_val):
        super().__init__()
        assert min_val + max_val == 0
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.op_name = 'Rotate'

    def _magnitude_to_arg(self, magnitude):
        if tf.random.uniform(()) < 0.5:
            magnitude = -magnitude
        return self.max_val * magnitude

    def call(self, x, magnitude):
        x = rotate(x, self._magnitude_to_arg(magnitude))
        return x


class Posterize(Layer):

    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.op_name = 'Posterize'

    def _magnitude_to_arg(self, magnitude):
        return tf.cast((self.max_val - self.min_val) * magnitude
                       + self.min_val, tf.int32)

    def call(self, x, magnitude):
        x = posterize(x, self._magnitude_to_arg(magnitude))
        return x


class Brightness(Layer):

    def __init__(self, min_val, max_val):
        super().__init__()
        assert min_val + max_val == 2
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.op_name = 'Brightness'

    def _magnitude_to_arg(self, magnitude):
        if tf.random.uniform(()) < 0.5:
            magnitude = -magnitude
        return (self.max_val - 1.0) * magnitude + 1.0

    def call(self, x, magnitude):
        x = brightness(x, self._magnitude_to_arg(magnitude))
        return x


class Color(Layer):

    def __init__(self, min_val, max_val):
        super().__init__()
        assert min_val + max_val == 2
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.op_name = 'Color'

    def _magnitude_to_arg(self, magnitude):
        if tf.random.uniform(()) < 0.5:
            magnitude = -magnitude
        return (self.max_val - 1.0) * magnitude + 1.0

    def call(self, x, magnitude):
        x = color(x, self._magnitude_to_arg(magnitude))
        return x


class Solarize(Layer):

    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.op_name = 'Solarize'

    def _magnitude_to_arg(self, magnitude):
        threshold = 256.0 - (self.max_val - self.min_val) * magnitude
        return threshold

    def call(self, x, magnitude):
        x = solarize(x, self._magnitude_to_arg(magnitude))
        return x


class SolarizeAdd(Layer):

    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.op_name = 'SolarizeAdd'

    def _magnitude_to_arg(self, magnitude):
        addition = (self.max_val - self.min_val) * magnitude
        return addition

    def call(self, x, magnitude):
        x = solarize_add(x, self._magnitude_to_arg(magnitude))
        return x


class Sharpness(Layer):

    def __init__(self, min_val, max_val):
        super().__init__()
        assert min_val + max_val == 2
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.op_name = 'Sharpness'

    def _magnitude_to_arg(self, magnitude):
        if tf.random.uniform(()) < 0.5:
            magnitude = -magnitude
        return (self.max_val - 1.0) * magnitude + 1.0

    def call(self, x, magnitude):
        x = sharpness(x, self._magnitude_to_arg(magnitude))
        return x


class Contrast(Layer):

    def __init__(self, min_val, max_val):
        super().__init__()
        assert min_val + max_val == 2
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.op_name = 'Contrast'

    def _magnitude_to_arg(self, magnitude):
        if tf.random.uniform(()) < 0.5:
            magnitude = -magnitude
        return (self.max_val - 1.0) * magnitude + 1.0

    def call(self, x, magnitude):
        x = contrast(x, self._magnitude_to_arg(magnitude))
        return x


class Invert(Layer):

    def __init__(self, min_val, max_val):
        super().__init__()
        self.op_name = 'Invert'
        self.min_val = min_val
        self.max_val = max_val

    def call(self, x, magnitude):
        x = 255.0 - x
        return x


class Identity(Layer):

    def __init__(self, min_val, max_val):
        super().__init__()
        self.op_name = 'Identity'
        self.min_val = min_val
        self.max_val = max_val

    def call(self, x, magnitude):
        return x


class Cutout(Layer):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.op_name = 'Cutout'

    def _magnitude_to_arg(self, h, magnitude):
        magnitude = (self.max_val - self.min_val) * magnitude + self.min_val
        length = tf.cast(magnitude * tf.cast(h, tf.float32), tf.int32)
        return length

    def call(self, x, magnitude):
        x = cutout(x, self._magnitude_to_arg(tf.shape(x)[1], magnitude))
        return x


class AutoContrast(Layer):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.op_name = 'AutoContrast'
        self.min_val = min_val
        self.max_val = max_val

    def call(self, x, magnitude):
        x = autocontrast(x)
        return x


class Equalize(Layer):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.op_name = 'Equalize'
        self.min_val = min_val
        self.max_val = max_val

    def call(self, x, magnitude):
        x = equalize(x)
        return x


