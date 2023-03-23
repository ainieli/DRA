# These implementations were stolen from https://github.com/google/automl/blob/master/efficientnetv2/autoaugment.py
# 1. Image ops are exactly the same.
# 2. Function to combine and apply policies is different.
# 3. The way to apply hparams (cutout_max and translate_max) is different.



import tensorflow as tf
from hanser.transform import sharpness, shear_x, shear_y, solarize, solarize_add, autocontrast, translate_x, \
    translate_y, rotate, color, posterize, contrast, brightness, equalize, invert, cutout3 as cutout, random_apply, solarize_add
from hanser.transform.common import image_dimensions


H_PARAMS = {
    "fill_color": (0, 0, 0),
    'cutout_max': 0.2,
    'shear_max': 0.3,
    'translate_max': 10.,
    'rotate_max': 30.,
    # 'enhance_min': 0.1,
    'enhance_max': 1.9,
    'posterize_min': 4.0,
    'posterize_max': 8.0,
    'solarize_max': 256,
    'solarize_add_max': 110,

}


def _randomly_negate_tensor(tensor):
    tensor = tf.convert_to_tensor(tensor)
    sign = tf.sign(tf.random.normal(()))
    sign = tf.convert_to_tensor(sign, tensor.dtype)
    return tensor * sign


def _rotate_magnitude_to_arg(magnitude, max_rotate_degree):
    degree = magnitude * max_rotate_degree
    degree = _randomly_negate_tensor(degree)
    return degree


def _scale_magnitude_to_arg(magnitude, max_scale):
    magnitude = magnitude * max_scale
    magnitude = _randomly_negate_tensor(magnitude)
    return magnitude


def _posterize_level_to_arg(magnitude, min_bits_pil, max_bits_pil):
    bits_pil = max_bits_pil - magnitude * (max_bits_pil - min_bits_pil)
    bits = tf.cast(tf.math.ceil(bits_pil), tf.int32)
    return bits


def _solarize_level_to_arg(magnitude, max_thre):
    thre = tf.cast(tf.math.ceil(256. - magnitude * max_thre), tf.int32)
    return thre


def _solarize_add_level_to_arg(magnitude, max_add):
    add = tf.cast(magnitude * max_add, tf.int32)
    return add


def _enhance_level_to_arg(magnitude, max_val):
    max_scale = _randomly_negate_tensor(max_val - 1.)
    magnitude = 1. + max_scale * magnitude
    return magnitude


def _cutout_level_to_arg(magnitude, max_val):
    return magnitude * max_val


def _shear_x(img, magnitude, hparams):
    fill_color = hparams['fill_color']
    magnitude = _scale_magnitude_to_arg(magnitude, hparams['shear_max'])
    return shear_x(img, magnitude, fill_color)


def _shear_y(img, magnitude, hparams):
    fill_color = hparams['fill_color']
    magnitude = _scale_magnitude_to_arg(magnitude, hparams['shear_max'])
    return shear_y(img, magnitude, fill_color)


def _translate_x(img, magnitude, hparams):
    fill_color = hparams['fill_color']
    magnitude = _scale_magnitude_to_arg(magnitude, hparams['translate_max'])
    # magnitude = tf.cast(tf.shape(img)[1], magnitude.dtype) * magnitude
    return translate_x(img, magnitude, fill_color)


def _translate_y(img, magnitude, hparams):
    fill_color = hparams['fill_color']
    magnitude = _scale_magnitude_to_arg(magnitude, hparams['translate_max'])
    # magnitude = tf.cast(tf.shape(img)[0], magnitude.dtype) * magnitude
    return translate_y(img, magnitude, fill_color)


def _rotate(img, magnitude, hparams):
    fill_color = hparams['fill_color']
    magnitude = _rotate_magnitude_to_arg(magnitude, hparams['rotate_max'])
    return rotate(img, magnitude, fill_color)


def _posterize(img, magnitude, hparams):
    magnitude = _posterize_level_to_arg(
        magnitude, hparams['posterize_min'], hparams['posterize_max'])
    return posterize(img, magnitude)


def _solarize(img, magnitude, hparams):
    magnitude = _solarize_level_to_arg(magnitude, hparams['solarize_max'])
    return solarize(img, magnitude)


def _solarize_add(img, magnitude, hparams):
    magnitude = _solarize_add_level_to_arg(magnitude, hparams['solarize_add_max'])
    return solarize_add(img, magnitude)


def _color(img, magnitude, hparams):
    magnitude = _enhance_level_to_arg(
        magnitude, hparams['enhance_max'])
    return color(img, magnitude)


def _contrast(img, magnitude, hparams):
    magnitude = _enhance_level_to_arg(
        magnitude, hparams['enhance_max'])
    return contrast(img, magnitude)


def _sharpness(img, magnitude, hparams):
    magnitude = _enhance_level_to_arg(
        magnitude, hparams['enhance_max'])
    return sharpness(img, magnitude)


def _brightness(img, magnitude, hparams):
    magnitude = _enhance_level_to_arg(
        magnitude, hparams['enhance_max'])
    return brightness(img, magnitude)


def _autocontrast(img, magnitude, hparams):
    return autocontrast(img)


def _equalize(img, magnitude, hparams):
    return equalize(img)


def _invert(img, magnitude, hparams):
    return invert(img)


def _cutout(img, magnitude, hparams):
    fill_color = hparams['fill_color']
    magnitude = _cutout_level_to_arg(magnitude, hparams['cutout_max'])
    h, w = image_dimensions(img, 3)[:2]
    magnitude = tf.cast(tf.cast(tf.maximum(h, w), tf.float32) * magnitude, tf.int32)
    return cutout(img, magnitude, fill_color)


def _identity(img, level, hparams):
    return img


NAME_TO_FUNC = {
    "Identity": _identity,
    "AutoContrast": _autocontrast,
    "Equalize": _equalize,
    "Invert": _invert,

    "Solarize": _solarize,
    "SolarizeAdd": _solarize_add,
    "Posterize": _posterize,

    "Color": _color,
    "Contrast": _contrast,
    "Brightness": _brightness,
    "Sharpness": _sharpness,

    "Rotate": _rotate,
    "ShearX": _shear_x,
    "ShearY": _shear_y,
    'TranslateX': _translate_x,
    'TranslateY': _translate_y,

    'Cutout': _cutout,
}
