from typing import Optional, Dict

import tensorflow as tf

from auto_augment.operations import NAME_TO_FUNC, H_PARAMS


# Sequence should be consistent to searching process
CADIDATE_OPS_LIST_G = [
            'ShearX',
            'ShearY',
            'TranslateX',
            'TranslateY',
            'Rotate',
            'Brightness',
            'Color',
            'Sharpness',
            'Contrast',
            # 'Cutout',
            'Solarize',
            'Posterize',
            'Equalize',
            'AutoContrast',
            # 'Invert'
            # 'Identity',
        ]

# basic candidate operations in our DRA
CADIDATE_OPS_LIST_RA = CADIDATE_OPS_LIST_G[:9] + [
    'Cutout'
] + CADIDATE_OPS_LIST_G[9:] + [
    'Invert',
    'SolarizeAdd'
]


def _apply_func_with_prob(func, p, image, level, hparams):
    augmented_image = tf.cond(
        tf.random.uniform([], dtype=tf.float32) < p,
        lambda: func(image, level, hparams),
        lambda: image)
    return augmented_image


def remove_duplicate_ops(op_list):
    op_list_no_dup = list(set(op_list))
    op_list_no_dup.sort(key=op_list.index)
    return op_list_no_dup


def apply_autoaugment_from_distribution_ra_mc(image, W, M_MEAN, M_STD, scale=1.0,
                    hparams: Optional[Dict]=None, cadidate_op_extra=[],
                    ra_depth=2, augment_style='RA', p_min_t=0.2, p_max_t=0.8):
    assert augment_style in ['RA', None]


    if augment_style == 'RA':
        CADIDATE_OPS_LIST = CADIDATE_OPS_LIST_RA + cadidate_op_extra
        hparams = {
            **H_PARAMS,
            **(hparams or {}),
        }
    else:
        CADIDATE_OPS_LIST = CADIDATE_OPS_LIST_G + cadidate_op_extra
        hparams = {
            **H_PARAMS,
            **(hparams or {}),
        }
    # Remove duplicated ops and keep original sequence
    CADIDATE_OPS_LIST = remove_duplicate_ops(CADIDATE_OPS_LIST)
    num_cadidate_ops = len(CADIDATE_OPS_LIST)

    depth = ra_depth
    if W is not None:
        op_to_select = tf.random.categorical(logits=W, num_samples=1, dtype=tf.int32)   # sampled op_num = depth
    else:
        op_to_select = tf.random.uniform((depth,), maxval=num_cadidate_ops, dtype=tf.int32)  # random sample op

    if isinstance(M_MEAN, list):
        assert depth == len(M_MEAN)
    elif isinstance(M_MEAN, (float, int)):     # a fixed value for RA style
        assert augment_style == 'RA'
        M_MEAN = tf.ones((depth, num_cadidate_ops), dtype=tf.float32) * M_MEAN
    else:
        raise NotImplementedError('Only support "learnable mean"(List), '
                                  '"TA style mean"(None) or "fixed mean"(float/int)!')

    if isinstance(M_STD, list):
        assert depth == len(M_STD)
    elif isinstance(M_STD, (float, int)):
        M_STD = tf.ones((depth, num_cadidate_ops), dtype=tf.float32) * M_STD
    elif M_STD is None:     # view "None" as not using std
        M_STD = tf.zeros((depth, num_cadidate_ops), dtype=tf.float32)
    else:
        raise NotImplementedError('Only support "learnable std"(List) or "fixed std"(None/float/int)!')

    for i in range(depth):
        for (j, op_name) in enumerate(CADIDATE_OPS_LIST):
            image = tf.cond(op_to_select[i] == j,
                lambda: apply_forward_ra_mc(image, op_name,
                        M_MEAN[i][j], M_STD[i][j],
                        hparams, scale, p_min_t, p_max_t),
                lambda: image)
    return image


def apply_forward_ra_mc(x, op_name, m_mean, m_std, hparams, scale=1.0, p_min_t=0.2, p_max_t=0.8):
    mag = tf.random.normal((), mean=m_mean, stddev=m_std, dtype=tf.float32)
    mag = tf.clip_by_value(mag, 0., 1. * scale)
    p = tf.random.uniform((), minval=p_min_t, maxval=p_max_t, dtype=tf.float32)     # sample a threshold to apply the op
    x = _apply_func_with_prob(NAME_TO_FUNC[op_name], p, x, mag, hparams)
    return x
