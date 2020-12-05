__author__ = 'Brian M Anderson'
# Created on 12/3/2020
from tensorflow.python.keras.losses import LossFunctionWrapper, losses_utils, nn, math_ops, cosine_similarity


def cosine_loss(y_true, y_pred, axis=-1):
    """
    :param y_true: Tensor of true targets.
    :param y_pred: Tensor of predicted targets.
    :param axis: Axis along which to determine similarity.
    :return: Cosine loss, scale from 0 to 1, with 1 being no overlap
    """
    return 1 + cosine_similarity(y_true=y_true, y_pred=y_pred, axis=axis)


class CosineLoss(LossFunctionWrapper):
    def __init__(self, axis=-1, reduction=losses_utils.ReductionV2.AUTO, name='cosine_loss'):
        super(CosineLoss, self).__init__(cosine_loss, reduction=reduction, name=name, axis=axis)


if __name__ == '__main__':
    pass
