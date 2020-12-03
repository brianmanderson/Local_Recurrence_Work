__author__ = 'Brian M Anderson'
# Created on 12/3/2020
from tensorflow.python.keras.losses import LossFunctionWrapper, cosine_similarity, losses_utils


def cosine_loss(y_true, y_pred, axis=-1):
    """
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return 1 + cosine_similarity(y_true=y_true, y_pred=y_pred, axis=axis)  # 1 + not - because it is already minimized


class CosineLoss(LossFunctionWrapper):
    def __init__(self, axis=-1, reduction=losses_utils.ReductionV2.AUTO, name='cosine_loss'):
        super(CosineLoss, self).__init__(cosine_similarity, reduction=reduction, name=name, axis=axis)


if __name__ == '__main__':
    pass
