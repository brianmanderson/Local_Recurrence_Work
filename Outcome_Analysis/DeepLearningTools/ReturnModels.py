__author__ = 'Brian M Anderson'
# Created on 11/24/2020
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.DenseNetModel.MyDenseNet import MyDenseNet121
from tensorflow.keras.mixed_precision import experimental


def return_model(model_type='2D_Vanilla'):
    policy = experimental.Policy('mixed_float16')
    experimental.set_policy(policy)
    if model_type == '2D_Vanilla':
        return MyDenseNet121(include_top=False, input_shape=(32, 128, 128, 3), classes=2)
    else:
        return None


if __name__ == '__main__':
    pass
