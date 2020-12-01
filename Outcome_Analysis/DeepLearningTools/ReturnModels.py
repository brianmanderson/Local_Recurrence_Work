__author__ = 'Brian M Anderson'
# Created on 11/24/2020
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.DenseNetModel.MyDenseNet import MyDenseNet121


def return_model(model_key=0):
    if model_key == 0:
        return MyDenseNet121(include_top=False, input_shape=(32, 128, 128, 3), classes=2, include_3d=False)
    elif model_key == 1:
        return MyDenseNet121(include_top=False, input_shape=(32, 128, 128, 3), classes=2, include_3d=True)
    else:
        return None


if __name__ == '__main__':
    pass
