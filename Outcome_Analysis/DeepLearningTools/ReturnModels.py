__author__ = 'Brian M Anderson'
# Created on 11/24/2020
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.DenseNetModel.MyDenseNet import MyDenseNet121
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.DenseNetModel.My3DDenseNet import mydensenet


def return_model(model_key=0):
    model = None
    if model_key == 0:
        model = MyDenseNet121(include_top=False, input_shape=(32, 64, 64, 3), classes=2, include_3d=False)
    elif model_key == 1:
        model = MyDenseNet121(include_top=False, input_shape=(32, 64, 64, 3), classes=2, include_3d=True)
    elif model_key == 2:
        model = MyDenseNet121(include_top=False, input_shape=(32, 64, 64, 3), classes=2, include_3d=False)
        freeze_name = 'final_average_pooling'
        trainable = False
        for index, layer in enumerate(model.layers):
            if layer.name.find(freeze_name) == 0:
                trainable = True
            model.layers[index].trainable = trainable
    elif model_key == 3:
        model = mydensenet
    return model


if __name__ == '__main__':
    pass
