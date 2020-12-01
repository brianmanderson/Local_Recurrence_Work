__author__ = 'Brian M Anderson'
# Created on 11/28/2020
from Deep_Learning.Base_Deeplearning_Code.Finding_Optimization_Parameters.HyperParameters import OrderedDict


def return_list_of_models(model_type=0):
    dictionary = []
    if model_type == 0:
        base_dict0 = lambda min_lr, max_lr, step_factor: \
            OrderedDict({'Model_Type': model_type, 'min_lr': min_lr, 'max_lr': max_lr, 'step_factor': step_factor})
        dictionary = [
            base_dict0(min_lr=8e-6, max_lr=2e-3, step_factor=5),
            base_dict0(min_lr=8e-6, max_lr=2e-3, step_factor=10),
            base_dict0(min_lr=8e-6, max_lr=2e-3, step_factor=20)
        ]
    elif model_type == 1:
        base_dict0 = lambda min_lr, max_lr, step_factor: \
            OrderedDict({'Model_Type': model_type, 'min_lr': min_lr, 'max_lr': max_lr, 'step_factor': step_factor})
        dictionary = [
            base_dict0(min_lr=8e-5, max_lr=4e-2, step_factor=10),
        ]
    model_dictionary = {model_type: dictionary}
    return model_dictionary


if __name__ == '__main__':
    pass
