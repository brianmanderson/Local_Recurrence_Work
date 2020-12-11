__author__ = 'Brian M Anderson'
# Created on 11/28/2020
from Deep_Learning.Base_Deeplearning_Code.Finding_Optimization_Parameters.HyperParameters import OrderedDict


def return_list_of_models(model_key=0):
    dictionary = []
    if model_key == 0:
        base_dict0 = lambda min_lr, max_lr, step_factor, opt, loss: \
            OrderedDict({'Model_Type': model_key, 'min_lr': min_lr, 'max_lr': max_lr, 'step_factor': step_factor,
                         'Optimizer': opt, 'Loss': loss})
        dictionary = [
            # base_dict0(min_lr=8e-6, max_lr=2e-3, step_factor=5),
            base_dict0(min_lr=8e-6, max_lr=2e-3, step_factor=10, opt='SGD', loss='CCE'),
            # base_dict0(min_lr=8e-6, max_lr=2e-3, step_factor=20)
        ]
    elif model_key == 1:
        base_dict0 = lambda min_lr, max_lr, step_factor, opt, loss: \
            OrderedDict({'Model_Type': model_key, 'min_lr': min_lr, 'max_lr': max_lr, 'step_factor': step_factor,
                         'Optimizer': opt, 'Loss': loss})
        dictionary = [
            base_dict0(min_lr=8e-5, max_lr=1e-3, step_factor=10, opt='SGD', loss='CCE'),
            base_dict0(min_lr=8e-5, max_lr=8e-3, step_factor=10, opt='SGD', loss='CCE'),
            base_dict0(min_lr=8e-5, max_lr=4e-2, step_factor=10, opt='SGD', loss='CCE'),
        ]
    elif model_key == 2:
        base_dict0 = lambda min_lr, max_lr, step_factor, opt: \
            OrderedDict({'Model_Type': model_key, 'min_lr': min_lr, 'max_lr': max_lr, 'step_factor': step_factor,
                         'Optimizer': opt})
        dictionary = [
            base_dict0(min_lr=3e-6, max_lr=2e-3, step_factor=10, opt='SGD'),
        ]
    elif model_key == 3:
        base_dict0 = lambda blocks_in_dense, dense_conv_blocks, dense_layers, num_dense_connections, filters, growth_rate, min_lr, max_lr, step_factor, opt, loss: \
            OrderedDict({'blocks_in_dense': blocks_in_dense, 'dense_conv_blocks': dense_conv_blocks,
                         'dense_layers': dense_layers, 'num_dense_connections': num_dense_connections,
                         'filters': filters, 'growth_rate': growth_rate, 'min_lr': min_lr, 'max_lr': max_lr,
                         'step_factor': step_factor, 'Model_Type': model_key, 'Optimizer': opt, 'Loss': loss})
        dictionary = [
            base_dict0(blocks_in_dense=3, dense_conv_blocks=3, dense_layers=3, num_dense_connections=256, filters=8,
                       growth_rate=8, min_lr=1e-5, max_lr=0.867, step_factor=10, opt='SGD', loss='CosineLoss'),
            base_dict0(blocks_in_dense=3, dense_conv_blocks=3, dense_layers=2, num_dense_connections=256, filters=8,
                       growth_rate=8, min_lr=1e-5, max_lr=1.41, step_factor=10, opt='SGD', loss='CosineLoss'),
            base_dict0(blocks_in_dense=3, dense_conv_blocks=2, dense_layers=3, num_dense_connections=256, filters=8,
                       growth_rate=8, min_lr=1e-5, max_lr=1.04, step_factor=10, opt='SGD', loss='CosineLoss'),
            base_dict0(blocks_in_dense=3, dense_conv_blocks=1, dense_layers=3, num_dense_connections=256, filters=8,
                       growth_rate=8, min_lr=1e-5, max_lr=0.769, step_factor=10, opt='SGD', loss='CosineLoss'),
            base_dict0(blocks_in_dense=3, dense_conv_blocks=1, dense_layers=2, num_dense_connections=256, filters=8,
                       growth_rate=8, min_lr=1e-5, max_lr=0.612, step_factor=10, opt='SGD', loss='CosineLoss'),
            base_dict0(blocks_in_dense=3, dense_conv_blocks=1, dense_layers=2, num_dense_connections=256, filters=8,
                       growth_rate=4, min_lr=1e-5, max_lr=0.505, step_factor=10, opt='SGD', loss='CosineLoss'),
            base_dict0(blocks_in_dense=3, dense_conv_blocks=1, dense_layers=3, num_dense_connections=256, filters=8,
                       growth_rate=4, min_lr=1e-5, max_lr=0.718, step_factor=10, opt='SGD', loss='CosineLoss')
        ]
    model_dictionary = {model_key: dictionary}
    return model_dictionary


if __name__ == '__main__':
    pass
