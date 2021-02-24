__author__ = 'Brian M Anderson'
# Created on 11/18/2020
'''
All parts here are in the DeepLearningTools folder
'''
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))
if len(sys.argv) > 1:
    gpu = int(sys.argv[1])
    model_key = int(sys.argv[2])
else:
    gpu = 0
model_key = 5
print('Running on {} for key {}'.format(gpu, model_key))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

if os.path.exists(r'K:\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work\Predicting_Recurrence'):
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnPaths import return_paths
    import shutil
    base_path, morfeus_drive, excel_path = return_paths()
    # shutil.copy(os.path.join(morfeus_drive, 'ModelParameters.xlsx'), os.path.join(base_path, 'ModelParameters.xlsx'))

sanity_check = False
if sanity_check:
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.Sanity_Checks.Print_Center_Images import print_center_images
    print_center_images()
    xxx = 1

batch_size = 32
find_lr = True
finished_lr = True
if find_lr:
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.FindBestLRs import find_best_lr
    finished_lr = find_best_lr(batch_size=batch_size, model_key=model_key)

add_lr = False
added_lr = True
if add_lr and finished_lr:
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.PlotLRs import plot_lrs, pd
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnPaths import return_paths, os
    base_path, morfeus_drive, excel_path = return_paths()
    df = pd.read_excel(excel_path, engine='openpyxl')
    not_filled_df = df.loc[(pd.isnull(df['min_lr'])) & (df['Optimizer'] == 'Adam') & (df['loss'] == 'CosineLoss')
                           & (df['Model_Type'] == model_key)]
    for index in not_filled_df.index.values:
        model_index = not_filled_df['Model_Index'][index]
        print(model_index)
        path = os.path.join(morfeus_drive, 'Learning_Rates', 'Model_Key_{}'.format(model_key),
                            'Model_Index_{}'.format(model_index))
        plot_lrs(input_path=path, excel_path=excel_path, add_to_excel=True, base_df=df,
                 save_path=os.path.join(morfeus_drive, 'Learning_Rates', 'Model_Key_{}'.format(model_key), 'Outputs'))
    added_lr = True

run_the_2D_model = False
if run_the_2D_model and added_lr:
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.Run2DModel import run_2d_model
    run_2d_model(batch_size=batch_size, model_type=model_key)

add_metrics_to_excel = False
if add_metrics_to_excel:
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.AddMetricsToExcel import add_metrics_to_excel
    add_metrics_to_excel()
    xxx = 1

review_models_via_cv = False
if review_models_via_cv:
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.Find_Mean_Std_Across_CV_Groups import \
        add_mean_std_across_cv_groups
    add_mean_std_across_cv_groups()

view_results_with_r = False
if view_results_with_r:
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnPaths import return_paths
    from plotnine import *
    import pandas as pd
    import numpy as np
    base_path, morfeus_drive, excel_path = return_paths()
    df = pd.read_excel(excel_path, engine='openpyxl')
    # df = df.dropna()
    df = df[(~pd.isnull(df['epoch_loss'])) & (df['Optimizer'] == 'Adam') & (df['loss'] == 'CosineLoss')]
    # df.epoch_loss = np.where((df.epoch_AUC < .51), 1, df.epoch_loss)
    # df.epoch_loss = np.where((df.epoch_loss > .6), .6, df.epoch_loss)
    xxx = 1
    (ggplot(df) + aes(x='blocks_in_dense', y='epoch_loss') + facet_wrap('dense_conv_blocks',
                                                                        labeller='label_both') + geom_point(
        mapping=aes(color='epoch_AUC')) + xlab('blocks_in_dense') + ylab('Validation Loss') +
     ggtitle('Validation Loss vs Blocks in Dense, and Dense Convolution Blocks') + scale_colour_gradient(low='blue',
                                                                                                         high='red'))

    for variable in ['Dropout', 'blocks_in_dense', 'dense_conv_blocks', 'dense_layers', 'reduction',
                     'num_dense_connections', 'filters', 'growth_rate']:
        xxx = 1
        (ggplot(df) + aes(x='{}'.format(variable), y='epoch_loss') + geom_point(
            mapping=aes(color='epoch_AUC')) + xlab('{}'.format(variable)) + ylab('Validation Loss') + scale_y_log10()
         + scale_colour_gradient(low='blue', high='red'))
"""
Best model notes
Dropout = 0
blocks_in_dense = 5, check out 1
dense_conv_blocks = 3, check out 4
dense_layers = 0
reduction = 1
num_dense_connections = 256
filters = 8/16
growth_rate = 16, check 8
"""
view_gradients = False
if view_gradients:
    from tensorflow.keras.models import load_model
    import numpy as np
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnGenerators import return_generators, plot_scroll_Image
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnModels import return_model
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnCosineLoss import CosineLoss
    from Deep_Learning.Base_Deeplearning_Code.Visualizing_Model.Visualize_Model import ModelVisualizationClass
    model_path = r'H:\Deeplearning_Recurrence_Work\Nifti_Exports\Records\Models\Model_Index_140\final_model.h5'
    model = load_model(model_path, custom_objects={'CosineLoss': CosineLoss})
    _, _, train_generator, val_generator = return_generators(batch_size=8, cross_validation_id=0, model_key=model_key)
    visualizer = ModelVisualizationClass(model=model, save_images=True,
                                         out_path=r'H:\Deeplearning_Recurrence_Work\Activation_Images')
    x, y = next(iter(val_generator.data_set))
    visualizer.predict_on_tensor(x)
    visualizer.plot_activations()
    xxx = 1
"""
0: 2
1: 26
2: 57
3: 69
4:
"""
evaluate_model = False
if evaluate_model:
    from tensorflow.keras.models import load_model
    import numpy as np
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnGenerators import return_generators
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnModels import mydensenet
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnCosineLoss import CosineLoss
    from tensorflow.keras import metrics, optimizers

    aucs = []
    parameters = {994: {'Dropout': 0., 'blocks_in_dense': 7, 'dense_conv_blocks': 3, 'dense_layers': 1, 'reduction': 1,
                        'num_dense_connections': 256, 'filters': 16, 'global_max': 0, 'growth_rate': 32, 'channels': 3,
                        'model_key': 5},
                  961: {'Dropout': 0.5, 'blocks_in_dense': 5, 'dense_conv_blocks': 3, 'dense_layers': 1,
                        'reduction': 0.5, 'model_key': 3,
                        'num_dense_connections': 128, 'filters': 16, 'global_max': 0, 'growth_rate': 32, 'channels': 2},
                  962: {'Dropout': 0, 'blocks_in_dense': 5, 'dense_conv_blocks': 3, 'dense_layers': 3,
                        'reduction': 1., 'model_key': 4,
                        'num_dense_connections': 256, 'filters': 16, 'global_max': 0, 'growth_rate': 32, 'channels': 3}
                  }
    for model_index in parameters:
        model_parameters = parameters[model_index]
        for key in model_parameters.keys():
            if type(model_parameters[key]) is np.int64:
                model_parameters[key] = int(model_parameters[key])
            elif type(model_parameters[key]) is np.float64:
                model_parameters[key] = float(model_parameters[key])
        model_base_path = r'H:\Deeplearning_Recurrence_Work\Models\Model_Index_{}'.format(model_index)
        pred_path = os.path.join(model_base_path, 'Predictions.npy')
        truth_path = os.path.join(model_base_path, 'Truth.npy')
        model_path = os.path.join(model_base_path, 'cp-best.cpkt')
        # model_path = os.path.join(model_base_path, 'final_model.h5')
        METRICS = [
            metrics.TruePositives(name='TruePositive'),
            metrics.FalsePositives(name='FalsePositive'),
            metrics.TrueNegatives(name='TrueNegative'),
            metrics.FalseNegatives(name='FalseNegative'),
            metrics.BinaryAccuracy(name='Accuracy'),
            metrics.Precision(name='Precision'),
            metrics.Recall(name='Recall'),
            metrics.AUC(name='AUC', multi_label=True),
        ]

        model = mydensenet(**model_parameters)
        model.load_weights(model_path)
        model.compile(optimizer=optimizers.Adam(), loss=CosineLoss(), metrics=METRICS)
        # model = load_model(model_path, custom_objects={'CosineLoss': CosineLoss})
        _, _, val_generator = return_generators(batch_size=8, return_validation_generators=True, model_key=model_parameters['model_key'])
        truth = []
        prediction = []
        val_iter = val_generator.data_set.as_numpy_iterator()
        for i in range(len(val_generator)):
            print(i)
            x, y = next(val_iter)
            truth.append(np.argmax(y[0]))
            pred = model.predict(x)
            prediction.append(pred)
        evaluation = model.evaluate(val_generator.data_set, steps=len(val_generator))
        prediction = np.squeeze(np.asarray(prediction))
        truth = np.squeeze(np.asarray(truth))
        np.save(file=pred_path, arr=prediction)
        np.save(file=truth_path, arr=truth)

        final_pred = np.asarray([np.argmax(i) for i in prediction])
        truth = np.asarray(truth)
        correct = np.sum(truth == final_pred)
        total = len(truth)
        missed = total - correct
        accuracy = correct/total * 100
        correct_recurred = np.sum(truth[truth == 1] == final_pred[truth == 1]) / np.sum(truth == 1) * 100
        correct_non_recurred = np.sum(truth[truth == 0] == final_pred[truth == 0]) / np.sum(truth == 0) * 100
        print('Guessed {}% correct'.format(accuracy))
        print('Guessed {}% of the recurrence correct'.format(correct_recurred))
        print('Guessed {}% of the non-recurrence correct'.format(correct_non_recurred))
        plot_roc = True
        if plot_roc:
            import matplotlib.pyplot as plt
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, threshold = roc_curve(truth, prediction[:, 1])
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.title('Receiver Operating Characteristic')
            color = 'b'
            plt.plot(fpr, tpr, color, label='AUC {} = %0.2f'.format(model_parameters['model_key']) % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()
            xxx = 1
            # plt.close()
"""
3 blocks in dense
2 dense layers
2 dense convolution blocks
16 filters
16 growth rate
0.5 reduction
256 connections
"""