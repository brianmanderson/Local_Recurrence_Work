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
model_key = 3
print('Running on {} for key {}'.format(gpu, model_key))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

if os.path.exists(r'K:\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work\Predicting_Recurrence'):
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnPaths import return_paths
    import shutil
    base_path, morfeus_drive, excel_path = return_paths()
    # shutil.copy(os.path.join(morfeus_drive, 'ModelParameters.xlsx'), os.path.join(base_path, 'ModelParameters.xlsx'))
batch_size = 16
find_lr = False
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
    df = pd.read_excel(excel_path)
    not_filled_df = df.loc[pd.isnull(df['min_lr'])]
    for index in not_filled_df.index.values:
        model_index = not_filled_df['Model_Index'][index]
        print(model_index)
        path = os.path.join(morfeus_drive, 'Learning_Rates', 'Model_Key_3', 'Model_Index_{}'.format(model_index))
        if len(os.listdir(path)) >= 2:
            plot_lrs(input_path=path, excel_path=excel_path, add_to_excel=True, base_df=df,
                     save_path=os.path.join(morfeus_drive, 'Learning_Rates', 'Model_Key_3', 'Outputs'))
    added_lr = True

run_the_2D_model = True
if run_the_2D_model and added_lr:
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.Run2DModel import run_2d_model
    run_2d_model(batch_size=batch_size)

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
    import shutil
    from plotnine import *
    import pandas as pd
    import numpy as np
    base_path, morfeus_drive, excel_path = return_paths()
    df = pd.read_excel(excel_path)
    # df = df.dropna()
    df = df[~pd.isnull(df['epoch_loss'])]
    df.epoch_loss = np.where((df.epoch_categorical_accuracy < .51), 1, df.epoch_loss)
    df.epoch_loss = np.where((df.epoch_loss > .6), .6, df.epoch_loss)
    xxx = 1
    (ggplot(df) + aes(x='blocks_in_dense', y='epoch_loss') + facet_wrap('dense_conv_blocks',
                                                                        labeller='label_both') + geom_point(
        mapping=aes(color='epoch_categorical_accuracy')) + xlab('blocks_in_dense') + ylab('Validation Loss') +
     ggtitle('Validation Loss vs Blocks in Dense, and Dense Convolution Blocks') + scale_colour_gradient(low='blue',
                                                                                                         high='red'))

    (ggplot(df) + aes(x='dense_layers', y='epoch_loss') + facet_wrap('blocks_in_dense',
                                                                        labeller='label_both') + geom_point(
        mapping=aes(color='epoch_categorical_accuracy')) + xlab('dense_layers') + ylab('Validation Loss') +
     ggtitle('Validation Loss vs Number of Layers, Number of Conv Blocks, and Conv Lambda') + scale_colour_gradient(low='blue',
                                                                                                      high='red'))

    (ggplot(df) + aes(x='reduction', y='epoch_loss') + facet_wrap('filters', labeller='label_both') + geom_point(
        mapping=aes(color='epoch_categorical_accuracy')) + xlab('reduction') + ylab('Validation Loss') +
     ggtitle('Validation Loss vs Number of Layers, Number of Conv Blocks, and Conv Lambda') + scale_colour_gradient(low='blue',
                                                                                                      high='red'))
    (ggplot(df) + aes(x='num_dense_connections', y='epoch_loss') + geom_point(
        mapping=aes(color='epoch_categorical_accuracy')) + xlab('dense_connections') + ylab('Validation Loss') +
     ggtitle('Validation Loss vs Number of Layers, Number of Conv Blocks, and Conv Lambda') + scale_colour_gradient(low='blue',
                                                                                                      high='red'))
    (ggplot(df) + aes(x='blocks_in_dense', y='epoch_loss') + geom_point(
        mapping=aes(color='epoch_categorical_accuracy')) + xlab('blocks_in_dense') + ylab('Validation Loss') +
     ggtitle('Validation Loss vs Number of Layers, Number of Conv Blocks, and Conv Lambda') + scale_colour_gradient(low='blue',
                                                                                                      high='red'))

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

    aucs = []
    model_base_path = r'H:\Deeplearning_Recurrence_Work\Models'
    for cv_id in range(4):
        print('Running for cv_id: {}'.format(cv_id))
        pred_path = os.path.join(model_base_path, 'cv_id_{}'.format(cv_id), 'Predictions.npy')
        truth_path = os.path.join(model_base_path, 'cv_id_{}'.format(cv_id), 'Truth.npy')
        model_path = os.path.join(model_base_path, 'cv_id_{}'.format(cv_id), 'cp-best.cpkt')
        if not os.path.exists(pred_path):
            model = mydensenet(blocks_in_dense=3, dense_layers=2, dense_conv_blocks=2,
                               filters=16, growth_rate=16, reduction=0.5, num_dense_connections=256)
            model.load_weights(model_path)
            # model = load_model(model_path, custom_objects={'CosineLoss': CosineLoss})
            _, _, val_recur_generator, val_non_recur_generator = return_generators(cross_validation_id=cv_id,
                                                                                   model_key=model_key,
                                                                                   return_validation_generators=True)
            truth = []
            prediction = []
            for val_generator in [val_recur_generator, val_non_recur_generator]:
                val_iter = iter(val_generator.data_set)
                for i in range(len(val_generator)):
                    # print(i)
                    x, y = next(val_iter)
                    truth.append(np.argmax(y[0].numpy()))
                    pred = model.predict(x)
                    prediction.append(pred)
            np.save(file=pred_path, arr=np.squeeze(np.asarray(prediction)))
            np.save(file=truth_path, arr=np.squeeze(np.asarray(truth)))
        else:
            prediction, truth = np.load(pred_path), np.load(truth_path)
            # break
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
        recurred_truth = truth[0::2]
        xxx = 1
        plot_roc = True
        if plot_roc:
            import matplotlib.pyplot as plt
            import sklearn.metrics as metrics
            fpr, tpr, threshold = metrics.roc_curve(truth, prediction[:, 1])
            roc_auc = metrics.auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.title('Receiver Operating Characteristic')
            plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()
            xxx = 1
            plt.close()
"""
3 blocks in dense
2 dense layers
2 dense convolution blocks
16 filters
16 growth rate
0.5 reduction
256 connections
"""