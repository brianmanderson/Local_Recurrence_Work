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
model_key = 0
print('Running on {} for key {}'.format(gpu, model_key))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

if os.path.exists(r'K:\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work\Predicting_Recurrence'):
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnPaths import return_paths
    import shutil
    base_path, morfeus_drive = return_paths()
    # shutil.copy(os.path.join(morfeus_drive, 'ModelParameters.xlsx'), os.path.join(base_path, 'ModelParameters.xlsx'))

batch_size = 16
find_lr = True
finished_lr = False
if find_lr:
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.FindBestLRs import find_best_lr
    finished_lr = find_best_lr(batch_size=batch_size, model_key=model_key)

add_lr = True
added_lr = False
if add_lr and finished_lr:
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.PlotLRs import plot_lrs
    added_lr = plot_lrs(input_path=r'K:\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work\Predicting_Recurrence\Learning_Rates\Model_Key_0')

run_the_2D_model = False
if run_the_2D_model and added_lr:
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.Run2DModel import run_2d_model
    run_2d_model(batch_size=batch_size, model_key=model_key)

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
    base_path, morfeus_drive = return_paths()
    excel_path = os.path.join(morfeus_drive, 'Reduced_Parameters.xlsx')
    df = pd.read_excel(excel_path)
    df = df.dropna()
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
        mapping=aes(color='dense_layers')) + xlab('blocks_in_dense') + ylab('Validation Loss') +
     ggtitle('Validation Loss vs Number of Layers, Number of Conv Blocks, and Conv Lambda') + scale_colour_gradient(low='blue',
                                                                                                      high='red'))

    (ggplot(data) + aes(x='layers', y='log_epoch_loss') + facet_wrap('conv_lambda', labeller='label_both') + geom_point(
        mapping=aes(color='num_conv_blocks')) + xlab('Layers') + ylab('Validation Loss') +
     ggtitle('Validation Loss vs Number of Layers, Filters, and Max Filters') + scale_colour_gradient(low='blue',
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

evaluate_model = False
if evaluate_model:
    from tensorflow.keras.models import load_model
    import numpy as np
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnGenerators import return_generators
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnModels import return_model
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnCosineLoss import CosineLoss
    model_path = r'H:\Deeplearning_Recurrence_Work\Nifti_Exports\Records\Models\Model_Index_140\final_model.h5'
    model = load_model(model_path, custom_objects={'CosineLoss': CosineLoss})
    # model = return_model(model_key=0)
    # model.load_weights(model_path)
    _, _, train_generator, val_generator = return_generators(batch_size=8, cross_validation_id=0, model_key=model_key)
    val_iter = iter(val_generator.data_set)
    truth = []
    prediction = []
    for i in range(len(val_generator)):
        print(i)
        x, y = next(val_iter)
        truth.append(np.argmax(y[0].numpy()))
        pred = model.predict(x)
        prediction.append(pred)
    final_pred = np.asarray([np.argmax(i) for i in prediction])
    truth = np.asarray(truth)
    correct = np.sum(truth == final_pred)
    total = len(truth)
    missed = total - correct
    accuracy = correct/total * 100
    correct_recurred = np.sum(truth[::2] == final_pred[::2]) / len(truth[::2]) * 100
    correct_non_recurred = np.sum(truth[1::2] == final_pred[1::2]) / len(truth[1::2]) * 100
    print('Guessed {}% correct'.format(accuracy))
    print('Guessed {}% of the recurrence correct'.format(correct_recurred))
    print('Guessed {}% of the non-recurrence correct'.format(correct_non_recurred))
    recurred_truth = truth[0::2]
    xxx = 1