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
model_key = 2
print('Running on {} for key {}'.format(gpu, model_key))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')


batch_size = 24
find_lr = False
if find_lr:
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.FindBestLRs import find_best_lr
    find_best_lr(batch_size=batch_size, model_key=model_key)

plot_lr = True
if plot_lr:
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.PlotLRs import plot_lrs
    plot_lrs(input_path=r'K:\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work\Predicting_Recurrence\Learning_Rates')

run_the_2D_model = False
if run_the_2D_model:
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.Run2DModel import run_2d_model
    run_2d_model(batch_size=batch_size, model_key=model_key)

evaluate_model = False
if evaluate_model:
    from tensorflow.keras.models import load_model
    import numpy as np
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnGenerators import return_generators
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnModels import return_model
    model_path = r'H:\Deeplearning_Recurrence_Work\Nifti_Exports\Records\Models\Model_Index_88\cp-best.cpkt'
    model = return_model(model_key=0)
    model.load_weights(model_path)
    _, _, train_generator, val_generator = return_generators(batch_size=12, cross_validation_id=4)
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
    print('Guessed {}% correct'.format(accuracy))
    xxx = 1