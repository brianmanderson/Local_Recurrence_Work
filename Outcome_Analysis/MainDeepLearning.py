__author__ = 'Brian M Anderson'
# Created on 11/18/2020
'''
All parts here are in the DeepLearningTools folder
'''
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))
if len(sys.argv) > 1:
    gpu = int(sys.argv[1])
else:
    gpu = 0
print('Running on {}'.format(gpu))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

find_lr = True
if find_lr:
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.FindBestLRs import find_best_lr
    find_best_lr(batch_size=24, model_type='2D_Vanilla')

plot_lr = False
if plot_lr:
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.PlotLRs import plot_lrs
    plot_lrs(input_path=r'K:\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work\Predicting_Recurrence\Learning_Rates')

workondeeplearning = False
if workondeeplearning:
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnGenerators import return_generators
    base_path, morfeus_drive, train_generator, validation_generator = return_generators()
    xxx = 1