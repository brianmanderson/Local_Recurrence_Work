__author__ = 'Brian M Anderson'
# Created on 11/28/2020
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnHparameters import return_list_of_models
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnGenerators import return_generators, return_paths
from Deep_Learning.Base_Deeplearning_Code.Finding_Optimization_Parameters.HyperParameters import return_pandas_df, \
    return_hparams, pd
import os


def run_2d_model(batch_size=24):
    model_dictionary = return_list_of_models()
    model_key = 0
    list_of_models = model_dictionary[model_key]  # A list of models to attempt to run
    base_path, morfeus_drive = return_paths()
    excel_path = os.path.join(morfeus_drive, 'ModelParameters.xlsx')
    base_df = pd.read_excel(excel_path)
    for cv_id in range(5):
        # _, _, train_generator, validation_generator = return_generators(batch_size=batch_size,
        #                                                                 cross_validation_id=cv_id,
        #                                                                 cache=True)
        for iteration in range(3):
            for model_parameters in list_of_models:
                run_df = pd.DataFrame(model_parameters)
                xxx = 1