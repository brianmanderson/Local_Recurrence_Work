__author__ = 'Brian M Anderson'
# Created on 11/28/2020
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnHparameters import return_list_of_models
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnGenerators import return_generators, return_paths
from Deep_Learning.Base_Deeplearning_Code.Finding_Optimization_Parameters.HyperParameters import return_pandas_df, \
    is_df_within_another
import os
import numpy as np
import pandas as pd


def run_2d_model(batch_size=24):
    model_dictionary = return_list_of_models()
    model_key = 0
    list_of_models = model_dictionary[model_key]  # A list of models to attempt to run
    base_path, morfeus_drive = return_paths()
    excel_path = os.path.join(morfeus_drive, 'ModelParameters.xlsx')
    features_list = ('Model_Type', 'min_lr', 'max_lr', 'step_factor', 'Iteration')
    for cv_id in range(5):
        # _, _, train_generator, validation_generator = return_generators(batch_size=batch_size,
        #                                                                 cross_validation_id=cv_id,
        #                                                                 cache=True)
        for iteration in range(3):
            for model_parameters in list_of_models:
                base_df = pd.read_excel(excel_path)
                base_df.set_index('Model_Index')
                model_parameters['Iteration'] = iteration
                current_run_df = pd.DataFrame(model_parameters, index=[0])
                contained = is_df_within_another(data_frame=base_df, current_run_df=current_run_df,
                                                 features_list=features_list)
                if contained:
                    print("Already ran this one")
                    continue
                current_model_indexes = base_df['Model_Index']
                model_index = 0
                while model_index in current_model_indexes:
                    model_index += 1
                current_run_df.insert(0, column='Model_Index', value=model_index)
                current_run_df.set_index('Model_Index')
                base_df = base_df.append(current_run_df)
                base_df.to_excel(excel_path)