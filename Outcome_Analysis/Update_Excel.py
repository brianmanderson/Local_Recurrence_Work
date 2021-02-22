__author__ = 'Brian M Anderson'
# Created on 2/22/2021
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnPaths import return_paths
import pandas as pd
import os


base_path, morfeus_drive, excel_path = return_paths()
df = pd.read_excel(excel_path, engine='openpyxl')
tensorflow_path = os.path.join(morfeus_drive, 'Tensorflow')
for model_path in os.listdir(tensorflow_path):
    model_key = int(model_path.split('Model_Key_')[-1])
    for model_folder in os.listdir(os.path.join(tensorflow_path, model_path)):
        model_index = int(model_folder.split('_')[-1])
        potentially_not_run = df.loc[(df['Model_Index'] == model_index) & (df['Model_Type'] == model_key)]
        if potentially_not_run.shape[0] == 0:
            new_model_index = 0
            while new_model_index in df['Model_Index'].values:
                new_model_index += 1
            new_dict = {'Model_Index': [new_model_index], 'Model_Type': [model_key]}
            os.rename(os.path.join(model_path, model_folder), os.path.join(model_path,
                                                                           'Model_Index_{}'.format(new_model_index)))
            xxx = 1