__author__ = 'Brian M Anderson'
# Created on 12/8/2020
from Deep_Learning.Base_Deeplearning_Code.Finding_Optimization_Parameters.History_Plotter_TF2 import np, \
    iterate_paths_add_to_dictionary
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnPaths import return_paths
import pandas as pd
import os


def add_metrics_to_excel():
    base_dictionary = {}
    base_path, morfeus_drive = return_paths()
    path_base = os.path.join(morfeus_drive, 'Tensorflow', 'Model_Key_3')
    excel_path = os.path.join(morfeus_drive, 'ModelParameters.xlsx')
    df = pd.read_excel(excel_path)
    not_filled_df = df.loc[pd.isnull(df['epoch_loss'])]
    df.set_index('Model_Index', inplace=True)
    for index in not_filled_df.index.values:
        model_index = not_filled_df['Model_Index'][index]
        path = os.path.join(path_base, 'Model_Index_{}'.format(model_index))
        if not os.path.exists(path):
            continue
        iterate_paths_add_to_dictionary(path=path, all_dictionaries=base_dictionary, fraction_start=0.5,
                                        metric_name_and_criteria={'epoch_loss': np.min,
                                                                  'epoch_categorical_accuracy': np.max})
    out_dictionary = {'Model_Index': [], 'epoch_loss': [], 'epoch_categorical_accuracy': []}
    for key in base_dictionary.keys():
        out_dictionary['Model_Index'].append(int(key.split('_')[-1]))
        out_dictionary['epoch_loss'].append(base_dictionary[key]['epoch_loss'])
        out_dictionary['epoch_categorical_accuracy'].append(base_dictionary[key]['epoch_categorical_accuracy'])
    new_df = pd.DataFrame(out_dictionary)
    new_df.set_index('Model_Index', inplace=True)
    df.update(new_df)
    df = df.reset_index()
    df.to_excel(excel_path, index=0)
    return None


if __name__ == '__main__':
    pass
