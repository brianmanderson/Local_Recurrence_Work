__author__ = 'Brian M Anderson'
# Created on 12/16/2020
import pandas as pd
import numpy as np
import os
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnPaths import return_paths


def add_mean_std_across_cv_groups():
    base_path, morfeus_drive, excel_path = return_paths()
    df = pd.read_excel(excel_path, engine='openpyxl')
    compare_list = ('Model_Type', 'min_lr', 'max_lr', 'blocks_in_dense',
                    'dense_conv_blocks', 'dense_layers', 'num_dense_connections', 'filters', 'growth_rate')
    for index in range(df.shape[0]):
        current_run = df.loc[index]
        all_runs = df
        for key in compare_list:
            all_runs = all_runs.loc[all_runs[key] == current_run[key]]
        if all_runs.loc[all_runs.cv_id == -1].shape[0] != 0:
            continue
        unique_cvs = np.unique(all_runs.cv_id.values)
        if len(unique_cvs) != 5:
            continue
        best_loss = []
        best_accuracy = []
        best_run = all_runs
        for cv in unique_cvs:
            cv_runs = all_runs.loc[all_runs.cv_id == cv]
            best_run = cv_runs.loc[cv_runs.epoch_loss == np.min(cv_runs.epoch_loss)]
            best_loss.append(best_run.epoch_loss.values[0])
            best_accuracy.append(best_run.epoch_categorical_accuracy.values[0])
        best_run.at[best_run.index[0], 'cv_id'] = -1
        best_run.at[best_run.index[0], 'Iteration'] = -1
        best_run.at[best_run.index[0], 'epoch_loss'] = np.mean(best_loss)
        best_run.at[best_run.index[0], 'epoch_categorical_accuracy'] = np.mean(best_accuracy)
        best_run.at[best_run.index[0], 'Model_Index'] = np.max(df.Model_Index) + 1
        best_run.at[best_run.index[0], 'epoch_loss_std'] = np.std(best_loss)
        best_run.at[best_run.index[0], 'epoch_categorical_accuracy_std'] = np.std(best_accuracy)
        df = pd.concat([df, best_run], axis=0, ignore_index=True)
        df.to_excel(excel_path, index=0)
    return None


if __name__ == '__main__':
    pass
