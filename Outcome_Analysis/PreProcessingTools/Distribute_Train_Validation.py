__author__ = 'Brian M Anderson'
# Created on 1/26/2021
import pandas as pd
import numpy as np
import os


def distribute_train_validation(nifti_export_path, excel_path):
    all_files = os.listdir(nifti_export_path)
    validation_path = os.path.join(nifti_export_path, 'Validation')
    train_path = os.path.join(nifti_export_path, 'Train')
    for path in [train_path, validation_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    df = pd.read_excel(excel_path, engine='openpyxl')
    unfilled = df.loc[(pd.isnull(df['Non_Recurrence_Sites'])) | (pd.isnull(df['Recurrence_Sites']))]
    assert unfilled.shape[0] == 0, 'Run Write_Sites_To_Sheet.py, should not have a null value here'
    '''
    First, we will split up the recurrence, as they are fewer... Then distribute the non_recurrence to whats left
    '''
    total_recurrence_in_validation = 0
    total_non_recurrence_in_validation = 0
    total_recurrence_in_train = 0
    total_non_recurrence_in_train = 0

    '''
    Check and see if patients have already been moved, and if so, add them to the count
    '''
    rewrite = False
    validation_already_moved = [i.split('_')[0] for i in os.listdir(validation_path) if i.endswith('Primary_Dicom.nii')]
    for val_index in validation_already_moved:
        moved = df.loc[df['PatientID'] == int(val_index)]
        index = moved.index[0]
        if pd.isnull(df['Folder'][index]):
            rewrite = True
            df.at[index, 'Folder'] = 0
        total_recurrence_in_validation += np.sum(moved['Recurrence_Sites'].values)
        total_non_recurrence_in_validation += np.sum(moved['Non_Recurrence_Sites'].values)

    train_already_moved = [i.split('_')[0] for i in os.listdir(train_path) if i.endswith('Primary_Dicom.nii')]
    for val_index in train_already_moved:
        moved = df.loc[df['PatientID'] == int(val_index)]
        index = moved.index[0]
        if pd.isnull(df['Folder'][index]):
            rewrite = True
            df.at[index, 'Folder'] = 1
        total_recurrence_in_train += np.sum(moved['Recurrence_Sites'].values)
        total_non_recurrence_in_train += np.sum(moved['Non_Recurrence_Sites'].values)
    if rewrite:
        df.to_excel(excel_path, index=0)

    '''
    Since we want to keep the patients together based on MRN, we will sort and distribute based on MRN
    '''
    recurrence_df = df.loc[df['Recurrence_Sites'] > 0]
    total_recurrence = np.sum(recurrence_df['Recurrence_Sites'].values)
    split_for_validation = total_recurrence // 3

    recurrence_MRNs = np.unique(recurrence_df['MRN'].values)
    np.random.shuffle(recurrence_MRNs)
    for MRN in recurrence_MRNs:
        indexes = df.loc[df['MRN'] == MRN].index
        if total_recurrence_in_validation < split_for_validation:
            for index in indexes:
                patient_id = df['PatientID'][index]
                total_recurrence_in_validation += df['Recurrence_Sites'][index]
                total_non_recurrence_in_validation += df['Non_Recurrence_Sites'][index]
                files = [i for i in all_files if i.split('_')[0] == str(patient_id)]
                for file in files:
                    os.rename(os.path.join(nifti_export_path, file), os.path.join(validation_path, file))
        else:
            for index in indexes:
                patient_id = df['PatientID'][index]
                total_recurrence_in_train += df['Recurrence_Sites'][index]
                total_non_recurrence_in_train += df['Non_Recurrence_Sites'][index]
                files = [i for i in all_files if i.split('_')[0] == str(patient_id)]
                for file in files:
                    os.rename(os.path.join(nifti_export_path, file), os.path.join(train_path, file))

    non_recurrence_df = df.loc[df['Recurrence_Sites'] == 0]
    total_non_recurrence = np.sum(non_recurrence_df['Non_Recurrence_Sites'].values)
    split_for_validation = total_non_recurrence // 3
    MRNs = np.unique(non_recurrence_df['MRN'].values)
    np.random.shuffle(MRNs)
    for MRN in MRNs:
        if MRN in recurrence_MRNs:
            continue
        indexes = df.loc[df['MRN'] == MRN].index
        if total_non_recurrence_in_validation < split_for_validation:
            for index in indexes:
                patient_id = df['PatientID'][index]
                total_recurrence_in_validation += df['Recurrence_Sites'][index]
                total_non_recurrence_in_validation += df['Non_Recurrence_Sites'][index]
                files = [i for i in all_files if i.split('_')[0] == str(patient_id)]
                for file in files:
                    os.rename(os.path.join(nifti_export_path, file), os.path.join(validation_path, file))
        else:
            for index in indexes:
                patient_id = df['PatientID'][index]
                total_recurrence_in_train += df['Recurrence_Sites'][index]
                total_non_recurrence_in_train += df['Non_Recurrence_Sites'][index]
                files = [i for i in all_files if i.split('_')[0] == str(patient_id)]
                for file in files:
                    os.rename(os.path.join(nifti_export_path, file), os.path.join(train_path, file))
    MRNs = np.unique(df['MRN'].values)
    for MRN in MRNs:
        folders = df.loc[df['MRN'] == MRN]['Folder']
        assert np.max(folders.values) == np.min(folders.values), 'Should not distribute a patient across train and ' \
                                                                 'validation! {} failed this check'.format(MRN)
    return None


if __name__ == '__main__':
    pass
